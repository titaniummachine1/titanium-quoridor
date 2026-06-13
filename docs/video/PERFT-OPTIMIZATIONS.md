# Perft optimization log — Titanium fundamentals

Chronological discoveries while making perft **correct, light, and fast** before αβ search.

**Gate:** depth 3 = **2_062_264** nodes (never trade correctness for speed).

---

## Baseline (naive Rust)

| Technique               | Depth 3 | Depth 4        |
| ----------------------- | ------- | -------------- |
| `Board::clone` per node | ~0.21s  | minutes / hung |

**Discovery:** Quoridor root branching ≈ **131**, not chess ~20. Depth 4 ≈ **250M nodes**. Language choice does not beat exponent.

---

## Layer 1 — Tree walk (Stockfish pattern)

| Change                                | Why                                         |
| ------------------------------------- | ------------------------------------------- |
| `make_move` / `unmake_move` + `Undo`  | Stop cloning 9×9 state every node           |
| Incremental Zobrist xor on make       | O(1) hash for TT                            |
| Perft TT `(hash, depth) → node count` | Memoize identical subtrees                  |
| `mem::take` move buffer               | Child recursion was clearing parent's `Vec` |

**Result:** depth 3 ~0.16s, depth 4 ~9s (first fast path).

**Discovery #A:** Shared `Vec` move buffer + recursion = index panic. **Fix:** snapshot moves per node (`mem::take`).

---

## Layer 2 — Move gen hot path

| Change                               | Why                                           |
| ------------------------------------ | --------------------------------------------- |
| `generate_legal_moves_into`          | Reuse buffer, no alloc in loop                |
| Wall trial: `set_wall` → BFS → unset | Was `board.clone()` per candidate wall        |
| `BfsScratch` in `PerftContext`       | Reuse queue + `u128` visited (81 cells)       |
| Reachability BFS drops depth array   | Boolean path check only needs visited + queue |
| Short-circuit dual BFS               | Fail fast if P1 blocked                       |

**Result:** depth 3 ~0.14s, depth 4 ~7s.

**Discovery #B:** **Wall legality dominates**, not tree walk. Perft spends most time in BFS inside wall loops, not in `make_move`.

**Discovery #C:** Floating walls (topology false) skip BFS entirely — already required for correctness (bug #1 in BUG-DIARY).

---

## Layer 3 — Zero-alloc perft node (this pass)

| Change                                      | Why                                                |
| ------------------------------------------- | -------------------------------------------------- |
| `MAX_LEGAL_MOVES = 140` stack buffer        | Eliminate `mem::take` heap alloc **per tree node** |
| `generate_legal_moves_slice`                | Write into `&mut [Move]`, return count             |
| Wall slots: `trailing_zeros` on `!bitboard` | Skip occupied wall bits; fewer iterations midgame  |
| TT **clusters** (4 slots/bucket)            | Stockfish pattern — fewer collisions at depth 4    |
| `lto = "fat"`, `codegen-units = 1`          | Free rustc inlining across crates                  |

**Measured (release, this machine):**

| Depth | Before layer 3 | After layer 3         |
| ----- | -------------- | --------------------- |
| 3     | ~0.14s         | **~0.10s** (~20M nps) |
| 4     | ~7s            | **~6s** (~41M nps)    |

**Discovery #D:** Eliminating **heap alloc per node** (`mem::take`) mattered as much as wall-trial clone removal at depth 4. Stack `[Move; 140]` is the Stockfish move-list pattern.

---

## Layer 4 — Bitwise flood fill + centered u128 layout

| Change                                                 | Why                                                                              |
| ------------------------------------------------------ | -------------------------------------------------------------------------------- |
| `DirMasks` (N/S/E/W `u128`) built once per wall trial  | Replace per-edge `can_step` in BFS loop                                          |
| Flood expand via shifts (`>>11`, `<<11`, `<<1`, `>>1`) | Word-parallel frontier (Ishtar/Canta style)                                      |
| **Centered 11×11 stride** in `u128` (`grid.rs`)        | 9×9 playable + **side buffer columns** — east/west shifts never wrap across rows |
| Ishtar component reuse                                 | If P2 pawn ∈ P1 flood mask, skip second full flood                               |
| Known-path wall skip                                   | If a wall misses one current path for each player, both paths survive            |
| `pack_flood_mask` at API boundary                      | Internal flood bits → compact 81-bit game mask for search                        |

**Layout:** playable `(row, col)` → bit `(row+1)*11 + (col+1)`. Max bit index **108** (fits in `u128` with headroom).

**Oracles (startpos, release, 1 thread):**

| Depth | Nodes             | Time (this machine)   |
| ----- | ----------------- | --------------------- |
| 1     | 131               | —                     |
| 2     | 16,677            | —                     |
| 3     | **2,062,264**     | **~0.06s**            |
| 4     | **247,569,030**   | **~3.1s**             |
| 5     | 28,837,934,502    | ~18s (Ishtar ref)     |
| 6     | 3,257,436,276,501 | ~691 min (Ishtar ref) |

Depth 4 matches Ishtar/Canta oracle. Regression: `PERFT4_STARTPOS` test (`cargo test --release perft_depth4 -- --ignored`).

**Discovery #E:** Wall-legality BFS dominated perft time. Queue BFS at stride-9 risked **row wrap** on `<<1`/`>>1`; centered stride-11 lets hardware shifts stay dumb — no per-expand boundary paranoia.

**Before → after (depth 4, same machine):** ~6–9s → **~3.1s** (~2×) with correct node count.

---

## Layer 5 — Pawn `ShiftCanStep` + timed regression (Jun 2026)

| Change | Why |
| ------ | --- |
| `PawnGenMode::ShiftCanStep` default *(superseded — `O1Lookup` is now the default, perft-proven faster)* | No `DirMasks` table per pawn node — blind shift + `can_step` wall check |
| Smart perft test `perft_depth4_matches_oracle` | Depths 1→4 sequential, per-depth timeout, core pinning, `exit(1)` on hang |
| `core_affinity` dev-dep | Pin worker to P-core (not last logical E-core on hybrid CPUs) |

**Measured:** d4 **~3.2–3.4s** on idle CPU; timed test passes in ~3.5s wall.

---

## Layer 5b — Incremental `DirMasks` on `Board` (REJECTED)

**Idea:** Patch masks in `set_wall`; read `board.dir_masks` in BFS instead of scratch `from_board`.

**Result:** perft d4 **~3s → ~20–40s** (~12× regression).

**Why:** Perft explores every wall edge in the tree — patching on each wall make/unmake costs more than one `DirMasks::from_board` per movegen node. The existing **`BfsScratch` hash-keyed cache** is the correct tradeoff.

**Lesson:** Incremental topology on `Board` may help search (fewer wall plies after pruning), not correctness perft.

---

## What we deliberately did NOT do (yet)

| Idea                                      | Why wait                                                      |
| ----------------------------------------- | ------------------------------------------------------------- |
| Probable-wall pruning                     | Breaks full perft legality — for **search** only (gorisanson) |
| Parallel perft                            | Correctness/debug pain; search parallelism comes later        |
| `Move` as `u16` packing                   | Nice, but not the bottleneck                                  |
| Incremental `DirMasks` on `Board`         | **Tried — regressed perft ~12×** (see Layer 5b)               |
| Reachability cache on `Board`             | Invalidation complexity; revisit for **eval in search**       |
| Skip wall gen when `walls_remaining == 0` | Already done                                                  |

---

## Architecture (current)

```
perft_fast_ctx
  ├─ TT probe (Zobrist hash, depth)
  ├─ generate_legal_moves_slice → stack [Move; 140]
  │    ├─ pawn moves (≤8)
  │    └─ wall moves: iterate empty bitboard slots
  │         ├─ collision / topology (cheap)
  │         └─ in-place wall trial + bitwise flood (centered u128)
  └─ for each move: make_move → recurse → unmake_move
```

---

## Commands

```bash
cd engine
cargo test
cargo run --release -- perft 3
cargo run --release -- perft 4      # stress only
cargo run --release -- bench 3 20
node ../benchmark/perft_triple.mjs
```

---

## Video beats

1. "Perft 3 is our Stockfish depth 6 — two million nodes, three plies."
2. "Rust didn't fail us — **cloning** failed us."
3. "Wall checks were cloning the board **inside** move generation. That's where the seconds went."
4. "Fundamentals first: when we turn on αβ, we don't rewrite move gen."
