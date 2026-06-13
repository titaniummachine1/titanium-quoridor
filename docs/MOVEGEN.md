# Move generation — production architecture

**Status:** Closed for playing engine (Jun 2026).  
**Policy:** Single-thread hot path only — no movegen multithreading, no GPU.

## Wall pipeline

```text
L1  empty slot          !horizontal_walls / !vertical_walls
L2  collision           whole-board shifts (overlap / cross / neighbor)
TOPO flood-skip         two-of-three anchor shifts (= scraped canWallBlock)
L3  path legality       parallel u128 flood + bit theft (lazy WallTrialCtx)
```

| Layer | File | Function |
| ----- | ---- | -------- |
| L1∧L2∧TOPO masks | `movegen/o1/lookup.rs` | `wall_masks(board)` |
| Wall emit | `movegen/legal.rs` | `collect_wall_orientation` (isolated → flood) |
| L3 flood | `path/parallel.rs` | `both_players_reach_goals_grids` |

**Walls use shift algebra only** — no runtime wall tables (tried; rejected as fake or unsound).

### L3 flood: step dilation vs Kogge-Stone occluded fill (rejected)

The L3 flood (`path/parallel.rs`) advances the frontier **one square per iteration**
(`expand_wave`: 4 gated shifts + a playable mask, early-exit on goal contact). An
obvious idea is the **Kogge-Stone occluded fill** — smear the frontier along a whole
open run per step in `O(log w)` shifts, so the loop runs once per *turn* in the path
instead of once per unit of *length*. Implemented and verified
(`*_ks` fns + `pext_…`-class build), but **rejected — measurably slower:**

| Measurement | step `expand_wave` | KS occluded fill |
| ----------- | -----------------: | ---------------: |
| flood micro-bench (`flood_modes`, 600-pos corpus) | **1.00×** | 1.08× slower |
| full perft(4) hot path, `o1_full_lut` row (native) | **1.039s** | 1.212s (+17%) |

Why KS loses for Quoridor specifically:

- Quoridor floods are **short and turn-heavy** (pawn is often a few rows from its
  goal; corridors zig-zag). KS only wins on long *straight* runs — it pays ~3× the
  ops per step to fill a whole ray, which is wasted when the goal is 1–2 steps away.
- `expand_wave`'s **per-ring early-exit** fires the instant the wave touches the goal
  row; KS overshoots (fills the whole ray) before it can check.
- Even with the propagators precomputed once per flood (they're constant — see
  `KsProp`) and the shift-8 round dropped, KS stayed slower.

The `*_ks` functions are kept as a **verified-correct reference + negative-result
record** (used only by `flood_modes` bench and the `random_walls_match_naive_reference`
test), not by production. No anti-wrap file masks are needed: the propagator is zero
on the buffer ring, and `p &= p<<s` propagates that zero, so E/W runs can't wrap rows
— this is why KS is *correct* here even though the POC's earlier subtraction-based ray
sweep was not (see `parallel.rs` header).

Run it: `cargo bench --bench flood_modes` (add `RUSTFLAGS="-C target-cpu=native"`).

## Pawns — production vs alternatives

| Mode | Used in play? | How |
| ---- | ------------- | --- |
| **`O1Lookup`** | **Yes — production default** | `PawnGenMode::default()` in `legal.rs`; search, CLI, perft. Fastest at perft d4/d5. |
| **`ShiftCanStep`** | No — alternative | a few bit shifts + `can_step`; bench/test baseline, no table/BMI2 dependency |
| **`O1LeanLut`** | **No — rejected experiment** | hybrid: `ShiftCanStep` for `enemy_key==0`, table otherwise |

### O1 pawn lookup — how it works

Offline `PAWN_LEGAL[sq][enemy_key][wall_key] -> u16` legal-destination bitmask
(`generated_tables_data.rs`, ~1.6MB `generated_remap.bin`). At runtime:

1. `enemy_key = encode_enemy_key(board, side, sq)` — which adjacent square (if any)
   holds the enemy pawn (jump/lateral special cases). `0` ⇒ no adjacent enemy.
2. `wall_key = pack_wall_key(board, sq, enemy_key)` — packs the handful of wall
   slots that can actually change this square's legal moves into a small index.
3. `PAWN_LEGAL[sq][enemy_key][wall_key]` → bitmask, iterate set bits through
   `PAWN_CATALOG[sq]` to emit destination squares.

#### `pack_wall_key`: PEXT packing (the win)

The original packer looped ~12 wall slots × 3 table lookups each — that overhead
made O1 *slower* than `ShiftCanStep`. Replaced with a two-instruction **PEXT**
extraction (`_pext_u64`, BMI2):

```rust
let h_bits = _pext_u64(board.horizontal_walls, PAWN_H_PEXT_MASK[sq][ek]);
let v_bits = _pext_u64(board.vertical_walls,   PAWN_V_PEXT_MASK[sq][ek]);
let phys   = h_bits | (v_bits << PAWN_H_SLOT_COUNT[sq][ek]);
wall_remap_byte(sq, ek, phys)        // physical combo → semantic wall_key
```

- `PAWN_H_PEXT_MASK` / `PAWN_V_PEXT_MASK` / `PAWN_H_SLOT_COUNT` are emitted by the
  generator (`build/movegen_o1/emit.rs`).
- The masks are **tight**: only wall slots that physically change a legal move are
  set, so an irrelevant wall in the neighborhood never causes a pattern miss.
- **H-first ordering invariant:** the generator sorts `wall_bits` H-before-V
  (`build/movegen_o1/pawn.rs`: `sort_by_key(|&(r,c,h)| (!h, r, c))`) so the remap
  table's bit order matches PEXT extraction order (H mask first, then V shifted up
  by `PAWN_H_SLOT_COUNT`). Get this wrong and the wall_key indexes the wrong row.
- A `#[cfg(...bmi2)]` scalar fallback (`pack_wall_key_scalar`) preserves the old
  loop for non-BMI2 builds; a test asserts the two agree.

**Build caveat:** the PEXT path only compiles in with BMI2 enabled
(`RUSTFLAGS="-C target-cpu=native"` or `-C target-feature=+bmi2`). A plain
`cargo build`/`cargo bench` on `x86_64-pc-windows-msvc` compiles the **scalar**
packer (you'll see "never used" warnings on the PEXT mask constants). O1 still
wins via the scalar packer, but PEXT roughly doubles the pawn-only throughput
(see below) — so the playing engine should be built with `target-cpu=native`.

### O1 is the production default (decision taken)

`PawnGenMode::default()` in `legal.rs` is **`O1Lookup`**. The earlier "research
only" status is **superseded** — O1 is decisively fastest and verified correct:

- perft(4) frontier (this machine, no TT): O1 full LUT **0.920s** (native /
  BMI2) and **0.986s** (plain) — fastest in *both* builds, ahead of scalar
  (0.977 / 1.018) and shift (1.004 / 1.054). All modes return the oracle count.
- perft(5) cross-verified: **28,837,934,502** (sub-12s, a public record);
  perft(6): **3,257,436,276,501**. See `PERFT5_STARTPOS`/`PERFT6_STARTPOS`.

The ~1.6MB remap + table is a fixed ~2MB footprint (not 400MB) and the LUT wins
even inside full search. `ShiftCanStep` is retained only as a portable
bench/test baseline (no table, no BMI2 dependency). If you must ship without the
tables, generate them at startup (`cargo run --bin movegen-o1-gen` produces
`generated_*`) — a cold-start option, not a reason to keep shift as default.

### O1LeanLut — rejected experiment

Hypothesis: skip the table when no enemy is adjacent (`enemy_key==0`) and use the
faster shift path there, reserving the table for jump/lateral cases. **Disproved.**
`encode_enemy_key` must run regardless, so the lean path pays that cost *plus* the
full `ShiftCanStep` cost in the common `ek==0` case — strictly worse than either
pure path (see table: lean is ~1.15–2.3× the full LUT). The full LUT is fastest
*because* PEXT + remap + table is cheaper than 4× `can_step` even when there's no
enemy. Kept only as a labelled bench mode for the record.

**O1 is the default and fully wired.** Supporting pieces:

- Table correctness tests (`movegen::o1::lookup`), incl. `pext_pack_wall_key_matches_scalar`
- `cargo run --bin movegen-o1-gen` (regenerate tables / cold-start generation)
- The generator also emits the PEXT mask constants (`build/movegen_o1/emit.rs`)

To force a non-default pawn mode (bench/test): `perft_no_tt_mode(..., PawnGenMode::ShiftCanStep)`.

## Performance (i7-4900MQ, release, 1 thread, Jun 2026)

| Command | Result |
| ------- | ------ |
| `titanium perft 3` | **2_062_264** nodes |
| `titanium perft 4` | **247_569_030** nodes (~0.30–0.35s CLI, TT + bulk d1) |
| `titanium bench 3 20` | ~**210–240M nps** (honest: movegen + make/unmake + Zobrist) |

Perft CLI time ≠ bench nps (bulk depth-1 + TT).

### Pawn-only bench (`cargo bench --bench perft_pawn_only`, depth 12, no walls/TT)

Isolates pawn-gen cost. All modes match **2_890_001** nodes. Jun 2026 run:

| Mode | default build (no BMI2) | `target-cpu=native` (PEXT) |
| ---- | ----------------------: | -------------------------: |
| `o1_full_lut` | **0.050s (58M nps, 1.00×)** | **0.028s (103M nps, 1.00×)** |
| `shift_can_step` (default) | 0.083s (35M nps, 1.67×) | 0.065s (44M nps, 2.31×) |
| `o1_lean_lut` | 0.086s (34M nps, 1.72×) | 0.064s (45M nps, 2.27×) |

PEXT roughly doubles O1 throughput (58→103M nps) and widens the gap to 2.3× over shift.

### Pawn mode bench (`cargo bench --bench perft_pawn_modes`, perft 4, no TT)

Full legal perft (wall legality dominates). All modes match **247_569_030** nodes.
`target-cpu=native` (PEXT), Jun 2026:

| Mode | seconds | vs fastest |
| ---- | ------: | ---------: |
| `o1_full_lut` | **0.984** | **1.00×** |
| `o1_lean_lut` | 1.129 | 1.15× |
| `scalar_can_step` | 1.174 | 1.19× |
| `shift_bit_can_step` (default) | 1.284 | 1.30× |
| `bitboard_*` | 2.83–2.85 | ~2.9× |

**Decision:** `ShiftCanStep` stays default *for now* (cache footprint + no BMI2
dependency), but O1 full LUT is now the measured winner at every depth. Flipping
the default for BMI2 builds is a documented, pending one-liner — see the pawn
section above.

## Offline pawn tables (research)

```bash
cargo run --release --bin movegen-o1-gen
```

Output: `src/movegen/o1/generated_tables_data.rs`, `generated_remap.bin`. Not required for `cargo build` if files are committed.

## Regression

```bash
cargo test --release
cargo test --release movegen::o1::lookup
cargo test --release perft_depth4_matches_oracle -- --ignored --nocapture
cargo run --release --bin titanium -- bench 3 20
```

## Next work

See `docs/MOVEGEN-HANDOFF.md` — L3 profiling in search, completeness oracle (not movegen).
