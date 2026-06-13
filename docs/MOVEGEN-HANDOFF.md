# Movegen + core handoff

**`main` @ `9302db0`** — movegen closed, §A (Zobrist/Undo) merged.  
**Branch `movgen-improvements`** — same as `main` after fast-forward.

---

## Done

| Item | Status |
| ---- | ------ |
| Shift L2 / TOPO wall masks | ✓ production |
| Lazy L3, `wall_masks`, split loops | ✓ |
| Perft bulk d1, gates exact | ✓ |
| §A const Zobrist, fused deltas, slim `Undo` | ✓ merged `main` |
| §B pawn default | ✓ **`O1Lookup`** — perft-proven fastest (see MOVEGEN.md) |
| Movegen multithread / GPU | ✗ policy: never |

### Gates

```text
perft 3 = 2_062_264
perft 4 = 247_569_030
perft 5 = 28_837_934_502   (sub-12s, public record)
perft 6 = 3_257_436_276_501
cargo test --release → all pass
titanium bench 3 20 → ~210–240M nps (honest)
```

---

## O1 pawn — PRODUCTION DEFAULT (was research-only; promoted)

**Superseded note:** O1 was previously kept research-only. It is now the default
because it is decisively faster *and* verified correct — do not be misled by any
lingering "research only" text elsewhere.

- `generate_legal_moves_slice` uses `PawnGenMode::default()` → **`O1Lookup`**.
- Fastest at perft(4) in both plain and BMI2/PEXT builds; correct vs the oracle
  at d3/d4 and cross-verified at d5/d6.
- Tables (~2MB) are a fixed offline artifact; regenerate / cold-start-generate
  with `cargo run --bin movegen-o1-gen`.
- `ShiftCanStep` / `Scalar` remain only as portable bench/test baselines.

---

## Next work (Fable or Cursor)

### 1. L3 flood fraction in **search** (not perft)

Profile wall-heavy replay positions — is §C incremental L3 worth a proof harness?

### 2. §C incremental L3

**Blocked** until harness spec + tests vs scalar flood (BUG-DIARY `a1h`/`a5h`).

### 3. §D completeness oracle

Batch exact solve + invariant hash collisions — research track.

### 4. Eval / search (STATE.md)

Distance cache, opening depth, Game A/B replays.

---

## Do not redo

Movegen tables, topo tables, movegen threads, O1 as default without in-search proof.
