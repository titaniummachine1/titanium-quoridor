# Episode 03 — Perft (divide) harness

- **branch:** `checkpoint/03-perft`
- **commit:** `5a4b0fc`
- **tag:** `checkpoint-03-perft`

## Hook

"Stockfish has perft. We don't have a standard Quoridor table — so we build our own divide and grow depth until we're confident."

## What we build

- `engine/src/perft.rs` — recursive node counter + divide output
- CLI: `titanium perft`, `titanium divide`

## Demo

```bash
cd engine
cargo run --release -- perft 1
cargo run --release -- divide 2
```

Explain: depth 3+ explodes because of walls — use depth 1–2 for CI, higher for bench only.

## Talking points

- Perft catches duplicate moves, illegal walls, broken jumps
- Notation: `e2` pawn, `d2h` horizontal wall at d2

## Next episode

"Cargo bench — proving Rust speed isn't theoretical."
