# Episode 04 — Benchmarks + JS cross-check

- **branch:** `checkpoint/04-bench`
- **commit:** `90193b0`
- **tag:** `checkpoint-04-bench`

## Hook

"A faster chess engine you never turn up to 11 is just a hobby. We bench every hot loop."

## What we build

- `engine/benches/path_bfs.rs` — criterion benches for BFS, legal moves, perft d2
- `benchmark/compare_moves.mjs` — Rust vs JS move set diff
- CLI `titanium bench <depth> <iterations>`

## Demo

```bash
cd engine
cargo bench
cargo run --release -- bench 2 20
node ../benchmark/compare_moves.mjs
```

## Numbers to show on screen

- Legal moves at startpos (should match JS)
- Nodes/sec at perft depth 2
- BFS both-players check latency

## Next episode teaser

"Alpha-beta + Zobrist TT — Phase 1 of the hybrid engine."
