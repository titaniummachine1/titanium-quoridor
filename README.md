# Titanium Quoridor Engine

Rust search engine for [Quoridor](https://en.wikipedia.org/wiki/Quoridor): iterative-deepening αβ, CAT corridor pruning, ACE v11 eval, UCI protocol, and WASM bindings.

Repo: [github.com/titaniummachine1/titanium-quoridor](https://github.com/titaniummachine1/titanium-quoridor)

Related repos:

| Repo                                                                                               | Purpose                                              |
| -------------------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| [Titanium-Quoridor-Website](https://github.com/titaniummachine1/Titanium-Quoridor-Website)         | Playable UI, benchmarks, vendored JS engines         |
| [Titanium-Quoridor-Coordinator](https://github.com/titaniummachine1/Titanium-Quoridor-Coordinator) | Cloudflare Worker for distributed SPRT testing       |
| [titanium-quoridor-test-client](https://github.com/titaniummachine1/titanium-quoridor-test-client) | CLI worker that runs matches against the coordinator |

## Build & test

```bash
cargo test --release
cargo build --release
cargo run --release --bin titanium -- perft 3    # → 2_062_264 nodes
cargo run --release --bin titanium -- perft 4    # → 247_569_030 nodes (stress)
cargo run --release --bin titanium -- bench 3 10 # honest movegen+make/unmake nps
cargo run --release --bin titanium -- uci        # UCI loop
```

WASM (install once: `cargo install wasm-pack`):

```bash
wasm-pack build --release --no-default-features --features wasm
```

## Move generation

Production movegen is **single-thread shift algebra** (no wall tables, no movegen GPU/threads):

- **Walls:** L1 empty → L2 collision shifts → TOPO flood-skip → L3 parallel flood + bit theft
- **Pawns:** `ShiftCanStep` default (`PawnGenMode` in `movegen/legal.rs`)

Full reference: [`docs/MOVEGEN.md`](docs/MOVEGEN.md)  
Handoff for follow-up work: [`docs/MOVEGEN-HANDOFF.md`](docs/MOVEGEN-HANDOFF.md)

Regenerate the offline pawn O(1) tables (`PawnGenMode::O1Lookup` is the
production default — perft-proven fastest):

```bash
cargo run --release --bin movegen-o1-gen
```

## Documentation

| Doc | Content |
| --- | ------- |
| [`docs/MOVEGEN.md`](docs/MOVEGEN.md) | Production movegen architecture |
| [`docs/MOVEGEN-HANDOFF.md`](docs/MOVEGEN-HANDOFF.md) | Next work for audit / make-unmake / L3 |
| [`docs/STATE.md`](docs/STATE.md) | Session handoff (search, eval) |
| [`docs/video/README.md`](docs/video/README.md) | Episode index |
| [`docs/video/PERFT-BENCHMARKS.md`](docs/video/PERFT-BENCHMARKS.md) | Perft gates + timings |

## License

**GPL-3.0-or-later** — same license family as [Stockfish](https://github.com/official-stockfish/Stockfish).
Copyleft applies when binaries or combined works are **distributed**; hosting
as a service without distribution is not covered by GPL (unlike AGPL). See
`LICENSE`.
