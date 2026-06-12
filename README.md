# Titanium Quoridor Engine

Rust search engine for [Quoridor](https://en.wikipedia.org/wiki/Quoridor): iterative-deepening αβ, CAT corridor pruning, ACE v10 eval, UCI protocol, and WASM bindings.

Repo: [github.com/titaniummachine1/titanium-quoridor](https://github.com/titaniummachine1/titanium-quoridor)

Related repos:

| Repo                                                                                               | Purpose                                              |
| -------------------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| [Titanium-Quoridor-Website](https://github.com/titaniummachine1/Titanium-Quoridor-Website)         | Playable UI, benchmarks, vendored JS engines         |
| [Titanium-Quoridor-Coordinator](https://github.com/titaniummachine1/Titanium-Quoridor-Coordinator) | Cloudflare Worker for distributed SPRT testing       |
| [titanium-quoridor-test-client](https://github.com/titaniummachine1/titanium-quoridor-test-client) | CLI worker that runs matches against the coordinator |

## Build & test

```bash
cargo test
cargo build --release
cargo run --release -- perft          # depth 3 → 2_062_264 nodes
cargo run --release -- uci            # UCI loop: uci / isready / position startpos / go movetime 500 / quit
```

WASM (install once: `cargo install wasm-pack`):

```bash
wasm-pack build --release --no-default-features --features wasm
```

## Documentation

See `docs/` — start with `docs/STATE.md` for session handoff and `docs/video/README.md` for the episode index.

## License

MIT OR Apache-2.0 (see `Cargo.toml`).
