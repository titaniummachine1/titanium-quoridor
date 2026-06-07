# Titanium Quoridor

**Titanium Engine** — hybrid Quoridor AI (αβ + guided MCTS) with a reverse-engineered play UI.

Repo: [github.com/titaniummachine1/titanium-quoridor](https://github.com/titaniummachine1/titanium-quoridor)

## Layout

| Path         | Purpose                                                                               |
| ------------ | ------------------------------------------------------------------------------------- |
| `engine/`    | **Titanium** — Rust search core (in development)                                      |
| `web/`       | Playable UI (scraped from [quoridor-ai.netlify.app](https://quoridor-ai.netlify.app)) |
| `scraped/`   | Deobfuscated extracts + raw bundle archive                                            |
| `extracted/` | Protocol docs + WebSocket client                                                      |
| `benchmark/` | Head-to-head tests (planned)                                                          |

## Quick start — web UI

```bash
cd web
npm install
npm run dev
```

Play Human vs **Ishtar** or **Ka** (remote WebSocket engines).

## Quick start — Titanium engine (Rust)

```bash
cd engine
cargo build --release
cargo test
cargo run --release -- perft 2
cargo run --release -- divide 1
cargo run --release -- bench 2 20
cargo bench
```

Cross-check move generation vs scraped JS:

```bash
node benchmark/compare_moves.mjs
```

## Engine roadmap

1. **Phase 1** — Board, eval (dual BFS), iterative deepening αβ, Zobrist TT, aspiration windows
2. **Phase 2** — Gorisanson-style guided MCTS, seeded from Phase 1 PV
3. **Hybrid** — Time-split tactical + rollout phases
4. **Bench** — vs MCTS JS, vs Ishtar@Short (external exam)

## References (ideas only — not ports)

- [gorisanson/quoridor-ai](https://github.com/gorisanson/quoridor-ai) — MCTS heuristics
- [pavlosdais/Quoridor](https://github.com/pavlosdais/Quoridor) — αβ + TT
- [quoridor-ai.netlify.app](https://quoridor-ai.netlify.app) — UI + wire protocol scrape

## License

Engine: TBD. Web scrape artifacts and third-party references retain their original licenses.
