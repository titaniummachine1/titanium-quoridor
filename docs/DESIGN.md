# Titanium Engine — design notes

## Hybrid search (planned)

```
Time budget T
├── Phase 1 (~40–50% T): ID + αβ + Zobrist TT + aspiration windows
└── Phase 2 (remainder): MCTS with gorisanson rollouts, seeded from Phase 1 PV
```

## Path / eval

- Dual BFS distance fields per position (cache in TT entry)
- Incremental invalidation on wall placement — D\* Lite only if profiling demands it

## Move ordering (Phase 1)

1. TT best move
2. Pawn steps that shorten distance to goal
3. Walls that lengthen opponent path
4. Probable walls (gorisanson heuristic)
5. Everything else (LMR candidates)

## Pondering (planned — not active)

Stockfish-style: think while opponent moves. Opponent compute is untouched.

| Engine      | Approach                                       | Status                                    |
| ----------- | ---------------------------------------------- | ----------------------------------------- |
| Ishtar / Ka | `go ponder` / `stop` on WebSocket              | `EngineClient.ponder()` ready, not called |
| Local MCTS  | Predicted reply + node-cap search + tree reuse | Blocked on one-shot worker                |

Prep: `docs/video/09-pondering-prep.md`, `web/src/lib/enginePonder.js`, `appController.maybePonderInactiveEngines()`.

Ponder budget: **rollout cap only**, no wall-clock limit.

## External benchmarks

| Opponent          | Role                   |
| ----------------- | ---------------------- |
| JS `gameLogic.js` | Rules oracle           |
| gorisanson MCTS   | Local OSS baseline     |
| Ishtar @ Short    | External strength exam |

## References

- [titaniummachine1/titanium-quoridor](https://github.com/titaniummachine1/titanium-quoridor)
- [gorisanson/quoridor-ai](https://github.com/gorisanson/quoridor-ai)
- [pavlosdais/Quoridor](https://github.com/pavlosdais/Quoridor)
