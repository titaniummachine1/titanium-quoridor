# Episode 09 — Pondering prep (Stockfish-style, not built yet)

- **status:** prepared, not implemented
- **tag:** _(future)_ `checkpoint-09-pondering`

## Hook

"Stockfish doesn't sleep on the opponent's clock — it ponders. We wire the same idea: think on **our** machine while they move, without touching **their** compute."

---

## What pondering means here

| Phase             | Who moves | Our engine                                          |
| ----------------- | --------- | --------------------------------------------------- |
| Opponent thinking | Them      | **Ponder** — search ahead (node cap, no wall clock) |
| Our turn          | Us        | **Stop** ponder → **go** with full time budget      |

Opponent is unaffected: remote `go ponder` uses **your** WebSocket slot; local MCTS runs in **your** worker thread.

---

## Two engines, two paths

### A. Remote Ishtar / Ka — do first

Protocol already supports it (`scraped/engine_client_extract.js`, `extracted/ENGINE_PROTOCOL.md`):

```
go ponder    # while opponent thinks
stop         # when opponent moved — before our go
go           # our turn, normal time + visits
```

**Prep in repo:** `EngineClient.ponder()` / `stop()` in `web/src/lib/engineClient.js` — call sites TBD in `appController.maybePonderInactiveEngines()`.

### B. Local MCTS (Gorisanson / Titanium) — later

MCTS root must be **our turn**. While opponent thinks:

1. Predict their reply (PV / shallow search).
2. Run MCTS on **that** child position — node cap only, no time limit.
3. On their move: reuse subtree if match, else discard and search fresh.

**Blocker today:** `gorisansonWorker.js` throws the tree away every ply. Needs persistent tree or predicted-line worker mode.

---

## Planned UI / settings

| Setting             | Ponder behaviour                      |
| ------------------- | ------------------------------------- |
| Time per move       | **Ignored** during ponder             |
| Rollout cap         | **Still applies** — background budget |
| Strength (Titanium) | Same UCT tiers                        |

Status line: `pondering` (already in `EngineStatus` enum).

---

## Integration checklist (when we build it)

- [ ] `appController.maybePonderInactiveEngines()` after each ply
- [ ] `stopPonder()` on inactive engines before `requestMove`
- [ ] `newGame` / `undo` → stop all ponders
- [ ] Remote: `syncRemoteEnginesAfterMove` then ponder on synced engines not on move
- [ ] Local: worker `mode: 'ponder'` + `expectedOpponentMove` + tree reuse
- [ ] Video segment: Human vs Ishtar — show eval ticking while you think

---

## Related

- [08-greedy-ui-lab.md](08-greedy-ui-lab.md) — testing lab UI
- [../DESIGN.md](../DESIGN.md) — pondering section
- `web/src/lib/enginePonder.js` — shared contract stubs
