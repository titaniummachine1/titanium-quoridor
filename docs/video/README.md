# Titanium Engine — video checkpoint scripts

Each file is a **standalone episode outline** tied to a git branch/tag and commit hash.
After each checkpoint commit, update the `commit:` line with `git rev-parse --short HEAD`.

| Episode | Branch                        | Commit     | Script                                                                            |
| ------- | ----------------------------- | ---------- | --------------------------------------------------------------------------------- |
| 01      | `checkpoint/01-path-bfs`      | `43a1b93`  | [01-path-bfs.md](01-path-bfs.md)                                                  |
| 02      | `checkpoint/02-legal-moves`   | `19864b8`  | [02-legal-moves.md](02-legal-moves.md)                                            |
| 03      | `checkpoint/03-perft`         | `5a4b0fc`  | [03-perft.md](03-perft.md)                                                        |
| 04      | `checkpoint/04-bench`         | `90193b0`  | [04-bench.md](04-bench.md)                                                        |
| 05      | `checkpoint/05-perft-bugfix`  | `6b9e00d`  | [05-first-perft-bug.md](05-first-perft-bug.md)                                    |
| 06      | `checkpoint/06-threading`     | `098477c`  | [06-threading-prep.md](06-threading-prep.md)                                      |
| 07      | `checkpoint/07-gorisanson-ui` | `7c85a20`  | [07-ai-opponents.md](07-ai-opponents.md)                                          |
| 08      | `checkpoint/08-greedy-ui`     | `10bcb23`  | [08-greedy-ui-lab.md](08-greedy-ui-lab.md)                                        |
| 09      | _(future)_                    | —          | [09-pondering-prep.md](09-pondering-prep.md) — Stockfish-style ponder (prep only) |
| 10+     | `checkpoint-10-alphabeta` …   | _(future)_ | αβ search — beat gorisanson at same time budget                                   |

**Perft debug:** `node benchmark/perft_diff.mjs 2` — divide diff vs JS oracle.

**Future Elo ladder:** [TOURNAMENT-ROADMAP.md](TOURNAMENT-ROADMAP.md) · machine list: `benchmark/checkpoints.json`

**Series index:** [00-SERIES-OVERVIEW.md](00-SERIES-OVERVIEW.md) · [BUG-DIARY.md](BUG-DIARY.md) · [PERFT-BENCHMARKS.md](PERFT-BENCHMARKS.md) · [00-HOW-THE-ENGINE-WORKS.md](00-HOW-THE-ENGINE-WORKS.md)

Checkout any episode: `git checkout checkpoint/03-perft`
