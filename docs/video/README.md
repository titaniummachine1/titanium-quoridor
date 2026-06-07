# Titanium Engine — video checkpoint scripts

Each file is a **standalone episode outline** tied to a git branch/tag and commit hash.
After each checkpoint commit, update the `commit:` line with `git rev-parse --short HEAD`.

| Episode | Branch | Commit | Script |
|---------|--------|--------|--------|
| 01 | `checkpoint/01-path-bfs` | `43a1b93` | [01-path-bfs.md](01-path-bfs.md) |
| 02 | `checkpoint/02-legal-moves` | `19864b8` | [02-legal-moves.md](02-legal-moves.md) |
| 03 | `checkpoint/03-perft` | `5a4b0fc` | [03-perft.md](03-perft.md) |
| 04 | `checkpoint/04-bench` | `90193b0` | [04-bench.md](04-bench.md) |

`main` is at `90193b0` (latest). Checkout any episode: `git checkout checkpoint/03-perft`
