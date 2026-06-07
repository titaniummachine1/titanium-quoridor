# Episode 01 — Bitboard walls + BFS pathfinding

- **branch:** `checkpoint/01-path-bfs`
- **commit:** `43a1b93`
- **tag:** `checkpoint-01-path-bfs`

## Hook (15s)

"We scraped a Quoridor site, but the real AI lives on a server. So we're building **Titanium Engine** in Rust — and the first thing a fast engine needs is: can each player still reach their goal?"

## What we build

1. `engine/src/board.rs` — pawn positions, wall bitboards, 0-indexed internal coords
2. `engine/src/grid.rs` — O(1) wall tests, `can_step` ported from scraped JS
3. `engine/src/path.rs` — stack BFS with `u128` visited mask (81 squares)

## Why BFS, not D* Lite (yet)

- Uniform step cost → BFS is correct and simple
- Same algorithm family as the scraped `isWallBlocking` in `gameLogic.js`
- D* Lite is for incremental replanning when the board changes slightly — overkill for v1

## Demo commands

```bash
cd engine
cargo test path::
cargo test board::
```

Show test: start position distance 8 for both players.

## Talking points

- Walls stored as two `u64` bitboards (horizontal + vertical)
- Internal row 0 = UI row 1 (`e1`)
- Goal: sub-microsecond reachability checks for wall validation later

## Next episode teaser

"Next we generate every legal pawn jump and wall — and prove it matches JavaScript."
