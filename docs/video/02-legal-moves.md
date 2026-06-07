# Episode 02 — Legal move generation (JS oracle parity)

- **branch:** `checkpoint/02-legal-moves`
- **commit:** `19864b8`
- **tag:** `checkpoint-02-legal-moves`

## Hook

"A chess engine without legal moves is useless. Quoridor is worse — ~130 moves at the start because of walls."

## What we build

`engine/src/moves.rs`:

- Pawn moves + jump-over + sideways jump when blocked
- Wall placement with `collidesWithExistingWall` + `canWallBlock` topology rules
- Path check: trial wall + `both_players_reach_goals`

## Demo

```bash
cd engine
cargo run -- moves
node ../benchmark/compare_moves.mjs
```

Show matching move count vs `web/src/lib/gameLogic.js`.

## Talking points

- Wall validation hot loop = BFS × candidate walls — why Rust matters
- We don't trust hand-wavy rules; we diff against scraped JS

## Next episode

"Perft — the chess engineer's unit test for move generation."
