# Episode 08 — Greedy genmove + testing lab UI

- **branch:** `checkpoint/08-greedy-ui`
- **commit:** `10bcb23`
- **tag:** `checkpoint-08-greedy-ui`

## Hook

"Episode 07 gave us a boss fight in the browser. Episode 08 turns the site into a **lab**: tune each player independently, match the scraped pro UI, and run Titanium's first real `genmove` in the terminal."

---

## What shipped

| Piece                        | Path                                                                        |
| ---------------------------- | --------------------------------------------------------------------------- |
| Greedy one-ply search        | `engine/src/greedy.rs`                                                      |
| CLI `titanium genmove`       | `engine/src/main.rs`                                                        |
| Titanium vs Gorisanson bench | `benchmark/titanium_vs_gorisanson.mjs`                                      |
| Per-player AI settings       | `web/src/lib/timeControl.js`, `appController.js`                            |
| Scraped-style sliders        | `web/src/ui/discreteSlider.js`, `sliderWire.js`, `scrapedSlider.css`        |
| Time-budget local MCTS       | `web/src/workers/gorisansonWorker.js`                                       |
| Coordinate bridge fix        | `web/src/lib/gorisansonBridge.js`                                           |
| Remote engine sync fix       | `web/src/game/appController.js`, `web/src/lib/engineClient.js`              |
| Titanium in web UI           | `web/src/lib/localMctsEngine.js` — Gorisanson MCTS + Ishtar strength slider |

**Titanium (MCTS)** is in the player dropdown — same worker/search as Gorisanson, plus **AI Strength** (UCT tiers). Terminal `titanium genmove` remains greedy Rust for benchmarks.

---

## Web UI — before vs after (video segment)

**Checkout for “before”:** `git checkout checkpoint-07-gorisanson-ui`  
**Checkout for “after”:** `git checkout checkpoint-08-greedy-ui`

| Area                        | Episode 07 (before)                                          | Episode 08 (after)                                                                                    |
| --------------------------- | ------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------- |
| **AI settings scope**       | One global **AI time preset** under the player dropdown      | **Per-player** controls under each `<select>`                                                         |
| **Remote (Ishtar / Ka)**    | Time preset only; strength fixed                             | **AI Strength** slider (Beg. → Alpha) + **AI Time** (Immediate → Long); hint shows visits + threads   |
| **Local (Gorisanson)**      | Fixed sim count from global preset (2.5k / 7.5k / 20k / 60k) | **Time per move** (0.5–60 s) + **Visit budget** (1k–60k); MCTS stops at whichever limit hits first    |
| **Slider UX**               | Native `<select>` or basic range                             | Discrete **scraped** sliders (labels under thumb, quoridor-ai.netlify.app parity)                     |
| **Slider drag**             | N/A or broken on range inputs                                | Drag works: `input` updates silently; full refresh on `change` only (no DOM wipe mid-drag)            |
| **Settings memory**         | Lost on player swap                                          | `playerAiSettingsMemory` — each slot remembers its last local vs remote settings                      |
| **Status line**             | Generic engine name                                          | Live hint: `Ishtar: Alpha · Short (~3,200 visits) · 32 threads` or `Gorisanson: 3s · ≤7,500 rollouts` |
| **Remote play reliability** | Broke after ~2 plies (engine not told about its own moves)   | `makemove` after **every** ply (human + AI), matching scraped `takeAction` middleware                 |
| **Local MCTS reliability**  | Infinite spinner if bridge sent wrong coordinates            | Row flip fix: UI row 1 = bottom ↔ Gorisanson row 0 = top                                              |

**Narration beat:** "We didn't just skin the controls — we split the test matrix. P1 on Immediate Ishtar vs P2 on 30 s / 20k-rollout Gorisanson is one click away."

---

## Video scenarios (demo script)

Record these on **`checkpoint-08-greedy-ui`** with `npm run dev` in `web/`.

### 1. Side-by-side UI tour (30 s)

1. Show Player 1 = Human, Player 2 = Gorisanson — point at **two** slider rows (time + visits).
2. Change Player 2 to Ishtar — controls **swap** to Strength + AI Time (not wall-clock).
3. Set Player 1 = Ishtar, Player 2 = Ka — **different** strength/time per cloud engine.
4. Toggle back to Gorisanson on P2 — sliders **restore** previous Gorisanson values (memory).

### 2. Titanium vs Gorisanson in the browser (45 s)

- P1 **Titanium (greedy)**, P2 **Gorisanson** at 3 s / 7.5k — watch greedy lose in real time.
- Same engines as `titanium_vs_gorisanson.mjs`, no terminal needed for a quick sanity check.

### 3. Uneven match smoke test (1 min)

- P1 Human, P2 Gorisanson: **0.5 s**, **1,000** visits → weak, fast replies (good for live coding).
- P1 Gorisanson **60 s** / **60k**, P2 Human → strong local boss, spinner shows progress.
- Narrate: "Same board, same rules — different budgets without recompiling."

### 4. Remote engine lab (1 min)

- P1 Human, P2 Ishtar: Strength **Beg.**, Time **Immediate**.
- Play 3–4 full moves — **no red error icon** after AI replies (sync fix).
- Bump P2 to **Alpha / Long** — hint updates visit estimate; next search uses new budget.
- Optional: P1 Ishtar vs P2 Ka — both remote; each ply syncs to **both** engines.

### 5. Terminal: Titanium enters the arena (45 s)

```bash
cd engine && cargo build --release
cd ..
node benchmark/titanium_vs_gorisanson.mjs --games 4 --gorisanson 7500
node benchmark/titanium_vs_gorisanson.mjs --games 4 --gorisanson 7500 -v
```

Show score + provisional Elo. Greedy loses badly — **that's the hook** for episode 09 (αβ).

### 6. Bug diary callbacks (optional B-roll)

- **Spinner forever:** open `gorisansonBridge.js`, show `PAWN_ROWS - row` — wrong row → illegal move → AI loop.
- **Red `!` on ply 3:** diagram: server still at ply 1 because AI's `bestmove` was never echoed as `makemove`.

---

## Terminal commands (no user input)

```bash
# Greedy move from start
engine/target/release/titanium genmove

# After a sequence
engine/target/release/titanium genmove e2 e8 e3

# Head-to-head (build release first)
node benchmark/titanium_vs_gorisanson.mjs --games 10 --gorisanson 20000
```

Compare with episode 07 bench:

```bash
node benchmark/head_to_head.mjs --games 4 --p1 7500 --p2 20000
```

---

## Files to flash on screen

| Moment              | File / symbol                                                                      |
| ------------------- | ---------------------------------------------------------------------------------- |
| Per-player settings | `timeControl.js` → `defaultPlayerAiSettings`                                       |
| Slider drag fix     | `sliderWire.js` → `{ silent: true }` on `input`                                    |
| MCTS dual stop      | `gorisansonWorker.js` → wall clock + `maxSimulations`                              |
| Scraped sync model  | `scraped/engine_client_extract.js` → `takeAction` → `makeMoves` on **all** engines |
| Greedy eval         | `greedy.rs` → `opp_dist - our_dist`                                                |

---

## Next

Episode 09: [09-pondering-prep.md](09-pondering-prep.md) — pondering hooks (Stockfish-style, not built yet). Episode 10+: **αβ search** in Rust.
