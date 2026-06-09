#!/usr/bin/env node
/**
 * Terminal minimax tuning harness — Titanium minimax vs Gorisanson MCTS.
 *
 *   node benchmark/tune_minimax.mjs --games 6 --time 10
 *   TITANIUM_BIN=engine/target/tune/release/titanium.exe node benchmark/tune_minimax.mjs
 */

import { playMatch } from './lib/match_engine.mjs';
import { RUST_TITANIUM_ID, GORISANSON_ID } from './lib/engine_ids.mjs';

function parseArgs(argv) {
  const opts = { games: 4, timeSec: 10, gorisansonTimeSec: 3, quiet: true };
  for (let i = 2; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--games' && argv[i + 1]) opts.games = Number(argv[++i]);
    else if (arg === '--time' && argv[i + 1]) opts.timeSec = Number(argv[++i]);
    else if (arg === '--gorisanson-time' && argv[i + 1]) {
      opts.gorisansonTimeSec = Number(argv[++i]);
    } else if (arg === '--verbose' || arg === '-v') opts.quiet = false;
    else if (arg === '--label' && argv[i + 1]) opts.label = argv[++i];
  }
  return opts;
}

async function main() {
  const opts = parseArgs(process.argv);
  const label = opts.label ?? process.env.TUNE_LABEL ?? 'default';
  const budget = {
    timeSec: opts.timeSec,
    timeMs: opts.timeSec * 1000,
    maxSimulations: Number(process.env.TITANIUM_MAX_NODES ?? 2_000_000_000),
  };

  const titanium = { id: RUST_TITANIUM_ID, engine: 'minimax', timeSec: opts.timeSec };
  const gorisanson = { id: GORISANSON_ID, timeSec: opts.gorisansonTimeSec };

  if (!opts.quiet && opts.label) {
    console.log(`════ ${opts.label} · Ti ${opts.timeSec}s vs Go ${opts.gorisansonTimeSec}s ════`);
  }

  const started = performance.now();
  const match = await playMatch(titanium, gorisanson, opts.games, {
    ...budget,
    engine: 'minimax',
    quiet: opts.quiet,
    logMoves: !opts.quiet,
    logReplay: false,
    logSearch: true,
  });
  const wallSec = (performance.now() - started) / 1000;

  let totalPlies = 0;
  let totalNodes = 0;
  let totalDepth = 0;
  let depthSamples = 0;

  for (const game of match.results) {
    totalPlies += game.plies ?? 0;
    const ti = game.stats?.byEngine?.[RUST_TITANIUM_ID];
    if (ti) {
      totalNodes += ti.nodes ?? 0;
    }
  }

  const summary = {
    label,
    games: opts.games,
    timeSec: opts.timeSec,
    score: `${match.scoreA}-${match.scoreB}`,
    draws: match.draws,
    winRate: match.scoreA / opts.games,
    wallSec: Number(wallSec.toFixed(1)),
    avgPlies: totalPlies / opts.games,
    avgNodesPerMove: totalPlies ? Math.round(totalNodes / totalPlies) : 0,
    bin: process.env.TITANIUM_BIN ?? 'default',
  };

  console.log(JSON.stringify(summary));
  process.exit(match.scoreA > match.scoreB ? 0 : 1);
}

main().catch((err) => {
  console.error(err?.stack || String(err));
  process.exit(2);
});
