#!/usr/bin/env node
/**
 * Rust Titanium vs Rust Titanium — symmetry / pierce regression worker.
 *
 *   node benchmark/tune_selfplay.mjs --games 1 --time 10 --report-dir benchmark/overnight/self
 */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { playMatch } from './lib/match_engine.mjs';
import { RUST_TITANIUM_ID } from './lib/engine_ids.mjs';
import { TITANIUM_MAX_NODES } from './lib/bench_limits.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

function parseArgs(argv) {
  const opts = {
    games: 2,
    timeSec: 10,
    quiet: true,
    reportDir: null,
    label: 'selfplay',
  };
  for (let i = 2; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--games' && argv[i + 1]) opts.games = Number(argv[++i]);
    else if (arg === '--time' && argv[i + 1]) opts.timeSec = Number(argv[++i]);
    else if (arg === '--report-dir' && argv[i + 1]) opts.reportDir = argv[++i];
    else if (arg === '--label' && argv[i + 1]) opts.label = argv[++i];
    else if (arg === '--verbose' || arg === '-v') opts.quiet = false;
  }
  return opts;
}

function summarizeGame(game, gameIndex) {
  const ti = game.stats?.byEngine?.[RUST_TITANIUM_ID] ?? {};
  return {
    gameIndex,
    winner: game.winner,
    winnerPawn: game.winnerPawn,
    plies: game.plies,
    finalMargin: game.finalPos?.margin,
    whiteDist: game.finalPos?.whiteDist,
    blackDist: game.finalPos?.blackDist,
    errors: game.errors?.length ?? 0,
    illegalMoves: game.errors,
    tiNodes: ti.nodes ?? 0,
    tiAvgNodesPerMove: ti.plies ? Math.round((ti.nodes ?? 0) / ti.plies) : 0,
    replay: game.replayCode,
    reportPath: null,
  };
}

async function main() {
  const opts = parseArgs(process.argv);
  const label = opts.label ?? 'selfplay';
  const titanium = {
    id: RUST_TITANIUM_ID,
    engine: 'minimax',
    timeSec: opts.timeSec,
    maxSimulations: Number(process.env.TITANIUM_MAX_NODES ?? TITANIUM_MAX_NODES),
    useCatGuidance: true,
  };

  const started = performance.now();
  const match = await playMatch(titanium, titanium, opts.games, {
    engine: 'minimax',
    quiet: opts.quiet,
    logMoves: !opts.quiet,
    logReplay: !opts.quiet,
    swapColors: true,
    useCatGuidance: true,
  });
  const wallSec = (performance.now() - started) / 1000;

  if (opts.reportDir) {
    fs.mkdirSync(opts.reportDir, { recursive: true });
  }

  const gameSummaries = [];
  let totalPlies = 0;
  let totalErrors = 0;
  let totalNodes = 0;

  for (let i = 0; i < match.results.length; i++) {
    const game = match.results[i];
    totalPlies += game.plies ?? 0;
    totalErrors += game.errors?.length ?? 0;
    const ti = game.stats?.byEngine?.[RUST_TITANIUM_ID];
    if (ti) totalNodes += ti.nodes ?? 0;

    const summary = summarizeGame(game, i + 1);
    if (opts.reportDir && game.report) {
      const reportPath = path.join(opts.reportDir, `${label}-game${i + 1}.txt`);
      fs.writeFileSync(reportPath, game.report, 'utf8');
      summary.reportPath = reportPath;
    }
    gameSummaries.push(summary);
  }

  const summary = {
    label,
    opponent: 'self',
    games: opts.games,
    timeSec: opts.timeSec,
    titaniumMaxNodes: Number(process.env.TITANIUM_MAX_NODES ?? TITANIUM_MAX_NODES),
    engine: 'minimax',
    score: `${match.scoreA}-${match.scoreB}`,
    draws: match.draws,
    winRate: opts.games ? match.scoreA / opts.games : 0,
    symmetryDelta: Math.abs(match.scoreA - match.scoreB),
    wallSec: Number(wallSec.toFixed(1)),
    avgPlies: opts.games ? totalPlies / opts.games : 0,
    avgNodesPerMove: totalPlies ? Math.round(totalNodes / totalPlies) : 0,
    illegalMoveCount: totalErrors,
    games_detail: gameSummaries,
    pierceEnv: {
      relax: process.env.TITANIUM_PIERCE_RELAX ?? null,
      hot: process.env.TITANIUM_PIERCE_HOT ?? null,
      aggr: process.env.TITANIUM_PIERCE_AGGR ?? null,
      pow: process.env.TITANIUM_PIERCE_POW ?? null,
    },
  };

  console.log(JSON.stringify(summary));
  process.exit(totalErrors > 0 ? 2 : 0);
}

main().catch((err) => {
  console.error(err?.stack || String(err));
  process.exit(2);
});
