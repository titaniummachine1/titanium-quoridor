#!/usr/bin/env node
/**
 * Titanium vs Ishtar worker — JSON output for overnight ladder.
 *
 *   node benchmark/tune_ishtar.mjs --games 1 --time 10 --ishtar short
 */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseAlgebraic, toAlgebraic } from '../web/src/lib/gameLogic.js';
import { actionToGorisansonMove, gorisansonMoveToAction } from './lib/gorisanson_bridge.mjs';
import {
  applyGorisansonMove,
  createGorisansonGame,
  winnerIndex,
} from './lib/gorisanson_ai.mjs';
import { requestIshtarMove, resolveIshtarOptions } from './lib/ishtar_remote.mjs';
import { chooseTitaniumMove } from './lib/titanium_ai.mjs';
import { TITANIUM_MAX_NODES } from './lib/bench_limits.mjs';
import { encodeReplayFromAlgebraic } from './lib/replay_code.mjs';
import { evalPosition } from './lib/path_eval.mjs';

const MAX_PLIES = 250;

function parseArgs(argv) {
  const opts = {
    games: 1,
    timeSec: 10,
    ishtarPreset: 'short',
    quiet: true,
    reportDir: null,
    label: 'ishtar',
    titaniumSide: 0,
  };
  for (let i = 2; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--games' && argv[i + 1]) opts.games = Number(argv[++i]);
    else if (arg === '--time' && argv[i + 1]) opts.timeSec = Number(argv[++i]);
    else if (arg === '--ishtar' && argv[i + 1]) opts.ishtarPreset = argv[++i];
    else if (arg === '--report-dir' && argv[i + 1]) opts.reportDir = argv[++i];
    else if (arg === '--label' && argv[i + 1]) opts.label = argv[++i];
    else if (arg === '--side' && argv[i + 1]) opts.titaniumSide = Number(argv[++i]);
    else if (arg === '--verbose' || arg === '-v') opts.quiet = false;
  }
  return opts;
}

async function playOneGame(titaniumSide, opts) {
  let game = createGorisansonGame();
  const algebraicHistory = [];
  let plies = 0;
  const stats = { ishtarMs: 0, titaniumMs: 0, tiNodes: 0, errors: 0 };

  while (winnerIndex(game) === null && plies < MAX_PLIES) {
    const side = game.pawnOfTurn.index;
    let move;

    if (side === titaniumSide) {
      const started = performance.now();
      const { move: algebraic, meta } = await chooseTitaniumMove(algebraicHistory, {
        log: false,
        timeSec: opts.timeSec,
        maxSims: Number(process.env.TITANIUM_MAX_NODES ?? TITANIUM_MAX_NODES),
      });
      stats.titaniumMs += performance.now() - started;
      stats.tiNodes += meta?.nodes ?? 0;
      move = actionToGorisansonMove(parseAlgebraic(algebraic));
    } else {
      const started = performance.now();
      const algebraic = await requestIshtarMove(algebraicHistory, { preset: opts.ishtarPreset });
      stats.ishtarMs += performance.now() - started;
      move = actionToGorisansonMove(parseAlgebraic(algebraic));
    }

    applyGorisansonMove(game, move);
    algebraicHistory.push(toAlgebraic(gorisansonMoveToAction(move)));
    plies += 1;
  }

  const winner = winnerIndex(game);
  const pos = evalPosition(game);
  const margin = pos.whiteDist < 200 && pos.blackDist < 200 ? pos.blackDist - pos.whiteDist : pos.margin;
  const replayCode = encodeReplayFromAlgebraic(algebraicHistory, {
    a: 'rust-titanium',
    b: 'ishtar',
    plies,
    winner: winner === null ? 'draw' : winner === titaniumSide ? 'rust-titanium' : 'ishtar',
  });

  return { winner, titaniumSide, plies, margin, pos, stats, replayCode };
}

async function main() {
  const opts = parseArgs(process.argv);
  const ishtarOpts = resolveIshtarOptions(opts.ishtarPreset);
  const gameSummaries = [];
  let tiWins = 0;
  let isWins = 0;
  const started = performance.now();

  for (let i = 0; i < opts.games; i++) {
    const titaniumSide = opts.titaniumSide ?? i % 2;
    const outcome = await playOneGame(titaniumSide, opts);
    if (outcome.winner === titaniumSide) tiWins += 1;
    else if (outcome.winner !== null) isWins += 1;

    gameSummaries.push({
      gameIndex: i + 1,
      winner: outcome.winner === titaniumSide ? 'rust-titanium' : 'ishtar',
      winnerPawn: outcome.winner,
      plies: outcome.plies,
      finalMargin: outcome.margin,
      whiteDist: outcome.pos.whiteDist,
      blackDist: outcome.pos.blackDist,
      errors: outcome.stats.errors,
      illegalMoves: [],
      tiNodes: outcome.stats.tiNodes,
      tiAvgNodesPerMove: outcome.plies ? Math.round(outcome.stats.tiNodes / outcome.plies) : 0,
      replay: outcome.replayCode,
    });
  }

  const summary = {
    label: opts.label,
    opponent: 'ishtar',
    games: opts.games,
    timeSec: opts.timeSec,
    ishtarPreset: opts.ishtarPreset,
    ishtarVisits: ishtarOpts.visits,
    titaniumMaxNodes: Number(process.env.TITANIUM_MAX_NODES ?? TITANIUM_MAX_NODES),
    engine: 'minimax',
    score: `${tiWins}-${isWins}`,
    draws: 0,
    winRate: opts.games ? tiWins / opts.games : 0,
    wallSec: Number(((performance.now() - started) / 1000).toFixed(1)),
    avgPlies: opts.games ? gameSummaries.reduce((s, g) => s + g.plies, 0) / opts.games : 0,
    illegalMoveCount: 0,
    games_detail: gameSummaries,
  };

  if (opts.reportDir) {
    fs.mkdirSync(opts.reportDir, { recursive: true });
    fs.writeFileSync(
      path.join(opts.reportDir, `${opts.label}-aggregate.json`),
      JSON.stringify(summary, null, 2),
    );
  }

  console.log(JSON.stringify(summary));
  process.exit(tiWins >= isWins ? 0 : 1);
}

main().catch((err) => {
  console.error(err?.stack || String(err));
  process.exit(2);
});
