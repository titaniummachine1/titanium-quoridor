#!/usr/bin/env node
/**
 * Rust Titanium vs Ishtar remote.
 *
 *   node benchmark/titanium_vs_ishtar.mjs
 *   node benchmark/titanium_vs_ishtar.mjs --games 2 --ishtar short -v
 */

import { parseAlgebraic, toAlgebraic } from '../web/src/lib/gameLogic.js';
import { actionToGorisansonMove, gorisansonMoveToAction } from './lib/gorisanson_bridge.mjs';
import {
  applyGorisansonMove,
  createGorisansonGame,
  winnerIndex,
} from './lib/gorisanson_ai.mjs';
import { requestIshtarMove, resolveIshtarOptions } from './lib/ishtar_remote.mjs';
import { chooseTitaniumMove } from './lib/titanium_ai.mjs';
import { BENCH_MAX_SIMULATIONS, BENCH_TIME_SEC } from './lib/bench_limits.mjs';
import {
  printFinalPosition,
  printGameHeader,
  printGameSummary,
  printMatchFooter,
  printPly,
  printPlyCompact,
} from './lib/terminal_reporter.mjs';
import { encodeReplayFromAlgebraic, formatReplayBlock } from './lib/replay_code.mjs';

const MAX_PLIES = 250;

function parseArgs(argv) {
  const opts = {
    games: 2,
    ishtarPreset: 'short',
    verbose: false,
    board: false,
    quiet: false,
  };

  for (let i = 2; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--games' && argv[i + 1]) {
      opts.games = Number(argv[++i]);
    } else if (arg === '--ishtar' && argv[i + 1]) {
      opts.ishtarPreset = argv[++i];
    } else if (arg === '--verbose' || arg === '-v') {
      opts.verbose = true;
    } else if (arg === '--board' || arg === '-b') {
      opts.board = true;
      opts.verbose = true;
    } else if (arg === '--quiet' || arg === '-q') {
      opts.quiet = true;
    }
  }

  return opts;
}

async function playOneGame(titaniumSide, opts) {
  let game = createGorisansonGame();
  const algebraicHistory = [];
  let plies = 0;
  const stats = { ishtarMs: 0, titaniumMs: 0, rustMoves: 0 };
  const ishtarOpts = resolveIshtarOptions(opts.ishtarPreset);

  while (winnerIndex(game) === null && plies < MAX_PLIES) {
    const side = game.pawnOfTurn.index;
    let move;
    let result = null;
    let elapsedMs = 0;

    if (side === titaniumSide) {
      const started = performance.now();
      const { move: algebraic, meta } = await chooseTitaniumMove(algebraicHistory, {
        log: true,
        timeSec: BENCH_TIME_SEC,
        maxSims: BENCH_MAX_SIMULATIONS,
      });
      elapsedMs = performance.now() - started;
      stats.titaniumMs += elapsedMs;
      stats.rustMoves += 1;
      move = actionToGorisansonMove(parseAlgebraic(algebraic));
      result = meta;
    } else {
      const started = performance.now();
      const algebraic = await requestIshtarMove(algebraicHistory, { preset: opts.ishtarPreset });
      elapsedMs = performance.now() - started;
      stats.ishtarMs += elapsedMs;
      move = actionToGorisansonMove(parseAlgebraic(algebraic));
      result = {
        stoppedBy: 'remote',
        simulations: ishtarOpts.visits,
        parallelism: ishtarOpts.parallelism,
      };
    }

    applyGorisansonMove(game, move);
    algebraicHistory.push(toAlgebraic(gorisansonMoveToAction(move)));
    plies += 1;

    if (!opts.quiet) {
      const engine = side === titaniumSide ? 'Rust Titanium' : 'Ishtar';
      if (opts.verbose) {
        printPly({
          ply: plies,
          who: side,
          engine,
          result,
          move,
          elapsedMs,
          game,
          showBoard: opts.board,
        });
      } else {
        printPlyCompact({ ply: plies, who: side, engine, result, move });
      }
    }
  }

  const winner = winnerIndex(game);
  const replayCode = encodeReplayFromAlgebraic(algebraicHistory, {
    game: 'titanium-vs-ishtar',
    winner: winner === null ? 'draw' : winner === titaniumSide ? 'Titanium' : 'Ishtar',
    plies,
  });

  return { winner, plies, stats, replayCode, algebraicHistory, game };
}

async function main() {
  const opts = parseArgs(process.argv);
  const ishtarOpts = resolveIshtarOptions(opts.ishtarPreset);

  console.log('Rust Titanium vs Ishtar');
  console.log(
    `  ${opts.games} games · Rust genmove · Ishtar ${ishtarOpts.label} (${ishtarOpts.visits.toLocaleString()} visits · ${ishtarOpts.parallelism} threads)`,
  );

  let titaniumWins = 0;
  let ishtarWins = 0;
  const totals = { kaMs: 0, titaniumMs: 0, rustMoves: 0 };
  const started = performance.now();

  for (let i = 0; i < opts.games; i += 1) {
    const titaniumSide = i % 2;
    if (!opts.quiet) {
      printGameHeader(i + 1, opts.games, titaniumSide, 'Ishtar (remote)');
    }

    const outcome = await playOneGame(titaniumSide, opts);
    totals.kaMs += outcome.stats.ishtarMs;
    totals.titaniumMs += outcome.stats.titaniumMs;
    totals.rustMoves += outcome.stats.rustMoves;

    if (outcome.winner === titaniumSide) {
      titaniumWins += 1;
    } else if (outcome.winner !== null) {
      ishtarWins += 1;
    }

    const resultLabel =
      outcome.winner === null ? 'draw' : outcome.winner === titaniumSide ? 'Titanium' : 'Ishtar';

    if (!opts.quiet) {
      printFinalPosition(outcome.game, {
        winnerSide: outcome.winner,
        winnerLabel: resultLabel === 'Titanium' ? 'Rust Titanium' : resultLabel,
        algebraicHistory: outcome.algebraicHistory,
      });
    }

    console.log(formatReplayBlock(outcome.replayCode, { label: `REPLAY game ${i + 1}` }));
    printGameSummary({
      gameIndex: i + 1,
      winnerLabel: resultLabel,
      plies: outcome.plies,
      stats: outcome.stats,
    });
  }

  printMatchFooter({
    titaniumWins,
    opponentWins: ishtarWins,
    opponentName: 'Ishtar',
    totals,
    elapsedSec: (performance.now() - started) / 1000,
    games: opts.games,
  });
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
