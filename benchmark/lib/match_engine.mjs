/**
 * Rust Titanium vs local opponent (gorisanson MCTS or rust-titanium self-play).
 */

import { parseAlgebraic, toAlgebraic } from '../../web/src/lib/gameLogic.js';
import { actionToGorisansonMove, gorisansonMoveToAction } from './gorisanson_bridge.mjs';
import {
  applyGorisansonMove,
  chooseGorisansonMoveWithMeta,
  createGorisansonGame,
  winnerIndex,
} from './gorisanson_ai.mjs';
import { moveLabel } from './gorisanson_moves.mjs';
import { chooseTitaniumMove } from './titanium_ai.mjs';
import { RUST_TITANIUM_ID, GORISANSON_ID, assertRustTitaniumId } from './engine_ids.mjs';
import { encodeReplayFromAlgebraic, formatReplayBlock } from './replay_code.mjs';
import { termLine, termThinking } from './terminal_log.mjs';
import { printPlyCompact, printFinalPosition, printSearchDepth } from './terminal_reporter.mjs';
import { resolveThinkBudget } from './bench_limits.mjs';

const MAX_PLIES = 250;

function engineLabel(cfg, budget) {
  if (cfg.id === GORISANSON_ID) {
    return `Gorisanson MCTS (${budget.timeSec}s/${formatSimsCap(budget.maxSimulations)})`;
  }
  if (cfg.id === RUST_TITANIUM_ID) {
    const mode = cfg.engine === 'minimax' ? 'Minimax' : 'MCTS';
    return `Rust Titanium ${mode} (${budget.timeSec}s/${formatSimsCap(budget.maxSimulations)})`;
  }
  return cfg.id;
}

function formatSimsCap(n) {
  if (n >= 1_000_000_000) {
    return `${(n / 1_000_000_000).toFixed(0)}B cap`;
  }
  if (n >= 1_000_000) {
    return `${(n / 1_000_000).toFixed(1)}M cap`;
  }
  return `${n} cap`;
}

function formatSims(n) {
  if (n >= 1_000_000) {
    return `${(n / 1_000_000).toFixed(1)}M`;
  }
  if (n >= 1000) {
    return `${(n / 1000).toFixed(1)}k`;
  }
  return String(n);
}

async function chooseMove(game, algebraicHistory, playerConfig, ply, options) {
  const logMoves = options.logMoves !== false && !options.quiet;
  const budget = resolveThinkBudget(options, playerConfig);
  const label = engineLabel(playerConfig, budget);

  if (logMoves) {
    termThinking({ ply, side: game.pawnOfTurn.index, engine: label });
  }

  if (playerConfig.id === GORISANSON_ID) {
    let lastProgressMs = -1;
    const { move, meta } = chooseGorisansonMoveWithMeta(game, {
      timeMs: budget.timeMs,
      maxSimulations: budget.maxSimulations,
      uct: playerConfig.uct,
      onProgress: logMoves
        ? (progress) => {
          const elapsedMs = progress.elapsedMs ?? 0;
          if (lastProgressMs >= 0 && elapsedMs - lastProgressMs < 900) {
            return;
          }
          lastProgressMs = elapsedMs;
          termLine(
            `      ply ${ply} progress ${playerConfig.id}: ${formatSims(progress.simulations ?? 0)} sims · ${(elapsedMs / 1000).toFixed(1)}s`,
          );
        }
        : undefined,
    });
    return { move, meta, elapsedMs: meta.elapsedMs };
  }

  if (playerConfig.id === RUST_TITANIUM_ID) {
    assertRustTitaniumId(playerConfig.id);
    const log = options.logSearch !== false;
    const started = performance.now();
    let lastProgressMs = -1;
    const engineMode = playerConfig.engine ?? options.engine;
    const { move: algebraic, meta } = await chooseTitaniumMove(algebraicHistory, {
      log,
      ply,
      engine: engineMode,
      timeSec: budget.timeSec,
      maxSims: budget.maxSimulations,
      uct: playerConfig.uct,
      disableBook: playerConfig.disableBook ?? options.disableBook,
      disableBridge: playerConfig.disableBridge ?? options.disableBridge,
      useCatGuidance: playerConfig.useCatGuidance ?? options.useCatGuidance,
      onDepth:
        logMoves && engineMode === 'minimax'
          ? (depth) => {
            printSearchDepth({ ply, ...depth });
          }
          : undefined,
      onProgress: logMoves
        ? (progress) => {
          const elapsedMs = progress.elapsedMs ?? 0;
          if (engineMode === 'minimax') {
            return;
          }
          if (lastProgressMs >= 0 && elapsedMs - lastProgressMs < 900) {
            return;
          }
          lastProgressMs = elapsedMs;
          termLine(
            `      ply ${ply} progress ${playerConfig.id}: ${formatSims(progress.simulations ?? 0)} sims · ${(elapsedMs / 1000).toFixed(1)}s`,
          );
        }
        : undefined,
    });
    const elapsedMs = performance.now() - started;
    return {
      move: actionToGorisansonMove(parseAlgebraic(algebraic)),
      meta,
      elapsedMs,
    };
  }

  throw new Error(`Unknown player id: ${playerConfig.id}`);
}

export async function playOneGame(playerA, playerB, options = {}) {
  let game = createGorisansonGame();
  const algebraicHistory = [];
  let plies = 0;
  const stats = {
    byEngine: {
      [playerA.id]: { plies: 0, simulations: 0, nodes: 0 },
      [playerB.id]: { plies: 0, simulations: 0, nodes: 0 },
    },
  };
  const logMoves = options.logMoves !== false && !options.quiet;
  const budget = resolveThinkBudget(options);

  while (winnerIndex(game) === null && plies < MAX_PLIES) {
    const side = game.pawnOfTurn.index;
    const cfg = side === 0 ? playerA : playerB;
    const ply = plies + 1;

    const { move, meta } = await chooseMove(game, algebraicHistory, cfg, ply, options);

    if (stats.byEngine[cfg.id]) {
      stats.byEngine[cfg.id].plies += 1;
      stats.byEngine[cfg.id].simulations += meta?.simulations ?? 0;
      stats.byEngine[cfg.id].nodes += meta?.nodes ?? 0;
    }

    applyGorisansonMove(game, move);
    algebraicHistory.push(toAlgebraic(gorisansonMoveToAction(move)));
    plies += 1;

    if (typeof options.onPly === 'function') {
      options.onPly({
        ply,
        whiteId: playerA.id,
        blackId: playerB.id,
        algebraicHistory: [...algebraicHistory],
      });
    }

    if (logMoves) {
      printPlyCompact({
        ply,
        who: side,
        engine: engineLabel(cfg, budget),
        result: meta,
        move,
      });
    } else if (options.verbose) {
      termLine(
        `  ply ${ply} P${side + 1} (${cfg.id}): ${moveLabel(move)} · ${meta.simulations ?? 0} sims · ${meta.stoppedBy}`,
      );
    }
  }

  const winner = winnerIndex(game);
  const replayCode = encodeReplayFromAlgebraic(algebraicHistory, {
    a: playerA.id,
    b: playerB.id,
    plies,
    winner: winner === null ? 'draw' : winner === 0 ? playerA.id : playerB.id,
  });

  if (winner === null) {
    return { result: 'draw', winner: null, plies, replayCode, algebraicHistory, game, stats };
  }
  return {
    result: 'decided',
    winner: winner === 0 ? playerA.id : playerB.id,
    winnerPawn: winner,
    plies,
    replayCode,
    algebraicHistory,
    game,
    stats,
  };
}

export async function playMatch(playerA, playerB, games, options = {}) {
  let scoreA = 0;
  let scoreB = 0;
  let draws = 0;
  const results = [];
  const swapColors = options.swapColors !== false;
  const logMoves = options.logMoves !== false && !options.quiet;
  const budget = resolveThinkBudget(options);

  for (let i = 0; i < games; i++) {
    const swap = swapColors && i % 2 === 1;
    const light = swap ? playerB : playerA;
    const dark = swap ? playerA : playerB;

    if (logMoves || options.verbose) {
      termLine('');
      termLine(
        `── Game ${i + 1}/${games} · White=${light.id} · Black=${dark.id} · budget ${budget.timeSec}s / ${formatSimsCap(budget.maxSimulations)} ──`,
      );
    }

    if (typeof options.onGameStart === 'function') {
      options.onGameStart({ gameIndex: i + 1, totalGames: games, whiteId: light.id, blackId: dark.id });
    }

    const outcome = await playOneGame(light, dark, options);
    results.push(outcome);

    if (logMoves) {
      const winnerLabel =
        outcome.winner === null
          ? null
          : outcome.winner === playerA.id
            ? playerA.id
            : playerB.id;
      printFinalPosition(outcome.game, {
        winnerSide: outcome.winnerPawn ?? null,
        winnerLabel,
        algebraicHistory: outcome.algebraicHistory,
      });
    }

    if (options.logReplay !== false) {
      termLine(
        formatReplayBlock(outcome.replayCode, {
          label: `REPLAY game ${i + 1} — paste in web Replay tab`,
        }),
      );
    }

    if (outcome.result === 'draw') {
      draws += 1;
      scoreA += 0.5;
      scoreB += 0.5;
      continue;
    }

    if (outcome.winner === playerA.id) {
      scoreA += 1;
    } else if (outcome.winner === playerB.id) {
      scoreB += 1;
    }
  }

  return { playerA, playerB, games, scoreA, scoreB, draws, results };
}

export function eloFromMatch(scoreA, scoreB, games, ratingA = 1500, ratingB = 1500, k = 32) {
  const expectedA = 1 / (1 + 10 ** ((ratingB - ratingA) / 400));
  const actualA = scoreA / games;
  return {
    ratingA: ratingA + k * (actualA - expectedA),
    ratingB: ratingB + k * ((1 - actualA) - (1 - expectedA)),
    expectedA,
  };
}
