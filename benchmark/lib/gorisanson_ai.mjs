/**
 * Gorisanson MCTS — read-only vendor play API for benchmarks.
 * Think budget: wall-clock + sim cap (whichever hits first).
 */

import { createRequire } from 'node:module';
import { BENCH_MAX_SIMULATIONS, BENCH_TIME_MS } from './bench_limits.mjs';

const require = createRequire(import.meta.url);
const g = require('./load_gorisanson.cjs');

export const GORISANSON_UCT = 0.2;

/** Legacy presets — benchmarks use BENCH_* limits instead. */
export const GORISANSON_TIME_SIMS = {
  intuition: 2_500,
  short: 7_500,
  medium: 20_000,
  long: 60_000,
};

export function createGorisansonGame() {
  return new g.Game(true);
}

export function cloneGorisansonGame(game) {
  return g.Game.clone(game);
}

function chooseOpeningPawnMove(game) {
  if (game.turn >= 2) {
    return null;
  }
  const nextPosition = g.AI.chooseShortestPathNextPawnPosition(game);
  const pawnMoveTuple = nextPosition.getDisplacementPawnMoveTupleFrom(game.pawnOfTurn.position);
  if (pawnMoveTuple[1] === 0) {
    return [[nextPosition.row, nextPosition.col], null, null];
  }
  return null;
}

function fallbackMove(game) {
  const nextPosition = g.AI.chooseShortestPathNextPawnPosition(game);
  const pawnMoveTuple = nextPosition.getDisplacementPawnMoveTupleFrom(game.pawnOfTurn.position);
  if (pawnMoveTuple[1] === 0) {
    return [[nextPosition.row, nextPosition.col], null, null];
  }
  const valids = game.getArrOfValidNextPositionTuples();
  if (valids.length > 0) {
    return [[valids[0][0], valids[0][1]], null, null];
  }
  const walls = game.getArrOfProbableValidNoBlockNextHorizontalWallPositions();
  if (walls.length > 0) {
    return [null, walls[0], null];
  }
  const verts = game.getArrOfProbableValidNoBlockNextVerticalWallPositions();
  if (verts.length > 0) {
    return [null, null, verts[0]];
  }
  return null;
}

function pickBestMoveFromTree(mcts, game) {
  if (mcts.root.children.length > 0) {
    const best = mcts.selectBestMove();
    if (best?.move) {
      return best.move;
    }
  }
  return fallbackMove(game);
}

function findImmediateWinMove(game) {
  const valids = game.getArrOfValidNextPositionTuples();
  for (const [row, col] of valids) {
    const trial = g.Game.clone(game);
    trial.doMove([[row, col], null, null], true);
    if (trial.winner !== null) {
      return [[row, col], null, null];
    }
  }
  return null;
}

function stmOneStepFromGoal(game) {
  const next = g.AI.chooseShortestPathNextPawnPosition(game);
  const goalRow = game.pawnOfTurn === game.pawn1 ? 8 : 0;
  return next.row === goalRow;
}

function shouldStopGorisansonSearch(mcts, game, simulations) {
  if (simulations < 100) {
    return false;
  }
  if (stmOneStepFromGoal(game)) {
    return true;
  }
  if (!mcts.root.children.length) {
    return false;
  }
  const best = mcts.selectBestMove();
  if (!best || best.n < 100) {
    return false;
  }
  const wr = best.wins / best.n;
  if (best.n >= 300 && wr >= 0.98) {
    return true;
  }
  if (best.n >= 150 && wr >= 0.99) {
    return true;
  }
  if (mcts.root.children.length === 1 && best.n >= 500 && wr >= 0.95) {
    return true;
  }
  return false;
}

/**
 * @param {object} game
 * @param {{ timeMs?: number, maxSimulations?: number, uct?: number, onProgress?: (p: { simulations: number, elapsedMs: number }) => void }} [budget]
 */
export function chooseGorisansonMoveWithMeta(game, budget = {}) {
  const timeMs = budget.timeMs ?? BENCH_TIME_MS;
  const maxSimulations = budget.maxSimulations ?? BENCH_MAX_SIMULATIONS;
  const uct = budget.uct ?? GORISANSON_UCT;
  const started = performance.now();

  const opening = chooseOpeningPawnMove(game);
  if (opening) {
    return {
      move: opening,
      meta: {
        stoppedBy: 'opening',
        simulations: 0,
        elapsedMs: performance.now() - started,
      },
    };
  }

  const immediateWin = findImmediateWinMove(game);
  if (immediateWin) {
    return {
      move: immediateWin,
      meta: {
        stoppedBy: 'win-in-1',
        simulations: 0,
        elapsedMs: performance.now() - started,
      },
    };
  }

  const prevLog = console.log;
  console.log = () => { };
  try {
    const mcts = new g.MonteCarloTreeSearch(game, uct);
    const deadline = started + timeMs;
    const batchSize = 50;
    let simulations = 0;
    let lastProgressMs = -1;

    while (performance.now() < deadline && simulations < maxSimulations) {
      const remainingMs = deadline - performance.now();
      const remainingSims = maxSimulations - simulations;
      const batch = Math.min(remainingMs < 250 ? 1 : batchSize, remainingSims);
      if (batch <= 0) {
        break;
      }
      mcts.search(batch);
      simulations += batch;

      if (shouldStopGorisansonSearch(mcts, game, simulations)) {
        break;
      }

      if (typeof budget.onProgress === 'function') {
        const elapsedMs = Math.max(0, Math.round(performance.now() - started));
        if (lastProgressMs < 0 || elapsedMs - lastProgressMs >= 800) {
          budget.onProgress({ simulations, elapsedMs });
          lastProgressMs = elapsedMs;
        }
      }
    }

    const stoppedBy = shouldStopGorisansonSearch(mcts, game, simulations)
      ? 'forced'
      : simulations >= maxSimulations
        ? 'visits'
        : 'time';
    const move = pickBestMoveFromTree(mcts, game);
    if (!move) {
      throw new Error('gorisanson: no legal move');
    }

    return {
      move,
      meta: {
        stoppedBy,
        simulations,
        elapsedMs: performance.now() - started,
      },
    };
  } finally {
    console.log = prevLog;
  }
}

/** @deprecated use chooseGorisansonMoveWithMeta — returns move tuple only */
export function chooseGorisansonMove(game, budget = {}) {
  const resolved =
    typeof budget === 'number'
      ? { maxSimulations: budget, timeMs: BENCH_TIME_MS }
      : budget;
  return chooseGorisansonMoveWithMeta(game, resolved).move;
}

export function applyGorisansonMove(game, move) {
  game.doMove(move, true);
}

export function winnerIndex(game) {
  return game.winner === null ? null : game.winner.index;
}
