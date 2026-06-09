/**
 * Gorisanson MCTS in a Web Worker — vanilla vendor logic only (read-only ai.js).
 */

import gameJs from '../../../_vendor/quoridor-mcts/src/js/game.js?raw';
import aiJs from '../../../_vendor/quoridor-mcts/src/js/ai.js?raw';

const PAWN_ROWS = 9;
const WALL_ROWS = 8;

function parseAlgebraic(move) {
  const coordinate = {
    column: move[0],
    row: Number.parseInt(move[1], 10),
  };
  if (move.length > 2) {
    return {
      coordinate,
      wallType: move[2] === 'h' ? 'h' : 'v',
    };
  }
  return { coordinate };
}

function toAlgebraic(action) {
  const base = `${action.coordinate.column}${action.coordinate.row}`;
  return action.wallType ? `${base}${action.wallType}` : base;
}

function actionToGorisansonMove(action) {
  const col = action.coordinate.column.charCodeAt(0) - 97;
  if (action.wallType === 'h') {
    const row = WALL_ROWS - action.coordinate.row;
    return [null, [row, col], null];
  }
  if (action.wallType === 'v') {
    const row = WALL_ROWS - action.coordinate.row;
    return [null, null, [row, col]];
  }
  const row = PAWN_ROWS - action.coordinate.row;
  return [[row, col], null, null];
}

function gorisansonMoveToAction(move) {
  const [pawn, horiz, vert] = move;
  if (pawn) {
    const [row, col] = pawn;
    return {
      coordinate: { column: String.fromCharCode(97 + col), row: PAWN_ROWS - row },
    };
  }
  if (horiz) {
    const [row, col] = horiz;
    return {
      coordinate: { column: String.fromCharCode(97 + col), row: WALL_ROWS - row },
      wallType: 'h',
    };
  }
  if (vert) {
    const [row, col] = vert;
    return {
      coordinate: { column: String.fromCharCode(97 + col), row: WALL_ROWS - row },
      wallType: 'v',
    };
  }
  throw new Error('Invalid move tuple from gorisanson engine');
}

const bootstrap = new Function(
  'postMessage',
  'performance',
  `${gameJs}\n${aiJs}\n
  function chooseOpeningPawnMove(game) {
    if (game.turn >= 2) {
      return null;
    }
    const nextPosition = AI.chooseShortestPathNextPawnPosition(game);
    const pawnMoveTuple = nextPosition.getDisplacementPawnMoveTupleFrom(game.pawnOfTurn.position);
    if (pawnMoveTuple[1] === 0) {
      return [[nextPosition.row, nextPosition.col], null, null];
    }
    return null;
  }

  function fallbackMove(game) {
    const nextPosition = AI.chooseShortestPathNextPawnPosition(game);
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
      if (best && best.move) {
        return best.move;
      }
    }
    return fallbackMove(game);
  }

  function findImmediateWinMove(game) {
    const valids = game.getArrOfValidNextPositionTuples();
    for (const [row, col] of valids) {
      const trial = Game.clone(game);
      trial.doMove([[row, col], null, null], true);
      if (trial.winner !== null) {
        return [[row, col], null, null];
      }
    }
    return null;
  }

  function stmOneStepFromGoal(game) {
    const next = AI.chooseShortestPathNextPawnPosition(game);
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

  function searchForTime(game, uctConst, timeMs, maxSimulations) {
    const opening = chooseOpeningPawnMove(game);
    if (opening) {
      return { move: opening, simulations: 0, stoppedBy: 'opening' };
    }

    const immediateWin = findImmediateWinMove(game);
    if (immediateWin) {
      return { move: immediateWin, simulations: 0, stoppedBy: 'win-in-1' };
    }

    const mcts = new MonteCarloTreeSearch(game, uctConst);
    const started = performance.now();
    const deadline = started + timeMs;
    const batchSize = 50;
    let simulations = 0;
    let tick = 0;
    const simCap =
      Number.isFinite(maxSimulations) && maxSimulations > 0 ? maxSimulations : Infinity;

    while (performance.now() < deadline && simulations < simCap) {
      const remainingMs = deadline - performance.now();
      const remainingSims = simCap - simulations;
      const batch = Math.min(remainingMs < 250 ? 1 : batchSize, remainingSims);
      if (batch <= 0) {
        break;
      }

      mcts.search(batch);
      simulations += batch;
      tick += 1;

      if (shouldStopGorisansonSearch(mcts, game, simulations)) {
        break;
      }

      if (tick % 5 === 0) {
        const elapsed = performance.now() - started;
        postMessage({ type: 'progress', value: Math.min(0.99, elapsed / timeMs), simulations });
      }
    }

    const stoppedBy = shouldStopGorisansonSearch(mcts, game, simulations)
      ? 'forced'
      : simulations >= simCap
        ? 'visits'
        : 'time';
    const move = pickBestMoveFromTree(mcts, game);
    if (!move) {
      throw new Error('no legal move');
    }
    return { move, simulations, stoppedBy };
  }

  return { Game, AI, searchForTime };
  `,
);

const { Game, AI, searchForTime } = bootstrap(
  (msg) => {
    if (typeof msg === 'number') {
      self.postMessage({ type: 'progress', value: msg });
    }
  },
  performance,
);

self.onmessage = (event) => {
  const { algebraicMoves = [], simulations, timeMs, maxSimulations, uctConst } = event.data;
  const game = new Game(true);
  for (const move of algebraicMoves) {
    game.doMove(actionToGorisansonMove(parseAlgebraic(move)), true);
  }

  if (game.winner !== null) {
    self.postMessage({ type: 'error', message: 'terminal position' });
    return;
  }

  if (Number.isFinite(timeMs) && timeMs > 0) {
    try {
      const result = searchForTime(game, uctConst ?? 0.2, timeMs, maxSimulations);
      self.postMessage({
        type: 'bestmove',
        move: result.move,
        algebraicMove: toAlgebraic(gorisansonMoveToAction(result.move)),
        simulations: result.simulations,
        stoppedBy: result.stoppedBy,
        timeMs,
      });
    } catch (err) {
      self.postMessage({ type: 'error', message: err.message ?? String(err) });
    }
    return;
  }

  const ai = new AI(simulations, uctConst, false, true);
  const move = ai.chooseNextMove(game);
  self.postMessage({
    type: 'bestmove',
    move,
    algebraicMove: toAlgebraic(gorisansonMoveToAction(move)),
    simulations,
  });
};
