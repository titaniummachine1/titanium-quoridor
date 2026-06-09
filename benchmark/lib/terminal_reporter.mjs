/**
 * Readable terminal layout for head-to-head benchmarks.
 */

import { moveAlgebraic, renderBoardAscii, sideName } from './terminal_board.mjs';
import { termLine } from './terminal_log.mjs';

const MODE_LABEL = {
  trivial: 'instant',
  race: 'win-path',
  hybrid: 'hybrid',
  minimax: 'minimax',
  mcts: 'MCTS',
  time: 'MCTS·time',
  visits: 'MCTS·cap',
  converged: 'MCTS·done',
  remote: 'Ka',
  mcts: 'MCTS',
  time: 'MCTS·time',
  visits: 'MCTS·cap',
  opening: 'opening',
  'rust-greedy': 'Rust greedy',
  opening: 'opening',
};

export function formatSearchMode(result) {
  const key = result.stoppedBy ?? 'mcts';
  const base = MODE_LABEL[key] ?? key;
  if ((key === 'minimax' || key === 'hybrid') && result.searchDepth) {
    return `${base} d${result.searchDepth}`;
  }
  if (key === 'remote' && result.simulations) {
    return `${result.simulations} visits`;
  }
  if (
    (key === 'mcts' ||
      key === 'time' ||
      key === 'visits' ||
      key === 'converged' ||
      key === 'hybrid' ||
      key === 'opening') &&
    result.simulations != null
  ) {
    return `${base} ${formatSims(result.simulations)}`;
  }
  return base;
}

export function formatScore(score) {
  if (score == null || !Number.isFinite(score)) {
    return '?';
  }
  if (Math.abs(score) >= 19_500) {
    return score > 0 ? '#+' : '#-';
  }
  return score > 0 ? `+${score}` : String(score);
}

/** Engine log: mode, dist, per-depth eval, nodes/sims. */
export function formatEngineLog(result) {
  if (!result) {
    return '';
  }
  const parts = [formatSearchMode(result)];

  if (result.whiteDist != null && result.blackDist != null) {
    parts.push(`dist W${result.whiteDist} B${result.blackDist}`);
  }

  if (result.depthLog?.length) {
    const depthStr = result.depthLog
      .map((entry) => `d${entry.depth}=${formatScore(entry.score)}`)
      .join(' ');
    parts.push(depthStr);
  } else if (result.rootScore != null) {
    parts.push(`eval=${formatScore(result.rootScore)}`);
  }

  if (result.nodes) {
    parts.push(`${result.nodes} nodes`);
  }

  if (result.prunedWalls) {
    parts.push(`${result.prunedWalls} wall-pruned`);
  }

  if (result.aspirationFails) {
    parts.push(`asp↺${result.aspirationFails}`);
  }

  if (result.lmrReSearches) {
    parts.push(`LMR↺${result.lmrReSearches}`);
  }

  if (result.mateExtensions) {
    parts.push(`mate+${result.mateExtensions}`);
  }

  if (result.pvMateFailures) {
    parts.push(`pv✗${result.pvMateFailures}`);
  }

  return parts.join(' · ');
}

/** Live iterative-deepening line (streamed during search). */
export function printSearchDepth({ ply, depth, score, nodes, aspirationFails, lmrReSearches, starting }) {
  const plyTag = ply != null ? `ply ${ply} ` : '';
  if (starting) {
    termLine(`      ${plyTag}… searching depth ${depth}  (${nodes} nodes so far)`);
    return;
  }
  const asp = aspirationFails ? ` asp↺${aspirationFails}` : '';
  const lmr = lmrReSearches ? ` LMR↺${lmrReSearches}` : '';
  termLine(`      ${plyTag}d${depth}  eval=${formatScore(score)}  ${nodes} nodes${asp}${lmr}`);
}

/** Classic one-liner with search diagnostics. */
export function printPlyCompact({ ply, who, engine, result, move }) {
  const algebraic = moveAlgebraic(move);
  const log = formatEngineLog(result);
  const tag = log ? `${engine} (${log})` : engine;
  termLine(`  ply ${ply} ${tag}: ${algebraic}`);
}

export function printMoveList(algebraicHistory) {
  if (!algebraicHistory.length) {
    return;
  }
  console.log('');
  console.log('  moves:');
  const chunks = [];
  for (let i = 0; i < algebraicHistory.length; i += 2) {
    const w = `${i + 1}. ${algebraicHistory[i]}`;
    const b = algebraicHistory[i + 1] ? `  ${i + 2}. ${algebraicHistory[i + 1]}` : '';
    chunks.push(`    ${w}${b}`);
  }
  console.log(chunks.join('\n'));
}

export function printFinalPosition(game, { winnerSide, winnerLabel, algebraicHistory }) {
  console.log('');
  console.log(`── final position${winnerLabel ? ` · ${winnerLabel} wins` : ''} ──`);
  console.log(renderBoardAscii(game));
  printMoveList(algebraicHistory);
}

export function formatSims(n) {
  if (n >= 1_000_000) {
    return `${(n / 1_000_000).toFixed(1)}M`;
  }
  if (n >= 1000) {
    return `${(n / 1000).toFixed(1)}k`;
  }
  return String(n);
}

export function printGameHeader(gameIndex, total, titaniumSide, opponent = 'Ka (remote)') {
  const color = sideName(titaniumSide);
  console.log('');
  console.log('┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓');
  console.log(
    `┃  Game ${gameIndex}/${total}  ·  Titanium ${color.padEnd(5)}  ·  vs ${opponent.padEnd(18)}┃`,
  );
  console.log('┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛');
}

export function printPly({
  ply,
  who,
  engine,
  result,
  move,
  elapsedMs,
  game,
  showBoard,
}) {
  const side = sideName(who);
  const algebraic = moveAlgebraic(move);
  const mode = formatSearchMode(result ?? { stoppedBy: 'remote' });
  const ms = elapsedMs != null ? `${elapsedMs.toFixed(0)}ms` : '—';

  console.log('');
  const log = formatEngineLog(result);
  console.log(`── ply ${String(ply).padStart(3)} ── ${side} / ${engine} ── ${log} (${ms}) ──`);
  console.log(`    ▶ ${algebraic}`);
  if (showBoard) {
    console.log(renderBoardAscii(game));
  }
}

export function printGameSummary({ gameIndex, winnerLabel, plies, stats }) {
  console.log('');
  console.log(`── result game ${gameIndex}: ${winnerLabel} in ${plies} plies ──`);
  if (stats.rustMoves != null) {
    console.log(`    Titanium Rust moves: ${stats.rustMoves}`);
  } else if (stats.race != null || stats.hybrid != null) {
    console.log(
      `    Titanium modes: ${stats.race ?? 0} win-path · ${stats.hybrid ?? 0} hybrid · ` +
        `${stats.minimax ?? 0} minimax · ${stats.mcts ?? 0} mcts · ${stats.trivial ?? 0} instant`,
    );
    if (stats.lmrReSearches) {
      console.log(`    LMR re-searches: ${stats.lmrReSearches}`);
    }
  }
}

export function printMatchFooter({
  titaniumWins,
  opponentWins,
  kaWins,
  opponentName = 'Ka',
  totals,
  elapsedSec,
  games,
}) {
  const oppWins = opponentWins ?? kaWins ?? 0;
  console.log('');
  console.log('┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓');
  console.log(
    `┃  Score:  Titanium ${titaniumWins}  —  ${opponentName} ${oppWins}`.padEnd(55) + '┃',
  );
  if (totals.rustMoves != null) {
    console.log(`┃  Rust moves: ${totals.rustMoves}`.padEnd(55) + '┃');
  } else if (totals.hybrid != null) {
    console.log(
      `┃  Hybrid: ${totals.race ?? 0} win-path · ${totals.hybrid ?? 0} hybrid · ` +
        `${totals.minimax ?? 0} mm · ${totals.mcts ?? 0} mcts`.padEnd(55) +
        '┃',
    );
  }
  console.log(
    `┃  Think:  Ti ${(totals.titaniumMs / 1000).toFixed(0)}s  Ka ${(totals.kaMs / 1000).toFixed(0)}s  ·  ` +
      `wall ${elapsedSec.toFixed(0)}s (${(elapsedSec / games).toFixed(0)}s/game)`.padEnd(55) +
      '┃',
  );
  console.log('┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛');
}
