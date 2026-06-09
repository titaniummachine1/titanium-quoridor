import { formatCoordinate, toAlgebraic } from '../lib/gameLogic.js';
import { playerColorName } from '../lib/playerColors.js';

export function renderGameFooter(container, state) {
  const {
    winner,
    playerToMove,
    actions,
    liveSearch,
    aiThinking,
    replay,
    uiMode,
    playReplayCode,
    moveThinkLog,
  } = state;

  let turnText;
  if (uiMode === 'replay' && replay) {
    turnText = `Replay ply ${replay.index} / ${replay.total}`;
  } else if (winner) {
    turnText = `Game over — ${playerColorName(winner)} wins`;
  } else {
    turnText = `Turn: ${playerColorName(playerToMove)}`;
  }

  const moveText =
    actions.length === 0
      ? '—'
      : actions.map((action, index) => `${index + 1}. ${toAlgebraic(action)}`).join('  ');

  let liveLine = '';
  if (aiThinking && liveSearch) {
    const who = liveSearch.playerLabel ?? 'AI';
    const modeLabel = formatSearchMode(liveSearch.mode);
    if (liveSearch.mode) {
      liveLine = `${who}: ${modeLabel}`;
      if (liveSearch.searchDepth) {
        liveLine += ` depth=${liveSearch.searchDepth}`;
      }
      if (liveSearch.nodes) {
        liveLine += ` · ${liveSearch.nodes.toLocaleString()} nodes`;
      } else if (liveSearch.simulations) {
        liveLine += ` · ${liveSearch.simulations.toLocaleString()} sims`;
      }
      if (liveSearch.depthLog?.length) {
        const last = liveSearch.depthLog[liveSearch.depthLog.length - 1];
        liveLine += ` · eval=${formatEngineScore(last.score)}`;
      } else if (liveSearch.rootWinRate != null) {
        liveLine += ` · wr ${(liveSearch.rootWinRate * 100).toFixed(0)}%`;
      }
      if (liveSearch.bestMove) {
        liveLine += ` · best=${liveSearch.bestMove}`;
      }
    } else if (liveSearch.simulations) {
      liveLine = `${who}: thinking… ${liveSearch.simulations.toLocaleString()} sims`;
    } else {
      liveLine = `${who}: thinking…`;
    }
  }

  const replayBlock =
    playReplayCode && uiMode === 'play'
      ? `<div class="game-footer__replay"><span class="game-footer__replay-label">Replay code (paste in Replay tab):</span> <code class="game-footer__replay-code">${escapeHtml(playReplayCode)}</code></div>`
      : '';

  const wasOpen = container.querySelector('details.think-chain')?.open ?? false;

  container.innerHTML = `
    <div class="game-footer__row game-footer__row--turn">
      <strong>${turnText}</strong>
      ${liveLine ? `<span class="game-footer__live">${escapeHtml(liveLine)}</span>` : ''}
    </div>
    <div class="game-footer__moves" title="${escapeHtml(moveText)}">${escapeHtml(moveText)}</div>
    ${replayBlock}
    ${buildThinkChainBlock(moveThinkLog, wasOpen)}
  `;

  const copyBtn = container.querySelector('[data-action="copy-think-chain"]');
  if (copyBtn) {
    copyBtn.addEventListener('click', () => {
      const text = buildGameExportText(state);
      navigator.clipboard.writeText(text).catch(() => {
        const ta = document.createElement('textarea');
        ta.value = text;
        document.body.appendChild(ta);
        ta.select();
        document.execCommand('copy');
        document.body.removeChild(ta);
      });
      copyBtn.textContent = 'Copied!';
      setTimeout(() => { copyBtn.textContent = 'Copy game report'; }, 1500);
    });
  }
}

function formatSearchMode(mode) {
  const labels = {
    searching: 'searching',
    mcts: 'MCTS',
    minimax: 'αβ+LMR',
    hybrid: 'hybrid',
    race: 'win path',
    trivial: 'instant',
    converged: 'MCTS ✓',
    visits: 'MCTS cap',
    time: 'MCTS',
  };
  return labels[mode] ?? mode;
}

function escapeHtml(text) {
  return String(text)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;');
}

function isMateScore(score) {
  return Math.abs(Number(score) || 0) >= 19_500;
}

function formatEngineScore(score) {
  if (score == null || !Number.isFinite(Number(score))) {
    return '?';
  }
  const n = Number(score);
  if (isMateScore(n)) {
    const sign = n > 0 ? '+' : '-';
    return `${sign}M${Math.max(0, 20_000 - Math.abs(n))}`;
  }
  const meters = n / 100;
  return `${meters > 0 ? '+' : ''}${meters.toFixed(2)}`;
}

function formatDepthLog(depthLog) {
  return depthLog
    .map((e) => `d${e.depth}=${formatEngineScore(e.score)}`)
    .join(' ');
}

function isTitaniumThinkEntry(entry) {
  return entry.engine?.includes('Titanium');
}

/** Top root candidates for copied reports: `roots: d5=-991 W6/B5 g0; h3h=-803 W5/B6 g2` */
function formatRootMovesSummary(rootMoves) {
  if (!rootMoves?.length) {
    return '';
  }
  const roots = [...rootMoves]
    .sort((a, b) => b.score - a.score)
    .slice(0, 5)
    .map((r) => `${r.move}=${formatEngineScore(r.score)} W${r.whiteDist}/B${r.blackDist} g${r.gain ?? 0}`)
    .join('; ');
  return ` roots: ${roots}`;
}

function formatThinkEntry(entry) {
  const who =
    entry.ply % 2 === 1 ? 'White' : 'Black';
  const engine = entry.engine ? ` [${entry.engine}]` : '';
  const dist =
    entry.whiteDist != null && entry.blackDist != null
      ? ` W${entry.whiteDist} B${entry.blackDist}`
      : '';

  const isMcts = entry.stoppedBy === 'mcts' || entry.stoppedBy === 'time' ||
    entry.stoppedBy === 'visits' || entry.stoppedBy === 'bridge' || entry.stoppedBy === 'bridge-visits' ||
    entry.stoppedBy === 'forced' || entry.stoppedBy === 'win-in-1' || entry.stoppedBy === 'opening';

  const sims = entry.nodes > 0 ? ` ${entry.nodes.toLocaleString()}nodes` : '';
  const wr = entry.rootWinRate != null && isMcts
    ? ` wr=${(entry.rootWinRate * 100).toFixed(0)}%`
    : '';
  const rootCands =
    isTitaniumThinkEntry(entry) ? formatRootMovesSummary(entry.rootMoves) : '';

  if (isMcts && !entry.depthLog?.length) {
    const stopped = entry.stoppedBy ? ` (${entry.stoppedBy})` : '';
    return `ply${entry.ply} ${who}${engine} ${entry.move}${dist}${sims}${wr}${stopped}${rootCands}`;
  }

  const depth = entry.searchDepth ? ` d${entry.searchDepth}` : '';
  const dlog =
    entry.depthLog?.length
      ? ' ' + formatDepthLog(entry.depthLog)
      : '';

  return `ply${entry.ply} ${who}${engine} ${entry.move}${dist}${depth}${sims}${dlog}${rootCands}`;
}

function engineLabelForSlot(state, playerNum) {
  const playerType = state.settings?.players?.[playerNum - 1];
  const opt = state.playerOptions?.find((entry) => entry.value === playerType);
  return opt?.label ?? playerType ?? '?';
}

function formatMargin(margin) {
  if (margin == null || !Number.isFinite(margin)) {
    return '?';
  }
  return margin > 0 ? `+${margin}` : String(margin);
}

function raceVerdict(winner, loserDist, closestMargin) {
  if (!winner) {
    if (closestMargin != null && Math.abs(closestMargin) <= 1) {
      return 'live — race within 1 step';
    }
    if (closestMargin != null && Math.abs(closestMargin) <= 3) {
      return 'live — close race';
    }
    return 'in progress';
  }
  if (loserDist <= 1) {
    return 'photo finish — loser 0–1 steps from goal';
  }
  if (loserDist <= 3) {
    return 'close — loser within 3 steps of goal';
  }
  if (loserDist >= 8) {
    return 'blowout — loser far from goal';
  }
  return 'decisive';
}

function summarizeRaceFromLog(log) {
  let closestMargin = null;
  let maxWhiteLead = null;
  let maxBlackLead = null;
  for (const entry of log ?? []) {
    if (entry.whiteDist == null || entry.blackDist == null) {
      continue;
    }
    const margin = entry.blackDist - entry.whiteDist;
    if (closestMargin === null || Math.abs(margin) < Math.abs(closestMargin)) {
      closestMargin = margin;
    }
    if (maxWhiteLead === null || margin > maxWhiteLead) {
      maxWhiteLead = margin;
    }
    if (maxBlackLead === null || margin < maxBlackLead) {
      maxBlackLead = margin;
    }
  }
  return { closestMargin, maxWhiteLead, maxBlackLead };
}

function buildGameHeader(state) {
  const {
    winner,
    actions,
    playerToMove,
    playerPositions,
    wallsRemaining,
    eval: evalState,
    playReplayCode,
    timeBudgetHint,
    moveThinkLog,
  } = state;

  const plies = actions?.length ?? 0;
  const whiteSq = playerPositions?.[0] ? formatCoordinate(playerPositions[0]) : '?';
  const blackSq = playerPositions?.[1] ? formatCoordinate(playerPositions[1]) : '?';
  const wDist = evalState?.whiteDist;
  const bDist = evalState?.blackDist;
  const margin = evalState?.margin;
  const wallsUsedW = wallsRemaining?.[0] != null ? 10 - wallsRemaining[0] : '?';
  const wallsUsedB = wallsRemaining?.[1] != null ? 10 - wallsRemaining[1] : '?';

  const { closestMargin, maxWhiteLead, maxBlackLead } = summarizeRaceFromLog(moveThinkLog);
  const loserDist =
    winner === 1 ? bDist : winner === 2 ? wDist : null;

  const lines = ['=== Quoridor game report ===', ''];

  if (winner) {
    lines.push(`Result: ${playerColorName(winner)} wins · ${plies} plies`);
  } else {
    lines.push(
      `Result: in progress · ply ${plies} · ${playerColorName(playerToMove)} to move`,
    );
  }

  lines.push(
    `White: ${engineLabelForSlot(state, 1)}`,
    `Black: ${engineLabelForSlot(state, 2)}`,
  );
  if (timeBudgetHint) {
    lines.push(`Budget: ${timeBudgetHint}`);
  }
  lines.push('');

  lines.push(
    `Final position: White=${whiteSq} Black=${blackSq}`,
    `Path distance: W=${wDist ?? '?'} B=${bDist ?? '?'} · margin=${formatMargin(margin)} (positive = White ahead)`,
    `Walls used: White=${wallsUsedW} Black=${wallsUsedB} · left W=${wallsRemaining?.[0] ?? '?'} B=${wallsRemaining?.[1] ?? '?'}`,
  );

  if (closestMargin != null) {
    lines.push(
      `Race swing: closest margin=${formatMargin(closestMargin)} · best White lead=${formatMargin(maxWhiteLead)} · best Black lead=${formatMargin(maxBlackLead)}`,
    );
  }

  lines.push(`Verdict: ${raceVerdict(winner, loserDist ?? 99, closestMargin)}`);

  if (playReplayCode) {
    lines.push('', `Replay: ${playReplayCode}`);
  } else if (actions?.length) {
    const compact = actions.map((a) => toAlgebraic(a)).join(' ');
    lines.push('', `Moves: ${compact}`);
  }

  return lines.join('\n');
}

export function buildGameExportText(state) {
  const header = buildGameHeader(state);
  const log = state.moveThinkLog;
  if (!log?.length) {
    return `${header}\n\n--- Think chain ---\n(no AI think log yet)`;
  }
  return `${header}\n\n--- Think chain ---\n${log.map(formatThinkEntry).join('\n')}`;
}

function buildThinkChainBlock(log, keepOpen = false) {
  if (!log?.length) {
    return '';
  }
  const rows = log
    .map((entry) => `<div class="think-chain__row">${escapeHtml(formatThinkEntry(entry))}</div>`)
    .join('');
  return `
    <details class="think-chain"${keepOpen ? ' open' : ''}>
      <summary class="think-chain__summary">
        Think chain (${log.length} ply)
        <button type="button" class="btn btn--small" data-action="copy-think-chain">Copy game report</button>
      </summary>
      <div class="think-chain__log">${rows}</div>
    </details>
  `;
}
