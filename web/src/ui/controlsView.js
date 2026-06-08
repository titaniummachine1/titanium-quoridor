import {
  STRENGTH_LEVEL_PRESETS,
  TIME_TO_MOVE_PRESETS,
  formatVisitsCap,
  formatWallClock,
  visitsFromSliderPosition,
} from '../lib/timeControl.js';
import { playerColorLabel, playerColorName } from '../lib/playerColors.js';
import { renderDiscreteSlider } from './discreteSlider.js';
import { wireRangeSlider } from './sliderWire.js';
import './scrapedSlider.css';

export function renderControls(container, state, controller) {
  const { settings, aiThinking, playerAiSettingsUi, playerOptionGroups, searchInfoLine, uiMode, replay } =
    state;
  const engineErrorLines = Object.entries(state.engineErrors ?? {})
    .filter(([, message]) => Boolean(message))
    .map(([playerType, message]) => `${playerType}: ${message}`)
    .join(' | ');
  const [p1Ui, p2Ui] = playerAiSettingsUi ?? [];
  const isReplay = uiMode === 'replay';

  container.innerHTML = `
    <section class="controls-card">
      <h1 class="app-title">Quoridor AI</h1>
      <p class="app-subtitle">Play · Titanium, Gorisanson, Ishtar, or Ka</p>

      <div class="mode-tabs">
        <button type="button" class="mode-tab ${!isReplay ? 'mode-tab--active' : ''}" data-ui-mode="play">Play</button>
        <button type="button" class="mode-tab ${isReplay ? 'mode-tab--active' : ''}" data-ui-mode="replay">Replay</button>
      </div>

      ${isReplay ? renderReplayPanel(replay) : ''}

      <div class="play-panel ${isReplay ? 'play-panel--hidden' : ''}">
      <div class="control-group">
        <label class="control-label">${playerColorLabel(1)}</label>
        ${renderPlayerSelect('player1', settings.players[0], playerOptionGroups)}
        ${renderPlayerAiSettings(p1Ui, 1)}
      </div>

      <div class="control-group">
        <label class="control-label">${playerColorName(2)}</label>
        ${renderPlayerSelect('player2', settings.players[1], playerOptionGroups)}
        ${renderPlayerAiSettings(p2Ui, 2)}
      </div>

      <div class="button-row">
        <button class="btn btn--primary" data-action="new-game">New Game</button>
        <button class="btn" data-action="undo" ${aiThinking ? 'disabled' : ''}>Undo</button>
      </div>

      <div class="toggle-group">
        <label class="toggle"><input type="checkbox" data-toggle="rotate" ${settings.rotateBoard ? 'checked' : ''} /> Rotate board</label>
        <label class="toggle"><input type="checkbox" data-toggle="coordinates" ${settings.displayCoordinates ? 'checked' : ''} /> Coordinates</label>
        <label class="toggle"><input type="checkbox" data-toggle="walls" ${settings.displayRemainingWalls ? 'checked' : ''} /> Wall count</label>
        <label class="toggle"><input type="checkbox" data-toggle="eval" ${settings.displayEvalBar ? 'checked' : ''} /> Eval bar</label>
      </div>

      <div class="status-panel">
        <div class="status-line">
          <span>Turn</span>
          <strong>${state.winner ? `Over (${playerColorName(state.winner)})` : playerColorName(state.playerToMove)}</strong>
        </div>
        <div class="status-line">
          <span>Dist (W−B)</span>
          <strong>${formatDistanceEval(state.eval)}</strong>
        </div>
        ${searchInfoLine ? `<div class="status-line status-line--muted"><span>AI</span><strong>${escapeHtml(searchInfoLine)}</strong></div>` : ''}
        ${engineErrorLines ? `<div class="status-line status-line--error"><span>Error</span><strong>${escapeHtml(engineErrorLines)}</strong></div>` : ''}
        ${state.eval.pv?.length
      ? `<div class="pv-line">PV: ${state.eval.pv.map((move) => (move.coordinate ? formatMove(move) : '?')).join(' ')}</div>`
      : ''
    }
      </div>
      </div>
    </section>
  `;

  container.querySelectorAll('[data-ui-mode]').forEach((btn) => {
    btn.addEventListener('click', () => {
      controller.setUiMode(btn.dataset.uiMode);
    });
  });

  wireReplayPanel(container, controller);

  container.querySelector('[data-setting="player1"]')?.addEventListener('change', (event) => {
    controller.setPlayer(1, event.target.value);
  });
  container.querySelector('[data-setting="player2"]')?.addEventListener('change', (event) => {
    controller.setPlayer(2, event.target.value);
  });

  wirePlayerAiSettings(container, controller, 1);
  wirePlayerAiSettings(container, controller, 2);

  container.querySelector('[data-action="new-game"]')?.addEventListener('click', () => {
    controller.newGame();
  });
  container.querySelector('[data-action="undo"]')?.addEventListener('click', () => {
    controller.undo();
  });

  container.querySelector('[data-toggle="rotate"]')?.addEventListener('change', () => {
    controller.toggleRotateBoard();
  });
  container.querySelector('[data-toggle="coordinates"]')?.addEventListener('change', () => {
    controller.toggleDisplayCoordinates();
  });
  container.querySelector('[data-toggle="walls"]')?.addEventListener('change', () => {
    controller.toggleDisplayRemainingWalls();
  });
  container.querySelector('[data-toggle="eval"]')?.addEventListener('change', () => {
    controller.toggleDisplayEvalBar();
  });
}

function wirePlayerAiSettings(container, controller, playerNum) {
  const refresh = () => controller.onChange?.();

  wireRangeSlider(
    container,
    `[data-setting="strength-level-${playerNum}"]`,
    (value) => controller.setPlayerStrengthLevel(playerNum, value, { silent: true }),
    refresh,
  );

  wireRangeSlider(
    container,
    `[data-setting="time-to-move-${playerNum}"]`,
    (value) => controller.setPlayerTimeToMove(playerNum, value, { silent: true }),
    refresh,
  );

  wireRangeSlider(
    container,
    `[data-setting="wallclock-${playerNum}"]`,
    (value) => {
      controller.setPlayerWallClock(playerNum, value, { silent: true });
      const label = container.querySelector(`[data-wallclock-label="${playerNum}"]`);
      if (label) {
        label.textContent = formatWallClock(Number(value));
      }
    },
    refresh,
  );

  wireRangeSlider(
    container,
    `[data-setting="visits-${playerNum}"]`,
    (value) => {
      const visits = visitsFromSliderPosition(value);
      controller.setPlayerVisitsBudget(playerNum, visits, { silent: true });
      const label = container.querySelector(`[data-visits-label="${playerNum}"]`);
      if (label) {
        label.textContent = formatVisitsCap(visits);
      }
    },
    refresh,
  );
}

function renderPlayerAiSettings(ui, playerNum) {
  if (!ui || ui.isHuman) {
    return '';
  }

  if (ui.isLocalMcts) {
    const { min: tMin, max: tMax, step: tStep } = ui.wallclockRange;
    const { min: vMin, max: vMax, step: vStep } = ui.visitsRange;
    const isMinimax = ui.playerType === 'titanium-minimax';
    const budgetLabel = isMinimax ? 'Node budget' : 'Rollout cap';
    return `
      <div class="player-ai-settings">
        ${ui.isTitanium
        ? renderDiscreteSlider({
          label: 'AI Strength',
          settingName: 'strength-level',
          playerNum,
          value: ui.strengthLevel,
          presets: STRENGTH_LEVEL_PRESETS,
        })
        : ''
      }
        <label class="control-label control-label--sub">Time per move</label>
        <div class="time-slider-row">
          <input
            type="range"
            class="time-slider scraped-slider"
            data-setting="wallclock-${playerNum}"
            min="${tMin}"
            max="${tMax}"
            step="${tStep}"
            value="${ui.wallClockSeconds}"
          />
          <output class="time-slider-value" data-wallclock-label="${playerNum}">${formatWallClock(ui.wallClockSeconds)}</output>
        </div>
        <label class="control-label control-label--sub">${budgetLabel}</label>
        <div class="time-slider-row">
          <input
            type="range"
            class="time-slider scraped-slider"
            data-setting="visits-${playerNum}"
            min="${vMin}"
            max="${vMax}"
            step="${vStep}"
            value="${ui.visitsSliderPosition}"
          />
          <output class="time-slider-value" data-visits-label="${playerNum}">${formatVisitsCap(ui.visitsBudget)}</output>
        </div>
        <p class="time-hint">${escapeHtml(ui.hint)}</p>
      </div>`;
  }

  return `
    <div class="player-ai-settings">
      ${renderDiscreteSlider({
    label: 'AI Strength',
    settingName: 'strength-level',
    playerNum,
    value: ui.strengthLevel,
    presets: STRENGTH_LEVEL_PRESETS,
  })}
      ${renderDiscreteSlider({
    label: 'AI Time',
    settingName: 'time-to-move',
    playerNum,
    value: ui.timeToMove,
    presets: TIME_TO_MOVE_PRESETS,
  })}
      <p class="time-hint">${escapeHtml(ui.hint)}</p>
    </div>`;
}

function renderPlayerSelect(name, value, groups) {
  const options = groups
    .map(
      (group) => `
      <optgroup label="${escapeHtml(group.label)}">
        ${group.options
          .map(
            (opt) =>
              `<option value="${opt.value}" ${opt.value === value ? 'selected' : ''} ${opt.disabled ? 'disabled' : ''}>${escapeHtml(opt.label)}</option>`,
          )
          .join('')}
      </optgroup>`,
    )
    .join('');

  return `<select class="control-select" data-setting="${name}">${options}</select>`;
}

function renderReplayPanel(replay) {
  const index = replay?.index ?? 0;
  const total = replay?.total ?? 0;
  const code = replay?.code ?? '';
  const metaLine = replay?.meta
    ? `<p class="replay-meta">${escapeHtml(JSON.stringify(replay.meta))}</p>`
    : '';

  return `
    <div class="replay-panel">
      <label class="control-label">Paste terminal replay code</label>
      <textarea class="replay-input" data-replay-input rows="4" placeholder="tq1 e2 e8 e3 …">${escapeHtml(code)}</textarea>
      ${metaLine}
      <div class="button-row">
        <button type="button" class="btn btn--primary" data-action="load-replay">Load</button>
        <button type="button" class="btn" data-action="copy-replay" ${code ? '' : 'disabled'}>Copy</button>
      </div>
      <div class="replay-scrub">
        <button type="button" class="btn btn--icon" data-action="replay-start" title="Start" ${total ? '' : 'disabled'}>⏮</button>
        <button type="button" class="btn btn--icon" data-action="replay-prev" ${total ? '' : 'disabled'}>◀</button>
        <input type="range" class="replay-slider" data-replay-slider min="0" max="${total}" value="${index}" ${total ? '' : 'disabled'} />
        <button type="button" class="btn btn--icon" data-action="replay-next" ${total ? '' : 'disabled'}>▶</button>
        <button type="button" class="btn btn--icon" data-action="replay-end" title="End" ${total ? '' : 'disabled'}>⏭</button>
      </div>
      <p class="replay-status">Ply <strong>${index}</strong> / ${total}${total ? ` · ${replayStatusLabel(replay)}` : ' — load a code'}</p>
      <p class="time-hint">Terminal prints <code>tq1 …</code> after each benchmark game. Paste here to step through on the board.</p>
    </div>`;
}

function replayStatusLabel(replay) {
  if (!replay || replay.total === 0) {
    return '';
  }
  if (replay.index === 0) {
    return 'start position';
  }
  if (replay.index >= replay.total) {
    return 'final position';
  }
  return `after move ${replay.index}`;
}

function wireReplayPanel(container, controller) {
  container.querySelector('[data-action="load-replay"]')?.addEventListener('click', () => {
    const text = container.querySelector('[data-replay-input]')?.value ?? '';
    try {
      controller.loadReplay(text);
    } catch (err) {
      window.alert(err.message ?? String(err));
    }
  });

  container.querySelector('[data-action="copy-replay"]')?.addEventListener('click', async () => {
    const code = controller.exportReplayCode();
    try {
      await navigator.clipboard.writeText(code);
    } catch {
      window.prompt('Copy replay code:', code);
    }
  });

  container.querySelector('[data-action="replay-prev"]')?.addEventListener('click', () => {
    controller.replayStep(-1);
  });
  container.querySelector('[data-action="replay-next"]')?.addEventListener('click', () => {
    controller.replayStep(1);
  });
  container.querySelector('[data-action="replay-start"]')?.addEventListener('click', () => {
    controller.setReplayIndex(0);
  });
  container.querySelector('[data-action="replay-end"]')?.addEventListener('click', () => {
    const total = controller.replay?.actions.length ?? 0;
    controller.setReplayIndex(total);
  });

  const slider = container.querySelector('[data-replay-slider]');
  slider?.addEventListener('input', () => {
    controller.setReplayIndex(Number(slider.value));
  });
}

function formatDistanceEval(evalState) {
  const w = evalState.whiteDist;
  const b = evalState.blackDist;
  if (!Number.isFinite(w) || !Number.isFinite(b)) {
    return `${Math.round((evalState.p1 ?? 0.5) * 100)}%`;
  }
  const margin = evalState.margin ?? b - w;
  const sign = margin > 0 ? '+' : '';
  return `W${w} B${b} (${sign}${margin})`;
}

function formatMove(action) {
  if (action.wallType) {
    const suffix = action.wallType === 'h' ? 'h' : 'v';
    return `${action.coordinate.column}${action.coordinate.row}${suffix}`;
  }
  return `${action.coordinate.column}${action.coordinate.row}`;
}

function escapeHtml(text) {
  return String(text)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;');
}
