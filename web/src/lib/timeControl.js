/**
 * Per-player AI settings — matches scraped quoridor-ai.netlify.app controls.
 *
 * Remote (Ishtar/Ka): AI Strength (Beg→Alpha) + AI Time (Immediate→Long) sliders.
 * Local (Gorisanson): wall-clock + visit-budget sliders.
 */

import { PlayerType, StrengthLevel, TimeToMove } from './engineConfig.js';

/** Scraped StrengthLevel slider — legacy label, kept for remote UI parity. */
export const STRENGTH_LEVEL_PRESETS = [
  { id: StrengthLevel.Beginner, label: 'Beg.' },
  { id: StrengthLevel.Intermediate, label: 'Inter.' },
  { id: StrengthLevel.Advanced, label: 'Adv.' },
  { id: StrengthLevel.Expert, label: 'Expert' },
  { id: StrengthLevel.Alpha, label: 'Alpha' },
];

/** Scraped timeToMove slider — drives visit count on cloud engines. */
export const TIME_TO_MOVE_PRESETS = [
  { id: TimeToMove.Intuition, label: 'Immediate' },
  { id: TimeToMove.Short, label: 'Short' },
  { id: TimeToMove.Medium, label: 'Medium' },
  { id: TimeToMove.Long, label: 'Long' },
];

export const WALL_CLOCK_RANGE = {
  min: 0.5,
  max: 60,
  step: 0.5,
  defaultSeconds: 10,
};

/** Exponential visit cap for local MCTS — slider is linear, stored value is log-spaced. */
export const LOCAL_VISITS_RANGE = {
  min: 1_000,
  max: 2_000_000_000,
  default: 66_000,
  sliderSteps: 1_000,
};

export const TITANIUM_NODE_CAP = 2_000_000_000;

export function clampVisits(visits) {
  const n = Number(visits);
  if (!Number.isFinite(n)) {
    return LOCAL_VISITS_RANGE.default;
  }
  return Math.min(LOCAL_VISITS_RANGE.max, Math.max(LOCAL_VISITS_RANGE.min, Math.round(n)));
}

/** Map slider position (0 … sliderSteps) → visit count. */
export function visitsFromSliderPosition(position) {
  const { min, max, sliderSteps } = LOCAL_VISITS_RANGE;
  const t = Math.min(1, Math.max(0, Number(position) / sliderSteps));
  if (t <= 0) {
    return min;
  }
  if (t >= 1) {
    return max;
  }
  return clampVisits(min * (max / min) ** t);
}

/** Map visit count → slider position for rendering. */
export function sliderPositionFromVisits(visits) {
  const { min, max, sliderSteps } = LOCAL_VISITS_RANGE;
  const clamped = clampVisits(visits);
  if (clamped <= min) {
    return 0;
  }
  if (clamped >= max) {
    return sliderSteps;
  }
  const t = Math.log(clamped / min) / Math.log(max / min);
  return Math.round(t * sliderSteps);
}

export function getEngineConfig(playerType, engineConfigs) {
  return engineConfigs.find((entry) => entry.key === playerType);
}

export function isRemoteEngine(playerType, engineConfigs) {
  return getEngineConfig(playerType, engineConfigs)?.kind === 'remote';
}

export function isLocalEngine(playerType, engineConfigs) {
  return getEngineConfig(playerType, engineConfigs)?.kind === 'local';
}

export function isTitaniumEngine(playerType, engineConfigs) {
  return getEngineConfig(playerType, engineConfigs)?.kind === 'titanium';
}

export function isLocalMctsEngine(playerType, engineConfigs) {
  const kind = getEngineConfig(playerType, engineConfigs)?.kind;
  return kind === 'local' || kind === 'titanium';
}

/** Higher UCT = more exploration — weaker play (mirrors strength tiers). */
export function uctFromStrengthLevel(strengthLevel) {
  const level = Math.min(4, Math.max(0, Number(strengthLevel ?? StrengthLevel.Alpha)));
  return [0.55, 0.45, 0.35, 0.28, 0.2][level];
}

export function defaultPlayerAiSettings(playerType, engineConfigs) {
  if (playerType === PlayerType.Human) {
    return null;
  }
  if (isTitaniumEngine(playerType, engineConfigs)) {
    return {
      strengthLevel: StrengthLevel.Alpha,
      wallClockSeconds: WALL_CLOCK_RANGE.defaultSeconds,
      visitsBudget: TITANIUM_NODE_CAP,
    };
  }
  if (isLocalEngine(playerType, engineConfigs)) {
    return {
      wallClockSeconds: WALL_CLOCK_RANGE.defaultSeconds,
      visitsBudget: LOCAL_VISITS_RANGE.default,
    };
  }
  return {
    strengthLevel: StrengthLevel.Alpha,
    timeToMove: TimeToMove.Short,
  };
}

export function formatWallClock(seconds) {
  if (seconds < 1) {
    return `${(seconds * 1000).toFixed(0)}ms`;
  }
  if (Number.isInteger(seconds)) {
    return `${seconds}s`;
  }
  return `${seconds.toFixed(1)}s`;
}

export function formatVisits(n) {
  const v = Number(n);
  if (!Number.isFinite(v)) {
    return '?';
  }
  if (v >= 1_000_000_000) {
    const billions = v / 1_000_000_000;
    return billions >= 10 ? `${Math.round(billions)}B` : `${billions.toFixed(1)}B`;
  }
  if (v >= 1_000_000) {
    return `${(v / 1_000_000).toFixed(1)}M`;
  }
  if (v >= 10_000) {
    return `${(v / 1_000).toFixed(1)}k`;
  }
  return Math.round(v).toLocaleString();
}

export function formatVisitsCap(n) {
  return `≤${formatVisits(clampVisits(n))}`;
}

function strengthLevelLabel(level) {
  return STRENGTH_LEVEL_PRESETS.find((preset) => preset.id === level)?.label ?? 'Alpha';
}

function timeToMoveLabel(timeMode) {
  return TIME_TO_MOVE_PRESETS.find((preset) => preset.id === timeMode)?.label ?? 'Short';
}

export function describePlayerAiSettings(playerType, aiSettings, engineConfigs) {
  if (playerType === PlayerType.Human || !aiSettings) {
    return '';
  }
  const config = getEngineConfig(playerType, engineConfigs);
  if (!config) {
    return '';
  }

  if (isLocalMctsEngine(playerType, engineConfigs)) {
    const time = formatWallClock(aiSettings.wallClockSeconds ?? WALL_CLOCK_RANGE.defaultSeconds);
    const cap = formatVisitsCap(aiSettings.visitsBudget ?? LOCAL_VISITS_RANGE.default);
    if (isTitaniumEngine(playerType, engineConfigs)) {
      const modeLabel =
        config.engineMode === 'minimax' ? 'Hybrid MCTS opening→AB' : 'Rust MCTS only';
      const budgetLabel = config.engineMode === 'minimax' ? 'nodes' : 'sims';
      return `${config.name}: ${time} · ${cap} ${budgetLabel} · ${modeLabel}`;
    }
    return `${config.name}: ${time} · ${cap}`;
  }

  if (isRemoteEngine(playerType, engineConfigs) && config.visits) {
    const timeMode = aiSettings.timeToMove ?? TimeToMove.Short;
    const visits = config.visits[timeMode];
    const parallelism = config.settings?.parallelism?.[timeMode];
    const strength = strengthLevelLabel(aiSettings.strengthLevel ?? StrengthLevel.Alpha);
    const time = timeToMoveLabel(timeMode);
    let text = `${config.name}: ${strength} · ${time} (~${visits.toLocaleString()} visits)`;
    if (parallelism) {
      text += ` · ${parallelism} threads`;
    }
    return text;
  }

  return config.name;
}

export function describeAiSettingsForPlayers(players, playerAiSettings, engineConfigs) {
  const lines = players
    .map((playerType, index) =>
      describePlayerAiSettings(playerType, playerAiSettings[index], engineConfigs),
    )
    .filter(Boolean);
  return lines.length ? lines.join(' · ') : 'No AI selected.';
}
