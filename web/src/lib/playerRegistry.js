/**
 * Opponent registry — local, remote, and future competition targets.
 */

import { PlayerType, getEngineList } from './engineConfig.js';
import {
  STRENGTH_LEVEL_PRESETS,
  TIME_TO_MOVE_PRESETS,
  describeAiSettingsForPlayers,
  formatWallClock,
} from './timeControl.js';

export { STRENGTH_LEVEL_PRESETS, TIME_TO_MOVE_PRESETS };
/** @deprecated use TIME_TO_MOVE_PRESETS */
export const TIME_PRESETS = TIME_TO_MOVE_PRESETS;

const GORISANSON_ENGINE = {
  kind: 'local',
  name: 'Gorisanson (JS, original)',
  key: PlayerType.GorisansonMCTS,
  tooltip: 'Original JavaScript MCTS — first boss (github.com/gorisanson/quoridor-ai)',
  uctConst: 0.2,
};

const TITANIUM_ENGINE = {
  kind: 'titanium',
  name: 'Titanium (MCTS, Rust)',
  key: PlayerType.Titanium,
  engineMode: 'mcts',
  tooltip: 'Updated local Rust MCTS engine — `titanium genmove` (cargo build --release in engine/)',
};

const TITANIUM_MINIMAX_ENGINE = {
  kind: 'titanium',
  name: 'Titanium Hybrid (strongest)',
  key: PlayerType.TitaniumMinimax,
  engineMode: 'minimax',
  tooltip:
    'Full pipeline: MCTS opening/book → minimax+CAT v2 (`cargo build --release` in engine/)',
};

const PLACEHOLDER_ENGINES = [
  {
    kind: 'placeholder',
    name: 'pavlosdais (C αβ)',
    key: PlayerType.Pavlosdais,
    tooltip: 'Competition baseline — not wired yet',
    disabled: true,
  },
];

export function getAllEngineConfigs() {
  const remote = getEngineList().map((entry) => ({
    ...entry,
    kind: 'remote',
  }));
  return [GORISANSON_ENGINE, TITANIUM_ENGINE, TITANIUM_MINIMAX_ENGINE, ...remote, ...PLACEHOLDER_ENGINES];
}

export function getPlayerOptionGroups() {
  return [
    {
      label: 'Human',
      options: [{ value: PlayerType.Human, label: 'Human', disabled: false }],
    },
    {
      label: 'Local — beat these first',
      options: [
        {
          value: PlayerType.GorisansonMCTS,
          label: 'Gorisanson (JS, original)',
          disabled: false,
          tooltip: GORISANSON_ENGINE.tooltip,
        },
        {
          value: PlayerType.Titanium,
          label: 'Titanium (MCTS, Rust)',
          disabled: false,
          tooltip: TITANIUM_ENGINE.tooltip,
        },
        {
          value: PlayerType.TitaniumMinimax,
          label: 'Titanium Hybrid (strongest)',
          disabled: false,
          tooltip: TITANIUM_MINIMAX_ENGINE.tooltip,
        },
      ],
    },
    {
      label: 'Remote',
      options: [
        { value: PlayerType.IshtarV3, label: 'Ishtar', disabled: false },
        { value: PlayerType.KaAI, label: 'Ka', disabled: false },
      ],
    },
    {
      label: 'Competition (planned)',
      options: [
        { value: PlayerType.Pavlosdais, label: 'pavlosdais C', disabled: true },
      ],
    },
  ];
}

export function flattenPlayerOptions(groups) {
  return groups.flatMap((group) => group.options);
}

export function describeTimeBudget(players, playerAiSettings, engineConfigs) {
  return describeAiSettingsForPlayers(players, playerAiSettings, engineConfigs);
}

export function describeActiveSearchInfo(players, searchInfoByType, engineConfigs) {
  const aiTypes = players.filter((p) => p !== PlayerType.Human);
  const lines = aiTypes
    .map((playerType) =>
      describeSearchInfo(playerType, searchInfoByType[playerType], engineConfigs),
    )
    .filter(Boolean);
  return lines.join(' · ');
}

export function describeSearchInfo(playerType, searchInfo, engineConfigs) {
  if (!searchInfo || playerType === PlayerType.Human) {
    return '';
  }
  const config = engineConfigs.find((entry) => entry.key === playerType);
  if ((config?.kind === 'local' || config?.kind === 'titanium') && searchInfo.time != null) {
    const isMinimax = searchInfo.stoppedBy === 'minimax';
    const budgetLabel = isMinimax
      ? `${(searchInfo.nodes ?? 0).toLocaleString()} nodes`
      : `${searchInfo.simulations?.toLocaleString() ?? '?'} sims`;
    const winPart =
      searchInfo.rootWinRate != null
        ? ` · wr ${(searchInfo.rootWinRate * 100).toFixed(0)}%`
        : '';
    const depthPart =
      searchInfo.depthLog?.length > 0
        ? ` · ${searchInfo.depthLog.map((e) => `d${e.depth}=${e.score > 0 ? '+' : ''}${e.score}`).join(' ')}`
        : searchInfo.searchDepth
          ? ` · d${searchInfo.searchDepth}`
          : '';
    const distPart =
      searchInfo.whiteDist != null
        ? ` · W${searchInfo.whiteDist} B${searchInfo.blackDist}`
        : '';
    const stopLabels = {
      visits: 'hit cap',
      time: 'time',
      converged: 'converged',
      trivial: 'instant',
      opening: 'instant',
      minimax: 'minimax',
      mcts: 'MCTS',
      time: 'MCTS·time',
      visits: 'MCTS·cap',
      opening: 'opening',
      hybrid: 'hybrid',
      race: 'win path',
    };
    const limit = stopLabels[searchInfo.stoppedBy] ?? '';
    const suffix = limit ? ` (${limit})` : '';
    const profile =
      searchInfo.profileName && isMinimax ? ` · ${searchInfo.profileName}` : '';
    return `Last think: ${formatWallClock(searchInfo.time / 1000)} · ${budgetLabel}${winPart}${depthPart}${distPart}${profile}${suffix}`;
  }
  if (config?.kind === 'remote') {
    const parts = [];
    if (searchInfo.time != null) {
      parts.push(`${searchInfo.time}ms`);
    }
    if (searchInfo.visits != null) {
      parts.push(`${searchInfo.visits.toLocaleString()} visits`);
    }
    return parts.length ? `Last think: ${parts.join(' · ')}` : '';
  }
  return '';
}
