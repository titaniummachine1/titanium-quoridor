import { GameSession } from './gameSession.js';
import { EngineClient } from '../lib/engineClient.js';
import { GorisansonEngineClient, TitaniumEngineClient } from '../lib/localMctsEngine.js';
import { PlayerType, StrengthLevel, TimeToMove } from '../lib/engineConfig.js';
import {
  STRENGTH_LEVEL_PRESETS,
  TIME_TO_MOVE_PRESETS,
  getAllEngineConfigs,
  getPlayerOptionGroups,
  flattenPlayerOptions,
  describeTimeBudget,
  describeActiveSearchInfo,
} from '../lib/playerRegistry.js';
import {
  WALL_CLOCK_RANGE,
  LOCAL_VISITS_RANGE,
  clampVisits,
  sliderPositionFromVisits,
  defaultPlayerAiSettings,
  describePlayerAiSettings,
  isLocalEngine,
  isLocalMctsEngine,
  isRemoteEngine,
  isTitaniumEngine,
} from '../lib/timeControl.js';
import { ponderCandidateSlots } from '../lib/enginePonder.js';

function isSavedSettingsValid(playerType, saved, engineConfigs) {
  if (isTitaniumEngine(playerType, engineConfigs)) {
    return (
      saved.strengthLevel != null &&
      saved.wallClockSeconds != null &&
      saved.visitsBudget != null
    );
  }
  if (isLocalEngine(playerType, engineConfigs)) {
    return saved.wallClockSeconds != null && saved.visitsBudget != null;
  }
  if (isRemoteEngine(playerType, engineConfigs)) {
    return saved.strengthLevel != null && saved.timeToMove != null;
  }
  return false;
}

export class AppController {
  constructor() {
    this.session = new GameSession();
    this.engines = new Map();
    this.engineConfigs = getAllEngineConfigs();

    this.settings = {
      players: [PlayerType.Human, PlayerType.IshtarV3],
      playerAiSettings: [
        null,
        defaultPlayerAiSettings(PlayerType.IshtarV3, this.engineConfigs),
      ],
      playerAiSettingsMemory: [{}, {}],
      rotateBoard: false,
      displayCoordinates: true,
      displayRemainingWalls: true,
      displayEvalBar: true,
    };

    this.engineStatus = {};
    this.searchInfo = {};
    this.eval = { score: 0.5, p1: 0.5, pv: [] };
    this.aiThinking = false;

    this.session.subscribe(() => this.onSessionChange());
  }

  getState() {
    return {
      ...this.session.getSnapshot(),
      settings: { ...this.settings },
      engineStatus: { ...this.engineStatus },
      eval: { ...this.eval },
      aiThinking: this.aiThinking,
      strengthLevelPresets: STRENGTH_LEVEL_PRESETS,
      timeToMovePresets: TIME_TO_MOVE_PRESETS,
      playerOptionGroups: getPlayerOptionGroups(),
      playerOptions: flattenPlayerOptions(getPlayerOptionGroups()),
      playerAiSettingsUi: this.getPlayerAiSettingsUi(),
      timeBudgetHint: describeTimeBudget(
        this.settings.players,
        this.settings.playerAiSettings,
        this.engineConfigs,
      ),
      searchInfoLine: describeActiveSearchInfo(
        this.settings.players,
        this.searchInfo,
        this.engineConfigs,
      ),
    };
  }

  onChange = null;

  setPlayer(playerNum, playerType) {
    if (playerType === PlayerType.Pavlosdais) {
      return;
    }
    this.settings.players[playerNum - 1] = playerType;
    this.ensurePlayerAiSettingsSlot(playerNum, playerType);

    if (playerType !== PlayerType.Human && playerType !== PlayerType.GorisansonMCTS) {
      this.syncRemoteEngine(playerType);
    }
    this.onChange?.();
    this.maybeRequestAiMove();
  }

  ensurePlayerAiSettingsSlot(playerNum, playerType) {
    const index = playerNum - 1;
    if (playerType === PlayerType.Human) {
      return;
    }

    const memory = this.settings.playerAiSettingsMemory[index] ?? {};
    let saved = memory[playerType];
    if (saved?.strength != null && saved.timeToMove == null) {
      saved = {
        strengthLevel: StrengthLevel.Alpha,
        timeToMove: saved.strength,
      };
      memory[playerType] = saved;
    }
    if (saved && isSavedSettingsValid(playerType, saved, this.engineConfigs)) {
      this.settings.playerAiSettings[index] = { ...saved };
      return;
    }

    const created = defaultPlayerAiSettings(playerType, this.engineConfigs);
    memory[playerType] = { ...created };
    this.settings.playerAiSettingsMemory[index] = memory;
    this.settings.playerAiSettings[index] = created;
  }

  rememberPlayerAiSettings(playerNum, aiSettings) {
    const index = playerNum - 1;
    const playerType = this.settings.players[index];
    if (playerType === PlayerType.Human || !aiSettings) {
      return;
    }
    const memory = this.settings.playerAiSettingsMemory[index] ?? {};
    memory[playerType] = { ...aiSettings };
    this.settings.playerAiSettingsMemory[index] = memory;
    this.settings.playerAiSettings[index] = { ...aiSettings };
  }

  getPlayerAiSettingsUiForSlot(playerNum) {
    const index = playerNum - 1;
    const playerType = this.settings.players[index];
    const current = this.settings.playerAiSettings[index];

    return {
      playerNum,
      playerType,
      isHuman: playerType === PlayerType.Human,
      isLocal: isLocalEngine(playerType, this.engineConfigs),
      isTitanium: isTitaniumEngine(playerType, this.engineConfigs),
      isLocalMcts: isLocalMctsEngine(playerType, this.engineConfigs),
      isRemote: isRemoteEngine(playerType, this.engineConfigs),
      strengthLevel: current?.strengthLevel ?? StrengthLevel.Alpha,
      timeToMove: current?.timeToMove ?? TimeToMove.Short,
      wallClockSeconds: current?.wallClockSeconds ?? WALL_CLOCK_RANGE.defaultSeconds,
      visitsBudget: clampVisits(current?.visitsBudget ?? LOCAL_VISITS_RANGE.default),
      visitsSliderPosition: sliderPositionFromVisits(
        current?.visitsBudget ?? LOCAL_VISITS_RANGE.default,
      ),
      wallclockRange: WALL_CLOCK_RANGE,
      visitsRange: {
        min: 0,
        max: LOCAL_VISITS_RANGE.sliderSteps,
        step: 1,
      },
      hint: describePlayerAiSettings(playerType, current, this.engineConfigs),
    };
  }

  getPlayerAiSettingsUi() {
    return [this.getPlayerAiSettingsUiForSlot(1), this.getPlayerAiSettingsUiForSlot(2)];
  }

  setPlayerStrengthLevel(playerNum, strengthLevel, { silent = false } = {}) {
    const index = playerNum - 1;
    const playerType = this.settings.players[index];
    if (!isRemoteEngine(playerType, this.engineConfigs) && !isTitaniumEngine(playerType, this.engineConfigs)) {
      return;
    }
    const current = this.settings.playerAiSettings[index] ?? {};
    this.rememberPlayerAiSettings(playerNum, {
      ...current,
      strengthLevel: Number(strengthLevel),
    });
    if (!silent) {
      this.onChange?.();
    }
  }

  setPlayerTimeToMove(playerNum, timeToMove, { silent = false } = {}) {
    const index = playerNum - 1;
    const playerType = this.settings.players[index];
    if (!isRemoteEngine(playerType, this.engineConfigs)) {
      return;
    }
    const current = this.settings.playerAiSettings[index] ?? {};
    this.rememberPlayerAiSettings(playerNum, {
      ...current,
      timeToMove: Number(timeToMove),
    });
    if (!silent) {
      this.onChange?.();
    }
  }

  setPlayerWallClock(playerNum, seconds, { silent = false } = {}) {
    const index = playerNum - 1;
    const playerType = this.settings.players[index];
    if (!isLocalMctsEngine(playerType, this.engineConfigs)) {
      return;
    }
    const current = this.settings.playerAiSettings[index] ?? {};
    this.rememberPlayerAiSettings(playerNum, {
      ...current,
      wallClockSeconds: Number(seconds),
    });
    if (!silent) {
      this.onChange?.();
    }
  }

  setPlayerVisitsBudget(playerNum, visits, { silent = false } = {}) {
    const index = playerNum - 1;
    const playerType = this.settings.players[index];
    if (!isLocalMctsEngine(playerType, this.engineConfigs)) {
      return;
    }
    const current = this.settings.playerAiSettings[index] ?? {};
    this.rememberPlayerAiSettings(playerNum, {
      ...current,
      visitsBudget: clampVisits(visits),
    });
    if (!silent) {
      this.onChange?.();
    }
  }

  toggleRotateBoard() {
    this.settings.rotateBoard = !this.settings.rotateBoard;
    this.onChange?.();
  }

  toggleDisplayCoordinates() {
    this.settings.displayCoordinates = !this.settings.displayCoordinates;
    this.onChange?.();
  }

  toggleDisplayRemainingWalls() {
    this.settings.displayRemainingWalls = !this.settings.displayRemainingWalls;
    this.onChange?.();
  }

  toggleDisplayEvalBar() {
    this.settings.displayEvalBar = !this.settings.displayEvalBar;
    this.onChange?.();
  }

  newGame() {
    this.aiThinking = false;
    this.eval = { score: 0.5, p1: 0.5, pv: [] };
    for (const engine of this.engines.values()) {
      engine.resetConnection();
    }
    this.session.reset();
    this.onChange?.();
    this.maybeRequestAiMove();
  }

  undo() {
    if (this.aiThinking) {
      return;
    }
    this.session.undo();
    for (const engine of this.engines.values()) {
      engine.resetConnection();
    }
    this.onChange?.();
    this.maybeRequestAiMove();
  }

  tryAction(action) {
    if (this.aiThinking || !this.session.isHumanTurn(this.settings.players)) {
      return;
    }

    const applied = this.session.applyAction(action);
    if (!applied) {
      return;
    }

    this.syncRemoteEnginesAfterMove(action);
    this.onChange?.();
    this.maybeRequestAiMove();
    this.maybePonderInactiveEngines();
  }

  onSessionChange() {
    this.onChange?.();
  }

  createEngineClient(config) {
    if (config.kind === 'local') {
      return new GorisansonEngineClient(config);
    }
    if (config.kind === 'titanium') {
      return new TitaniumEngineClient(config);
    }
    return new EngineClient(config);
  }

  getEngine(playerType) {
    if (playerType === PlayerType.Human) {
      return null;
    }

    if (!this.engines.has(playerType)) {
      const config = this.engineConfigs.find((entry) => entry.key === playerType);
      if (!config || config.disabled) {
        return null;
      }

      const engine = this.createEngineClient(config);
      engine.onStatus = (status) => {
        const prev = this.engineStatus[playerType];
        this.engineStatus[playerType] = status;
        if (prev !== status) {
          this.onChange?.();
        }
      };
      engine.onInfo = (info) => {
        this.searchInfo[playerType] = { ...this.searchInfo[playerType], ...info };
        if (info.progress !== undefined && info.p1 === undefined && !info.pv) {
          return;
        }
        if (info.p1 !== undefined) {
          this.eval.p1 = info.p1;
          this.eval.score = info.score ?? info.p1;
        }
        if (info.pv) {
          this.eval.pv = info.pv;
        }
        this.onChange?.();
      };
      engine.onError = () => {
        this.aiThinking = false;
        this.onChange?.();
      };
      this.engines.set(playerType, engine);
    }

    return this.engines.get(playerType);
  }

  /** Keep remote engines in sync after every ply (human or AI), matching scraped takeAction middleware. */
  syncRemoteEnginesAfterMove(action) {
    for (const playerType of this.settings.players) {
      if (
        playerType === PlayerType.Human ||
        playerType === PlayerType.GorisansonMCTS ||
        playerType === PlayerType.Titanium
      ) {
        continue;
      }
      const engine = this.getEngine(playerType);
      engine?.makeMoves([action]);
    }
  }

  syncRemoteEngine(playerType) {
    const engine = this.getEngine(playerType);
    if (!engine?.syncGameState) {
      return;
    }

    const moveHistory = this.session.actions;
    engine.syncGameState({
      moveHistory,
      gameSnapshot: this.session.getEngineSnapshot(),
      isFreshGame: moveHistory.length === 0,
    });
  }

  /** Stop background ponder on all engines before a real search. Safe no-op until pondering ships. */
  stopAllPonders() {
    for (const engine of this.engines.values()) {
      engine.stopPonder?.();
    }
  }

  /**
   * Future: remote `go ponder` + local predicted-line MCTS (node cap only).
   * @see docs/video/09-pondering-prep.md
   */
  maybePonderInactiveEngines() {
    if (this.session.winner || this.aiThinking) {
      return;
    }
    const { playerToMove } = this.session.getSnapshot();
    for (const { playerType } of ponderCandidateSlots(this.settings.players, playerToMove)) {
      const engine = this.getEngine(playerType);
      if (!engine?.ponder) {
        continue;
      }
      // Not enabled yet — wire aiSettings + sync before calling engine.ponder(...)
    }
  }

  maybeRequestAiMove() {
    if (this.session.winner) {
      this.aiThinking = false;
      return;
    }

    this.stopAllPonders();

    const playerType = this.session.getCurrentPlayerType(this.settings.players);
    if (playerType === PlayerType.Human) {
      this.aiThinking = false;
      return;
    }

    const engine = this.getEngine(playerType);
    if (!engine) {
      this.aiThinking = false;
      return;
    }

    this.aiThinking = true;
    this.onChange?.();

    engine.onBestMove = (action) => {
      this.aiThinking = false;
      if (this.session.winner) {
        return;
      }

      const applied = this.session.applyAction(action);
      if (!applied) {
        this.engineStatus[playerType] = 'error';
        this.onChange?.();
        return;
      }

      this.syncRemoteEnginesAfterMove(action);
      this.onChange?.();
      this.maybeRequestAiMove();
      this.maybePonderInactiveEngines();
    };

    const playerIndex = this.session.getSnapshot().playerToMove - 1;
    let aiSettings = this.settings.playerAiSettings[playerIndex];
    if (!aiSettings) {
      aiSettings = defaultPlayerAiSettings(playerType, this.engineConfigs);
      this.settings.playerAiSettings[playerIndex] = aiSettings;
    }
    const moveHistory = this.session.actions;
    const isFreshGame = moveHistory.length === 0;

    engine.requestMove({
      aiSettings,
      gameSnapshot: this.session.getEngineSnapshot(),
      moveHistory,
      isFreshGame,
    });
  }
}
