import { GameSession } from './gameSession.js';
import { naiveDistanceEval } from '../lib/gameLogic.js';
import { decodeReplayCode, encodeReplayFromActions } from '../lib/replayCode.js';
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
      players: [PlayerType.TitaniumMinimax, PlayerType.GorisansonMCTS],
      playerAiSettings: [
        defaultPlayerAiSettings(PlayerType.TitaniumMinimax, this.engineConfigs),
        defaultPlayerAiSettings(PlayerType.GorisansonMCTS, this.engineConfigs),
      ],
      playerAiSettingsMemory: [{}, {}],
      rotateBoard: false,
      displayCoordinates: true,
      displayRemainingWalls: true,
      displayEvalBar: true,
      uiMode: 'play',
    };

    this.replay = null;

    this.engineStatus = {};
    this.engineErrors = {};
    this.searchInfo = {};
    this.moveThinkLog = [];
    this.eval = { score: 0.5, p1: 0.5, pv: [] };
    this.aiThinking = false;
    this.liveSearch = null;
    this.thinkingPlayerType = null;
    this._moveRequestSeq = 0;

    this.session.subscribe(() => this.onSessionChange());
  }

  getState() {
    const snapshot = this.session.getSnapshot();
    const distanceEval = naiveDistanceEval(this.session.board);

    return {
      ...snapshot,
      settings: { ...this.settings },
      engineStatus: { ...this.engineStatus },
      engineErrors: { ...this.engineErrors },
      eval: {
        p1: distanceEval.p1,
        margin: distanceEval.margin,
        whiteDist: distanceEval.whiteDist,
        blackDist: distanceEval.blackDist,
        pv: this.eval.pv ?? [],
      },
      liveSearch: this.liveSearch,
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
      moveThinkLog: this.moveThinkLog,
      uiMode: this.settings.uiMode,
      replay: this.replay
        ? {
          index: this.replay.index,
          total: this.replay.actions.length,
          code: this.replay.code,
          meta: this.replay.meta,
        }
        : null,
      playReplayCode:
        this.session.winner && this.settings.uiMode === 'play'
          ? encodeReplayFromActions(this.session.actions, {
            winner: this.session.winner === 1 ? 'white' : 'black',
            plies: this.session.actions.length,
          })
          : null,
    };
  }

  onChange = null;
  onLiveUpdate = null;
  _liveUpdateLastMs = 0;

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
    this._moveRequestSeq += 1;
    this.aiThinking = false;
    this.liveSearch = null;
    this.engineErrors = {};
    this.replay = null;
    this.moveThinkLog = [];
    this.settings.uiMode = 'play';
    this.eval = { score: 0.5, p1: 0.5, pv: [] };
    for (const engine of this.engines.values()) {
      engine.resetConnection();
    }
    this.session.reset();
    this.onChange?.();
    this.maybeRequestAiMove();
  }

  setUiMode(mode) {
    this.settings.uiMode = mode;
    this.onChange?.();
  }

  loadReplay(code) {
    this._moveRequestSeq += 1;
    const trimmed = code.trim();
    const { actions, meta, algebraic } = decodeReplayCode(trimmed);
    this.replay = {
      actions,
      algebraic,
      index: actions.length,
      code: trimmed.startsWith('tq1') ? trimmed : encodeReplayFromActions(actions, meta),
      meta,
    };
    this.settings.uiMode = 'replay';
    this.aiThinking = false;
    this.liveSearch = null;
    this.engineErrors = {};
    for (const engine of this.engines.values()) {
      engine.resetConnection();
    }
    this.applyReplayIndex();
    this.onChange?.();
  }

  applyReplayIndex() {
    if (!this.replay) {
      return;
    }
    const slice = this.replay.actions.slice(0, this.replay.index);
    this.session.rebuildFromActions(slice);
  }

  setReplayIndex(index) {
    if (!this.replay) {
      return;
    }
    this.replay.index = Math.max(0, Math.min(index, this.replay.actions.length));
    this.applyReplayIndex();
    this.onChange?.();
  }

  replayStep(delta) {
    if (!this.replay) {
      return;
    }
    this.setReplayIndex(this.replay.index + delta);
  }

  exportReplayCode() {
    if (!this.replay) {
      return encodeReplayFromActions(this.session.actions);
    }
    return this.replay.code;
  }

  undo() {
    if (this.aiThinking) {
      return;
    }
    this._moveRequestSeq += 1;
    this.liveSearch = null;
    this.engineErrors = {};
    this.session.undo();
    for (const engine of this.engines.values()) {
      engine.resetConnection();
    }
    this.onChange?.();
    this.maybeRequestAiMove();
  }

  tryAction(action) {
    if (this.replay || this.aiThinking || !this.session.isHumanTurn(this.settings.players)) {
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
        if (info.thinking) {
          this.liveSearch = {
            playerLabel: this.engineLabel(playerType),
            simulations: info.simulations,
            nodes: info.nodes,
            progress: info.progress,
            mode: info.mode ?? info.stoppedBy ?? 'mcts',
            searchDepth: info.searchDepth,
            depthLog: info.depthLog,
            rootWinRate: info.rootWinRate,
            whiteDist: info.whiteDist,
            blackDist: info.blackDist,
            rootMoves: info.rootMoves,
          };
          const now = performance.now();
          if (now - this._liveUpdateLastMs >= 250) {
            this._liveUpdateLastMs = now;
            (this.onLiveUpdate ?? this.onChange)?.();
          }
          return;
        }
        if (info.progress !== undefined && info.p1 === undefined && !info.pv && !info.stoppedBy) {
          return;
        }
        if (info.pv) {
          this.eval.pv = info.pv;
        }
        if (info.stoppedBy) {
          this.liveSearch = {
            playerLabel: this.engineLabel(playerType),
            mode: info.stoppedBy,
            searchDepth: info.searchDepth,
            simulations: info.simulations,
            nodes: info.nodes,
            depthLog: info.depthLog,
            rootWinRate: info.rootWinRate,
            whiteDist: info.whiteDist,
            blackDist: info.blackDist,
            rootMoves: info.rootMoves,
          };
        }
        this.onChange?.();
      };
      engine.onError = () => {
        this.aiThinking = false;
        this.engineErrors[playerType] = 'Engine error';
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
        isLocalEngine(playerType, this.engineConfigs) ||
        isTitaniumEngine(playerType, this.engineConfigs)
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

  engineLabel(playerType) {
    const config = this.engineConfigs.find((entry) => entry.key === playerType);
    return config?.name ?? playerType;
  }

  maybeRequestAiMove() {
    if (this.replay) {
      this.aiThinking = false;
      return;
    }
    if (this.session.winner) {
      this.aiThinking = false;
      this.liveSearch = null;
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

    const requestSnapshot = this.session.getSnapshot();
    const requestSeq = ++this._moveRequestSeq;
    const requestPly = requestSnapshot.actions.length;
    const requestPlayerToMove = requestSnapshot.playerToMove;

    this.aiThinking = true;
    this.thinkingPlayerType = playerType;
    this.liveSearch = { playerLabel: this.engineLabel(playerType), mode: 'searching' };
    this.onChange?.();

    engine.onBestMove = (action) => {
      const current = this.session.getSnapshot();
      const currentPlayerType = this.session.getCurrentPlayerType(this.settings.players);
      const stale =
        requestSeq !== this._moveRequestSeq ||
        current.actions.length !== requestPly ||
        current.playerToMove !== requestPlayerToMove ||
        currentPlayerType !== playerType;
      if (stale) {
        console.warn('Ignoring stale engine move response', {
          playerType,
          requestSeq,
          currentSeq: this._moveRequestSeq,
          requestPly,
          currentPly: current.actions.length,
          requestPlayerToMove,
          currentPlayerToMove: current.playerToMove,
          currentPlayerType,
          suggested: this.session.actionToLabel(action),
        });
        return;
      }

      this.aiThinking = false;
      this.thinkingPlayerType = null;
      if (this.session.winner) {
        return;
      }

      const applied = this.session.applyAction(action);
      if (applied) {
        const plyNum = this.session.actions.length;
        const si = this.searchInfo[playerType] ?? {};
        this.moveThinkLog.push({
          ply: plyNum,
          move: this.session.actionToLabel(action),
          engine: this.engineLabel(playerType),
          stoppedBy: si.stoppedBy ?? si.mode ?? '?',
          nodes: si.nodes ?? si.simulations ?? 0,
          searchDepth: si.searchDepth,
          whiteDist: si.whiteDist,
          blackDist: si.blackDist,
          rootScore: si.rootScore,
          rootWinRate: si.rootWinRate,
          depthLog: si.depthLog ? [...si.depthLog] : [],
          rootMoves: si.rootMoves ? [...si.rootMoves] : [],
        });
      }
      if (!applied) {
        const snapshot = this.session.getSnapshot();
        const suggested = this.session.actionToLabel(action);
        const legal = snapshot.validActions.map((mv) => this.session.actionToLabel(mv));
        console.error('Engine produced illegal move', {
          playerType,
          suggested,
          ply: snapshot.actions.length + 1,
          playerToMove: snapshot.playerToMove,
          playerPositions: snapshot.playerPositions,
          wallsRemaining: snapshot.wallsRemaining,
          legalCount: legal.length,
          legalSample: legal.slice(0, 60),
        });
        this.searchInfo[playerType] = {
          ...(this.searchInfo[playerType] ?? {}),
          illegalMove: suggested,
          illegalMovePly: snapshot.actions.length + 1,
          legalMovesCount: legal.length,
        };
        this.engineErrors[playerType] = `Illegal move ${suggested} on ply ${snapshot.actions.length + 1}`;
        this.engineStatus[playerType] = 'error';
        this.onChange?.();
        return;
      }

      this.engineErrors[playerType] = null;

      this.syncRemoteEnginesAfterMove(action);
      this.onChange?.();
      this.maybeRequestAiMove();
      this.maybePonderInactiveEngines();
    };

    engine.onError = (err) => {
      if (requestSeq !== this._moveRequestSeq) {
        return;
      }
      this.aiThinking = false;
      this.thinkingPlayerType = null;
      this.engineErrors[playerType] = err?.message ?? String(err ?? 'Engine error');
      this.engineStatus[playerType] = 'error';
      this.onChange?.();
    };

    const playerIndex = requestPlayerToMove - 1;
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
