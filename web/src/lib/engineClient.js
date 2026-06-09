/**
 * Browser WebSocket client for remote Ishtar/Ka engines.
 * Reconstructed from scraped quoridor-ai.netlify.app bundle (class mT).
 *
 * Sync model matches the original: incremental makemove after every ply,
 * full replay only after reconnect/undo/new game; AI turns call go() only.
 */

import {
  AUTH_TOKEN,
  INFO_LINE_RE,
  BESTMOVE_LINE_RE,
  Notation,
  TimeToMove,
  buildPositionString,
  parseInfoLine,
} from './engineConfig.js';

import {
  Direction,
  toAlgebraic,
  parseAlgebraic,
  transformCoordinate,
  isWallAction,
} from './gameLogic.js';

function toEngineAlgebraic(action, notation) {
  let normalized = action;
  if (isWallAction(action) && notation === Notation.Glendenning) {
    normalized = {
      ...action,
      coordinate: transformCoordinate(action.coordinate, [Direction.Up]),
    };
  }
  return toAlgebraic(normalized);
}

function fromEngineAlgebraic(move, notation) {
  const action = parseAlgebraic(move);
  if (isWallAction(action) && notation === Notation.Glendenning) {
    action.coordinate = transformCoordinate(action.coordinate, [Direction.Down]);
  }
  return action;
}

export class EngineClient {
  constructor(engineConfig) {
    this.config = engineConfig;
    this.ws = null;
    this.sendBuffer = [];
    this.outstandingSearches = 0;
    this.isPondering = false;
    this.hasSynced = false;
    this.lastTimeMode = null;
    this.pendingSearch = null;
    this._lastSearch = null;
    this._reconnectAttempts = 0;

    this.onInfo = null;
    this.onBestMove = null;
    this.onStatus = null;
    this.onError = null;
  }

  destroy() {
    this.stop();
    this.ws?.close();
    this.ws = null;
    this.sendBuffer = [];
    this.outstandingSearches = 0;
    this.hasSynced = false;
    this.pendingSearch = null;
    this.setStatus('idle');
  }

  resetConnection() {
    this.destroy();
  }

  send(command) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(command);
      return;
    }
    this.sendBuffer.push(command);
    this.connect();
  }

  connect() {
    if (this.ws) {
      return;
    }

    this.setStatus('connecting');
    const socket = new WebSocket(this.config.uri);
    this.ws = socket;

    socket.addEventListener('open', () => this.onOpen());
    socket.addEventListener('message', (event) => this.onMessage(event.data));
    socket.addEventListener('error', () => {
      if (this.ws === socket) {
        this.setStatus('error');
        this.onError?.(new Error('WebSocket connection failed'));
      }
    });
    socket.addEventListener('close', () => {
      if (this.ws !== socket) {
        return;
      }
      const wasSearching = this.outstandingSearches > 0;
      this.ws = null;

      if (wasSearching && this._lastSearch && this._reconnectAttempts < 3) {
        this._reconnectAttempts += 1;
        this.hasSynced = false;
        this.outstandingSearches = 0;
        this.pendingSearch = () => {
          const ctx = this._lastSearch;
          this.syncGameState({
            moveHistory: ctx.moveHistory,
            gameSnapshot: ctx.gameSnapshot,
            isFreshGame: ctx.isFreshGame,
          });
          this.go(ctx.timeMode);
        };
        this.connect();
        return;
      }

      this.setStatus('error');
      if (this.pendingSearch || wasSearching) {
        this.pendingSearch = null;
        this.onError?.(new Error('WebSocket closed before bestmove'));
      }
    });
  }

  makeMoves(actions) {
    const moves = actions.map((action) => toEngineAlgebraic(action, this.config.notation)).join(' ');
    if (moves) {
      this.send(`makemove ${moves}`);
      this.hasSynced = true;
    }
    this.setStatus('idle');
  }

  setPosition(gameSnapshot) {
    const position = buildPositionString(gameSnapshot, this.config.notation);
    this.send(`setposition ${position}`);
    this.hasSynced = true;
  }

  /** Full replay — used after reconnect, undo, or selecting a remote opponent mid-game. */
  syncGameState({ moveHistory, gameSnapshot, isFreshGame }) {
    if (isFreshGame && moveHistory.length === 0) {
      this.hasSynced = true;
      return;
    }

    if (moveHistory.length > 0) {
      this.makeMoves(moveHistory);
      return;
    }

    if (gameSnapshot) {
      this.setPosition(gameSnapshot);
    }
  }

  go(timeMode) {
    if (timeMode == null) {
      timeMode = TimeToMove.Short;
    }
    this.lastTimeMode = timeMode;
    const visits = this.config.visits?.[timeMode];
    this.outstandingSearches++;

    if (Number.isFinite(visits)) {
      this.send(`setoption name visits value ${visits}`);
    }

    this.sendTimeToMoveSettings(timeMode);
    this.send('go');
    this.setStatus('searching');
  }

  requestMove({ aiSettings, gameSnapshot, moveHistory, isFreshGame }) {
    const timeMode = aiSettings?.timeToMove;
    this._lastSearch = { aiSettings, gameSnapshot, moveHistory, isFreshGame, timeMode };
    this._reconnectAttempts = 0;
    const runSearch = () => {
      if (!this.hasSynced) {
        this.syncGameState({ moveHistory, gameSnapshot, isFreshGame });
      }
      this.go(timeMode);
    };

    if (this.ws?.readyState === WebSocket.OPEN) {
      runSearch();
      return;
    }

    this.pendingSearch = runSearch;
    this.connect();
  }

  /** Stockfish-style — search on opponent time (your WS slot only). Not called yet. */
  ponder(timeMode) {
    if (this.outstandingSearches > 0 || this.isPondering) {
      return;
    }
    if (timeMode == null) {
      timeMode = this.lastTimeMode ?? TimeToMove.Short;
    }
    this.lastTimeMode = timeMode;
    this.sendTimeToMoveSettings(timeMode);
    this.send('go ponder');
    this.isPondering = true;
    this.setStatus('pondering');
  }

  stopPonder() {
    this.stop();
  }

  stop() {
    if (!this.isPondering) {
      return;
    }
    this.send('stop');
    this.isPondering = false;
    this.setStatus('idle');
  }

  onOpen() {
    this.ws.send(JSON.stringify({ token: AUTH_TOKEN, version: '0.0.0' }));
    this.sendStaticSettings();

    if (this.lastTimeMode != null) {
      this.sendTimeToMoveSettings(this.lastTimeMode);
    }

    for (const command of this.sendBuffer) {
      if (this.ws?.readyState === WebSocket.OPEN) {
        this.ws.send(command);
      }
    }
    this.sendBuffer = [];
    this.setStatus('idle');

    if (this.pendingSearch) {
      const runSearch = this.pendingSearch;
      this.pendingSearch = null;
      runSearch();
    }
  }

  onMessage(rawMessage) {
    const isBenignLog =
      /\bWARN\b/i.test(rawMessage) ||
      /already-known hash/i.test(rawMessage) ||
      /tensorflow/i.test(rawMessage);
    if (/log Error/i.test(rawMessage) && !isBenignLog) {
      this.setStatus('error');
      this.onError?.(new Error(rawMessage));
      return;
    }

    const infoMatch = INFO_LINE_RE.exec(rawMessage);
    if (infoMatch) {
      const info = parseInfoLine(infoMatch[1]);
      if (info.pv && typeof info.pv === 'string') {
        info.pv = info.pv.split(' ').map((move) => fromEngineAlgebraic(move, this.config.notation));
      }
      if (info.p1 !== undefined) {
        info.winChance = info.p1;
      } else if (info.score !== undefined) {
        info.winChance = info.score;
        info.p1 = info.score;
      }
      this.onInfo?.(info);
      return;
    }

    const bestMoveMatch = BESTMOVE_LINE_RE.exec(rawMessage);
    if (!bestMoveMatch) {
      return;
    }

    this.outstandingSearches = Math.max(0, this.outstandingSearches - 1);
    this._reconnectAttempts = 0;
    this.setStatus('idle');

    const moveText = bestMoveMatch[1].trim().split(/\s+/)[0];
    if (!moveText) {
      return;
    }

    const action = fromEngineAlgebraic(moveText, this.config.notation);
    this.onBestMove?.(action, bestMoveMatch[1]);
  }

  sendStaticSettings() {
    if (!this.config.settings) {
      return;
    }
    for (const [name, value] of Object.entries(this.config.settings)) {
      if (typeof value === 'string') {
        this.send(`setoption name ${name} value ${value}`);
      }
    }
  }

  sendTimeToMoveSettings(timeMode) {
    if (!this.config.settings || timeMode == null) {
      return;
    }
    for (const [name, value] of Object.entries(this.config.settings)) {
      if (typeof value !== 'string') {
        const optionValue = value[timeMode];
        if (optionValue != null) {
          this.send(`setoption name ${name} value ${optionValue}`);
        }
      }
    }
  }

  setStatus(status) {
    this.onStatus?.(status);
  }
}
