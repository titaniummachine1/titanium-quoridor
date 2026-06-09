/**
 * Ishtar engine over WebSocket — matches scraped quoridor-ai.netlify.app protocol.
 *
 * One persistent session per game (auth → static options → incremental makemove → go).
 * Strength is UI-only on the scraped site; only the 4 time presets set visits/parallelism.
 */

import {
  AUTH_TOKEN,
  PlayerType,
  TimeToMove,
  getEngineList,
} from '../../scraped/engine_config_extract.js';

const ISHTAR_URI = 'wss://quoridor-ai.com/ishtar-v3';

const ISHTAR_CONFIG = getEngineList().find((engine) => engine.key === PlayerType.IshtarV3);

if (!ISHTAR_CONFIG) {
  throw new Error('Ishtar engine config missing from scraped engine_config_extract.js');
}

/** CLI / benchmark preset name → scraped TimeToMove id. */
export const ISHTAR_TIME_PRESETS = {
  intuition: TimeToMove.Intuition,
  immediate: TimeToMove.Intuition,
  short: TimeToMove.Short,
  medium: TimeToMove.Medium,
  long: TimeToMove.Long,
};

/** @deprecated Use resolveIshtarOptions — kept for callers reading visit counts. */
export const ISHTAR_VISITS = ISHTAR_CONFIG.visits;

const TIME_LABELS = ['Immediate', 'Short', 'Medium', 'Long'];

/** Resolve scraped-site time control (visits + parallelism threads). */
export function resolveIshtarOptions(presetOrMode = 'long') {
  let timeMode = presetOrMode;
  if (typeof presetOrMode === 'string') {
    timeMode = ISHTAR_TIME_PRESETS[presetOrMode.toLowerCase()] ?? TimeToMove.Long;
  }
  if (!Number.isFinite(timeMode) || timeMode < 0 || timeMode > 3) {
    timeMode = TimeToMove.Long;
  }
  const visits = ISHTAR_CONFIG.visits[timeMode];
  const parallelism = ISHTAR_CONFIG.settings?.parallelism?.[timeMode] ?? '1';
  return {
    timeMode,
    visits,
    parallelism,
    label: TIME_LABELS[timeMode] ?? 'Long',
  };
}

/** Official → Glendenning (walls only — row +1). */
export function toGlendenningAlgebraic(official) {
  if (official.length <= 2) {
    return official;
  }
  const col = official[0];
  const row = Number(official[1]);
  const suffix = official.slice(2);
  return `${col}${row + 1}${suffix}`;
}

/** Glendenning → official (walls only — row -1). */
export function fromGlendenningAlgebraic(glendenning) {
  if (glendenning.length <= 2) {
    return glendenning;
  }
  const col = glendenning[0];
  const row = Number(glendenning[1]) - 1;
  const suffix = glendenning.slice(2);
  return `${col}${row}${suffix}`;
}

function defaultTimeoutMs(visits) {
  if (visits >= 1_000_000) return 600_000;
  if (visits >= 200_000) return 300_000;
  return 180_000;
}

function isBenignLog(message) {
  return (
    /\bWARN\b/i.test(message) ||
    /already-known hash/i.test(message) ||
    /tensorflow/i.test(message)
  );
}

function sendStaticSettings(ws) {
  const settings = ISHTAR_CONFIG.settings ?? {};
  for (const [name, value] of Object.entries(settings)) {
    if (typeof value === 'string') {
      ws.send(`setoption name ${name} value ${value}`);
    }
  }
}

function sendTimeToMoveSettings(ws, timeMode) {
  const settings = ISHTAR_CONFIG.settings ?? {};
  for (const [name, value] of Object.entries(settings)) {
    if (typeof value !== 'string') {
      const optionValue = value[timeMode];
      if (optionValue != null) {
        ws.send(`setoption name ${name} value ${optionValue}`);
      }
    }
  }
}

/**
 * Persistent Ishtar session — one WebSocket per game, incremental makemove.
 * Mirrors scraped EngineClient lifecycle (connect → makemove* → go*).
 */
export class IshtarGameSession {
  constructor(options = {}) {
    this.preset = options.preset ?? 'long';
    this.resolved = resolveIshtarOptions(this.preset);
    this.timeoutMs = options.timeoutMs ?? defaultTimeoutMs(this.resolved.visits);
    this.ws = null;
    this.sentPlies = 0;
    this.pendingBestmove = null;
  }

  async connect() {
    if (this.ws?.readyState === WebSocket.OPEN) {
      return;
    }
    this.close();

    await new Promise((resolve, reject) => {
      const ws = new WebSocket(ISHTAR_URI);
      this.ws = ws;
      let settled = false;

      const timer = setTimeout(() => {
        if (!settled) {
          settled = true;
          reject(new Error(`Ishtar connect timeout after ${this.timeoutMs}ms`));
        }
      }, 30_000);

      ws.addEventListener('open', () => {
        ws.send(JSON.stringify({ token: AUTH_TOKEN, version: '0.0.0' }));
        sendStaticSettings(ws);
        sendTimeToMoveSettings(ws, this.resolved.timeMode);
        if (!settled) {
          settled = true;
          clearTimeout(timer);
          resolve();
        }
      });

      ws.addEventListener('message', (event) => this._onMessage(event.data.toString()));
      ws.addEventListener('error', () => {
        if (!settled) {
          settled = true;
          clearTimeout(timer);
          reject(new Error('Ishtar WebSocket error on connect'));
        }
      });
      ws.addEventListener('close', () => {
        if (this.pendingBestmove) {
          this.pendingBestmove.reject(new Error('Ishtar WebSocket closed before bestmove'));
          this.pendingBestmove = null;
        }
      });
    });
  }

  _onMessage(message) {
    if (/log Error/i.test(message) && !isBenignLog(message)) {
      if (this.pendingBestmove) {
        this.pendingBestmove.reject(new Error(message));
        this.pendingBestmove = null;
      }
      return;
    }

    if (!message.startsWith('bestmove')) {
      return;
    }

    const glendenning = message.trim().split(/\s+/)[1];
    if (!glendenning || !this.pendingBestmove) {
      return;
    }

    const { resolve, timer } = this.pendingBestmove;
    this.pendingBestmove = null;
    clearTimeout(timer);
    resolve(fromGlendenningAlgebraic(glendenning));
  }

  _send(command) {
    if (this.ws?.readyState !== WebSocket.OPEN) {
      throw new Error('Ishtar session not connected');
    }
    this.ws.send(command);
  }

  async _goAndWait() {
    const { timeMode, visits } = this.resolved;
    this._send(`setoption name visits value ${visits}`);
    sendTimeToMoveSettings(this.ws, timeMode);
    this._send('go');

    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        if (this.pendingBestmove) {
          this.pendingBestmove = null;
          reject(new Error(`Ishtar timeout after ${this.timeoutMs}ms`));
        }
      }, this.timeoutMs);

      this.pendingBestmove = { resolve, reject, timer };
    });
  }

  /**
   * Request best move for `algebraicHistory` (official notation).
   * Sends only new plies since last call (incremental makemove).
   */
  async requestMove(algebraicHistory, { retries = 3 } = {}) {
    let lastErr;
    for (let attempt = 1; attempt <= retries; attempt += 1) {
      try {
        await this.connect();
        const newMoves = algebraicHistory.slice(this.sentPlies);
        if (newMoves.length > 0) {
          const glendenning = newMoves.map(toGlendenningAlgebraic).join(' ');
          this._send(`makemove ${glendenning}`);
          this.sentPlies += newMoves.length;
        }
        return await this._goAndWait();
      } catch (err) {
        lastErr = err;
        const retryable = /WebSocket|timeout|closed before bestmove/i.test(
          String(err?.message ?? err),
        );
        this.close();
        if (!retryable || attempt === retries) {
          throw err;
        }
        await new Promise((r) => setTimeout(r, 3000 * attempt));
      }
    }
    throw lastErr;
  }

  /** Close session — call between games (scraped `newGame` resets WS). */
  close() {
    if (this.pendingBestmove) {
      clearTimeout(this.pendingBestmove.timer);
      this.pendingBestmove = null;
    }
    try {
      this.ws?.close();
    } catch {
      // ignore
    }
    this.ws = null;
    this.sentPlies = 0;
  }
}

/**
 * One-shot move request (opens + closes session). Prefer IshtarGameSession for mining.
 */
export async function requestIshtarMove(algebraicHistory, options = {}) {
  const retries = options.retries ?? 3;
  let lastErr;

  for (let attempt = 1; attempt <= retries; attempt += 1) {
    const session = new IshtarGameSession({
      preset: options.preset ?? options.timeMode ?? 'long',
      timeoutMs: options.timeoutMs,
    });
    try {
      const move = await session.requestMove(algebraicHistory);
      session.close();
      return move;
    } catch (err) {
      lastErr = err;
      session.close();
      const retryable = /WebSocket|timeout|closed before bestmove/i.test(
        String(err?.message ?? err),
      );
      if (!retryable || attempt === retries) {
        throw err;
      }
      await new Promise((r) => setTimeout(r, 2000 * attempt));
    }
  }
  throw lastErr;
}

export { TimeToMove };
