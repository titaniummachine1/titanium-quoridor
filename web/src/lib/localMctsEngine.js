/**
 * Gorisanson MCTS in a Web Worker — shared by Gorisanson + Titanium slots.
 */

import GorisansonWorker from '../workers/gorisansonWorker.js?worker';
import { actionToGorisansonMove, gorisansonMoveToAction } from './gorisansonBridge.js';
import { LOCAL_VISITS_RANGE, clampVisits, uctFromStrengthLevel } from './timeControl.js';

export class LocalMctsEngineClient {
  constructor(engineConfig, { resolveUct } = {}) {
    this.config = engineConfig;
    this.resolveUct = resolveUct ?? (() => engineConfig.uctConst ?? 0.2);
    this.worker = null;
    this.gorisansonMoves = [];
    this.isPondering = false;
  }

  /**
   * Future: node-cap-only MCTS on predicted opponent reply (no wall clock).
   * @see docs/video/09-pondering-prep.md
   */
  ponder() {
    this.isPondering = false;
  }

  stopPonder() {
    if (!this.isPondering) {
      return;
    }
    this.worker?.terminate();
    this.worker = null;
    this.isPondering = false;
    this.setStatus('idle');
  }

  destroy() {
    this.worker?.terminate();
    this.worker = null;
    this.gorisansonMoves = [];
    this.setStatus('idle');
  }

  resetConnection() {
    this.destroy();
    this.gorisansonMoves = [];
  }

  makeMoves(actions) {
    for (const action of actions) {
      this.gorisansonMoves.push(actionToGorisansonMove(action));
    }
    this.setStatus('idle');
  }

  requestMove({ aiSettings, moveHistory, isFreshGame }) {
    if (isFreshGame) {
      this.gorisansonMoves = [];
    } else if (moveHistory?.length) {
      this.gorisansonMoves = moveHistory.map(actionToGorisansonMove);
    }

    const timeMs = Math.round((aiSettings?.wallClockSeconds ?? 3) * 1000);
    const maxSimulations = clampVisits(aiSettings?.visitsBudget ?? LOCAL_VISITS_RANGE.default);
    const uctConst = this.resolveUct(aiSettings);

    this.setStatus('searching');
    const started = performance.now();

    this.worker?.terminate();
    this.worker = new GorisansonWorker();

    this.worker.onmessage = (event) => {
      const data = event.data;
      if (data.type === 'progress') {
        return;
      }
      if (data.type === 'error') {
        this.setStatus('error');
        this.onError?.(new Error(data.message));
        return;
      }
      if (data.type === 'bestmove') {
        const elapsed = performance.now() - started;
        this.onInfo?.({
          time: elapsed,
          simulations: data.simulations,
          stoppedBy: data.stoppedBy,
          progress: 1,
        });
        this.setStatus('idle');
        const action = gorisansonMoveToAction(data.move);
        this.gorisansonMoves.push(data.move);
        this.onBestMove?.(action);
      }
    };

    this.worker.onerror = (err) => {
      this.setStatus('error');
      this.onError?.(err);
    };

    this.worker.postMessage({
      gorisansonMoves: this.gorisansonMoves,
      timeMs,
      maxSimulations,
      uctConst,
    });
  }

  setStatus(status) {
    this.onStatus?.(status);
  }
}

export class GorisansonEngineClient extends LocalMctsEngineClient {
  constructor(engineConfig) {
    super(engineConfig, {
      resolveUct: () => engineConfig.uctConst ?? 0.2,
    });
  }
}

export class TitaniumEngineClient extends LocalMctsEngineClient {
  constructor(engineConfig) {
    super(engineConfig, {
      resolveUct: (aiSettings) => uctFromStrengthLevel(aiSettings?.strengthLevel),
    });
  }
}
