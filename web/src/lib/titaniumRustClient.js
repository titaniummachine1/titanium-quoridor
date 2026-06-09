/**
 * Titanium — Rust engine (`titanium genmove` via dev-server SSE proxy).
 */

import { parseAlgebraic, toAlgebraic } from './gameLogic.js';
import { LOCAL_VISITS_RANGE, clampVisits, uctFromStrengthLevel } from './timeControl.js';

const GENMOVE_URL = '/api/titanium/genmove';

export class TitaniumEngineClient {
  constructor(engineConfig) {
    this.config = engineConfig;
    this.pendingRequest = null;
    this.abortController = null;
  }

  destroy() {
    this.abortController?.abort();
    this.abortController = null;
    this.pendingRequest = null;
    this.setStatus('idle');
  }

  resetConnection() {
    this.destroy();
  }

  makeMoves() {
    this.setStatus('idle');
  }

  ponder() { }

  stopPonder() { }

  requestMove({ aiSettings, moveHistory, isFreshGame }) {
    const history =
      isFreshGame || !moveHistory?.length
        ? []
        : moveHistory.map((action) => toAlgebraic(action));

    const timeSec = Number(aiSettings?.wallClockSeconds) || 10;
    const maxBudget = clampVisits(
      aiSettings?.visitsBudget ?? LOCAL_VISITS_RANGE.default,
    );
    const uct = uctFromStrengthLevel(aiSettings?.strengthLevel);
    const engineMode = this.config?.engineMode === 'minimax' ? 'minimax' : 'mcts';

    this.setStatus('searching');
    const started = performance.now();
    this.pendingRequest = { started };
    this.abortController = new AbortController();

    fetch(GENMOVE_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'text/event-stream',
      },
      body: JSON.stringify({
        moves: history,
        timeSec,
        maxSimulations: maxBudget,
        maxNodes: maxBudget,
        uct,
        engine: engineMode,
        stream: true,
      }),
      signal: this.abortController.signal,
    })
      .then(async (res) => {
        if (!res.ok) {
          const data = await res.json().catch(() => ({}));
          throw new Error(data.error ?? `HTTP ${res.status}`);
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let finalMeta = {
          stoppedBy: engineMode,
          simulations: 0,
          nodes: 0,
        };

        while (true) {
          const { done, value } = await reader.read();
          if (done) {
            break;
          }
          buffer += decoder.decode(value, { stream: true });
          const parts = buffer.split('\n\n');
          buffer = parts.pop() ?? '';

          for (const part of parts) {
            const line = part.split('\n').find((l) => l.startsWith('data: '));
            if (!line) {
              continue;
            }
            const data = JSON.parse(line.slice(6));

            if (data.type === 'progress') {
              this.onInfo?.({
                thinking: true,
                mode: engineMode,
                stoppedBy: engineMode,
                simulations: data.simulations,
                progress: Math.min(0.99, data.elapsedMs / (timeSec * 1000)),
                rootWinRate: data.winRate,
              });
              continue;
            }

            if (data.type === 'info') {
              finalMeta = { ...finalMeta, ...data, stoppedBy: data.stoppedBy ?? 'mcts' };
              this.onInfo?.({
                thinking: true,
                mode: data.stoppedBy ?? engineMode,
                simulations: data.simulations,
                nodes: data.nodes,
                searchDepth: data.searchDepth,
                depthLog: data.depthLog,
                whiteDist: data.whiteDist,
                blackDist: data.blackDist,
                rootWinRate: data.rootWinRate,
                rootMoves: data.rootMoves,
              });
              continue;
            }

            if (data.type === 'error') {
              throw new Error(data.error);
            }

            if (data.type === 'bestmove') {
              const elapsed = performance.now() - started;
              this.pendingRequest = null;
              this.setStatus('idle');
              this.onInfo?.({
                time: elapsed,
                stoppedBy: data.stoppedBy ?? finalMeta.stoppedBy ?? engineMode,
                simulations: finalMeta.simulations ?? 0,
                nodes: finalMeta.nodes ?? 0,
                searchDepth: finalMeta.searchDepth,
                depthLog: finalMeta.depthLog,
                whiteDist: finalMeta.whiteDist,
                blackDist: finalMeta.blackDist,
                rootWinRate: finalMeta.rootWinRate,
                rootMoves: finalMeta.rootMoves,
                progress: 1,
              });
              const action = parseAlgebraic(data.algebraic);
              this.onBestMove?.(action);
              return;
            }
          }
        }

        throw new Error('stream ended without bestmove');
      })
      .catch((err) => {
        if (err.name === 'AbortError') {
          return;
        }
        this.pendingRequest = null;
        this.setStatus('error');
        const message =
          err?.message === 'Failed to fetch'
            ? 'Cannot reach dev server (/api/titanium/genmove) — run npm run dev and ensure engine is built (cargo build --release in engine/)'
            : err?.message ?? String(err);
        this.onError?.(new Error(message));
      });
  }

  setStatus(status) {
    this.onStatus?.(status);
  }
}
