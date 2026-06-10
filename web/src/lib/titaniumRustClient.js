/**

 * Titanium — Rust engine via dev-server session proxy (warm TT per seat).

 * Falls back to one-shot `genmove` for non-minimax engine modes.

 */



import { parseAlgebraic, toAlgebraic } from './gameLogic.js';

import { LOCAL_VISITS_RANGE, clampVisits, uctFromStrengthLevel } from './timeControl.js';



const SESSION_URL = '/api/titanium/session';

const GENMOVE_URL = '/api/titanium/genmove';



export class TitaniumEngineClient {

  constructor(engineConfig, { seatId = 'seat-0' } = {}) {

    this.config = engineConfig;

    this.seatId = seatId;

    this.pendingRequest = null;

    this.queuedRequest = null;

    this.abortController = null;

    /** Plies applied to the server-side session board. */

    this.appliedPlies = 0;

    /** Serialize position sync vs search so makemove/position never races `go`. */

    this._syncChain = Promise.resolve();

  }



  cancelSearch() {

    this.queuedRequest = null;

    this.abortController?.abort();

    this.abortController = null;

    this.pendingRequest = null;

    this.setStatus('idle');

  }



  clearQueuedSearches() {

    this.queuedRequest = null;

  }



  destroy() {

    this.cancelSearch();

    this.sessionOp({ op: 'destroy' }).catch(() => {});

  }



  resetConnection() {

    this.cancelSearch();

    this.appliedPlies = 0;

    this.sessionOp({ op: 'reset' }).catch(() => {});

  }



  makeMoves(actions) {

    if (!actions?.length) {

      return Promise.resolve();

    }

    const moves = actions.map((action) => toAlgebraic(action));

    return this.enqueueSync(() => this.syncMovesToSession(moves, { incremental: true }));

  }



  ponder() {}



  stopPonder() {}



  requestMove(params) {

    if (this.pendingRequest) {

      this.queuedRequest = params;

      return;

    }

    this.startRequest(params);

  }



  drainQueuedRequest() {

    if (!this.queuedRequest) {

      return;

    }

    const next = this.queuedRequest;

    this.queuedRequest = null;

    this.startRequest(next);

  }



  async sessionOp(body, { stream = false, signal } = {}) {

    const res = await fetch(SESSION_URL, {

      method: 'POST',

      headers: {

        'Content-Type': 'application/json',

        Accept: stream ? 'text/event-stream' : 'application/json',

      },

      body: JSON.stringify({ seatId: this.seatId, ...body }),

      signal,

    });

    if (!res.ok && !stream) {

      const data = await res.json().catch(() => ({}));

      throw new Error(data.error ?? `HTTP ${res.status}`);

    }

    return res;

  }



  enqueueSync(fn) {

    this._syncChain = this._syncChain.then(fn).catch((err) => {

      this.appliedPlies = 0;

      throw err;

    });

    return this._syncChain;

  }



  /**

   * @param {string[]} algebraicMoves

   * @param {{ incremental?: boolean, forceFull?: boolean }} [opts]

   */

  async syncMovesToSession(algebraicMoves, { incremental = false, forceFull = false } = {}) {

    const moves = algebraicMoves ?? [];

    if (!forceFull && !incremental && moves.length === 0 && this.appliedPlies === 0) {

      return;

    }

    // Full replay before search — never trust appliedPlies alone (session respawn / race).

    if (forceFull || !incremental || moves.length < this.appliedPlies) {

      await this.sessionOp({ op: 'position', moves });

      this.appliedPlies = moves.length;

      return;

    }



    const delta = moves.slice(this.appliedPlies);

    if (delta.length === 0) {

      return;

    }

    for (const move of delta) {

      await this.sessionOp({ op: 'makemove', move });

    }

    this.appliedPlies = moves.length;

  }



  startRequest({ aiSettings, moveHistory, isFreshGame }) {

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

    this.pendingRequest = { started, timeSec };

    this.abortController = new AbortController();



    if (engineMode !== 'minimax') {

      this.startOneShotGenmove(history, { timeSec, maxBudget, uct, engineMode, started });

      return;

    }



    this.onInfo?.({

      thinking: true,

      mode: 'minimax',

      stoppedBy: 'minimax',

      nodes: 0,

      simulations: 0,

    });



    const run = async () => {

      if (isFreshGame) {

        await this.sessionOp({ op: 'reset' }, { signal: this.abortController.signal });

        this.appliedPlies = 0;

      }

      await this.enqueueSync(() =>

        this.syncMovesToSession(history, { forceFull: true }),

      );



      const res = await this.sessionOp(

        {

          op: 'go',

          timeSec,

          maxNodes: maxBudget,

          stream: true,

        },

        { stream: true, signal: this.abortController.signal },

      );



      if (!res.ok) {

        const data = await res.json().catch(() => ({}));

        throw new Error(data.error ?? `HTTP ${res.status}`);

      }



      const reader = res.body.getReader();

      const decoder = new TextDecoder();

      let buffer = '';

      let finalMeta = {

        stoppedBy: 'minimax',

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



          if (data.type === 'info') {

            finalMeta = { ...finalMeta, ...data, stoppedBy: data.stoppedBy ?? 'minimax' };

            this.onInfo?.({

              thinking: true,

              mode: 'minimax',

              stoppedBy: finalMeta.stoppedBy,

              simulations: data.simulations,

              nodes: data.nodes,

              searchDepth: data.searchDepth,

              depthLog: data.depthLog,

              whiteDist: data.whiteDist,

              blackDist: data.blackDist,

              rootScore: data.rootScore,

              rootWinRate: null,

              rootMoves: data.rootMoves,

              lmrProfile: data.lmrProfile,

              lmrReSearches: data.lmrReSearches,

              elapsedMs: data.elapsedMs,

            });

            continue;

          }



          if (data.type === 'error') {

            throw new Error(data.error);

          }



          if (data.type === 'bestmove') {

            const elapsed = performance.now() - started;

            this.pendingRequest = null;

            this.abortController = null;

            this.setStatus('idle');

            const stoppedBy = finalMeta.stoppedBy ?? 'minimax';

            this.onInfo?.({

              time: elapsed,

              elapsedMs: finalMeta.elapsedMs ?? Math.round(elapsed),

              stoppedBy,

              simulations: finalMeta.simulations ?? 0,

              nodes: finalMeta.nodes ?? 0,

              searchDepth: finalMeta.searchDepth,

              depthLog: finalMeta.depthLog,

              whiteDist: finalMeta.whiteDist,

              blackDist: finalMeta.blackDist,

              rootWinRate: null,

              rootMoves: finalMeta.rootMoves,

              lmrProfile: finalMeta.lmrProfile,

              lmrReSearches: finalMeta.lmrReSearches,

              progress: 1,

            });

            const action = parseAlgebraic(data.algebraic);

            const result = this.onBestMove?.(action);

            if (result === 'stale') {

              this.clearQueuedSearches();

              return;

            }

            if (result === false) {

              this.clearQueuedSearches();

            } else {

              this.drainQueuedRequest();

            }

            return;

          }

        }

      }



      throw new Error('session stream ended without bestmove');

    };



    run().catch((err) => {

      this.pendingRequest = null;

      this.abortController = null;

      if (err.name === 'AbortError') {

        this.setStatus('idle');

        this.drainQueuedRequest();

        return;

      }

      this.setStatus('error');

      const message =

        err?.message === 'Failed to fetch'

          ? 'Cannot reach dev server (/api/titanium/session) — run npm run dev and build engine (cargo build --release)'

          : err?.message ?? String(err);

      this.onError?.(new Error(message));

      this.drainQueuedRequest();

    });

  }



  startOneShotGenmove(history, { timeSec, maxBudget, uct, engineMode, started }) {

    if (engineMode === 'minimax') {

      this.onInfo?.({

        thinking: true,

        mode: 'minimax',

        stoppedBy: 'minimax',

        nodes: 0,

        simulations: 0,

      });

    }



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

              if (engineMode === 'minimax') {

                continue;

              }

              this.onInfo?.({

                thinking: true,

                mode: 'mcts',

                stoppedBy: 'time',

                simulations: data.simulations,

                progress: Math.min(0.99, data.elapsedMs / (timeSec * 1000)),

                rootWinRate: data.winRate,

              });

              continue;

            }



            if (data.type === 'info') {

              const stoppedBy = data.stoppedBy ?? engineMode;

              finalMeta = { ...finalMeta, ...data, stoppedBy };

              const isMinimax = stoppedBy === 'minimax';

              this.onInfo?.({

                thinking: true,

                mode: stoppedBy,

                stoppedBy,

                simulations: data.simulations,

                nodes: data.nodes,

                searchDepth: data.searchDepth,

                depthLog: data.depthLog,

                whiteDist: data.whiteDist,

                blackDist: data.blackDist,

                rootScore: data.rootScore,

                rootWinRate: isMinimax ? null : data.rootWinRate,

                rootMoves: data.rootMoves,

                lmrProfile: data.lmrProfile,

                lmrReSearches: data.lmrReSearches,

                elapsedMs: data.elapsedMs,

              });

              continue;

            }



            if (data.type === 'error') {

              throw new Error(data.error);

            }



            if (data.type === 'bestmove') {

              const elapsed = performance.now() - started;

              this.pendingRequest = null;

              this.abortController = null;

              this.setStatus('idle');

              const stoppedBy = finalMeta.stoppedBy ?? data.stoppedBy ?? engineMode;

              this.onInfo?.({

                time: elapsed,

                elapsedMs: finalMeta.elapsedMs ?? Math.round(elapsed),

                stoppedBy,

                simulations: finalMeta.simulations ?? 0,

                nodes: finalMeta.nodes ?? 0,

                searchDepth: finalMeta.searchDepth,

                depthLog: finalMeta.depthLog,

                whiteDist: finalMeta.whiteDist,

                blackDist: finalMeta.blackDist,

                rootWinRate: stoppedBy === 'minimax' ? null : finalMeta.rootWinRate,

                rootMoves: finalMeta.rootMoves,

                lmrProfile: finalMeta.lmrProfile,

                lmrReSearches: finalMeta.lmrReSearches,

                progress: 1,

              });

              const action = parseAlgebraic(data.algebraic);

              const result = this.onBestMove?.(action);

              if (result === 'stale') {

                this.clearQueuedSearches();

                return;

              }

              if (result === false) {

                this.clearQueuedSearches();

              } else {

                this.drainQueuedRequest();

              }

              return;

            }

          }

        }



        throw new Error('stream ended without bestmove');

      })

      .catch((err) => {

        this.pendingRequest = null;

        this.abortController = null;

        if (err.name === 'AbortError') {

          this.setStatus('idle');

          this.drainQueuedRequest();

          return;

        }

        this.setStatus('error');

        const message =

          err?.message === 'Failed to fetch'

            ? 'Cannot reach dev server (/api/titanium/genmove) — run npm run dev and ensure engine is built'

            : err?.message ?? String(err);

        this.onError?.(new Error(message));

        this.drainQueuedRequest();

      });

  }



  setStatus(status) {

    this.onStatus?.(status);

  }

}


