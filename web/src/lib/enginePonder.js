/**
 * Pondering contract — Stockfish-style think on opponent time.
 * Not wired in appController yet; see docs/video/09-pondering-prep.md
 */

import { PlayerType } from './engineConfig.js';

/** EngineClient / local MCTS surface for future pondering. */
export function engineSupportsPonder(engine) {
  return typeof engine?.ponder === 'function';
}

export function engineSupportsStopPonder(engine) {
  return typeof engine?.stopPonder === 'function';
}

/**
 * Player slots that are AI and not currently on move — candidates to ponder.
 * @param {string[]} players — settings.players
 * @param {number} playerToMove — 1 or 2
 */
export function ponderCandidateSlots(players, playerToMove) {
  const candidates = [];
  for (let slot = 0; slot < players.length; slot++) {
    const playerType = players[slot];
    if (playerType === PlayerType.Human) {
      continue;
    }
    if (slot + 1 === playerToMove) {
      continue;
    }
    candidates.push({ slot, playerType });
  }
  return candidates;
}
