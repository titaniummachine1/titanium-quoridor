#!/usr/bin/env node
/** Probe Canta turn_bytes → gorisanson perft oracle. */
import { createRequire } from 'node:module';
import { gorisansonPerft } from './lib/gorisanson_moves.mjs';

const require = createRequire(import.meta.url);
const g = require('./lib/load_gorisanson.cjs');

const TURN_BYTES = [
  [0x8d, 0x9d, 0xb, 0x5, 0x5f, 0xd5, 0x19, 0x81, 0x33, 0xc3, 0x27, 0x9b, 0x49, 0xcd, 0xbd],
];

const PERFT = [{ 1: 79n, 2: 5978n, 3: 432338n }];

function getNewPosition(index, x, y) {
  const t = [
    [x, y + 2], [x + 1, y + 1], [x - 1, y + 1], [x, y + 1],
    [x, y - 2], [x + 1, y - 1], [x - 1, y - 1], [x, y - 1],
    [x + 2, y], [x + 1, y + 1], [x + 1, y - 1], [x + 1, y],
    [x - 2, y], [x - 1, y + 1], [x - 1, y - 1], [x - 1, y],
  ];
  return t[index];
}

function decodeWall(index, orientation, mode) {
  const modes = {
    idx8: (i, o) => {
      const row = Math.floor(i / 8);
      const col = i % 8;
      return o ? [null, null, [row, col]] : [null, [row, col], null];
    },
    idx8flip: (i, o) => {
      const row = 7 - Math.floor(i / 8);
      const col = i % 8;
      return o ? [null, null, [row, col]] : [null, [row, col], null];
    },
    oriFlip: (i, o) => {
      const row = Math.floor(i / 8);
      const col = i % 8;
      return o ? [null, [row, col], null] : [null, null, [row, col]];
    },
    oriFlipFlip: (i, o) => {
      const row = 7 - Math.floor(i / 8);
      const col = i % 8;
      return o ? [null, [row, col], null] : [null, null, [row, col]];
    },
  };
  return modes[mode]?.(index, orientation) ?? null;
}

function gPosToCanta(pos) {
  return [pos.col, 8 - pos.row];
}

function cantaToGPos(x, y) {
  return [8 - y, x];
}

function applyTurn(game, byte, wallMode, pawnFlip = true) {
  const placeWalls = byte & 1;
  const orientation = (byte >> 1) & 1;
  const index = (byte >> 2) & 0x3f;
  if (placeWalls) {
    const mv = decodeWall(index, orientation, wallMode);
    if (!mv) throw new Error('bad wall mode');
    game.doMove(mv, true);
    return;
  }
  const pos = game.pawnOfTurn.position;
  let [gr, gc];
  if (pawnFlip) {
    const [x, y] = gPosToCanta(pos);
    const [nx, ny] = getNewPosition(index, x, y);
    [gr, gc] = cantaToGPos(nx, ny);
  } else {
    const [nx, ny] = getNewPosition(index, pos.col, pos.row);
    gr = nx;
    gc = ny;
  }
  game.doMove([[gr, gc], null, null], true);
}

function replay(wallMode, pawnFlip = true) {
  const game = new g.Game(false);
  for (const b of TURN_BYTES[0]) {
    applyTurn(game, b, wallMode, pawnFlip);
  }
  return game;
}

for (const mode of ['idx8', 'idx8flip', 'oriFlip', 'oriFlipFlip']) {
  try {
    const game = replay(mode);
    const p1 = game.pawns[0].position;
    const p2 = game.pawns[1].position;
    const d1 = gorisansonPerft(game, 1);
    const d2 = gorisansonPerft(game, 2);
    const d3 = gorisansonPerft(game, 3);
    const ok = d1 === PERFT[0][1] && d2 === PERFT[0][2] && d3 === PERFT[0][3];
    console.log(mode, { p1, p2, d1: String(d1), d2: String(d2), d3: String(d3), ok });
  } catch (e) {
    console.log(mode, 'ERR', e.message);
  }
}
