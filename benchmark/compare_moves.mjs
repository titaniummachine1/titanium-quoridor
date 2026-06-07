/**
 * Cross-validate Rust move count vs scraped JS rules engine.
 * Run: node benchmark/compare_moves.mjs
 * Requires: cargo build --release (for titanium moves count via CLI)
 */

import { createRequire } from 'node:module';
import { execSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const require = createRequire(import.meta.url);
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const { QuoridorBoard } = require(path.join(root, 'web/src/lib/gameLogic.js'));

function jsMoveLabel(action) {
  if (action.wallType) {
    const { column, row } = action.coordinate;
    return `${column}${row}${action.wallType}`;
  }
  const { column, row } = action.coordinate;
  return `${column}${row}`;
}

function jsCounts() {
  const board = new QuoridorBoard();
  const pawn = board.validPawnMoveActions();
  const walls = board.validWallActions();
  const all = board.validActions();
  return {
    pawn: pawn.length,
    walls: walls.length,
    total: all.length,
    labels: new Set(all.map(jsMoveLabel)),
  };
}

function rustMoveCount() {
  const out = execSync('cargo run --quiet -- moves', {
    cwd: path.join(root, 'engine'),
    encoding: 'utf8',
  });
  const lines = out.trim().split('\n');
  const header = lines[0];
  const labels = new Set(lines.slice(1));
  const total = Number(header.match(/(\d+)/)?.[1] ?? 0);
  return { total, labels };
}

const js = jsCounts();
const rust = rustMoveCount();

console.log('JS  pawn / walls / total:', js.pawn, js.walls, js.total);
console.log('Rust total:', rust.total);

const onlyJs = [...js.labels].filter((m) => !rust.labels.has(m)).sort();
const onlyRust = [...rust.labels].filter((m) => !js.labels.has(m)).sort();

if (onlyJs.length || onlyRust.length) {
  console.error('MISMATCH');
  if (onlyJs.length) console.error('only JS:', onlyJs.slice(0, 20));
  if (onlyRust.length) console.error('only Rust:', onlyRust.slice(0, 20));
  process.exit(1);
}

console.log('OK — move sets match at startpos');
