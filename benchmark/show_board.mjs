#!/usr/bin/env node
/**
 * Visual board renderer for post-game analysis.
 *
 * Usage:
 *   node benchmark/show_board.mjs e2 e8 d7h f8h ...   (algebraic move list)
 *   node benchmark/show_board.mjs "tq1#... e2 e8 ..."  (paste tq1 replay code)
 *   node benchmark/show_board.mjs                       (start position)
 *
 * Output: rich Unicode board printed to stdout + written to benchmark/board.txt
 * (AI can read benchmark/board.txt to see the position)
 */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const OUT = path.join(path.dirname(fileURLToPath(import.meta.url)), 'board.txt');

// ── Board model ─────────────────────────────────────────────────────────────

class Board {
  constructor() {
    // row 0 = row 1 (white's start), row 8 = row 9 (black's start)
    this.pawns = [
      { row: 0, col: 4 },   // White (player 1) — wins on row 8
      { row: 8, col: 4 },   // Black (player 2) — wins on row 0
    ];
    this.wallsLeft = [10, 10];
    // hWalls[r][c] = true → horizontal wall between rows r and r+1, cols c and c+1
    this.hWalls = Array.from({ length: 8 }, () => new Array(8).fill(false));
    this.vWalls = Array.from({ length: 8 }, () => new Array(8).fill(false));
    this.turn = 0; // 0 = White, 1 = Black
    this.plyCount = 0;
  }

  applyAlgebraic(move) {
    const m = move.trim().toLowerCase();
    if (m.length === 3) {
      // Wall: e.g. "d7h" or "e5v"
      const col = m.codePointAt(0) - 97;          // a=0 … i=8
      const row = Number.parseInt(m[1], 10) - 1;  // 1-indexed → 0-indexed
      const ori = m[2];
      if (ori === 'h') {
        this.hWalls[row][col] = true;
      } else {
        this.vWalls[row][col] = true;
      }
      this.wallsLeft[this.turn] -= 1;
    } else if (m.length === 2) {
      // Pawn: e.g. "e2"
      const col = m.codePointAt(0) - 97;
      const row = Number.parseInt(m[1], 10) - 1;
      this.pawns[this.turn] = { row, col };
    } else {
      throw new Error(`Cannot parse move: "${move}"`);
    }
    this.turn = 1 - this.turn;
    this.plyCount += 1;
  }
}

// ── Move parsing ─────────────────────────────────────────────────────────────

function parseMoves(argv) {
  // Support pasting either a full tq1 code or a quoted move list.
  const joined = argv.join(' ').trim();
  if (!joined) return [];

  if (joined.startsWith('tq1')) {
    // Strip tq1#{"..."} prefix, keep everything after the first space
    const rest = joined.replace(/^tq1[^\s]*\s*/, '');
    return rest.trim().split(/\s+/).filter(Boolean);
  }

  return joined.split(/\s+/).filter(Boolean);
}

// ── Rendering ────────────────────────────────────────────────────────────────

const COL_LABELS = 'abcdefghi';

function pawnGlyph(isWhite, isBlack) {
  if (isWhite) return '(W)';
  if (isBlack) return '(B)';
  return ' · ';
}

/**
 * Full Unicode board showing walls as lines between cells.
 *
 * Each cell is 3 chars wide, each gap is 1 char:
 *   ·   (empty)   W   (white pawn)   B   (black pawn)
 * Horizontal wall below row r, cols c..c+1: ═══
 * Vertical wall right of row r, col c:      ║
 */
function renderBoard(board) {
  const { pawns, hWalls, vWalls, wallsLeft, plyCount, turn } = board;

  const lines = [];

  // Header
  lines.push(`  Ply ${plyCount} — ${turn === 0 ? 'White' : 'Black'} to move   W walls: ${wallsLeft[0]}   B walls: ${wallsLeft[1]}`);
  lines.push('');

  for (let r = 8; r >= 0; r--) {
    // ── Cell row ──────────────────────────────────────────────────────────
    let cellLine = `${String(r + 1).padStart(2)} `;
    for (let c = 0; c < 9; c++) {
      const isW = pawns[0].row === r && pawns[0].col === c;
      const isB = pawns[1].row === r && pawns[1].col === c;
      cellLine += pawnGlyph(isW, isB);

      // Vertical wall to the right of (r, c)?
      // vWalls[row][col] anchors at row-row+1, col-col (right edge of col)
      if (c < 8) {
        const wallRight = vWallBlocksRight(vWalls, r, c);
        cellLine += wallRight ? '║' : ' ';
      }
    }
    lines.push(cellLine);

    // ── Wall row below ────────────────────────────────────────────────────
    if (r > 0) {
      let wallLine = '   ';
      for (let c = 0; c < 9; c++) {
        const wallBelow = hWallBlocksBelow(hWalls, r, c);
        wallLine += wallBelow ? '═══' : '   ';
        if (c < 8) {
          // Intersection: show + if any wall meets here
          const hLeft  = hWallBlocksBelow(hWalls, r, c);
          const hRight = hWallBlocksBelow(hWalls, r, c + 1);
          const vUp    = vWallBlocksRight(vWalls, r, c);
          const vDown  = vWallBlocksRight(vWalls, r - 1, c);
          wallLine += (hLeft || hRight || vUp || vDown) ? '+' : ' ';
        }
      }
      lines.push(wallLine);
    }
  }

  // Column labels
  lines.push('');
  let colLine = '    ';
  for (let c = 0; c < 9; c++) {
    colLine += ` ${COL_LABELS[c]}  `;
  }
  lines.push(colLine);

  // Wall list
  const placed = listPlacedWalls(board);
  if (placed.length > 0) {
    lines.push('');
    lines.push(`  Walls on board: ${placed.join(' ')}`);
  }

  return lines.join('\n');
}

/**
 * A horizontal wall at hWalls[row][col] blocks movement between
 * rows `row` and `row+1`, covering columns col and col+1.
 * "Below row r, at column c" means: is there a wall segment here?
 */
function hWallBlocksBelow(hWalls, r, c) {
  // Anchor row index = r-1 (wall below row r = between r-1 and r in 0-indexed)
  const wr = r - 1;
  if (wr < 0 || wr > 7) return false;
  // Wall at [wr][c] covers cols c, c+1 → blocks below cols c and c+1
  if (c <= 7 && hWalls[wr][c]) return true;
  // Wall at [wr][c-1] covers cols c-1, c → also blocks below col c
  if (c >= 1 && hWalls[wr][c - 1]) return true;
  return false;
}

/**
 * A vertical wall at vWalls[row][col] blocks movement between
 * columns `col` and `col+1`, covering rows row and row+1.
 * "Right of col c at row r" means: is there a wall segment here?
 */
function vWallBlocksRight(vWalls, r, c) {
  if (c < 0 || c > 7) return false;
  // Wall at [r][c] covers rows r, r+1 → blocks right of col c at rows r and r+1
  if (r <= 7 && vWalls[r][c]) return true;
  // Wall at [r-1][c] covers rows r-1, r → also blocks right of col c at row r
  if (r >= 1 && vWalls[r - 1][c]) return true;
  return false;
}

function listPlacedWalls(board) {
  const out = [];
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      if (board.hWalls[r][c]) {
        out.push(`${COL_LABELS[c]}${r + 1}h`);
      }
      if (board.vWalls[r][c]) {
        out.push(`${COL_LABELS[c]}${r + 1}v`);
      }
    }
  }
  return out;
}

// ── Shortest path (BFS) for distance display ──────────────────────────────────

function bfsDistance(board, playerIndex) {
  const { row: startRow, col: startCol } = board.pawns[playerIndex];
  const goalRow = playerIndex === 0 ? 8 : 0;

  const visited = Array.from({ length: 9 }, () => new Array(9).fill(false));
  const queue = [{ row: startRow, col: startCol, dist: 0 }];
  visited[startRow][startCol] = true;

  while (queue.length > 0) {
    const { row, col, dist } = queue.shift();
    if (row === goalRow) return dist;

    const dirs = [
      { dr: 1, dc: 0 },
      { dr: -1, dc: 0 },
      { dr: 0, dc: 1 },
      { dr: 0, dc: -1 },
    ];

    for (const { dr, dc } of dirs) {
      if (!canStep(board, row, col, dr, dc)) continue;
      const nr = row + dr;
      const nc = col + dc;
      if (nr < 0 || nr > 8 || nc < 0 || nc > 8) continue;
      if (visited[nr][nc]) continue;
      visited[nr][nc] = true;
      queue.push({ row: nr, col: nc, dist: dist + 1 });
    }
  }
  return 99; // unreachable
}

function canStep(board, r, c, dr, dc) {
  // Moving north (dr=+1): blocked by hWall at anchor row=r, cols c and c-1
  if (dr === 1) {
    if (r > 7) return false;
    return !hWallBlocksBelow(board.hWalls, r + 1, c);
  }
  // Moving south (dr=-1): blocked by hWall below row r
  if (dr === -1) {
    if (r < 1) return false;
    return !hWallBlocksBelow(board.hWalls, r, c);
  }
  // Moving east (dc=+1): blocked by vWall at col c
  if (dc === 1) {
    if (c > 7) return false;
    return !vWallBlocksRight(board.vWalls, r, c);
  }
  // Moving west (dc=-1): blocked by vWall at col c-1
  if (dc === -1) {
    if (c < 1) return false;
    return !vWallBlocksRight(board.vWalls, r, c - 1);
  }
  return false;
}

// ── CAT v2 heatmap ────────────────────────────────────────────────────────────
// Mirrors the Rust engine's `build_consensus_attention_v2` logic so the AI
// (or developer) can inspect the attention map for any position.

const CAT_CORRIDOR_CM = 200;
const CAT_BROAD_FLOOR_CM = 10;

const DIRS = [{ dr: 1, dc: 0 }, { dr: -1, dc: 0 }, { dr: 0, dc: 1 }, { dr: 0, dc: -1 }];

/** BFS from all goal-row squares → dist_to[r][c] = steps to reach any goal cell. */
function fillDistToGoal(board, goalRow) {
  const dist = Array.from({ length: 9 }, () => new Array(9).fill(Infinity));
  const queue = Array.from({ length: 9 }, (_, c) => {
    dist[goalRow][c] = 0;
    return { row: goalRow, col: c };
  });
  let head = 0;
  while (head < queue.length) {
    const { row, col } = queue[head++];
    for (const { dr, dc } of DIRS) {
      const nr = row + dr;
      const nc = col + dc;
      if (nr < 0 || nr > 8 || nc < 0 || nc > 8) continue;
      if (dist[nr][nc] !== Infinity) continue;
      if (!canStep(board, nr, nc, -dr, -dc)) continue;
      dist[nr][nc] = dist[row][col] + 1;
      queue.push({ row: nr, col: nc });
    }
  }
  return dist;
}

/** BFS from the pawn's square → dist_from[r][c] = steps from pawn. */
function fillDistFromPawn(board, startRow, startCol) {
  const dist = Array.from({ length: 9 }, () => new Array(9).fill(Infinity));
  dist[startRow][startCol] = 0;
  const queue = [{ row: startRow, col: startCol }];
  let head = 0;
  while (head < queue.length) {
    const { row, col } = queue[head++];
    for (const { dr, dc } of DIRS) {
      if (!canStep(board, row, col, dr, dc)) continue;
      const nr = row + dr;
      const nc = col + dc;
      if (nr < 0 || nr > 8 || nc < 0 || nc > 8) continue;
      if (dist[nr][nc] !== Infinity) continue;
      dist[nr][nc] = dist[row][col] + 1;
      queue.push({ row: nr, col: nc });
    }
  }
  return dist;
}

/** Heat contributed by one player for one square (centi-squares). */
function squareHeat(shortest, dfp, dtg) {
  if (shortest === Infinity || dtg === Infinity) return CAT_BROAD_FLOOR_CM;
  const delta = Math.max(0, dfp + dtg - shortest);
  return Math.max(Math.floor(CAT_CORRIDOR_CM / (1 + delta * delta)), CAT_BROAD_FLOOR_CM);
}

/**
 * Build the combined CAT v2 heat map: cat[r][c] in centi-squares.
 *
 * For each player: delta = distFromPawn[sq] + distToGoal[sq] - shortest.
 * Heat per player = CAT_CORRIDOR_CM / (1 + delta²), min CAT_BROAD_FLOOR_CM.
 * Both players are summed into one map.
 */
function buildCatV2(board) {
  const cat = Array.from({ length: 9 }, () => new Array(9).fill(0));

  for (let playerIndex = 0; playerIndex < 2; playerIndex++) {
    const { row: pr, col: pc } = board.pawns[playerIndex];
    const goalRow = playerIndex === 0 ? 8 : 0;
    const distFrom = fillDistFromPawn(board, pr, pc);
    const distTo = fillDistToGoal(board, goalRow);
    const shortest = distTo[pr][pc];

    for (let r = 0; r < 9; r++) {
      for (let c = 0; c < 9; c++) {
        const dfp = distFrom[r][c];
        if (dfp === Infinity) continue;
        const heat = squareHeat(shortest, dfp, distTo[r][c]);
        cat[r][c] += heat;
      }
    }
  }
  return cat;
}

const CAT_HOT_CM = 160;
const CAT_COLD_CM = 60;

function catCellGlyph(v, isW, isB) {
  if (isW) return ' W  ';
  if (isB) return ' B  ';
  if (v === 0) return '  ·  ';
  let tag;
  if (v >= CAT_HOT_CM) {
    tag = '!';
  } else if (v < CAT_COLD_CM) {
    tag = '~';
  } else {
    tag = ' ';
  }
  return `${String(v).padStart(3)}${tag} `;
}

/** Render the CAT heat map as a grid beside short labels. */
function renderCatHeatmap(board, cat) {
  const colLabels = '     ' + [...COL_LABELS].map((l) => `  ${l}   `).join('');
  const legend = '  ! = hot (≥160)   ~ = cold (<60)   · = unreachable/zero';

  const rowLines = [];
  const { pawns } = board;
  for (let r = 8; r >= 0; r--) {
    let rowStr = `${String(r + 1).padStart(2)} `;
    for (let c = 0; c < 9; c++) {
      const isW = pawns[0].row === r && pawns[0].col === c;
      const isB = pawns[1].row === r && pawns[1].col === c;
      rowStr += catCellGlyph(cat[r][c], isW, isB);
    }
    rowLines.push(rowStr);
  }

  return [
    '  CAT v2 heat (cm)  · HOT ≥160 · COLD <60 ·',
    '',
    ...rowLines,
    '',
    colLabels,
    legend,
  ].join('\n');
}

// ── Main ─────────────────────────────────────────────────────────────────────

const showCat = process.argv.includes('--cat');
const moves = parseMoves(process.argv.slice(2).filter((a) => a !== '--cat'));

const board = new Board();
const errors = [];
for (const mv of moves) {
  try {
    board.applyAlgebraic(mv);
  } catch (e) {
    errors.push(e.message);
  }
}

const wDist = bfsDistance(board, 0);
const bDist = bfsDistance(board, 1);

const rendered = renderBoard(board);
const distLine = `  BFS distances — White: ${wDist} steps to row 9   Black: ${bDist} steps to row 1`;
const moveLine = `  Move history (${moves.length}): ${moves.join(' ')}`;

const parts = [rendered, distLine, moveLine];
if (errors.length) {
  parts.push(`  ERRORS: ${errors.join(', ')}`);
}
if (showCat) {
  const cat = buildCatV2(board);
  parts.push('', renderCatHeatmap(board, cat));
}
const output = parts.join('\n');

process.stdout.write(output + '\n');
fs.writeFileSync(OUT, output + '\n', 'utf8');
process.stdout.write(`\n  → written to ${OUT}\n`);
