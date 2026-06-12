//! Legal move generation — pawn jumps + wall placements with path validation.

use crate::core::board::{Board, Move, Player, WallOrientation};
use crate::core::zobrist;
use crate::movegen::pawn_bits::{
    generate_pawn_moves_bitboard_with_masks, generate_pawn_moves_shift_slice,
};
use crate::path::masks::DirMasks;
use crate::path::BfsScratch;
use crate::util::grid::{can_step, goal_row, has_wall, set_wall, square_index, unpack_square};

const DIRS: [(i8, i8); 4] = [(1, 0), (0, 1), (-1, 0), (0, -1)];

/// Upper bound on legal moves in any Quoridor position (startpos ≈ 131).
pub const MAX_LEGAL_MOVES: usize = 140;

/// Pawn-generation strategy — production uses [`PawnGenMode::ShiftCanStep`]; other modes for benches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PawnGenMode {
    /// ~4× `can_step` per node — no mask table.
    Scalar,
    /// Full-board `DirMasks::from_board` + bitmask axis logic (current default).
    BitboardFreshDirMasks,
    /// Reuse `BfsScratch::dir_masks` — incorrect if stale after wall trials.
    BitboardCachedDirMasks,
    /// Blind bit shift + `can_step` wall check — no `DirMasks`.
    ShiftCanStep,
}

impl Default for PawnGenMode {
    fn default() -> Self {
        Self::ShiftCanStep
    }
}

fn generate_pawn_moves_with_mode(
    board: &Board,
    scratch: &mut BfsScratch,
    out: &mut [Move],
    mode: PawnGenMode,
) -> usize {
    match mode {
        PawnGenMode::Scalar => generate_pawn_moves_scalar_for(board, board.side_to_move, out),
        PawnGenMode::BitboardFreshDirMasks => {
            let masks = DirMasks::from_board(board);
            generate_pawn_moves_bitboard_with_masks(board, &masks, out)
        }
        PawnGenMode::BitboardCachedDirMasks => {
            let masks = scratch.dir_masks(board);
            generate_pawn_moves_bitboard_with_masks(board, &masks, out)
        }
        PawnGenMode::ShiftCanStep => generate_pawn_moves_shift_slice(board, out),
    }
}

pub fn generate_legal_moves(board: &Board) -> Vec<Move> {
    let mut copy = board.clone();
    let mut out = Vec::new();
    let mut scratch = BfsScratch::new();
    generate_legal_moves_into(&mut copy, &mut out, &mut scratch);
    out
}

/// Hot-path API — stack buffer in perft, zero heap allocs per node.
pub fn generate_legal_moves_slice(
    board: &mut Board,
    out: &mut [Move],
    scratch: &mut BfsScratch,
) -> usize {
    if board.is_terminal().is_some() {
        return 0;
    }

    generate_legal_moves_slice_mode(board, out, scratch, PawnGenMode::default())
}

/// Legal moves with a selectable pawn generator — wall path logic unchanged.
pub fn generate_legal_moves_slice_mode(
    board: &mut Board,
    out: &mut [Move],
    scratch: &mut BfsScratch,
    mode: PawnGenMode,
) -> usize {
    if board.is_terminal().is_some() {
        return 0;
    }

    let mut n = generate_pawn_moves_with_mode(board, scratch, out, mode);
    if board.walls_remaining[board.side_to_move as usize] > 0 {
        n += generate_wall_moves_slice(board, &mut out[n..], scratch);
    }
    debug_assert!(n <= MAX_LEGAL_MOVES);
    n
}

/// Reuses `out` buffer and `scratch` BFS pool — board restored after wall trials.
pub fn generate_legal_moves_into(board: &mut Board, out: &mut Vec<Move>, scratch: &mut BfsScratch) {
    out.clear();
    let mut buf = [Move::Pawn { row: 0, col: 0 }; MAX_LEGAL_MOVES];
    let n = generate_legal_moves_slice(board, &mut buf, scratch);
    out.extend_from_slice(&buf[..n]);
}

pub fn generate_pawn_moves(board: &Board) -> Vec<Move> {
    let mut out = Vec::with_capacity(4);
    generate_pawn_moves_into(board, &mut out);
    out
}

pub fn generate_pawn_moves_into(board: &Board, out: &mut Vec<Move>) {
    let mut buf = [Move::Pawn { row: 0, col: 0 }; 8];
    let n = generate_pawn_moves_slice(board, &mut buf);
    out.extend_from_slice(&buf[..n]);
}

pub(crate) fn generate_pawn_moves_slice(board: &Board, out: &mut [Move]) -> usize {
    generate_pawn_moves_scalar_for(board, board.side_to_move, out)
}

/// Pawn moves for an arbitrary player — no board clone, no wall generation.
/// Hot path for mobility eval: counting pawn moves must never trigger the
/// full legal movegen (which BFS-validates every wall placement).
pub(crate) fn generate_pawn_moves_for(board: &Board, player: Player, out: &mut [Move]) -> usize {
    generate_pawn_moves_scalar_for(board, player, out)
}

/// Scalar pawn moves — kept for mobility eval and differential tests vs bitboard.
pub(crate) fn generate_pawn_moves_scalar_for(
    board: &Board,
    player: Player,
    out: &mut [Move],
) -> usize {
    let side = player as usize;
    let (fr, fc) = board.pawns[side];
    let (or, oc) = board.pawns[1 - side];
    let mut n = 0usize;

    for (dr, dc) in DIRS {
        if !can_step(board, fr, fc, dr, dc) {
            continue;
        }
        let nr = (fr as i8 + dr) as u8;
        let nc = (fc as i8 + dc) as u8;

        if (nr, nc) != (or, oc) {
            out[n] = Move::Pawn { row: nr, col: nc };
            n += 1;
            continue;
        }

        if can_step(board, nr, nc, dr, dc) {
            let jr = (nr as i8 + dr) as u8;
            let jc = (nc as i8 + dc) as u8;
            out[n] = Move::Pawn { row: jr, col: jc };
            n += 1;
            continue;
        }

        let perp = if dr != 0 {
            [(0i8, 1i8), (0, -1)]
        } else {
            [(1, 0), (-1, 0)]
        };
        for (pdr, pdc) in perp {
            if can_step(board, nr, nc, pdr, pdc) {
                let sr = (nr as i8 + pdr) as u8;
                let sc = (nc as i8 + pdc) as u8;
                out[n] = Move::Pawn { row: sr, col: sc };
                n += 1;
            }
        }
    }
    n
}

pub fn generate_wall_moves(board: &Board) -> Vec<Move> {
    let mut copy = board.clone();
    let mut out = Vec::with_capacity(64);
    let mut scratch = BfsScratch::new();
    generate_wall_moves_into(&mut copy, &mut out, &mut scratch);
    out
}

pub fn generate_wall_moves_into(board: &mut Board, out: &mut Vec<Move>, scratch: &mut BfsScratch) {
    let mut buf = [Move::Wall {
        row: 0,
        col: 0,
        orientation: WallOrientation::Horizontal,
    }; MAX_LEGAL_MOVES];
    let n = generate_wall_moves_slice(board, &mut buf, scratch);
    out.extend_from_slice(&buf[..n]);
}

fn generate_wall_moves_slice(
    board: &mut Board,
    out: &mut [Move],
    scratch: &mut BfsScratch,
) -> usize {
    let mut path_cache = None;
    let mut n = 0usize;
    n += collect_wall_orientation(
        board,
        !board.horizontal_walls,
        WallOrientation::Horizontal,
        &mut out[n..],
        scratch,
        &mut path_cache,
    );
    n += collect_wall_orientation(
        board,
        !board.vertical_walls,
        WallOrientation::Vertical,
        &mut out[n..],
        scratch,
        &mut path_cache,
    );
    n
}

/// Iterate only **empty** wall slots via `trailing_zeros` — skips occupied bits early.
fn collect_wall_orientation(
    board: &mut Board,
    mut free: u64,
    orientation: WallOrientation,
    out: &mut [Move],
    scratch: &mut BfsScratch,
    path_cache: &mut Option<WallPathCache>,
) -> usize {
    let mut n = 0usize;
    while free != 0 {
        let bit = free.trailing_zeros();
        free &= free - 1;
        let row = (bit / 8) as u8;
        let col = (bit % 8) as u8;
        if is_legal_wall(board, row, col, orientation, scratch, path_cache) {
            out[n] = Move::Wall {
                row,
                col,
                orientation,
            };
            n += 1;
        }
    }
    n
}

fn is_legal_wall(
    board: &mut Board,
    row: u8,
    col: u8,
    orientation: WallOrientation,
    scratch: &mut BfsScratch,
    _path_cache: &mut Option<WallPathCache>,
) -> bool {
    if wall_collides(board, row, col, orientation) {
        return false;
    }
    // Matches scraped `canWallBlock` — isolated walls cannot cage anyone (perft fast path).
    if !can_wall_block_topology(board, row, col, orientation) {
        return true;
    }
    // Do NOT skip BFS when the wall is off the witness shortest path — unsound (a1h/a5h).
    path_ok_after_wall(board, row, col, orientation, scratch)
}

struct WallPathCache {
    p1: [u8; 81],
    p2: [u8; 81],
    p1_len: usize,
    p2_len: usize,
}

impl WallPathCache {
    fn new(board: &Board, scratch: &mut BfsScratch) -> Self {
        let mut p1 = [u8::MAX; 81];
        let mut p2 = [u8::MAX; 81];
        let p1_len = shortest_path(board, Player::One, scratch, &mut p1);
        let p2_len = shortest_path(board, Player::Two, scratch, &mut p2);
        Self {
            p1,
            p2,
            p1_len,
            p2_len,
        }
    }

    #[inline]
    fn wall_intersects_either_path(&self, row: u8, col: u8, orientation: WallOrientation) -> bool {
        wall_intersects_path(row, col, orientation, &self.p1, self.p1_len)
            || wall_intersects_path(row, col, orientation, &self.p2, self.p2_len)
    }
}

fn shortest_path(
    board: &Board,
    player: Player,
    scratch: &mut BfsScratch,
    path_out: &mut [u8; 81],
) -> usize {
    let mut next_out = [u8::MAX; 81];
    scratch.fill_next_toward_goal(board, player, &mut next_out);

    let (pr, pc) = board.pawn(player);
    let mut current = square_index(pr, pc);
    let mut len = 0usize;
    while len < path_out.len() {
        path_out[len] = current;
        len += 1;

        let (row, _) = unpack_square(current);
        if row == goal_row(player) {
            break;
        }

        let next = next_out[current as usize];
        if next == u8::MAX {
            break;
        }
        current = next;
    }
    len
}

#[inline]
fn wall_intersects_path(
    row: u8,
    col: u8,
    orientation: WallOrientation,
    path: &[u8; 81],
    len: usize,
) -> bool {
    if len <= 1 {
        return false;
    }
    for i in 0..(len - 1) {
        if wall_blocks_path_step(row, col, orientation, path[i], path[i + 1]) {
            return true;
        }
    }
    false
}

#[inline]
fn wall_blocks_path_step(row: u8, col: u8, orientation: WallOrientation, sq1: u8, sq2: u8) -> bool {
    let (r1, c1) = unpack_square(sq1);
    let (r2, c2) = unpack_square(sq2);
    match orientation {
        WallOrientation::Horizontal => {
            if c1 == c2 && r1.abs_diff(r2) == 1 {
                let min_r = r1.min(r2);
                min_r == row && (c1 == col || c1 == col + 1)
            } else {
                false
            }
        }
        WallOrientation::Vertical => {
            if r1 == r2 && c1.abs_diff(c2) == 1 {
                let min_c = c1.min(c2);
                min_c == col && (r1 == row || r1 == row + 1)
            } else {
                false
            }
        }
    }
}

/// Trial wall in-place — set, BFS both goals, unset. No `Board::clone`.
#[inline]
fn path_ok_after_wall(
    board: &mut Board,
    row: u8,
    col: u8,
    orientation: WallOrientation,
    scratch: &mut BfsScratch,
) -> bool {
    // `set_wall` does not update `board.hash`; XOR so `dir_masks` rebuilds for the trial
    // topology instead of reusing a stale cache (a1h ply-22 illegal-move bug).
    zobrist::xor_wall(&mut board.hash, orientation, row, col);
    set_wall(board, row, col, orientation, true);
    let ok = scratch.both_players_reach_goals(board);
    set_wall(board, row, col, orientation, false);
    zobrist::xor_wall(&mut board.hash, orientation, row, col);
    ok
}

/// Trial wall placement — both players must still reach goals (website rules oracle).
pub fn wall_path_ok_after_place(
    board: &mut Board,
    row: u8,
    col: u8,
    orientation: WallOrientation,
) -> bool {
    let mut scratch = BfsScratch::new();
    path_ok_after_wall(board, row, col, orientation, &mut scratch)
}

/// Matches scraped `collidesWithExistingWall`.
fn wall_collides(board: &Board, row: u8, col: u8, orientation: WallOrientation) -> bool {
    let perpendicular = match orientation {
        WallOrientation::Horizontal => WallOrientation::Vertical,
        WallOrientation::Vertical => WallOrientation::Horizontal,
    };

    if has_wall(board, row, col, orientation) || has_wall(board, row, col, perpendicular) {
        return true;
    }

    match orientation {
        WallOrientation::Horizontal => {
            if col > 0 && has_wall(board, row, col - 1, WallOrientation::Horizontal) {
                return true;
            }
            if col < 7 && has_wall(board, row, col + 1, WallOrientation::Horizontal) {
                return true;
            }
        }
        WallOrientation::Vertical => {
            if row > 0 && has_wall(board, row - 1, col, WallOrientation::Vertical) {
                return true;
            }
            if row < 7 && has_wall(board, row + 1, col, WallOrientation::Vertical) {
                return true;
            }
        }
    }
    false
}

/// Matches scraped `canWallBlock` — wall must touch existing topology to matter.
pub fn can_wall_block_topology(
    board: &Board,
    row: u8,
    col: u8,
    orientation: WallOrientation,
) -> bool {
    let js_col = col + 1;
    let js_row = row + 1;

    let (on_a, on_b) = match orientation {
        // Scraped `sideOnEdge`: horizontal right edge is col 9 (`numCols`), not wall slot h (js_col 8).
        WallOrientation::Horizontal => (js_col == 1, js_col == 9),
        WallOrientation::Vertical => (js_row == 8, js_row == 1),
    };

    let side_a = on_a || touching_side_a(board, row, col, orientation);
    let side_b = on_b || touching_side_b(board, row, col, orientation);
    let middle = touching_middle(board, row, col, orientation);

    (side_a && side_b) || (side_a && middle) || (side_b && middle)
}

fn touching_side_a(board: &Board, row: u8, col: u8, orientation: WallOrientation) -> bool {
    match orientation {
        WallOrientation::Horizontal => {
            wall_at_offset(board, row, col, &[(0, -1)], WallOrientation::Vertical)
                || wall_at_offset(
                    board,
                    row,
                    col,
                    &[(1, 0), (0, -1)],
                    WallOrientation::Vertical,
                )
                || wall_at_offset(
                    board,
                    row,
                    col,
                    &[(-1, 0), (0, -1)],
                    WallOrientation::Vertical,
                )
                || wall_at_offset(
                    board,
                    row,
                    col,
                    &[(0, -1), (0, -1)],
                    WallOrientation::Horizontal,
                )
        }
        WallOrientation::Vertical => {
            wall_at_offset(board, row, col, &[(1, 0)], WallOrientation::Horizontal)
                || wall_at_offset(
                    board,
                    row,
                    col,
                    &[(0, -1), (1, 0)],
                    WallOrientation::Horizontal,
                )
                || wall_at_offset(
                    board,
                    row,
                    col,
                    &[(0, 1), (1, 0)],
                    WallOrientation::Horizontal,
                )
                || wall_at_offset(
                    board,
                    row,
                    col,
                    &[(1, 0), (1, 0)],
                    WallOrientation::Vertical,
                )
        }
    }
}

fn touching_side_b(board: &Board, row: u8, col: u8, orientation: WallOrientation) -> bool {
    match orientation {
        WallOrientation::Horizontal => {
            wall_at_offset(board, row, col, &[(0, 1)], WallOrientation::Vertical)
                || wall_at_offset(
                    board,
                    row,
                    col,
                    &[(1, 0), (0, 1)],
                    WallOrientation::Vertical,
                )
                || wall_at_offset(
                    board,
                    row,
                    col,
                    &[(-1, 0), (0, 1)],
                    WallOrientation::Vertical,
                )
                || wall_at_offset(
                    board,
                    row,
                    col,
                    &[(0, 1), (0, 1)],
                    WallOrientation::Horizontal,
                )
        }
        WallOrientation::Vertical => {
            wall_at_offset(board, row, col, &[(-1, 0)], WallOrientation::Horizontal)
                || wall_at_offset(
                    board,
                    row,
                    col,
                    &[(0, -1), (-1, 0)],
                    WallOrientation::Horizontal,
                )
                || wall_at_offset(
                    board,
                    row,
                    col,
                    &[(0, 1), (-1, 0)],
                    WallOrientation::Horizontal,
                )
                || wall_at_offset(
                    board,
                    row,
                    col,
                    &[(-1, 0), (-1, 0)],
                    WallOrientation::Vertical,
                )
        }
    }
}

fn touching_middle(board: &Board, row: u8, col: u8, orientation: WallOrientation) -> bool {
    match orientation {
        WallOrientation::Horizontal => {
            wall_at_offset(board, row, col, &[(1, 0)], WallOrientation::Vertical)
                || wall_at_offset(board, row, col, &[(-1, 0)], WallOrientation::Vertical)
        }
        WallOrientation::Vertical => {
            wall_at_offset(board, row, col, &[(0, -1)], WallOrientation::Horizontal)
                || wall_at_offset(board, row, col, &[(0, 1)], WallOrientation::Horizontal)
        }
    }
}

fn wall_at_offset(
    board: &Board,
    row: u8,
    col: u8,
    offsets: &[(i8, i8)],
    orientation: WallOrientation,
) -> bool {
    let (wr, wc) = apply_offsets(row, col, offsets);
    if wr > 7 || wc > 7 {
        return false;
    }
    has_wall(board, wr, wc, orientation)
}

fn apply_offsets(mut row: u8, mut col: u8, offsets: &[(i8, i8)]) -> (u8, u8) {
    for (dr, dc) in offsets {
        row = (row as i16 + *dr as i16) as u8;
        col = (col as i16 + *dc as i16) as u8;
    }
    (row, col)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn start_has_three_pawn_moves_for_white() {
        let board = Board::new();
        let pawns = generate_pawn_moves(&board);
        assert_eq!(pawns.len(), 3);
    }

    #[test]
    fn start_has_many_wall_moves() {
        let board = Board::new();
        let walls = generate_wall_moves(&board);
        assert!(walls.len() > 100);
    }
    #[test]
    fn slice_matches_vec_at_startpos() {
        let mut board = Board::new();
        let mut scratch = BfsScratch::new();
        let mut slice_buf = [Move::Pawn { row: 0, col: 0 }; MAX_LEGAL_MOVES];
        let n = generate_legal_moves_slice(&mut board, &mut slice_buf, &mut scratch);
        let vec_moves = generate_legal_moves(&board);
        assert_eq!(n, vec_moves.len());
        assert_eq!(&slice_buf[..n], vec_moves.as_slice());
        assert!(n <= MAX_LEGAL_MOVES);
    }

    #[test]
    fn wall_trial_leaves_board_unchanged() {
        let mut board = Board::new();
        let before = board.clone();
        let mut scratch = BfsScratch::new();
        let mut moves = Vec::new();
        generate_wall_moves_into(&mut board, &mut moves, &mut scratch);
        assert_eq!(board, before);
    }
}
