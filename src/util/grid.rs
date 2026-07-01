//! O(1) wall checks and step logic — matches scraped `gameLogic.js` pawnCanMove / hasWall.

use crate::core::board::{Board, Player, WallOrientation};

/// P1 races to row 8, P2 to row 0 (internal 0..8 indexing).
#[inline]
pub fn goal_row(player: Player) -> u8 {
    match player {
        Player::One => 8,
        Player::Two => 0,
    }
}

#[inline]
pub fn is_goal(player: Player, row: u8) -> bool {
    row == goal_row(player)
}

/// True when a pawn step moves toward that player's goal row (not lateral/back).
#[inline]
pub fn pawn_geometrically_advances(stm: Player, from_row: u8, to_row: u8) -> bool {
    match stm {
        Player::One => to_row > from_row,
        Player::Two => to_row < from_row,
    }
}

#[inline]
fn has_horizontal(board: &Board, js_row: u8, col: u8) -> bool {
    if !(1..=8).contains(&js_row) || col >= 8 {
        return false;
    }
    let bit = ((js_row - 1) as u32) * 8 + col as u32;
    (board.horizontal_walls >> bit) & 1 != 0
}

#[inline]
fn has_vertical(board: &Board, js_row: u8, col: u8) -> bool {
    if !(1..=8).contains(&js_row) || col >= 8 {
        return false;
    }
    let bit = ((js_row - 1) as u32) * 8 + col as u32;
    (board.vertical_walls >> bit) & 1 != 0
}

/// Can the pawn at `(row, col)` step by `(dr, dc)`? Both in 0..8, steps are -1/0/1.
#[inline]
pub fn can_step(board: &Board, row: u8, col: u8, dr: i8, dc: i8) -> bool {
    let nr = row as i16 + dr as i16;
    let nc = col as i16 + dc as i16;
    if !(0..=8).contains(&nr) || !(0..=8).contains(&nc) {
        return false;
    }
    let nr = nr as u8;
    let nc = nc as u8;
    let js_from = row + 1;
    let js_to = nr + 1;

    match (dr, dc) {
        (1, 0) => {
            !has_horizontal(board, js_from, col)
                && (col == 0 || !has_horizontal(board, js_from, col - 1))
        }
        (-1, 0) => {
            !has_horizontal(board, js_to, col)
                && (col == 0 || !has_horizontal(board, js_to, col - 1))
        }
        // Lateral steps: match scraped `pawnCanMove` wall anchors (see docs/video/05-first-perft-bug.md).
        // Right — wallAnchor = from, sideAnchor = step(from, Down).
        (0, 1) => !has_vertical(board, js_from, col) && !has_vertical(board, row, col),
        // Left — wallAnchor = to, sideAnchor = step(to, Down).
        (0, -1) => !has_vertical(board, js_to, nc) && !has_vertical(board, nr, nc),
        _ => false,
    }
}

#[inline]
pub fn square_index(row: u8, col: u8) -> u8 {
    row * 9 + col
}

#[inline]
pub fn unpack_square(sq: u8) -> (u8, u8) {
    crate::bench_instr::count(|b| &mut b.unpack_square, || (sq / 9, sq % 9))
}

// ── Centered u128 flood layout (11×11 stride, 9×9 playable) ─────────────────
//
// Playable square (r,c) maps to bit `(r + ROW_PAD) * STRIDE + (c + COL_PAD)`.
// Buffer columns absorb east/west shifts so `<< 1` / `>> 1` never wrap rows.

/// Columns per row in the flood bitboard (9 playable + 1 buffer each side).
pub const FLOOD_STRIDE: u32 = 11;
pub const FLOOD_COL_PAD: u32 = 1;
pub const FLOOD_ROW_PAD: u32 = 1;

#[inline]
pub const fn flood_bit_index(row: u8, col: u8) -> u32 {
    (row as u32 + FLOOD_ROW_PAD) * FLOOD_STRIDE + col as u32 + FLOOD_COL_PAD
}

#[inline]
pub fn flood_bit_sq(sq: u8) -> u128 {
    crate::bench_instr::count(|b| &mut b.flood_bit_sq, || FLOOD_BIT_BY_SQ[sq as usize])
}

#[inline]
pub fn flood_sq_from_bit(bit: u32) -> Option<u8> {
    crate::bench_instr::record(
        |b| &mut b.flood_sq_from_bit,
        || flood_sq_from_bit_inner(bit),
    )
}

#[inline]
fn flood_sq_from_bit_inner(bit: u32) -> Option<u8> {
    if bit >= 128 {
        return None;
    }
    let sq = FLOOD_SQ_BY_BIT[bit as usize];
    if sq == u8::MAX {
        return None;
    }
    Some(sq)
}

const fn flood_bit_by_sq_table() -> [u128; 81] {
    let mut out = [0u128; 81];
    let mut sq = 0u8;
    while sq < 81 {
        let row = sq / 9;
        let col = sq % 9;
        out[sq as usize] = 1u128 << flood_bit_index(row, col);
        sq += 1;
    }
    out
}

const fn flood_sq_by_bit_table() -> [u8; 128] {
    let mut out = [u8::MAX; 128];
    let mut sq = 0u8;
    while sq < 81 {
        let row = sq / 9;
        let col = sq % 9;
        out[flood_bit_index(row, col) as usize] = sq;
        sq += 1;
    }
    out
}

pub const FLOOD_BIT_BY_SQ: [u128; 81] = flood_bit_by_sq_table();
pub const FLOOD_SQ_BY_BIT: [u8; 128] = flood_sq_by_bit_table();

const fn flood_playable_mask() -> u128 {
    let mut mask = 0u128;
    let mut row = 0u8;
    while row < 9 {
        let mut col = 0u8;
        while col < 9 {
            let bit = flood_bit_index(row, col);
            mask |= 1u128 << bit;
            col += 1;
        }
        row += 1;
    }
    mask
}

/// All 81 playable pawn squares in centered flood-bit layout.
pub const FLOOD_PLAYABLE: u128 = flood_playable_mask();

/// Pack centered flood bits → compact game-square mask (sq 0..80).
#[inline]
pub fn pack_flood_mask(bits: u128) -> u128 {
    let mut out = 0u128;
    let mut b = bits & FLOOD_PLAYABLE;
    while b != 0 {
        let fb = b.trailing_zeros();
        if let Some(sq) = flood_sq_from_bit(fb) {
            out |= 1u128 << sq;
        }
        b &= b - 1;
    }
    out
}

/// Pawn squares adjacent to a wall segment (internal wall coords).
#[inline]
pub fn wall_touch_squares(row: u8, col: u8, orientation: WallOrientation) -> [(u8, u8); 4] {
    match orientation {
        WallOrientation::Horizontal | WallOrientation::Vertical => [
            (row, col),
            (row, col + 1),
            (row + 1, col),
            (row + 1, col + 1),
        ],
    }
}

pub fn set_wall(board: &mut Board, row: u8, col: u8, orientation: WallOrientation, place: bool) {
    debug_assert!((1..=8).contains(&(row + 1)) && col < 8);
    let js_row = row + 1;
    let bit = 1u64 << (((js_row - 1) as u32) * 8 + col as u32);
    match orientation {
        WallOrientation::Horizontal => {
            if place {
                board.horizontal_walls |= bit;
            } else {
                board.horizontal_walls &= !bit;
            }
        }
        WallOrientation::Vertical => {
            if place {
                board.vertical_walls |= bit;
            } else {
                board.vertical_walls &= !bit;
            }
        }
    }
}

pub fn has_wall(board: &Board, row: u8, col: u8, orientation: WallOrientation) -> bool {
    let js_row = row + 1;
    match orientation {
        WallOrientation::Horizontal => has_horizontal(board, js_row, col),
        WallOrientation::Vertical => has_vertical(board, js_row, col),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::board::{Board, WallOrientation};

    #[test]
    fn flood_layout_centered_with_side_buffers() {
        assert_eq!(FLOOD_PLAYABLE.count_ones(), 81);
        assert!(flood_bit_index(8, 8) < 128);
        // (0,0) and (0,8) share a row in flood layout — east shift stays on row 0.
        let left = flood_bit_index(0, 0);
        let right = flood_bit_index(0, 8);
        assert_eq!(right - left, 8);
        // Buffer column west of (0,0).
        assert_eq!(flood_sq_from_bit(left - 1), None);
        // Buffer column east of (0,8).
        assert_eq!(flood_sq_from_bit(right + 1), None);
        for sq in 0u8..81 {
            let packed = pack_flood_mask(flood_bit_sq(sq));
            assert_eq!(packed, 1u128 << sq);
        }
    }

    #[test]
    fn vertical_d8v_blocks_black_left_from_e9() {
        let mut board = Board::new();
        set_wall(&mut board, 7, 3, WallOrientation::Vertical, true);
        board.side_to_move = crate::core::board::Player::Two;
        // P2 at e9 (internal 8,4) — left to d9 must be blocked by d8v.
        assert!(!can_step(&board, 8, 4, 0, -1));
        assert!(can_step(&board, 8, 4, 0, 1));
    }
}
