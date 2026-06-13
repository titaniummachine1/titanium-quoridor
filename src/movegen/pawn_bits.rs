//! Pawn move generation — scalar reference, `DirMasks` bitboard, and shift+wall-check.
//!
//! Compare against `generate_pawn_moves_scalar_for` in tests/benches.

use crate::core::board::{Board, Move};
use crate::path::masks::DirMasks;
use crate::util::grid::{
    can_step, flood_bit_sq, flood_sq_from_bit, square_index, unpack_square, FLOOD_STRIDE,
};

type StepFn = fn(u128, &DirMasks) -> u128;

#[inline]
fn step_north(bit: u128, m: &DirMasks) -> u128 {
    (bit & m.north) >> FLOOD_STRIDE
}

#[inline]
fn step_south(bit: u128, m: &DirMasks) -> u128 {
    (bit & m.south) << FLOOD_STRIDE
}

#[inline]
fn step_east(bit: u128, m: &DirMasks) -> u128 {
    (bit & m.east) << 1
}

#[inline]
fn step_west(bit: u128, m: &DirMasks) -> u128 {
    (bit & m.west) >> 1
}

#[inline]
fn push_flood_target(bit: u128, out: &mut [Move], n: &mut usize) {
    if bit == 0 {
        return;
    }
    let fb = bit.trailing_zeros();
    if let Some(sq) = flood_sq_from_bit(fb) {
        let (r, c) = unpack_square(sq);
        out[*n] = Move::Pawn { row: r, col: c };
        *n += 1;
    }
}

/// One axis: step, straight jump over opponent, or lateral slides when jump blocked.
fn axis_moves(
    from: u128,
    opp: u128,
    masks: &DirMasks,
    step_fwd: StepFn,
    step_perp_a: StepFn,
    step_perp_b: StepFn,
    out: &mut [Move],
    n: &mut usize,
) {
    let neighbor = step_fwd(from, masks);
    if neighbor == 0 {
        return;
    }
    if neighbor != opp {
        push_flood_target(neighbor, out, n);
        return;
    }
    let jump = step_fwd(opp, masks);
    if jump != 0 {
        push_flood_target(jump, out, n);
        return;
    }
    push_flood_target(step_perp_a(opp, masks), out, n);
    push_flood_target(step_perp_b(opp, masks), out, n);
}

/// Pawn targets using pre-built direction masks (cache `DirMasks` across nodes in search).
pub fn generate_pawn_moves_bitboard_with_masks(
    board: &Board,
    masks: &DirMasks,
    out: &mut [Move],
) -> usize {
    let side = board.side_to_move as usize;
    let (fr, fc) = board.pawns[side];
    let (or, oc) = board.pawns[1 - side];
    let from = flood_bit_sq(square_index(fr, fc));
    let opp = flood_bit_sq(square_index(or, oc));
    let mut n = 0usize;

    axis_moves(from, opp, masks, step_north, step_west, step_east, out, &mut n);
    axis_moves(from, opp, masks, step_south, step_west, step_east, out, &mut n);
    axis_moves(from, opp, masks, step_east, step_north, step_south, out, &mut n);
    axis_moves(from, opp, masks, step_west, step_north, step_south, out, &mut n);

    n
}

/// Builds fresh `DirMasks` each call — matches scalar API shape for benchmarking.
pub fn generate_pawn_moves_bitboard_slice(board: &Board, out: &mut [Move]) -> usize {
    let masks = DirMasks::from_board(board);
    generate_pawn_moves_bitboard_with_masks(board, &masks, out)
}

#[inline]
fn push_pawn(row: u8, col: u8, out: &mut [Move], n: &mut usize) {
    out[*n] = Move::Pawn { row, col };
    *n += 1;
}

/// One axis: `can_step` wall + boundary check, then jump/lateral logic.
fn axis_shift(
    board: &Board,
    fr: u8,
    fc: u8,
    or: u8,
    oc: u8,
    dr: i8,
    dc: i8,
    perp_a: (i8, i8),
    perp_b: (i8, i8),
    out: &mut [Move],
    n: &mut usize,
) {
    if !can_step(board, fr, fc, dr, dc) {
        return;
    }
    let nr = (fr as i8 + dr) as u8;
    let nc = (fc as i8 + dc) as u8;

    if (nr, nc) != (or, oc) {
        push_pawn(nr, nc, out, n);
        return;
    }

    if can_step(board, nr, nc, dr, dc) {
        push_pawn((nr as i8 + dr) as u8, (nc as i8 + dc) as u8, out, n);
        return;
    }

    for (pdr, pdc) in [perp_a, perp_b] {
        if can_step(board, nr, nc, pdr, pdc) {
            push_pawn((nr as i8 + pdr) as u8, (nc as i8 + pdc) as u8, out, n);
        }
    }
}

/// Validate each direction with `can_step` — no `DirMasks` table, no flood bits.
pub fn generate_pawn_moves_shift_slice(board: &Board, out: &mut [Move]) -> usize {
    let side = board.side_to_move as usize;
    let (fr, fc) = board.pawns[side];
    let (or, oc) = board.pawns[1 - side];
    let mut n = 0usize;

    axis_shift(board, fr, fc, or, oc, -1, 0, (0, -1), (0, 1), out, &mut n);
    axis_shift(board, fr, fc, or, oc, 1, 0, (0, -1), (0, 1), out, &mut n);
    axis_shift(board, fr, fc, or, oc, 0, 1, (-1, 0), (1, 0), out, &mut n);
    axis_shift(board, fr, fc, or, oc, 0, -1, (-1, 0), (1, 0), out, &mut n);

    n
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::movegen::legal::generate_pawn_moves_scalar_for;
    use crate::core::board::Player;
    use crate::movegen::MAX_LEGAL_MOVES;
    use crate::path::BfsScratch;

    fn same_pawn_multiset(a: &[Move], b: &[Move]) -> bool {
        if a.len() != b.len() {
            return false;
        }
        a.iter().all(|mv| b.contains(mv))
    }

    fn assert_pawn_moves_match(board: &Board) {
        let mut scalar = [Move::Pawn { row: 0, col: 0 }; 8];
        let mut bitboard = [Move::Pawn { row: 0, col: 0 }; 8];
        let mut shift = [Move::Pawn { row: 0, col: 0 }; 8];
        let ns = generate_pawn_moves_scalar_for(board, board.side_to_move, &mut scalar);
        let nb = generate_pawn_moves_bitboard_slice(board, &mut bitboard);
        let nsh = generate_pawn_moves_shift_slice(board, &mut shift);
        assert!(
            same_pawn_multiset(&scalar[..ns], &bitboard[..nb]),
            "bitboard mismatch stm={:?} pawns={:?}",
            board.side_to_move,
            board.pawns
        );
        assert!(
            same_pawn_multiset(&scalar[..ns], &shift[..nsh]),
            "shift mismatch stm={:?} pawns={:?}",
            board.side_to_move,
            board.pawns
        );
    }

    fn walk_compare(board: &mut Board, depth: u32, scratch: &mut BfsScratch) {
        if board.is_terminal().is_some() {
            return;
        }
        assert_pawn_moves_match(board);

        if depth == 0 {
            return;
        }

        let mut legal = [Move::Pawn { row: 0, col: 0 }; MAX_LEGAL_MOVES];
        let n = crate::movegen::generate_legal_moves_slice(board, &mut legal, scratch);
        for &mv in &legal[..n] {
            let undo = board.make_move(mv);
            walk_compare(board, depth - 1, scratch);
            board.unmake_move(undo);
        }
    }

    #[test]
    fn bitboard_matches_scalar_startpos() {
        assert_pawn_moves_match(&Board::new());
    }

    #[test]
    fn bitboard_matches_scalar_perft_depth3() {
        let mut board = Board::new();
        let mut scratch = BfsScratch::new();
        walk_compare(&mut board, 3, &mut scratch);
    }

    #[test]
    fn bitboard_jump_lateral_when_forward_blocked() {
        let mut board = Board::new();
        // Face-to-face (adjacent!) with a wall directly behind black so the
        // straight jump is blocked — lateral jumps beside the opponent only.
        board.pawns = [(4, 4), (5, 4)];
        board.side_to_move = crate::core::board::Player::One;
        crate::util::grid::set_wall(
            &mut board,
            5,
            4,
            crate::core::board::WallOrientation::Horizontal,
            true,
        );
        assert_pawn_moves_match(&board);
        let mut out = [Move::Pawn { row: 0, col: 0 }; 8];
        let n = generate_pawn_moves_bitboard_slice(&board, &mut out);
        let targets: Vec<_> = out[..n]
            .iter()
            .map(|mv| match mv {
                Move::Pawn { row, col } => (*row, *col),
                _ => panic!("pawn only"),
            })
            .collect();
        // Straight jump (6,4) must be blocked; laterals land beside black.
        assert!(!targets.contains(&(6, 4)), "straight jump must be blocked");
        assert!(targets.contains(&(5, 3)), "west lateral jump");
        assert!(targets.contains(&(5, 5)), "east lateral jump");
    }
}
