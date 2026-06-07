//! Legal move generation — pawn jumps + wall placements with path validation.

use crate::board::{Board, Move, WallOrientation};
use crate::grid::{can_step, has_wall, set_wall};
use crate::path::both_players_reach_goals;

const DIRS: [(i8, i8); 4] = [(1, 0), (0, 1), (-1, 0), (0, -1)];

pub fn generate_legal_moves(board: &Board) -> Vec<Move> {
    if board.is_terminal().is_some() {
        return Vec::new();
    }

    let mut moves = generate_pawn_moves(board);
    if board.walls_remaining[board.side_to_move as usize] > 0 {
        moves.extend(generate_wall_moves(board));
    }
    moves
}

pub fn generate_pawn_moves(board: &Board) -> Vec<Move> {
    let side = board.side_to_move as usize;
    let (fr, fc) = board.pawns[side];
    let (or, oc) = board.pawns[1 - side];
    let mut out = Vec::with_capacity(4);

    for (dr, dc) in DIRS {
        if !can_step(board, fr, fc, dr, dc) {
            continue;
        }
        let nr = (fr as i8 + dr) as u8;
        let nc = (fc as i8 + dc) as u8;

        if (nr, nc) != (or, oc) {
            out.push(Move::Pawn { row: nr, col: nc });
            continue;
        }

        if can_step(board, nr, nc, dr, dc) {
            let jr = (nr as i8 + dr) as u8;
            let jc = (nc as i8 + dc) as u8;
            out.push(Move::Pawn { row: jr, col: jc });
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
                out.push(Move::Pawn { row: sr, col: sc });
            }
        }
    }

    out
}

pub fn generate_wall_moves(board: &Board) -> Vec<Move> {
    let mut out = Vec::with_capacity(64);
    for row in 0..8u8 {
        for col in 0..8u8 {
            for orientation in [WallOrientation::Horizontal, WallOrientation::Vertical] {
                if is_legal_wall(board, row, col, orientation) {
                    out.push(Move::Wall {
                        row,
                        col,
                        orientation,
                    });
                }
            }
        }
    }
    out
}

fn is_legal_wall(board: &Board, row: u8, col: u8, orientation: WallOrientation) -> bool {
    if wall_collides(board, row, col, orientation) {
        return false;
    }
    // Matches scraped JS: if `canWallBlock` is false, `isWallBlocking` short-circuits to false
    // (floating walls are legal). Only run path check when topology can matter.
    if !can_wall_block_topology(board, row, col, orientation) {
        return true;
    }

    let mut trial = board.clone();
    set_wall(&mut trial, row, col, orientation, true);
    both_players_reach_goals(&trial)
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
fn can_wall_block_topology(board: &Board, row: u8, col: u8, orientation: WallOrientation) -> bool {
    let js_col = col + 1;
    let js_row = row + 1;

    let (on_a, on_b) = match orientation {
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
                || wall_at_offset(board, row, col, &[(1, 0), (0, -1)], WallOrientation::Vertical)
                || wall_at_offset(board, row, col, &[(-1, 0), (0, -1)], WallOrientation::Vertical)
                || wall_at_offset(board, row, col, &[(0, -1), (0, -1)], WallOrientation::Horizontal)
        }
        WallOrientation::Vertical => {
            wall_at_offset(board, row, col, &[(1, 0)], WallOrientation::Horizontal)
                || wall_at_offset(board, row, col, &[(0, -1), (1, 0)], WallOrientation::Horizontal)
                || wall_at_offset(board, row, col, &[(0, 1), (1, 0)], WallOrientation::Horizontal)
                || wall_at_offset(board, row, col, &[(1, 0), (1, 0)], WallOrientation::Vertical)
        }
    }
}

fn touching_side_b(board: &Board, row: u8, col: u8, orientation: WallOrientation) -> bool {
    match orientation {
        WallOrientation::Horizontal => {
            wall_at_offset(board, row, col, &[(0, 1)], WallOrientation::Vertical)
                || wall_at_offset(board, row, col, &[(1, 0), (0, 1)], WallOrientation::Vertical)
                || wall_at_offset(board, row, col, &[(-1, 0), (0, 1)], WallOrientation::Vertical)
                || wall_at_offset(board, row, col, &[(0, 1), (0, 1)], WallOrientation::Horizontal)
        }
        WallOrientation::Vertical => {
            wall_at_offset(board, row, col, &[(-1, 0)], WallOrientation::Horizontal)
                || wall_at_offset(board, row, col, &[(0, -1), (-1, 0)], WallOrientation::Horizontal)
                || wall_at_offset(board, row, col, &[(0, 1), (-1, 0)], WallOrientation::Horizontal)
                || wall_at_offset(board, row, col, &[(-1, 0), (-1, 0)], WallOrientation::Vertical)
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
}
