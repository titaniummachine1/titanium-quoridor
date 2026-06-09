//! Perft oracle positions from Canta (15 random games × 15 plies).
//! Turn byte: `place_walls:1 | orientation:1 | index:6`
//! Wall index = `row * 8 + col` (gorisanson / Titanium internal wall grid).

use crate::board::{Board, Move, Player, WallOrientation};
use crate::perft::perft_fast;
use crate::path::BfsScratch;

const TURN_BYTES: [[u8; 15]; 15] = [
    [0x8d, 0x9d, 0xb, 0x5, 0x5f, 0xd5, 0x19, 0x81, 0x33, 0xc3, 0x27, 0x9b, 0x49, 0xcd, 0xbd],
    [0xc5, 0x87, 0xbb, 0x23, 0xf9, 0x95, 0x69, 0xf, 0x6f, 0x61, 0x7b, 0x59, 0xd9, 0xd3, 0x35],
    [0x2d, 0x77, 0x85, 0x69, 0x1, 0x43, 0xd1, 0x9f, 0x91, 0xdd, 0x45, 0x1d, 0xc1, 0xf5, 0x3d],
    [0xfb, 0xa1, 0x67, 0xfd, 0x3f, 0x73, 0x45, 0xbb, 0xc5, 0xb7, 0x39, 0xcd, 0x75, 0x3, 0x69],
    [0x3d, 0x7d, 0x17, 0x7, 0x73, 0xc1, 0x3b, 0x97, 0x99, 0x67, 0xf3, 0xcd, 0x5f, 0x6d, 0xfb],
    [0x95, 0x45, 0x81, 0xd9, 0x8d, 0xb9, 0x13, 0x2b, 0x9f, 0xa1, 0xa9, 0x7, 0x3d, 0x3c, 0x19],
    [0x9f, 0xcb, 0x29, 0xfd, 0xe7, 0xdd, 0x55, 0x7d, 0xd5, 0x5, 0x99, 0x87, 0x3, 0xc5, 0x61],
    [0xc, 0xaf, 0xa5, 0xe9, 0xc9, 0xdf, 0x2f, 0x55, 0x7, 0xd1, 0x1, 0x2c, 0x7f, 0x8d, 0x5d],
    [0x33, 0xcb, 0xcd, 0xfb, 0x23, 0x83, 0xd, 0x89, 0x51, 0x2c, 0x95, 0x65, 0x35, 0x71, 0xef],
    [0x8b, 0xbd, 0xf9, 0x3, 0xc7, 0x9f, 0x35, 0x77, 0x87, 0x2b, 0x83, 0x55, 0xbb, 0x8d, 0x49],
    [0xe1, 0xcd, 0x9f, 0x3, 0x2d, 0x77, 0x73, 0x5d, 0xdb, 0x41, 0x5, 0x1b, 0x9b, 0xd, 0xf5],
    [0x6f, 0x9f, 0xcb, 0x9, 0xf1, 0x21, 0x27, 0xcf, 0x1d, 0x11, 0x53, 0x7b, 0x85, 0xd1, 0x77],
    [0x3b, 0x1f, 0xef, 0x87, 0xb9, 0x1, 0x4d, 0xd1, 0x75, 0xf5, 0xfd, 0xdf, 0x8d, 0x3d, 0x57],
    [0x29, 0x41, 0xfb, 0x23, 0xd1, 0x13, 0x17, 0x4f, 0x7d, 0x81, 0x5d, 0xf7, 0x3c, 0x1d, 0x6d],
    [0x2c, 0x33, 0x83, 0xb1, 0x49, 0x5f, 0x7d, 0x51, 0xd3, 0xa7, 0x25, 0xa1, 0x69, 0xab, 0x93],
];

const PERFT_VALUES: [[u64; 3]; 15] = [
    [79, 5978, 432338],
    [78, 5745, 409363],
    [77, 5697, 404581],
    [82, 6451, 486291],
    [80, 6229, 460083],
    [82, 6365, 478510],
    [82, 6454, 487137],
    [87, 7272, 583286],
    [80, 6064, 445703],
    [79, 6005, 438600],
    [77, 5612, 396652],
    [74, 5259, 358646],
    [76, 5612, 391949],
    [85, 6903, 535126],
    [84, 6794, 528318],
];

fn canta_pawn_dest(index: u8, col: u8, row: u8) -> (u8, u8) {
    let x = col;
    let y = row;
    match index {
        0 => (x, y.saturating_add(2)),
        1 => (x.saturating_add(1), y.saturating_add(1)),
        2 => (x.saturating_sub(1), y.saturating_add(1)),
        3 => (x, y.saturating_add(1)),
        4 => (x, y.saturating_sub(2)),
        5 => (x.saturating_add(1), y.saturating_sub(1)),
        6 => (x.saturating_sub(1), y.saturating_sub(1)),
        7 => (x, y.saturating_sub(1)),
        8 => (x.saturating_add(2), y),
        9 => (x.saturating_add(1), y.saturating_add(1)),
        10 => (x.saturating_add(1), y.saturating_sub(1)),
        11 => (x.saturating_add(1), y),
        12 => (x.saturating_sub(2), y),
        13 => (x.saturating_sub(1), y.saturating_add(1)),
        14 => (x.saturating_sub(1), y.saturating_sub(1)),
        _ => (x.saturating_sub(1), y),
    }
}

fn turn_to_move(board: &Board, byte: u8) -> Move {
    let place_walls = byte & 1 != 0;
    let orientation = (byte >> 1) & 1 != 0;
    let index = (byte >> 2) & 0x3f;

    if place_walls {
        let row = index / 8;
        let col = index % 8;
        let orient = if orientation {
            WallOrientation::Vertical
        } else {
            WallOrientation::Horizontal
        };
        Move::Wall {
            row,
            col,
            orientation: orient,
        }
    } else {
        let (row, col) = board.pawn(board.side());
        let (nc, nr) = canta_pawn_dest(index, col, row);
        Move::Pawn { row: nr, col: nc }
    }
}

/// Replay `game_idx` (0..14) opening — 15 plies from Canta's corpus.
pub fn board_after_canta_game(game_idx: usize) -> Board {
    assert!(game_idx < 15);
    let mut board = Board::new();
    for &byte in &TURN_BYTES[game_idx] {
        let mv = turn_to_move(&board, byte);
        let _ = board.make_move(mv);
    }
    board
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::moves::generate_legal_moves;

    #[test]
    fn canta_oracle_all_games_depth1_to_3() {
        for game in 0..15 {
            let mut board = board_after_canta_game(game);
            for depth in 1..=3u32 {
                let expected = PERFT_VALUES[game][depth as usize - 1];
                let got = perft_fast(&mut board, depth);
                assert_eq!(
                    got, expected,
                    "game {game} depth {depth}: got {got}, expected {expected}"
                );
            }
            let legal = generate_legal_moves(&board).len() as u64;
            assert_eq!(legal, PERFT_VALUES[game][0], "game {game} depth1 legal");
        }
    }

    #[test]
    fn canta_game0_matches_manual_replay() {
        let board = board_after_canta_game(0);
        let mut scratch = BfsScratch::new();
        assert!(scratch.both_players_reach_goals(&board));
    }
}
