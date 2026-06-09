//! Test specific replay for illegal moves

use crate::board::{Board, Move, WallOrientation};
use crate::moves::generate_legal_moves_slice;
use crate::path::BfsScratch;
use crate::perft::format_move;

fn parse_algebraic(move_str: &str) -> Option<Move> {
    let chars: Vec<char> = move_str.chars().collect();
    if chars.len() < 2 {
        return None;
    }

    let col = (chars[0] as u8).saturating_sub(b'a') as u8;
    if col > 8 {
        return None;
    }

    let row = (chars[1] as u8).saturating_sub(b'1') as u8;
    if row > 8 {
        return None;
    }

    if chars.len() >= 3 {
        let orientation = match chars.get(2) {
            Some(&'h') => WallOrientation::Horizontal,
            Some(&'v') => WallOrientation::Vertical,
            _ => return None,
        };
        Some(Move::Wall {
            row,
            col,
            orientation,
        })
    } else {
        Some(Move::Pawn { row, col })
    }
}

#[test]
fn g1v_correctly_rejected_after_replay_prefix() {
    // Move 24 from an external replay — g1v blocks a goal path under correct rules.
    let prefix = [
        "e2", "e8", "e3", "e7", "e4", "e6", "d1h", "d6h", "f4", "f6h", "f5", "e5", "d5", "c5v",
        "d4", "c3v", "d3", "e4", "c1v", "f4", "f1h", "h1v", "h2h", "g3v",
    ];
    let mut board = Board::new();
    let mut bfs = BfsScratch::new();
    let mut buf = [Move::Pawn { row: 0, col: 0 }; crate::moves::MAX_LEGAL_MOVES];

    for move_str in prefix {
        let mv = parse_algebraic(move_str).unwrap();
        let n = generate_legal_moves_slice(&mut board, &mut buf, &mut bfs);
        assert!(buf[..n].contains(&mv), "{move_str} must be legal in prefix");
        let _ = board.make_move(mv);
    }

    let g1v = parse_algebraic("g1v").unwrap();
    let n = generate_legal_moves_slice(&mut board, &mut buf, &mut bfs);
    assert!(
        !buf[..n].contains(&g1v),
        "g1v must be rejected — it blocks a goal path"
    );
}

#[test]
#[ignore = "external replay used pre-boundary-fix rules at move 24 (g1v)"]
fn test_replay_legality() {
    let moves = [
        "e2", "e8", "e3", "e7", "e4", "e6", "d1h", "d6h", "f4", "f6h", "f5", "e5", "d5", "c5v",
        "d4", "c3v", "d3", "e4", "c1v", "f4", "f1h", "h1v", "h2h", "g3v", "g1v", "h8h", "c7v",
        "d4h", "e3", "g4", "f3", "e6v", "f4", "f8h", "g5", "g6", "h6", "g5", "g5h", "h5", "h5v",
        "g5", "h7", "f5", "h8", "f6", "g8", "g6", "g6v", "f6", "f8", "g6", "e8", "f6", "e9",
    ];

    let mut board = Board::new();
    let mut bfs = BfsScratch::new();
    let mut buf = [Move::Pawn { row: 0, col: 0 }; crate::moves::MAX_LEGAL_MOVES];

    for (i, move_str) in moves.iter().enumerate() {
        let legal_count = generate_legal_moves_slice(&mut board, &mut buf, &mut bfs);

        // Parse the move
        let mv =
            parse_algebraic(move_str).expect(&format!("Failed to parse move {}: {}", i, move_str));

        // Check if move is in legal moves
        let is_legal = buf[..legal_count].contains(&mv);

        if !is_legal {
            panic!(
                "Move {} ({}) is not legal! Legal moves: {:?}",
                i,
                move_str,
                buf[..legal_count]
                    .iter()
                    .map(|m| format_move(*m))
                    .collect::<Vec<_>>()
            );
        }

        // Make the move
        let _undo = board.make_move(mv);

        // Verify both players can still reach goals after each move
        if !bfs.both_players_reach_goals(&board) {
            panic!(
                "After move {} ({}), one player cannot reach their goal!",
                i, move_str
            );
        }
    }

    println!("All {} moves in replay are legal", moves.len());
}
