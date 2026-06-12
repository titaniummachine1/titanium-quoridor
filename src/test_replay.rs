//! Test specific replay for illegal moves

use crate::core::board::{Board, Move, WallOrientation};
use crate::movegen::generate_legal_moves_slice;
use crate::path::BfsScratch;
use crate::util::perft::format_move;

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
fn d3h_legal_off_topology_matches_js() {
    // Scraped Gorisanson rules: `canWallBlock` false → no path trial → wall is legal
    // even if it looks strategically like a "cage" (see scraped/game_logic_extract.js).
    let prefix = ["e2", "e8", "e3", "e7", "e4", "e6", "e5", "e4"];
    let mut board = Board::new();
    let mut bfs = BfsScratch::new();
    let mut buf = [Move::Pawn { row: 0, col: 0 }; crate::movegen::MAX_LEGAL_MOVES];

    for move_str in prefix {
        let mv = parse_algebraic(move_str).unwrap();
        let _ = board.make_move(mv);
    }

    let d3h = parse_algebraic("d3h").unwrap();
    let n = generate_legal_moves_slice(&mut board, &mut buf, &mut bfs);
    assert!(
        buf[..n].contains(&d3h),
        "d3h must be legal when off topology — matches JS canWallBlock shortcut"
    );
}

#[test]
fn user_replay_a5h_ply14_js_mismatch() {
    let prefix = [
        "e2", "e8", "e3", "e7", "e4", "e6", "e5", "e4", "e3h", "e5h", "c3h", "c5h", "g3h",
    ];
    let mut board = Board::new();
    let mut bfs = BfsScratch::new();
    let mut buf = [Move::Pawn { row: 0, col: 0 }; crate::movegen::MAX_LEGAL_MOVES];

    for move_str in prefix {
        let mv = parse_algebraic(move_str).unwrap();
        let n = generate_legal_moves_slice(&mut board, &mut buf, &mut bfs);
        assert!(buf[..n].contains(&mv), "{move_str} must be legal in prefix");
        let _ = board.make_move(mv);
    }

    let a5h = parse_algebraic("a5h").unwrap();
    let n = generate_legal_moves_slice(&mut board, &mut buf, &mut bfs);
    let legal_a5h = buf[..n].contains(&a5h);

    use crate::path::flood::{flood_to_goal, goal_square_mask};
    use crate::path::masks::DirMasks;
    use crate::util::grid::{flood_bit_sq, square_index};
    use crate::core::board::Player;

    let masks = DirMasks::from_board(&board);
    let (w1, sc1) = board.pawn(Player::One);
    let (w2, sc2) = board.pawn(Player::Two);
    let start1 = square_index(w1, sc1);
    let start2 = square_index(w2, sc2);
    let (ok1, comp1) = flood_to_goal(start1, masks, goal_square_mask(Player::One));
    let in_comp = comp1 & flood_bit_sq(start2) != 0;
    let goal2_in = comp1 & goal_square_mask(Player::Two) != 0;
    eprintln!(
        "ply13 a5h legal={legal_a5h} ok1={ok1} black_in_white_comp={in_comp} goal2_in_comp={goal2_in} both={}",
        bfs.both_players_reach_goals(&board)
    );

    let mut trial = board.clone();
    let _ = trial.make_move(a5h);
    let masks2 = DirMasks::from_board(&trial);
    let (ok1b, comp1b) = flood_to_goal(start1, masks2, goal_square_mask(Player::One));
    let in_comp_b = comp1b & flood_bit_sq(start2) != 0;
    let goal2_in_b = comp1b & goal_square_mask(Player::Two) != 0;
    eprintln!(
        "after a5h ok1={ok1b} black_in_white_comp={in_comp_b} goal2_in_comp={goal2_in_b} both={}",
        bfs.both_players_reach_goals(&trial)
    );

    assert!(
        !legal_a5h,
        "a5h must be rejected at ply 14 — blocks a goal path (JS incorrectly allows this)"
    );
}

#[test]
fn a5h_correctly_rejected_after_tq1_ply19() {
    let prefix = [
        "e2", "e8", "e3", "e7", "e4", "e6", "c3h", "e7h", "e3h", "c7h", "f4", "g7h", "f5", "h8h",
        "f6", "b6v", "g3h", "h7v", "a3h",
    ];
    let mut board = Board::new();
    let mut bfs = BfsScratch::new();
    let mut buf = [Move::Pawn { row: 0, col: 0 }; crate::movegen::MAX_LEGAL_MOVES];

    for move_str in prefix {
        let mv = parse_algebraic(move_str).unwrap();
        let n = generate_legal_moves_slice(&mut board, &mut buf, &mut bfs);
        assert!(buf[..n].contains(&mv), "{move_str} must be legal in prefix");
        let _ = board.make_move(mv);
    }

    let a5h = parse_algebraic("a5h").unwrap();
    let n = generate_legal_moves_slice(&mut board, &mut buf, &mut bfs);
    assert!(
        !buf[..n].contains(&a5h),
        "a5h must be rejected — it blocks White's goal path"
    );
}

#[test]
fn a1h_correctly_rejected_ply22_wall_maze() {
    let prefix = [
        "e2", "e8", "e3", "e7", "e4", "e6", "e3h", "e4h", "d4", "c4h", "e5v", "a5h", "h8h", "d6",
        "b5v", "f3v", "e7v", "c3h", "d7h", "b2v", "h6h",
    ];
    let mut board = Board::new();
    let mut bfs = BfsScratch::new();
    let mut buf = [Move::Pawn { row: 0, col: 0 }; crate::movegen::MAX_LEGAL_MOVES];

    for move_str in prefix {
        let mv = parse_algebraic(move_str).unwrap();
        let n = generate_legal_moves_slice(&mut board, &mut buf, &mut bfs);
        assert!(buf[..n].contains(&mv), "{move_str} must be legal in prefix");
        let _ = board.make_move(mv);
    }

    let a1h = parse_algebraic("a1h").unwrap();
    let n = generate_legal_moves_slice(&mut board, &mut buf, &mut bfs);
    assert!(
        !buf[..n].contains(&a1h),
        "a1h cages White — must be illegal (DirMasks cache bug allowed this)"
    );
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
    let mut buf = [Move::Pawn { row: 0, col: 0 }; crate::movegen::MAX_LEGAL_MOVES];

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
    let mut buf = [Move::Pawn { row: 0, col: 0 }; crate::movegen::MAX_LEGAL_MOVES];

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
