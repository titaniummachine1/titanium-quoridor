//! ACE v11 hybrid port — quoridor_5.html "pathfix gen11_ghi" search
//! (_vendor/acev11_engine.js) plus optional Titanium movegen.
//!
//! v11 over the v10 base (same HalfPW net, byte-identical NET_DATA):
//! ZeroFence-A GHI guard on the TT, RaceProof exact race endgames (root
//! solve, eval verdicts, last-wall commitment gate), and the unsound
//! `wallCanBlockTopology` fast path removed from `wall_legal`. ThreatPrice
//! and WallSense ship disabled in the JS (falsifier/SPRT-killed) and are
//! not ported.
//!
//! Self-contained: own board representation, search, and HalfPW net eval.
//! Only this module's `genmove` entry translates between Titanium algebraic
//! notation and ACE move encoding.
//!
//! Coordinate mapping (ACE row 0 = top, Titanium row 0 = bottom):
//!   pawn  m = (8 - row) * 9 + col
//!   wall  m = base + (7 - row) * 8 + col   (base 100 = h, 200 = v)

pub mod game;
pub mod net;
pub mod perft;
pub mod race;
pub mod search;
pub mod session;

pub use game::AceGame;
pub use perft::{
    default_timeout, oracle_nodes, perft_ace_ti_timed, perft_ace_timed, perft_engine_timed,
    perft_titanium_timed, TimedPerftResult, ACE_PERFT4_STARTPOS,
};
pub use search::{board_move_to_ace, AceSearch, ThinkResult};
pub use session::run_ace_session_stdio;

use crate::core::board::{Move as BoardMove, WallOrientation};

/// ACE move encoding → Titanium board move (row flip between coordinate systems).
pub fn ace_move_to_board(m: i16) -> BoardMove {
    if m < 100 {
        BoardMove::Pawn {
            row: 8 - (m / 9) as u8,
            col: (m % 9) as u8,
        }
    } else {
        let (base, orientation) = if m < 200 {
            (100, WallOrientation::Horizontal)
        } else {
            (200, WallOrientation::Vertical)
        };
        let slot = m - base;
        BoardMove::Wall {
            row: 7 - (slot / 8) as u8,
            col: (slot % 8) as u8,
            orientation,
        }
    }
}

/// Algebraic ("e2", "e3h") → ACE move encoding.
pub fn algebraic_to_ace(text: &str) -> i16 {
    let b = text.as_bytes();
    let col = (b[0] - b'a') as i16;
    let row = (b[1] - b'1') as i16;
    if b.len() > 2 {
        let slot = (7 - row) * 8 + col;
        match b[2] {
            b'h' => 100 + slot,
            b'v' => 200 + slot,
            _ => panic!("bad wall suffix in {text}"),
        }
    } else {
        (8 - row) * 9 + col
    }
}

/// ACE move encoding → algebraic.
pub fn ace_to_algebraic(m: i16) -> String {
    if m < 100 {
        let r = m / 9;
        let c = m % 9;
        format!("{}{}", (b'a' + c as u8) as char, 9 - r)
    } else {
        let (base, suffix) = if m < 200 { (100, 'h') } else { (200, 'v') };
        let slot = m - base;
        let r = slot / 8;
        let c = slot % 8;
        format!("{}{}{}", (b'a' + c as u8) as char, 8 - r, suffix)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct AceParams {
    pub time_ms: u64,
    pub max_depth: i32,
    /// Disable the easy-move early stop (search the full time budget).
    pub full: bool,
    /// Hybrid: CAT-filter wall moves at inner nodes.
    pub cat: bool,
    /// Titanium `movegen` on mirrored board (fast full-legal generation).
    pub ti_movegen: bool,
    /// Stream iterative-deepening progress on stderr (`info json`).
    pub log: bool,
    /// Early Move Extensions on ordered wall moves (mirror of graduated LMR).
    pub eme: bool,
}

impl Default for AceParams {
    fn default() -> Self {
        Self {
            time_ms: 4000,
            max_depth: 30,
            full: false,
            cat: false,
            ti_movegen: false,
            log: false,
            eme: false,
        }
    }
}

/// CLI entry — plays `moves` (algebraic) from startpos, thinks, returns best move.
pub fn ace_genmove(
    moves: &[String],
    params: AceParams,
    engine_label: &str,
) -> Option<(String, ThinkResult)> {
    let mut g = AceGame::new();
    for text in moves {
        g.make_move(algebraic_to_ace(text));
    }
    if g.winner() >= 0 {
        return None;
    }
    let mut search = if params.ti_movegen && params.cat {
        AceSearch::with_ti_movegen_and_cat(g)
    } else if params.ti_movegen {
        AceSearch::with_ti_movegen(g)
    } else if params.cat {
        AceSearch::with_cat(g)
    } else {
        AceSearch::new(g)
    };
    if params.eme {
        search.enable_eme();
    }
    let result = search.think(
        params.time_ms,
        params.max_depth,
        params.full,
        params.log,
        engine_label,
    );
    if result.mv == 0 && search.g.winner() >= 0 {
        return None;
    }
    Some((ace_to_algebraic(result.mv), result))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn move_translation_round_trips() {
        // pawn: e1 = our (0,4) = ACE cell 76
        assert_eq!(algebraic_to_ace("e1"), 76);
        assert_eq!(ace_to_algebraic(76), "e1");
        // pawn: e9 = our (8,4) = ACE cell 4
        assert_eq!(algebraic_to_ace("e9"), 4);
        assert_eq!(ace_to_algebraic(4), "e9");
        // wall: d8v = our wall (7,3) = ACE vw slot 3
        assert_eq!(algebraic_to_ace("d8v"), 203);
        assert_eq!(ace_to_algebraic(203), "d8v");
        // wall: a1h = our wall (0,0) = ACE hw slot 56
        assert_eq!(algebraic_to_ace("a1h"), 156);
        assert_eq!(ace_to_algebraic(156), "a1h");
    }

    #[test]
    fn startpos_has_pawn_and_wall_moves() {
        let mut g = AceGame::new();
        let mut buf = [0i16; 160];
        let n = g.gen_pawn_moves(&mut buf, 0);
        assert_eq!(n, 3);
        let mut walls = 0;
        for slot in 0..64 {
            if g.wall_legal(0, slot) {
                walls += 1;
            }
            if g.wall_legal(1, slot) {
                walls += 1;
            }
        }
        assert_eq!(walls, 128);
    }

    #[test]
    fn h6h_legal_after_a2h_line() {
        use crate::core::board::Board;
        use crate::movegen::generate_legal_moves;
        use crate::util::perft::format_move;

        let moves = [
            "e2", "e8", "e3", "e7", "e4", "e6", "d3h", "d6h", "f3h", "f6h", "b3h", "b6h", "h3h",
            "d4v", "a2h",
        ];
        let mut g = AceGame::new();
        let mut board = Board::new();
        for m in moves {
            g.make_move(algebraic_to_ace(m));
            board.apply_algebraic(m);
        }
        let slot = (algebraic_to_ace("h6h") - 100) as usize;
        assert!(
            g.wall_legal(0, slot),
            "ACE must accept h6h (off-topology fast path)"
        );
        let ti_legal: Vec<_> = generate_legal_moves(&board)
            .iter()
            .map(|mv| format_move(*mv))
            .collect();
        assert!(
            ti_legal.iter().any(|m| m == "h6h"),
            "Titanium oracle must accept h6h after onB edge fix"
        );
    }

    #[test]
    fn a6h_path_parity_after_h3v_line() {
        use crate::core::board::Board;
        use crate::core::board::WallOrientation;
        use crate::movegen::generate_legal_moves;
        use crate::movegen::legal::can_wall_block_topology;
        use crate::util::perft::format_move;

        let moves = [
            "e2", "e8", "e3", "e7", "e4", "e6", "e3h", "e6h", "c3h", "c6h", "g3h", "g6h", "a3h",
            "e4v", "h3v",
        ];
        let mut g = AceGame::new();
        let mut board = Board::new();
        for m in moves {
            g.make_move(algebraic_to_ace(m));
            board.apply_algebraic(m);
        }
        let slot = (algebraic_to_ace("a6h") - 100) as usize;
        let row = 7 - (slot / 8) as u8;
        let col = (slot % 8) as u8;
        let ti_legal: Vec<_> = generate_legal_moves(&board)
            .iter()
            .map(|mv| format_move(*mv))
            .collect();
        let ace_ok = g.wall_legal(0, slot);
        let can_block = can_wall_block_topology(&board, row, col, WallOrientation::Horizontal);
        // a6h keeps both goal paths open here (naive BFS confirms); the old
        // rejection was V10's partial-component false negative. ACE and the
        // Titanium oracle must agree on acceptance.
        assert!(can_block, "a6h touches topology — path flood must run");
        assert!(ace_ok, "ACE must accept a6h when both goal paths survive");
        assert!(
            ti_legal.iter().any(|m| m == "a6h"),
            "Titanium oracle must accept a6h on h3v line"
        );
    }
}
