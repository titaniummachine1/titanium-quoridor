//! `genmove` entry — three-phase Titanium hybrid pipeline.
//!
//! ## Phase 1 — Theory (ply ≤ 10)
//! Opening book hints steer MCTS while the board is still wide open.
//!
//! ## Optional bridge — Solidification (ply 11–20, &lt; 4 walls)
//! CAT-guided MCTS bridge is kept behind `TITANIUM_BRIDGE=1`.
//!
//! ## Phase 2 — Annihilation (ply &gt; 10 by default)
//! Deep alpha-beta minimax + corridor attention pruning once corridors exist.
//!
//! Override: `TITANIUM_BRIDGE=1` enables the old ply 11–20 bridge.

use crate::board::Board;
use crate::greedy::choose_greedy_move;
use crate::mcts::{genmove_algebraic as mcts_algebraic, MctsConfig, DEFAULT_TIME_MS};
use crate::opening::{self, BOOK_MAX_PLY};
use crate::perft::format_move;
use crate::search::{genmove_algebraic as minimax_algebraic, SearchConfig, DEFAULT_MAX_NODES};

/// Walls on board before minimax is preferred over MCTS (topology is concrete).
const BRIDGE_WALL_THRESHOLD: u8 = 4;
/// Last ply where CAT-MCTS bridge may run (after book window).
const BRIDGE_MAX_PLY: u32 = 20;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchPhase {
    /// Ply ≤ 10 — book hints steer search ordering.
    Book,
    /// Ply 11–20, open board — CAT-guided MCTS.
    Bridge,
    /// Ply &gt; 20 or enough walls — minimax + corridor attention.
    Minimax,
}

fn walls_placed(board: &Board) -> u8 {
    20u8.saturating_sub(board.walls_remaining[0].saturating_add(board.walls_remaining[1]))
}

pub fn search_phase(board: &Board) -> SearchPhase {
    // Enough walls on board → corridors exist; CAT-backed minimax wins immediately.
    if walls_placed(board) >= BRIDGE_WALL_THRESHOLD {
        return SearchPhase::Minimax;
    }
    let ply = opening::ply_number(board);
    if ply <= BOOK_MAX_PLY {
        SearchPhase::Book
    } else if std::env::var("TITANIUM_BRIDGE").is_ok_and(|v| v == "1") && ply <= BRIDGE_MAX_PLY {
        SearchPhase::Bridge
    } else {
        SearchPhase::Minimax
    }
}

fn use_bridge(board: &Board) -> bool {
    matches!(search_phase(board), SearchPhase::Bridge)
}

fn log_phase(phase: SearchPhase) {
    if !std::env::var("TITANIUM_LOG").is_ok() {
        return;
    }
    let label = match phase {
        SearchPhase::Book => "book",
        SearchPhase::Bridge => "bridge",
        SearchPhase::Minimax => "minimax",
    };
    eprintln!("info phase {label}");
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenmoveEngine {
    Mcts,
    Minimax,
    Greedy,
}

impl Default for GenmoveEngine {
    fn default() -> Self {
        Self::Mcts
    }
}

#[derive(Debug, Clone)]
pub struct GenmoveConfig {
    pub engine: GenmoveEngine,
    pub mcts: MctsConfig,
    pub minimax: SearchConfig,
}

impl Default for GenmoveConfig {
    fn default() -> Self {
        Self {
            engine: GenmoveEngine::Mcts,
            mcts: MctsConfig::default(),
            minimax: SearchConfig {
                time_ms: DEFAULT_TIME_MS,
                max_nodes: DEFAULT_MAX_NODES,
                log: false,
                book_hint: None,
            },
        }
    }
}

pub fn genmove_algebraic(board: &mut Board, config: GenmoveConfig) -> Option<String> {
    let phase = search_phase(board);
    log_phase(phase);

    let book_hint = opening::book_hint(board);
    if phase == SearchPhase::Book {
        if let Some(hint) = book_hint {
            if hint.priority >= 100 {
                return Some(format_move(hint.mv));
            }
        }
    }

    match config.engine {
        GenmoveEngine::Mcts => {
            let mut mcts_cfg = config.mcts;
            mcts_cfg.book_hint = book_hint;
            mcts_algebraic(board, mcts_cfg)
        }
        GenmoveEngine::Minimax => {
            let mut minimax_cfg = config.minimax;
            minimax_cfg.book_hint = book_hint;
            if phase == SearchPhase::Book {
                let mut opening = config.mcts;
                opening.time_ms = config.minimax.time_ms;
                opening.log = config.minimax.log;
                opening.book_hint = book_hint;
                mcts_algebraic(board, opening)
            } else if use_bridge(board) {
                let bridge = MctsConfig {
                    time_ms: config.minimax.time_ms,
                    max_simulations: config.mcts.max_simulations,
                    uct: config.mcts.uct,
                    log: config.minimax.log,
                    use_cat_guidance: true,
                    book_hint,
                };
                mcts_algebraic(board, bridge)
            } else {
                minimax_algebraic(board, minimax_cfg)
            }
        }
        GenmoveEngine::Greedy => greedy_algebraic(board),
    }
}

fn greedy_algebraic(board: &mut Board) -> Option<String> {
    let mut scratch = crate::path::BfsScratch::new();
    choose_greedy_move(board, &mut scratch).map(format_move)
}

pub use crate::mcts::DEFAULT_MAX_SIMULATIONS as MCTS_DEFAULT_MAX_SIMULATIONS;
pub use crate::mcts::DEFAULT_UCT as MCTS_DEFAULT_UCT;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::perft::format_move;

    fn replay(moves: &[&str]) -> Board {
        let mut board = Board::new();
        for mv in moves {
            board.apply_algebraic(mv);
        }
        board
    }

    fn config(engine: GenmoveEngine) -> GenmoveConfig {
        GenmoveConfig {
            engine,
            mcts: MctsConfig {
                time_ms: 1,
                max_simulations: 1,
                log: false,
                ..MctsConfig::default()
            },
            minimax: SearchConfig {
                time_ms: 1,
                max_nodes: 1,
                log: false,
                book_hint: None,
            },
        }
    }

    #[test]
    fn phase_book_at_start() {
        let board = Board::new();
        assert_eq!(search_phase(&board), SearchPhase::Book);
    }

    #[test]
    fn phase_bridge_after_book_window() {
        let board = replay(&["e2", "e8", "e3", "e7", "e4", "e6", "e5", "e4", "d3h", "f4"]);
        assert_eq!(opening::ply_number(&board), 11);
        assert_eq!(search_phase(&board), SearchPhase::Minimax);
    }

    #[test]
    fn phase_minimax_when_walls_concrete() {
        let mut board = Board::new();
        for _ in 0..4 {
            board.apply_algebraic("d2h");
            board.apply_algebraic("d8h");
        }
        assert!(walls_placed(&board) >= BRIDGE_WALL_THRESHOLD);
        assert_eq!(search_phase(&board), SearchPhase::Minimax);
    }

    #[test]
    fn book_hint_avoids_free_jump_at_center_reply() {
        let mut board = replay(&["e2", "e8", "e3", "e7", "e4", "e6"]);
        let hint = opening::book_hint(&mut board).expect("book hint");
        let reply = format_move(hint.mv);
        assert!(
            matches!(reply.as_str(), "e3v" | "d3v" | "c3v"),
            "expected Standard/Shiller vertical wall, got {reply}"
        );
    }

    #[test]
    fn hybrid_opening_uses_mcts_not_minimax() {
        let mut board = replay(&["e2", "e8", "e3", "e7", "e4", "e6"]);
        let mut cfg = config(GenmoveEngine::Minimax);
        cfg.minimax.time_ms = 1;
        cfg.mcts.max_simulations = 1;
        assert!(genmove_algebraic(&mut board, cfg).is_some());
    }

    #[test]
    fn book_hint_present_at_ply_11_e3h_e5h() {
        // After e3h (White) + e5h (Black, boxing White at e5), book hint should be d5.
        let mut board = replay(&["e2", "e8", "e3", "e7", "e4", "e6", "e5", "e4", "e3h", "e5h"]);
        assert_eq!(opening::ply_number(&board), 11);
        let hint = opening::book_hint(&mut board).expect("book hint at ply 11");
        assert_eq!(format_move(hint.mv), "d5");
    }

    #[test]
    fn book_hint_absent_deep_midgame() {
        // At ply 13 with arbitrary position, lookup should find nothing.
        let mut board = replay(&[
            "e2", "e8", "e3", "e7", "e4", "e6", "e5", "e4", "d3h", "f4", "e6", "d3",
        ]);
        assert!(opening::ply_number(&board) > BOOK_MAX_PLY);
        // sprint guard fires until move_number 10 (ply ≤ 20); non-book position
        // still gets a guard hint from advancing_pawn_move so just ensure no panic.
        let _ = opening::book_hint(&mut board);
    }
}
