//! Persistent search state across plies — TT, killers, history, PV (Stockfish-style).

use crate::core::board::{Board, Move};
use crate::movegen::generate_legal_moves;
use crate::search::alphabeta::{run_search, SearchConfig, SearchReport};
use crate::search::move_pack::{pack_move, unpack_move};
use crate::search::search_tt::SearchTt;

pub const MAX_KILLER_PLY: usize = 256;
pub const HISTORY_SIZE: usize = 8192;

/// Cross-ply memory for iterative search — one instance per engine seat in the web UI.
pub struct GameSearchSession {
    pub board: Board,
    pub tt: SearchTt,
    /// Killer moves per ply (packed); preserved between searches in the same game.
    pub killers: [[u32; 2]; MAX_KILLER_PLY],
    /// Butterfly history — packed-move index.
    pub history: [i16; HISTORY_SIZE],
    /// Previous search best move — root move-ordering hint on the next ply.
    pub pv_move: Option<Move>,
    /// Last verified root score — soft aspiration anchor when still plausible.
    pub prev_score: Option<i32>,
}

impl Default for GameSearchSession {
    fn default() -> Self {
        Self::new()
    }
}

impl GameSearchSession {
    pub fn new() -> Self {
        Self {
            board: Board::new(),
            tt: SearchTt::new(),
            killers: [[0; 2]; MAX_KILLER_PLY],
            history: [0; HISTORY_SIZE],
            pv_move: None,
            prev_score: None,
        }
    }

    pub fn reset(&mut self) {
        self.board = Board::new();
        self.tt.clear();
        self.killers = [[0; 2]; MAX_KILLER_PLY];
        self.history = [0; HISTORY_SIZE];
        self.pv_move = None;
        self.prev_score = None;
    }

    pub fn set_position(&mut self, moves: &[String]) {
        self.board = Board::new();
        for mv in moves {
            if mv.is_empty() {
                continue;
            }
            if !self.apply_algebraic(mv) {
                break;
            }
        }
    }

    pub fn apply_algebraic(&mut self, algebraic: &str) -> bool {
        if algebraic.is_empty() {
            return true;
        }
        if self.board.is_terminal().is_some() {
            return false;
        }
        let legal = generate_legal_moves(&self.board);
        let Some(mv) = legal
            .iter()
            .find(|m| crate::util::perft::format_move(**m).eq_ignore_ascii_case(algebraic))
        else {
            return false;
        };
        let _ = self.board.make_move(*mv);
        true
    }

    pub fn killers_at(&self, ply: usize) -> [Option<Move>; 2] {
        let slot = self.killers.get(ply).copied().unwrap_or([0, 0]);
        [unpack_move(slot[0]), unpack_move(slot[1])]
    }

    pub fn record_killer(&mut self, ply: usize, mv: Move) {
        if ply >= MAX_KILLER_PLY {
            return;
        }
        let packed = pack_move(mv);
        if packed == 0 {
            return;
        }
        let slot = &mut self.killers[ply];
        if slot[0] == packed {
            return;
        }
        slot[1] = slot[0];
        slot[0] = packed;
    }

    pub fn bump_history(&mut self, mv: Move, depth: u32) {
        let packed = pack_move(mv);
        if packed == 0 {
            return;
        }
        let idx = packed as usize % HISTORY_SIZE;
        let bonus = (depth * depth).min(i16::MAX as u32) as i16;
        self.history[idx] = self.history[idx].saturating_add(bonus);
    }

    pub fn history_bonus(&self, mv: Move) -> i32 {
        let packed = pack_move(mv);
        if packed == 0 {
            return 0;
        }
        i32::from(self.history[packed as usize % HISTORY_SIZE])
    }

    /// Full-strength search; `run_search` updates `pv_move` / `prev_score`.
    pub fn search(&mut self, config: SearchConfig) -> Option<SearchReport> {
        run_search(self, config)
    }
}
