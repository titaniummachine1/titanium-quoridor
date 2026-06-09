//! Titanium Engine — Quoridor search core.
//!
//! Fundamentals: Zobrist hash, make/unmake, TT, iterative deepening perft.
//! Layout: `SharedState` (TT) + `WorkerContext` (per-thread scratch) — Lazy SMP ready.
//! Next: αβ search on the same `Engine` entry point.

pub mod board;
pub mod context;
pub mod engine;
pub mod genmove;
pub mod greedy;
pub mod grid;
pub mod mcts;
pub mod moves;
pub mod opening;
pub mod path;
pub mod perft;
pub mod search;
pub mod tt;
pub mod zobrist;

#[cfg(test)]
mod test_replay;

pub use board::{Board, Column, Move, Player, Row, Undo, WallOrientation};
pub use context::{EngineLimits, SharedState, ThreadBenchResult, WorkerContext};
pub use engine::Engine;
pub use genmove::{
    genmove_algebraic, GenmoveConfig, GenmoveEngine, SearchPhase, MCTS_DEFAULT_MAX_SIMULATIONS,
    MCTS_DEFAULT_UCT,
};
pub use opening::{ply_number, BOOK_MAX_PLY};
pub use greedy::choose_greedy_move;
pub use mcts::{search_mcts, MctsConfig, MctsReport};
pub use moves::{
    generate_legal_moves, generate_legal_moves_into, generate_legal_moves_slice, MAX_LEGAL_MOVES,
};
pub use path::{
    both_players_reach_goals, can_reach_goal, shortest_distance, BfsScratch, CorridorAttention,
};
pub use perft::{
    format_move, perft, perft_divide, perft_fast, perft_fast_ctx, perft_iterative, perft_naive,
    perft_parallel_root, PerftContext, PERFT3_STARTPOS, PERFT4_STARTPOS,
};
pub use search::{
    search_best_move, SearchConfig, SearchReport, DEFAULT_MAX_NODES, DEFAULT_TIME_MS,
};
pub use tt::TranspositionTable;
