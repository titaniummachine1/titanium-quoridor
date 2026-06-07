//! Titanium Engine — Quoridor search core.
//!
//! Checkpoint stack:
//! 1. `grid` + `path` — bitboard walls + stack BFS reachability
//! 2. `moves` — legal pawn/wall generation (JS-oracle parity)
//! 3. `perft` — divide harness for correctness + benches
//! 4. Search (αβ + TT) and guided MCTS — upcoming

pub mod board;
pub mod grid;
pub mod moves;
pub mod path;
pub mod perft;

pub use board::{Board, Column, Move, Player, Row, WallOrientation};
pub use moves::generate_legal_moves;
pub use path::{both_players_reach_goals, can_reach_goal, shortest_distance};
pub use perft::{format_move, perft, perft_divide};
