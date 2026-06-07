//! Checkpoint 02 — legal move generation.

pub mod board;
pub mod grid;
pub mod moves;
pub mod path;

pub use board::{Board, Column, Move, Player, Row, WallOrientation};
pub use moves::generate_legal_moves;
pub use path::{both_players_reach_goals, can_reach_goal, shortest_distance};
