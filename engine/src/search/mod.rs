//! Search — αβ negamax, TT, pipeline, genmove entry.

pub mod alphabeta;
pub mod context;
pub mod move_pack;
pub mod search_tt;
pub mod session;
pub mod session_stdio;
pub mod deprecated;
pub mod genmove;
pub mod greedy;
pub mod lmr_profile;
pub mod lmr_viz;
pub mod pipeline;
pub mod root_cap;
pub mod runtime;
pub mod tt;

pub use alphabeta::{
    run_search, search_best_move, SearchConfig, SearchReport, DEFAULT_MAX_ID_DEPTH,
    DEFAULT_MAX_NODES, DEFAULT_TIME_MS,
};
pub use session::GameSearchSession;
pub use context::{EngineLimits, SharedState, ThreadBenchResult, WorkerContext};
pub use genmove::{
    genmove_algebraic, GenmoveConfig, GenmoveEngine, MCTS_DEFAULT_MAX_SIMULATIONS, MCTS_DEFAULT_UCT,
};
#[allow(deprecated)]
pub use deprecated::mcts::{search_mcts, MctsConfig, MctsReport};
pub use pipeline::{lmr_stage_inputs, search_phase, walls_placed, LmrStageInputs, SearchPhase};
pub use runtime::Engine;
pub use tt::TranspositionTable;
