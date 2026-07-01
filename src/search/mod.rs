//! Search — αβ negamax, TT, pipeline, genmove entry.

pub mod alphabeta;
pub mod cat_index_lmr;
pub mod context;
pub mod deprecated;
pub mod genmove;
pub mod greedy;
pub mod lmr_profile;
pub mod lmr_viz;
pub mod move_pack;
pub mod pipeline;
pub mod rollout;
pub mod root_cap;
pub mod runtime;
pub mod search_tt;
pub mod session;
pub mod session_stdio;
pub mod tt;
pub mod uci;
pub mod v16_lmr;

pub use alphabeta::{
    run_search, search_best_move, SearchConfig, SearchReport, DEFAULT_MAX_ID_DEPTH,
    DEFAULT_MAX_NODES, DEFAULT_TIME_MS,
};
pub use context::{EngineLimits, SharedState, ThreadBenchResult, WorkerContext};
#[allow(deprecated)]
pub use deprecated::mcts::{search_mcts, MctsConfig, MctsReport};
pub use genmove::{
    genmove_algebraic, GenmoveConfig, GenmoveEngine, MCTS_DEFAULT_MAX_SIMULATIONS, MCTS_DEFAULT_UCT,
};
pub use pipeline::{lmr_stage_inputs, search_phase, walls_placed, LmrStageInputs, SearchPhase};
pub use runtime::Engine;
pub use session::GameSearchSession;
pub use tt::TranspositionTable;
