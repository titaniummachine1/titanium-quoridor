//! Titanium Engine — Quoridor search core.
//!
//! ```text
//! core/     board, zobrist
//! util/     grid, perft
//! movegen/  legal moves only
//! path/     BFS reachability
//! cat/      Corridor Attention Table v3 + pruning + viz
//! eval/     static evaluation (see search::alphabeta)
//! search/   αβ negamax, TT, pipeline, genmove
//! opening/  book
//! ```

pub mod ace;
pub mod bench_instr;
pub mod cat;
pub mod core;
pub mod eval;
pub mod friend_perft;
pub mod movegen;
pub mod opening;
pub mod oracle;
pub mod path;
pub mod search;
pub mod titanium;
pub mod util;

#[cfg(feature = "wasm")]
pub mod wasm;

#[cfg(all(feature = "wasm-threads", target_arch = "wasm32"))]
pub use wasm_bindgen_rayon::init_thread_pool;

#[cfg(test)]
mod test_replay;

// ── Public API (stable re-exports) ───────────────────────────────────────────

pub use cat::{
    cat_snapshot_json, collect_search_moves, move_corridor_attention, wall_net_race,
    wall_should_search, CorridorAttention, CAT_COLD_CM, CAT_HOT_CM,
};
pub use core::board::{Board, Column, Move, Player, Row, Undo, WallOrientation};
pub use movegen::{
    generate_legal_moves, generate_legal_moves_into, generate_legal_moves_slice,
    generate_legal_moves_slice_mode, PawnGenMode, MAX_LEGAL_MOVES,
};
pub use opening::{ply_number, BOOK_MAX_PLY};
pub use path::{both_players_reach_goals, can_reach_goal, shortest_distance, BfsScratch};
pub use search::greedy::choose_greedy_move;
#[allow(deprecated)]
pub use search::lmr_viz::lmr_snapshot_json;
pub use search::session_stdio::run_session_stdio;
pub use search::uci::run_uci_stdio;
pub use search::{
    genmove_algebraic, run_search, search_best_move, search_mcts, search_phase, walls_placed,
    Engine, EngineLimits, GameSearchSession, GenmoveConfig, GenmoveEngine, MctsConfig, MctsReport,
    SearchConfig, SearchPhase, SearchReport, SharedState, ThreadBenchResult, TranspositionTable,
    WorkerContext, DEFAULT_MAX_NODES, DEFAULT_TIME_MS, MCTS_DEFAULT_MAX_SIMULATIONS,
    MCTS_DEFAULT_UCT,
};
#[cfg(feature = "parallel")]
pub use util::perft::perft_parallel_root;
pub use util::perft::{
    format_move, perft, perft_divide, perft_fast, perft_fast_ctx, perft_fast_mode,
    perft_fast_mode_ctx, perft_iterative, perft_naive, perft_no_tt_mode, perft_pawn_only_mode,
    PerftContext, PERFT3_STARTPOS, PERFT4_STARTPOS,
};

// Titanium v15 production API (formerly `acev13` module path).
pub use titanium::fields_viz;
#[cfg(not(target_arch = "wasm32"))]
pub use titanium::opening_book;
#[cfg(not(target_arch = "wasm32"))]
pub use titanium::reduction_shadow_probe;
pub use titanium::{
    algebraic_to_move_id, board_move_to_move_id, decode_packed_state, move_id_to_algebraic,
    move_id_to_board, pack_state, reduction_counterfactual_probe, run_titanium_session_stdio,
    titanium_game_from_packed, titanium_genmove, GameState, TitaniumParams, TitaniumSearch,
    FEATURE_SCHEMA, PACKED_STATE_LEN, POSITION_SCHEMA_VERSION, TITANIUM_NO_MOVE,
};
