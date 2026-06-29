//! ACE v11 search — 1:1 port of the JS `Search` object (quoridor_5.html,
//! pathfix gen11_ghi).
//!
//! Iterative-deepening αβ with aspiration windows, typed TT, killers/history/
//! countermoves, null move, graduated LMR / EME, frontier LMP, reverse futility,
//! lazy wall legality, repetition detection, wall-stamp dist caching,
//! easy-move early stop, HalfPW net eval. Mirrors the JS node-for-node.
//!
//! gen11 additions over the v10 base:
//! - ZeroFence-A GHI guard (PLAIN variant, `ghiAnchor` shipped false): TT
//!   entries whose subtree leaned on a path-dependent repetition-zero are
//!   stored flag-demoted or tainted; tainted entries never give score cutoffs.
//! - RaceProof (`raceProof = true`, SPRT-passed): exact race-endgame tables
//!   when both hands are empty (eval verdicts, root solve, last-wall
//!   commitment gate with the budget reserve).
//! - ThreatPrice / WallSense ship FALSE in the JS (falsifier/SPRT-killed) and
//!   no-op cleanly when false — their machinery is intentionally NOT ported.
//! - RaceProof(c) certificates (`certify_win.js`) are node-only; the browser
//!   build runs with `RP_CERT === null`, which this port mirrors (the
//!   commitment gate keeps the wall when no certifier exists).

use crate::titanium::dist::{
    fill_ace_dist_from_pawn, fill_ace_dist_to_goal_with_masks, fill_choke_points, fill_contested,
    fill_corridor_delta, fill_distance_layers, fill_path_crossing, fill_sparse_route_masks,
    shortest_route_bits,
};
use crate::titanium::move_id_to_board;
use crate::util::clock::{Duration, Instant};

use crate::cat::prune::{
    cat_heat_refs_from_scores, cat_v16_lmr_fringe_pct_for_worker, cat_v16_lmr_reduction_plies,
    gap_play_zone_mask, get_shortest_path, move_corridor_attention_with_denial,
    move_corridor_attention_with_path, move_impact_heat, wall_in_dead_zone, wall_should_search,
    CatHeatRefs,
};
use crate::cat::CorridorAttention;
use crate::core::board::{Board, Move as BoardMove, Player, Undo, WallOrientation};
use crate::movegen::{
    generate_legal_moves_slice_cached, geometric_wall_len_cached, GeometricWallCache,
    GeometricWallCacheRole, GeometricWallCacheStats, MAX_LEGAL_MOVES,
};
use crate::path::flood::expand_frontier;
use crate::path::masks::DirMasks;
use crate::path::BfsScratch;
use crate::titanium::certify::{certify, CertifyOpts};
use crate::titanium::game::{GameState, ZOBRIST};
use crate::titanium::net::{net, net_frozen, Net, NET_BKT, NET_H, NET_MIRC, NET_MIRS};
use crate::titanium::packed_state::FEATURE_SCHEMA;
use crate::titanium::race::{
    race_outcome, solve_race_config, RaceBound, RaceScratch, RACE_MATE, RACE_STATES,
};
use crate::titanium::reduction_sidecar::ReductionSidecar;
use crate::util::grid::{flood_bit_sq, flood_sq_from_bit, FLOOD_PLAYABLE};
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
use std::sync::{
    atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    Arc, RwLock,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm-threads"))]
use std::sync::Mutex;
pub const MATE: i32 = 100_000;
pub const MAX_PLY: usize = 64;
const INF: i32 = 2 * MATE;
/// Proven-outcome band for stubborn-loser tie breaks (matches `search::alphabeta`).
const CERT_WIN_SCORE: i32 = 15_000;
const CERT_BAND: i32 = 4_000;

/// Graduated LMR starts after this move index (JS acev10: `i >= 4`).
const ACE_LMR_AFTER_MOVE: usize = 4;
/// Both LMR and EME require at least this remaining depth.
const ACE_LMR_MIN_DEPTH: i32 = 3;

/// Late-move reduction plies — same formula as JS graduated LMR.
fn ace_graduated_lmr_reduction(move_index: usize, depth: i32) -> i32 {
    let mut red = 1;
    if move_index >= 12 {
        red += 1;
    }
    if depth >= 6 && move_index >= 24 {
        red += 1;
    }
    red
}

/// EME extends only the first ordered wall moves after the TT/best move.
/// Index 0 (TT move) already gets full depth; extending more siblings
/// compounds multiplicatively down the tree and explodes the node count.
const ACE_EME_TOP_MOVES: usize = 2;

/// Early Move Extension — +1 ply for the top ordered walls; +2 only for
/// the very first non-TT wall when there is real depth left to spend.
fn ace_graduated_eme_extension(move_index: usize, depth: i32) -> i32 {
    if move_index == 1 && depth >= 8 {
        2
    } else {
        1
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod lazy_smp_tests {
    use super::*;

    fn fresh() -> Box<TitaniumSearch> {
        TitaniumSearch::grafted(GameState::new(), Some(18))
    }

    #[test]
    fn root_width_calculation_uses_ceiling_and_min_one() {
        let cases = [
            (30, 100, 30),
            (30, 80, 24),
            (30, 60, 18),
            (30, 40, 12),
            (30, 20, 6),
            (1, 20, 1),
            (2, 20, 1),
            (3, 20, 1),
            (4, 20, 1),
            (5, 20, 1),
            (6, 20, 2),
            (0, 100, 0),
        ];
        for (root_count, percent, expected) in cases {
            assert_eq!(lazy_smp_allowed_root_moves(root_count, percent), expected);
        }
    }

    #[test]
    fn root_filtering_limits_each_worker_to_its_width() {
        let mut search = fresh();
        let result = search.think_with_threads(1_000, 1, true, false, "titanium-v15", 5);
        assert_eq!(result.root_widths.len(), 5);
        for plan in &result.root_widths {
            let visits = &result.root_visits[plan.worker_id];
            let allowed = plan.allowed_root_moves();
            assert!(
                visits.iter().all(|&idx| idx < plan.root_move_count),
                "worker {} visited outside root list of {}: {:?}",
                plan.worker_id,
                plan.root_move_count,
                visits
            );
            let unique = visits
                .iter()
                .copied()
                .collect::<std::collections::HashSet<_>>();
            assert!(unique.len() <= allowed);
            if plan.worker_id == 0 {
                assert_eq!(allowed, plan.root_move_count);
            }
        }
    }

    #[test]
    fn helper_root_profiles_are_diversified() {
        let root_moves = (0..20).collect::<Vec<i16>>();
        let (main_moves, main_idx) =
            TitaniumSearch::lazy_smp_profile_root_moves(&root_moves, 0, 20);
        let (helper_moves, helper_idx) =
            TitaniumSearch::lazy_smp_profile_root_moves(&root_moves, 1, 12);
        assert_eq!(main_moves, root_moves);
        assert_eq!(main_idx, (0..20).collect::<Vec<_>>());
        assert_eq!(helper_moves.len(), 12);
        assert_eq!(helper_idx.len(), 12);
        assert_ne!(helper_idx, (0..12).collect::<Vec<_>>());
    }

    #[test]
    fn cat_v16_worker_profiles_raise_fringe_threshold() {
        let mut search = *TitaniumSearch::grafted_v16(GameState::new(), Some(18));
        search.set_cat_lmr_worker_profile(0);
        assert_eq!(search.cat_lmr_fringe_pct, 5);
        search.set_cat_lmr_worker_profile(1);
        assert_eq!(search.cat_lmr_fringe_pct, 10);
        search.set_cat_lmr_worker_profile(2);
        assert_eq!(search.cat_lmr_fringe_pct, 20);
        search.set_cat_lmr_worker_profile(4);
        assert_eq!(search.cat_lmr_fringe_pct, 40);
        search.set_cat_lmr_worker_profile(8);
        assert_eq!(search.cat_lmr_fringe_pct, 70);
    }

    #[test]
    fn shared_tt_allocation_and_probe_are_shared() {
        let mut search = fresh();
        search.resize_tt(18);
        let shared = Arc::new(SharedTitaniumTt::from_search(&search));
        let runtime = Arc::new(LazySmpRuntime::new(
            Instant::now() + Duration::from_millis(100),
        ));
        let root_moves = Arc::new(vec![0i16]);
        let root_visit_map = Arc::new(vec![0usize]);
        let mut worker = search.fork_lazy_worker(&GameState::new());
        search.install_lazy_smp_context(
            0,
            shared.clone(),
            runtime.clone(),
            root_moves.clone(),
            root_visit_map.clone(),
            1,
        );
        worker.install_lazy_smp_context(1, shared.clone(), runtime, root_moves, root_visit_map, 1);
        assert!(Arc::ptr_eq(
            search.shared_tt.as_ref().expect("main shared TT"),
            worker.shared_tt.as_ref().expect("helper shared TT")
        ));

        shared.store(
            123,
            456,
            7,
            false,
            SharedTtEntry {
                key_hi: 456,
                key_lo: 123,
                meta: 42 | (0 << 10) | (5 << 12),
                score: 99,
                rep: 0,
                anc_lo: 0,
                anc_hi: 0,
                entry_gen: 7,
            },
        );
        let entry = shared.probe(123, 456).expect("stored helper entry");
        assert_eq!(entry.score, 99);
        assert_eq!(entry.meta >> 12, 5);
    }

    #[test]
    fn shared_stop_flag_is_observed() {
        let mut search = fresh();
        let runtime = Arc::new(LazySmpRuntime::new(Instant::now() + Duration::from_secs(1)));
        runtime.stop.store(true, Ordering::Relaxed);
        search.lazy_runtime = Some(runtime);
        assert!(search.check_time().is_err());
    }

    #[test]
    fn helper_depth_does_not_replace_main_authority() {
        let mut search = fresh();
        let result = search.think_with_threads(1_000, 2, true, false, "titanium-v15", 4);
        assert_eq!(result.depth, result.main_completed_depth);
        assert_eq!(
            result.main_thread_nodes + result.helper_nodes.iter().sum::<u64>(),
            result.total_nodes
        );
        assert_eq!(result.nodes, result.total_nodes);
    }

    #[test]
    fn helper_partial_is_used_only_when_main_has_no_completed_move() {
        fn result(mv: i16, depth: i32, nodes: u64) -> ThinkResult {
            ThinkResult {
                mv,
                score: depth * 10,
                depth,
                nodes,
                main_thread_nodes: 0,
                helper_nodes: Vec::new(),
                total_nodes: 0,
                main_completed_depth: 0,
                helper_completed_depths: Vec::new(),
                root_widths: Vec::new(),
                root_visits: Vec::new(),
                ms: 0,
                white_dist: 0,
                black_dist: 0,
                depth_log: Vec::new(),
                stop_reason: "test",
            }
        }

        let legal_roots = [11, 22, 33];
        let helpers = vec![
            (1, result(99, 8, 99), Vec::new()),
            (2, result(22, 3, 300), Vec::new()),
            (3, result(33, 4, 200), Vec::new()),
        ];
        let main_ready = result(11, 1, 10);
        assert!(
            TitaniumSearch::lazy_smp_helper_partial(&main_ready, &helpers, &legal_roots).is_none()
        );

        let main_empty = result(crate::titanium::TITANIUM_NO_MOVE, 0, 0);
        let adopted = TitaniumSearch::lazy_smp_helper_partial(&main_empty, &helpers, &legal_roots)
            .expect("legal helper result");
        assert_eq!(adopted.mv, 33);
        assert_eq!(adopted.depth, 4);
    }

    #[test]
    fn one_thread_matches_existing_search_at_fixed_depth() {
        let mut old = fresh();
        let mut new = fresh();
        let a = old.think(10_000, 2, true, false, "titanium-v15");
        let b = new.think_with_threads(10_000, 2, true, false, "titanium-v15", 1);
        assert_eq!(a.mv, b.mv);
        assert_eq!(a.score, b.score);
        assert_eq!(a.depth, b.depth);
        assert_eq!(a.nodes, b.nodes);
        assert_eq!(a.depth_log.len(), b.depth_log.len());
    }

    #[test]
    fn race_stress_no_illegal_moves_or_hangs() {
        for _ in 0..8 {
            let mut search = fresh();
            let result = search.think_with_threads(250, 3, true, false, "titanium-v15", 4);
            let mut legal = [0i16; 160];
            let n = search.gen_moves(0, 1, result.mv, &mut legal);
            assert!(n > 0);
            assert!(legal[..n].contains(&result.mv));
        }
    }
}

/// Count moves in `walls` that cross `route` (a flood u128 bitset, bit = flood_bit_sq(sq)).
/// A wall crosses a route when both cells across one of its blocked edges are on the route.
fn wall_crossing_count(walls: &[BoardMove], route: u128) -> u32 {
    crate::bench_instr::record(
        |b| &mut b.wall_crossing_count,
        || wall_crossing_count_inner(walls, route),
    )
}

fn wall_crossing_count_inner(walls: &[BoardMove], route: u128) -> u32 {
    let mut n = 0u32;
    for mv in walls {
        let BoardMove::Wall {
            row,
            col,
            orientation,
        } = *mv
        else {
            continue;
        };
        let r = row as usize;
        let c = col as usize;
        let bit = |r: usize, c: usize| (route & flood_bit_sq((r * 9 + c) as u8)) != 0;
        let crosses = match orientation {
            WallOrientation::Horizontal => {
                (bit(r, c) && bit(r + 1, c)) || (bit(r, c + 1) && bit(r + 1, c + 1))
            }
            WallOrientation::Vertical => {
                (bit(r, c) && bit(r, c + 1)) || (bit(r + 1, c) && bit(r + 1, c + 1))
            }
        };
        if crosses {
            n += 1;
        }
    }
    n
}

/// Same as `wall_crossing_count` but takes a `[u8;81]` route array (route[r*9+c] != 0).
fn wall_crossing_count_arr(walls: &[BoardMove], route: &[u8; 81]) -> u32 {
    let mut n = 0u32;
    for mv in walls {
        let BoardMove::Wall {
            row,
            col,
            orientation,
        } = *mv
        else {
            continue;
        };
        let r = row as usize;
        let c = col as usize;
        let cell = |r: usize, c: usize| route[r * 9 + c] != 0;
        let crosses = match orientation {
            WallOrientation::Horizontal => {
                (cell(r, c) && cell(r + 1, c)) || (cell(r, c + 1) && cell(r + 1, c + 1))
            }
            WallOrientation::Vertical => {
                (cell(r, c) && cell(r, c + 1)) || (cell(r + 1, c) && cell(r + 1, c + 1))
            }
        };
        if crosses {
            n += 1;
        }
    }
    n
}

const TT_BITS: usize = 20;
const TT_SIZE: usize = 1 << TT_BITS;
const TT_MASK: u32 = (TT_SIZE - 1) as u32;

const LAZY_SMP_WIDTHS: [usize; 5] = [100, 80, 60, 40, 20];

#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
pub const LAZY_SMP_MAX_THREADS: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkerPlan {
    pub worker_id: usize,
    pub root_move_count: usize,
    pub root_width_percent: usize,
}

impl WorkerPlan {
    pub fn allowed_root_moves(&self) -> usize {
        lazy_smp_allowed_root_moves(self.root_move_count, self.root_width_percent)
    }
}

pub fn lazy_smp_allowed_root_moves(root_move_count: usize, root_width_percent: usize) -> usize {
    if root_move_count == 0 {
        return 0;
    }
    let allowed = root_move_count
        .saturating_mul(root_width_percent)
        .saturating_add(99)
        / 100;
    allowed.max(1).min(root_move_count)
}

#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
#[derive(Debug, Clone, Copy, Default)]
struct SharedTtEntry {
    key_hi: u32,
    key_lo: u32,
    meta: i32,
    score: i32,
    rep: u8,
    anc_lo: u32,
    anc_hi: u32,
    entry_gen: u8,
}

#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
struct SharedTitaniumTt {
    slots: Vec<RwLock<SharedTtEntry>>,
    mask: u32,
    bits: usize,
    filled: AtomicUsize,
}

#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
impl SharedTitaniumTt {
    fn from_search(search: &TitaniumSearch) -> Self {
        let slots = (0..search.tt_meta.len())
            .map(|i| {
                RwLock::new(SharedTtEntry {
                    key_hi: search.tt_key_hi[i],
                    key_lo: search.tt_key_lo[i],
                    meta: search.tt_meta[i],
                    score: search.tt_score[i],
                    rep: search.tt_rep[i],
                    anc_lo: search.tt_anc_lo[i],
                    anc_hi: search.tt_anc_hi[i],
                    entry_gen: search.tt_entry_gen[i],
                })
            })
            .collect();
        Self {
            slots,
            mask: search.tt_mask,
            bits: search.tt_bits,
            filled: AtomicUsize::new(search.tt_filled),
        }
    }

    fn probe(&self, hash_lo: u32, hash_hi: u32) -> Option<SharedTtEntry> {
        let idx = (hash_lo & self.mask) as usize;
        let entry = *self.slots[idx]
            .read()
            .expect("shared TT read lock poisoned");
        if entry.meta != 0 && entry.key_hi == hash_hi && entry.key_lo == hash_lo {
            Some(entry)
        } else {
            None
        }
    }

    fn store(&self, hash_lo: u32, hash_hi: u32, tt_gen: u8, pure_mode: bool, entry: SharedTtEntry) {
        let idx = (hash_lo & self.mask) as usize;
        let mut slot = self.slots[idx]
            .write()
            .expect("shared TT write lock poisoned");
        let was_empty = slot.meta == 0;
        let stale_gen = !pure_mode && !was_empty && slot.entry_gen != tt_gen;
        let deeper = !was_empty && !stale_gen && (entry.meta >> 12) >= (slot.meta >> 12);
        if was_empty || stale_gen || deeper {
            *slot = entry;
            if was_empty {
                self.filled.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
struct LazySmpRuntime {
    stop: AtomicBool,
    global_nodes: AtomicU64,
    deadline: Instant,
}

#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
impl LazySmpRuntime {
    fn new(deadline: Instant) -> Self {
        Self {
            stop: AtomicBool::new(false),
            global_nodes: AtomicU64::new(0),
            deadline,
        }
    }
}

/// Time-abort marker — propagates like the JS `throw "time"`.
pub struct TimeUp;

/// Titanium `Board` kept in sync with the ACE game — fast movegen + optional CAT.
pub struct TiBridge {
    pub board: Board,
    pub bfs: BfsScratch,
    undo_stack: Vec<Undo>,
    geometric_walls: Option<GeometricWallCache>,
    pub wall_cache_stats: GeometricWallCacheStats,
}

impl TiBridge {
    fn from_game(g: &GameState) -> Box<Self> {
        let mut board = Board::new();
        for i in 0..g.hist_len {
            let _ = board.make_move(move_id_to_board(g.hist_m[i]));
        }
        Box::new(Self {
            board,
            bfs: BfsScratch::new(),
            undo_stack: Vec::with_capacity(256),
            geometric_walls: None,
            wall_cache_stats: GeometricWallCacheStats::default(),
        })
    }

    fn push(&mut self, m: i16) {
        let undo = self.board.make_move(move_id_to_board(m));
        self.undo_stack.push(undo);
    }

    fn pop(&mut self) {
        if let Some(undo) = self.undo_stack.pop() {
            self.board.unmake_move(undo);
        }
    }

    /// Full legal moves via Titanium `movegen` → ACE encoding.
    fn gen_legal_ace(&mut self, out: &mut [i16; 160]) -> usize {
        let mut ti_buf = [BoardMove::Pawn { row: 0, col: 0 }; MAX_LEGAL_MOVES];
        let n = generate_legal_moves_slice_cached(
            &mut self.geometric_walls,
            &mut self.board,
            &mut ti_buf,
            &mut self.bfs,
            Some(&mut self.wall_cache_stats),
        );
        for i in 0..n {
            out[i] = board_move_to_move_id(ti_buf[i]);
        }
        n
    }
}

/// Titanium board move → ACE numeric encoding.
pub fn board_move_to_move_id(mv: BoardMove) -> i16 {
    match mv {
        BoardMove::Pawn { row, col } => ((8 - row as i16) * 9 + col as i16) as i16,
        BoardMove::Wall {
            row,
            col,
            orientation,
        } => {
            let slot = (7 - row as i16) * 8 + col as i16;
            match orientation {
                WallOrientation::Horizontal => 100 + slot,
                WallOrientation::Vertical => 200 + slot,
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct AceDepthLogEntry {
    pub depth: i32,
    pub score: i32,
    pub nodes: u64,
    pub elapsed_ms: u64,
    pub marginal_nodes: u64,
    pub pv: String,
}

#[derive(Clone)]
pub struct ThinkResult {
    pub mv: i16,
    pub score: i32,
    pub depth: i32,
    pub nodes: u64,
    pub main_thread_nodes: u64,
    pub helper_nodes: Vec<u64>,
    pub total_nodes: u64,
    pub main_completed_depth: i32,
    pub helper_completed_depths: Vec<i32>,
    pub root_widths: Vec<WorkerPlan>,
    pub root_visits: Vec<Vec<usize>>,
    pub ms: u64,
    pub white_dist: u8,
    pub black_dist: u8,
    pub depth_log: Vec<AceDepthLogEntry>,
    pub stop_reason: &'static str,
}

/// One complete late-move pipeline observation. These records are emitted only
/// by the offline counterfactual collector; production search leaves probing off.
#[derive(Debug, Clone)]
pub struct ReductionProbeEvent {
    pub ordinal: u64,
    pub parent_hash_lo: u32,
    pub parent_hash_hi: u32,
    pub child_hash_lo: u32,
    pub child_hash_hi: u32,
    pub mv: i16,
    pub depth: i32,
    pub ply: usize,
    pub alpha: i32,
    pub beta: i32,
    pub move_index: usize,
    pub base_reduction: i32,
    pub applied_extra_reduction: bool,
    pub verification_triggered: bool,
    pub score: i32,
    pub nodes: u64,
    pub hidden: [f64; NET_H],
    /// Total legal moves generated at this node (enables rank_percentile computation).
    pub total_legal_moves: usize,
    /// Raw history-table score for this wall move (proxy for ordering confidence).
    pub history_score: i32,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ReductionShadowStats {
    pub evaluations: u64,
    pub hypothetical_activations: u64,
    pub inference_nanos: u64,
}

pub fn score_label(score: i32) -> String {
    let abs = score.abs();
    if abs >= MATE - 1_000 {
        let plies = MATE - abs;
        if score > 0 {
            format!("mate in {}", plies.max(0))
        } else {
            format!("mated in {}", plies.max(0))
        }
    } else if abs >= RACE_MATE - 1_000 && abs <= RACE_MATE {
        let plies = RACE_MATE - abs;
        if score > 0 {
            format!("race win in {}", plies.max(0))
        } else {
            format!("race loss in {}", plies.max(0))
        }
    } else {
        format!("cp {score}")
    }
}

#[cfg(test)]
mod score_label_tests {
    use super::*;

    #[test]
    fn labels_race_scores_as_forced_races() {
        assert_eq!(score_label(RACE_MATE - 30), "race win in 30");
        assert_eq!(score_label(-(RACE_MATE - 17)), "race loss in 17");
    }

    #[test]
    fn labels_true_mate_scores_separately_from_races() {
        assert_eq!(score_label(MATE - 5), "mate in 5");
        assert_eq!(score_label(-(MATE - 9)), "mated in 9");
        assert_eq!(score_label(42), "cp 42");
    }
}

pub fn think_result_progress_json(engine_label: &str, result: &ThinkResult) -> String {
    ace_progress_json(
        engine_label,
        &result.depth_log,
        result.depth,
        result.nodes,
        result.main_thread_nodes,
        &result.helper_nodes,
        result.total_nodes,
        result.score,
        result.white_dist,
        result.black_dist,
        result.ms,
    )
}

fn ace_progress_json(
    engine_label: &str,
    depth_log: &[AceDepthLogEntry],
    search_depth: i32,
    nodes: u64,
    main_thread_nodes: u64,
    helper_nodes: &[u64],
    total_nodes: u64,
    root_score: i32,
    white_dist: u8,
    black_dist: u8,
    elapsed_ms: u64,
) -> String {
    let mut depth_json = String::new();
    for (i, e) in depth_log.iter().enumerate() {
        if i > 0 {
            depth_json.push(',');
        }
        const ESC_DQ: &str = "\\\"";
        let pv = e.pv.replace('\\', "\\\\").replace('"', ESC_DQ);
        let score_text = score_label(e.score);
        depth_json.push_str(&format!(
            r#"{{"depth":{},"score":{},"scoreText":"{}","nodes":{},"elapsedMs":{},"marginalNodes":{},"pv":"{}"}}"#,
            e.depth, e.score, score_text, e.nodes, e.elapsed_ms, e.marginal_nodes, pv
        ));
    }
    let mut helper_json = String::new();
    for (i, nodes) in helper_nodes.iter().enumerate() {
        if i > 0 {
            helper_json.push(',');
        }
        helper_json.push_str(&nodes.to_string());
    }
    let root_score_text = score_label(root_score);
    format!(
        r#"{{"engine":"{engine_label}","stoppedBy":"{engine_label}","searchDepth":{search_depth},"nodes":{nodes},"mainThreadNodes":{main_thread_nodes},"helperNodes":[{helper_json}],"totalNodes":{total_nodes},"totalNodesAcrossWorkers":{total_nodes},"rootScore":{root_score},"rootScoreText":"{root_score_text}","whiteDist":{white_dist},"blackDist":{black_dist},"elapsedMs":{elapsed_ms},"depthLog":[{depth_json}]}}"#
    )
}

#[cfg(feature = "wasm")]
fn emit_ace_progress_wasm(json: &str, wasm_cb: Option<&js_sys::Function>) {
    if let Some(f) = wasm_cb {
        let _ = f.call1(
            &wasm_bindgen::JsValue::NULL,
            &wasm_bindgen::JsValue::from_str(json),
        );
    }
}

fn emit_ace_progress(
    engine_label: &str,
    depth_log: &[AceDepthLogEntry],
    search_depth: i32,
    nodes: u64,
    root_score: i32,
    white_dist: u8,
    black_dist: u8,
    elapsed_ms: u64,
    #[cfg(feature = "wasm")] wasm_cb: Option<&js_sys::Function>,
) {
    let json = ace_progress_json(
        engine_label,
        depth_log,
        search_depth,
        nodes,
        nodes,
        &[],
        nodes,
        root_score,
        white_dist,
        black_dist,
        elapsed_ms,
    );
    #[cfg(feature = "wasm")]
    emit_ace_progress_wasm(&json, wasm_cb);
    #[cfg(not(feature = "wasm"))]
    {
        eprintln!("info json {json}");
        let _ = std::io::Write::flush(&mut std::io::stderr());
    }
}

/// RaceProof race-table LRU slots (keyed by wall-config zobrist).
const RC_SLOTS: usize = 64;

pub struct TitaniumSearch {
    pub g: GameState,
    tt_key_hi: Vec<u32>,
    tt_key_lo: Vec<u32>,
    tt_meta: Vec<i32>, // move | flag<<10 | depth<<12, 0 = empty
    tt_score: Vec<i32>,
    // ZeroFence-A: 1 = tainted-zero entry (move-only, never a score cutoff)
    tt_rep: Vec<u8>,
    tt_anc_lo: Vec<u32>,
    tt_anc_hi: Vec<u32>,
    /// Generation counter — wraps; incremented every think(). Stored per TT slot.
    /// Depth-preferred replacement: within the same generation only deeper entries
    /// overwrite; entries from a prior generation are always replaced.
    tt_gen: u8,
    tt_entry_gen: Vec<u8>,
    /// Index mask for the TT vecs (`size - 1`). Runtime so the TT can be resized
    /// (Titanium-style larger table) without recompiling — `1<<TT_BITS` default.
    tt_mask: u32,
    /// Current TT index bits (`tt_mask == (1<<tt_bits)-1`).
    tt_bits: usize,
    /// Occupied slots (meta != 0). Drives overflow-triggered growth.
    tt_filled: usize,
    /// Overflow-driven cache-tier growth targets (Titanium strategy): start in L1,
    /// jump L1→L2→L3→d4(18)→d5(22) on overflow, then +1 per overflow past d5. Each
    /// jump lands on a calibrated size that won't immediately re-overflow. Inactive
    /// unless [`enable_adaptive_tt`](TitaniumSearch::enable_adaptive_tt) is called.
    tt_l2: usize,
    tt_l3: usize,
    tt_d4: usize,
    tt_d5: usize,
    tt_max: usize,
    tt_adaptive: bool,
    // per-ply open-subtree dependency window: min external path-rep target ply
    sub_min: [i32; MAX_PLY],
    sub_anc_lo: [u32; MAX_PLY],
    sub_anc_hi: [u32; MAX_PLY],
    history_tbl: [i32; 512],
    cm: [i16; 512], // countermove table
    killers: [[i16; 2]; MAX_PLY],
    // Offline-only LMR counterfactual probe. A target ordinal receives exactly
    // one provisional extra reduction; verification always uses native depth.
    reduction_probe_enabled: bool,
    reduction_probe_target: Option<u64>,
    reduction_probe_next: u64,
    reduction_probe_limit: usize,
    reduction_probe_min_depth: i32,
    reduction_probe_events: Vec<ReductionProbeEvent>,
    reduction_sidecar: Option<ReductionSidecar>,
    reduction_shadow_stats: ReductionShadowStats,
    path_lo: [u32; MAX_PLY],
    path_hi: [u32; MAX_PLY],
    d0: [[u8; 81]; MAX_PLY],
    d1: [[u8; 81]; MAX_PLY],
    d0_layers: [[u128; 81]; MAX_PLY],
    d1_layers: [[u128; 81]; MAX_PLY],
    dist0_idx: usize, // active ply slot in d0 (JS: this.dist0 array ref)
    dist1_idx: usize,
    cached_stamp: i32,
    // HalfPW accumulator cache
    np_acc0: [f64; NET_H],
    np_acc1: [f64; NET_H],
    np_hw: [u8; 64],
    np_vw: [u8; 64],
    np_b0: i32,
    np_b1v: i32,
    net: &'static Net,
    /// Mirrored Titanium board (movegen and/or CAT).
    bridge: Option<Box<TiBridge>>,
    /// Use Titanium `generate_legal_moves_slice` instead of ACE `wall_legal`.
    ti_movegen: bool,
    /// CAT-filter walls at inner nodes (requires `bridge`).
    cat_walls: bool,
    /// Titanium v16: CAT-scaled LMR with ceiling normalization (500/800/1000 cm).
    cat_lmr_v16: bool,
    cat_lmr_ceiling: u16,
    cat_lmr_fringe_pct: u16,
    /// SOUND dead-zone wall prune at inner nodes (requires `bridge`): drop only
    /// walls in an unreachable void / sealed interior — provably irrelevant (they
    /// change no path and only burn inventory, never the best move). NPS-only;
    /// cannot cost Elo. Distinct from `cat_walls` (heat filter, which can).
    dead_zone_prune: bool,
    /// Grafted-engine flag: in the hands-empty endgame, use Titanium's cheap
    /// path-aware tempo classifier ([`cert_bridge::hands_empty_race`]) instead of
    /// the full recursive `certify`. Same result, a fraction of the nodes — frees
    /// NPS for the rest of the search. Off = faithful gen13 (always `certify`).
    cheap_cert: bool,
    /// When true, recursive certify + k=0 race oracle run only at quiescence
    /// leaves with both hands empty. Inner nodes use the HalfPW net (search
    /// + EME resolve tempo ambiguity). Set in [`Self::grafted_with_weights`].
    cert_eval_leaves_only: bool,
    /// Override for experimental wall-ignorance certificate (`None` = env only).
    wall_ignore_cert_override: Option<bool>,
    /// Early Move Extensions on the first ordered wall moves (mirror of graduated LMR).
    eme: bool,
    pub nodes: u64,
    deadline: Instant,
    root_best: i16,
    root_score: i32,
    /// Lague partial-iteration: on time-abort, adopt the best FULLY-searched
    /// root move from the unfinished deepest iteration instead of discarding it.
    use_partial_iter: bool,
    /// Pure-JS-port mode: disables all Rust-side state-retention extras
    /// (gen TT, history aging, dynamic ID startup, accumulator retention).
    /// Use with `ti_movegen=true` as the fair baseline opponent.
    pure_mode: bool,
    /// Ponder mode: suppresses tt_gen advance and history decay so all ponder
    /// chunks share one TT generation and history accumulates uninterrupted.
    /// Set true before the ponder loop, false before the real think() call.
    is_pondering: bool,
    // ---------- pathfix feature flags (gen11 shipping config) ----------
    /// Exact k=0 race endgame + last-wall gate (JS `raceProof`, ships true).
    race_proof: bool,
    // ZeroFence diagnostics (parity-debug counters, match JS fields)
    refused_cuts: u64,
    rb1_stores: u64,
    dg_el: u64,
    dg_eu: u64,
    rep_path_c: u64,
    rep_game_c: u64,
    // RaceProof: race-table LRU (keyed by wall-config zobrist = hash sans pawn/turn)
    rc_key_lo: [u32; RC_SLOTS],
    rc_key_hi: [u32; RC_SLOTS],
    rc_tbl: Vec<Option<Box<[i16]>>>,
    rc_use: [u64; RC_SLOTS],
    rc_tick: u64,
    rc_last: i32,
    rc_build_ms: u64,
    rc_hits: u64,
    rc_solves: u64,
    rc_budget_miss: u64,
    rc_solve_ms: u64,
    rc_think_solve_ms: u64,
    rc_solve_cap: f64,
    rc_blocked: bool,
    rc_miss_lo: u32,
    rc_miss_hi: u32,
    rc_think_solves: u32,
    /// deterministic per-think in-tree solve cap (LRU holds 64: stops config-thrash)
    rc_count_cap: u32,
    rp_build_ok: bool,
    rp_root_empty: bool,
    pub rp_demotions: u64,
    pub rp_root_solves: u64,
    /// -1 sentinel: cell 0 (a1) is a legal pawn-move id
    root_pawn_best: i16,
    root_pawn_score: i32,
    race_scratch: Option<Box<RaceScratch>>,
    // RaceProof(c): certificate memo (gen13 — `certify_win.js` inlined, so the
    // certifier exists in node AND browser). Key = (lo, hi, side, wl0, wl1);
    // value = 1 proven (permanent, sound) / -work for a failure (richer retries
    // re-run; weaker-or-equal retries inherit the false).
    cw_cache: std::collections::HashMap<(u32, u32, usize, i32, i32), i32>,
    cw_proven: u64,
    cw_calls: u64,
    cw_think_calls: u32,
    cw_cap: u32,
    /// Live `info json` during `think(..., log=true)` — cleared when search ends.
    stream_log: bool,
    stream_label: String,
    stream_t0: Instant,
    stream_root_score: i32,
    stream_search_depth: i32,
    stream_depth_log: Vec<AceDepthLogEntry>,
    stream_last_emit_nodes: u64,
    stream_last_emit_ms: u64,
    stream_last_best: i16,
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    shared_tt: Option<Arc<SharedTitaniumTt>>,
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    lazy_runtime: Option<Arc<LazySmpRuntime>>,
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    lazy_root_moves: Option<Arc<Vec<i16>>>,
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    lazy_root_visit_map: Option<Arc<Vec<usize>>>,
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    lazy_root_allowed: usize,
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    lazy_worker_id: usize,
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    lazy_skip_setup: bool,
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    lazy_root_visits: Vec<usize>,
    /// GitHub Pages: live `info json` payloads forwarded to the browser worker.
    #[cfg(feature = "wasm")]
    wasm_progress: Option<js_sys::Function>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm-threads"))]
unsafe impl Send for TitaniumSearch {}

/// Periodic progress cadence: every 64K nodes AND ≥ 100ms apart — stdout/stderr
/// writes are expensive; spamming them steals think time from the search.
const STREAM_EMIT_NODE_MASK: u64 = 65535;
const STREAM_EMIT_MIN_INTERVAL_MS: u64 = 100;

impl TitaniumSearch {
    pub fn new(g: GameState) -> Box<Self> {
        Box::new(Self {
            g,
            tt_key_hi: vec![0; TT_SIZE],
            tt_key_lo: vec![0; TT_SIZE],
            tt_meta: vec![0; TT_SIZE],
            tt_score: vec![0; TT_SIZE],
            tt_rep: vec![0; TT_SIZE],
            tt_anc_lo: vec![0; TT_SIZE],
            tt_anc_hi: vec![0; TT_SIZE],
            tt_gen: 0,
            tt_entry_gen: vec![0; TT_SIZE],
            tt_mask: TT_MASK,
            tt_bits: TT_BITS,
            tt_filled: 0,
            // Defaults overwritten by enable_adaptive_tt(); harmless when inactive.
            tt_l2: TT_BITS,
            tt_l3: TT_BITS,
            tt_d4: 18,
            tt_d5: 22,
            tt_max: 25,
            tt_adaptive: false,
            sub_min: [MAX_PLY as i32; MAX_PLY],
            sub_anc_lo: [0; MAX_PLY],
            sub_anc_hi: [0; MAX_PLY],
            history_tbl: [0; 512],
            cm: [0; 512],
            killers: [[0; 2]; MAX_PLY],
            reduction_probe_enabled: false,
            reduction_probe_target: None,
            reduction_probe_next: 0,
            reduction_probe_limit: 0,
            reduction_probe_min_depth: 0,
            reduction_probe_events: Vec::new(),
            reduction_sidecar: None,
            reduction_shadow_stats: ReductionShadowStats::default(),
            path_lo: [0; MAX_PLY],
            path_hi: [0; MAX_PLY],
            d0: [[0; 81]; MAX_PLY],
            d1: [[0; 81]; MAX_PLY],
            d0_layers: [[0; 81]; MAX_PLY],
            d1_layers: [[0; 81]; MAX_PLY],
            dist0_idx: 0,
            dist1_idx: 0,
            cached_stamp: -1,
            np_acc0: [0.0; NET_H],
            np_acc1: [0.0; NET_H],
            np_hw: [0; 64],
            np_vw: [0; 64],
            np_b0: -1,
            np_b1v: -1,
            net: net(),
            bridge: None,
            ti_movegen: false,
            cat_walls: false,
            cat_lmr_v16: false,
            cat_lmr_ceiling: crate::cat::CAT_V16_LMR_CEILING_DEFAULT,
            cat_lmr_fringe_pct: crate::cat::CAT_V16_FRINGE_PCT_DEFAULT,
            dead_zone_prune: false,
            cheap_cert: false,
            cert_eval_leaves_only: false,
            wall_ignore_cert_override: None,
            eme: false,
            nodes: 0,
            deadline: Instant::now(),
            root_best: super::TITANIUM_NO_MOVE,
            root_score: 0,
            use_partial_iter: true,
            pure_mode: false,
            is_pondering: false,
            race_proof: true,
            refused_cuts: 0,
            rb1_stores: 0,
            dg_el: 0,
            dg_eu: 0,
            rep_path_c: 0,
            rep_game_c: 0,
            rc_key_lo: [0; RC_SLOTS],
            rc_key_hi: [0; RC_SLOTS],
            rc_tbl: (0..RC_SLOTS).map(|_| None).collect(),
            rc_use: [0; RC_SLOTS],
            rc_tick: 0,
            rc_last: -1,
            rc_build_ms: 6,
            rc_hits: 0,
            rc_solves: 0,
            rc_budget_miss: 0,
            rc_solve_ms: 0,
            rc_think_solve_ms: 0,
            rc_solve_cap: f64::INFINITY,
            rc_blocked: false,
            rc_miss_lo: 0,
            rc_miss_hi: 0,
            rc_think_solves: 0,
            rc_count_cap: 48,
            rp_build_ok: false,
            rp_root_empty: false,
            rp_demotions: 0,
            rp_root_solves: 0,
            root_pawn_best: -1,
            root_pawn_score: i32::MIN,
            race_scratch: None,
            cw_cache: std::collections::HashMap::new(),
            cw_proven: 0,
            cw_calls: 0,
            cw_think_calls: 0,
            cw_cap: 24,
            stream_log: false,
            stream_label: String::new(),
            stream_t0: Instant::now(),
            stream_root_score: 0,
            stream_search_depth: 0,
            stream_depth_log: Vec::new(),
            stream_last_emit_nodes: 0,
            stream_last_emit_ms: 0,
            stream_last_best: 0,
            #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
            shared_tt: None,
            #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
            lazy_runtime: None,
            #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
            lazy_root_moves: None,
            #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
            lazy_root_visit_map: None,
            #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
            lazy_root_allowed: usize::MAX,
            #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
            lazy_worker_id: 0,
            #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
            lazy_skip_setup: false,
            #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
            lazy_root_visits: Vec::new(),
            #[cfg(feature = "wasm")]
            wasm_progress: None,
        })
    }

    /// Wire browser progress callback (`go(..., on_progress)` on GitHub Pages).
    #[cfg(feature = "wasm")]
    pub fn set_wasm_progress(&mut self, cb: Option<js_sys::Function>) {
        self.wasm_progress = cb;
    }

    /// Enable Early Move Extensions — same gates/tuning as graduated LMR, early indices.
    pub fn enable_eme(&mut self) {
        self.eme = true;
    }

    /// Enable offline observation of complete native LMR move pipelines.
    /// `target=None` records baseline events; `Some(n)` applies +1 only to event n.
    /// `min_depth` skips events at local depth < min_depth so shallow-tree events
    /// (which dominate post-order traversal) do not fill the limit before useful ones.
    pub fn enable_reduction_probe(&mut self, target: Option<u64>, limit: usize, min_depth: i32) {
        self.reduction_probe_enabled = true;
        self.reduction_probe_target = target;
        self.reduction_probe_next = 0;
        self.reduction_probe_limit = limit;
        self.reduction_probe_min_depth = min_depth;
        self.reduction_probe_events.clear();
    }

    pub fn reduction_probe_events(&self) -> &[ReductionProbeEvent] {
        &self.reduction_probe_events
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn enable_reduction_shadow(&mut self, path: &std::path::Path) -> Result<(), String> {
        self.reduction_sidecar = Some(ReductionSidecar::load(path)?);
        self.reduction_shadow_stats = ReductionShadowStats::default();
        Ok(())
    }

    pub fn reduction_shadow_stats(&self) -> ReductionShadowStats {
        self.reduction_shadow_stats
    }

    /// Titanium movegen on a mirrored board — same legal set, much faster than `wall_legal`.
    pub fn with_ti_movegen(g: GameState) -> Box<Self> {
        let mut search = Self::new(g);
        search.bridge = Some(TiBridge::from_game(&search.g));
        search.ti_movegen = true;
        search
    }

    /// ACE v13 reference tier with Titanium movegen acceleration, pinned to the
    /// frozen HalfPW blob used by the JS reference instead of live training weights.
    pub fn with_ti_movegen_frozen(g: GameState) -> Box<Self> {
        let mut search = Self::with_ti_movegen(g);
        search.net = net_frozen();
        search
    }

    /// Pure JS-port baseline + O1 movegen only. Uses **frozen** v13 HalfPW weights
    /// (`net_weights_frozen.bin`) — never picks up live training/deploy updates.
    pub fn with_ti_movegen_pure(g: GameState) -> Box<Self> {
        let mut search = Self::with_ti_movegen_frozen(g);
        search.pure_mode = true;
        search
    }

    /// CAT hybrid: walls at inner nodes must pass `wall_should_search`.
    pub fn with_cat(g: GameState) -> Box<Self> {
        let mut search = Self::new(g);
        search.bridge = Some(TiBridge::from_game(&search.g));
        search.cat_walls = true;
        search
    }

    /// Fast Titanium movegen + CAT wall filter.
    pub fn with_ti_movegen_and_cat(g: GameState) -> Box<Self> {
        let mut search = Self::with_ti_movegen(g);
        search.cat_walls = true;
        search
    }

    /// **Grafted engine** — gen13 net search + Titanium's *logically-safe* extras:
    ///   - cheap hands-empty cert: replaces the recursive `certify` with the exact
    ///     race classifier when no walls remain — IDENTICAL verdict, fewer nodes.
    ///     A strict non-regression (can't produce a worse move; frees NPS).
    ///   - adaptive cache-tier TT: identical TT semantics, better cache locality
    ///     and safe growth. Also can't hurt.
    ///
    /// EXCLUDED:
    ///   - CAT heat-prune: removes wall candidates the net wants (drops Elo).
    ///   - dead-zone prune: unsound (block-a-blocker) AND its apparent gain was
    ///     measurement noise — a single-seed +76 became −25 on another seed.
    ///
    /// NOTE on measurement: 112-game runs carry a ±~64 Elo 95% CI, so per-run Elo
    /// deltas are not individually trustworthy. These two extras are kept because
    /// they are *provably* non-harmful, not because a single match "won".
    /// `tt_bits = Some(n)` pins a fixed TT instead of the adaptive one.
    pub fn grafted(g: GameState, tt_bits: Option<usize>) -> Box<Self> {
        Self::grafted_with_weights(g, tt_bits, net())
    }

    /// Same as [`grafted_frozen`] but uses the frozen v13 HalfPW blob (training A/B control).
    pub fn grafted_frozen(g: GameState, tt_bits: Option<usize>) -> Box<Self> {
        Self::grafted_with_weights(g, tt_bits, net_frozen())
    }

    /// Medium tier — runtime-installed weights (`net_weights_medium.bin`).
    pub fn grafted_medium(g: GameState, tt_bits: Option<usize>) -> Box<Self> {
        let weights = crate::titanium::net::net_medium()
            .expect("medium NNUE weights not installed — fetch net_weights_medium.bin first");
        Self::grafted_with_weights(g, tt_bits, weights)
    }

    /// Production graft minus RaceProof/cert gates. Experimental only: useful for
    /// measuring whether search can replace the proof layer before removing it.
    pub fn grafted_no_raceproof(g: GameState, tt_bits: Option<usize>) -> Box<Self> {
        let mut search = Self::grafted(g, tt_bits);
        search.race_proof = false;
        search
    }

    /// Titanium v15 experimental — wall-ignorance loss certificate (frozen net).
    pub fn grafted_wall_ignore_experimental(g: GameState, tt_bits: Option<usize>) -> Box<Self> {
        let mut search = Self::grafted_frozen(g, tt_bits);
        search.wall_ignore_cert_override = Some(true);
        search
    }

    pub fn grafted_with_weights(
        g: GameState,
        tt_bits: Option<usize>,
        weights: &'static Net,
    ) -> Box<Self> {
        let mut search = Self::with_ti_movegen(g);
        search.net = weights;
        search.cheap_cert = true;
        search.cert_eval_leaves_only = true;
        match tt_bits {
            Some(bits) => search.resize_tt(bits),
            None => search.enable_adaptive_tt(),
        }
        search
    }

    /// **Titanium v16** — v15 graft + CAT-scaled late-move reductions.
    ///
    /// Main-thread cold walls (at or below 5% of the attention ceiling, default 800 cm)
    /// are searched at child depth 1; helper workers use 10%, 20%, 30%... capped
    /// at 70%. Corridor-hot walls keep proportional depth. Same NNUE weights as
    /// v15 live (`net_weights.bin`) until a dedicated v16 blob ships.
    pub fn grafted_v16(g: GameState, tt_bits: Option<usize>) -> Box<Self> {
        #[cfg(not(target_arch = "wasm32"))]
        let ceiling = crate::cat::cat_v16_lmr_ceiling_from_env();
        #[cfg(target_arch = "wasm32")]
        let ceiling = crate::cat::CAT_V16_LMR_CEILING_DEFAULT;
        Self::grafted_v16_with_ceiling(g, tt_bits, ceiling)
    }

    pub fn grafted_v16_with_ceiling(
        g: GameState,
        tt_bits: Option<usize>,
        ceiling: u16,
    ) -> Box<Self> {
        let mut search = Self::grafted(g, tt_bits);
        search.cat_lmr_v16 = true;
        search.cat_lmr_ceiling = if crate::cat::CAT_V16_LMR_CEILINGS.contains(&ceiling) {
            ceiling
        } else {
            crate::cat::CAT_V16_LMR_CEILING_DEFAULT
        };
        search
    }

    /// gen13 net search + O1 movegen + cheap hands-empty cert, but **no CAT**.
    /// Isolates the certificate contribution from CAT wall-pruning.
    pub fn with_ti_movegen_cheap_cert(g: GameState, tt_bits: Option<usize>) -> Box<Self> {
        let mut search = Self::with_ti_movegen(g);
        search.cheap_cert = true;
        if let Some(bits) = tt_bits {
            search.resize_tt(bits);
        }
        search
    }

    /// gen13 net search + O1 movegen + adaptive cache-tier TT (no CAT, no cert).
    /// Isolates the TT-growth contribution.
    pub fn with_ti_movegen_adaptive_tt(g: GameState) -> Box<Self> {
        let mut search = Self::with_ti_movegen(g);
        search.enable_adaptive_tt();
        search
    }

    /// gen13 net search + O1 movegen + SOUND dead-zone wall prune (no CAT heat).
    /// Isolates the dead-zone pruner's contribution (NPS-only, can't cost Elo).
    pub fn with_ti_movegen_deadzone(g: GameState) -> Box<Self> {
        let mut search = Self::with_ti_movegen(g);
        search.dead_zone_prune = true;
        search
    }

    /// Reallocate the transposition table to `1 << bits` entries. Clears all TT
    /// state — call before search starts, not mid-think.
    pub fn resize_tt(&mut self, bits: usize) {
        let size = 1usize << bits;
        self.tt_key_hi = vec![0; size];
        self.tt_key_lo = vec![0; size];
        self.tt_meta = vec![0; size];
        self.tt_score = vec![0; size];
        self.tt_rep = vec![0; size];
        self.tt_anc_lo = vec![0; size];
        self.tt_anc_hi = vec![0; size];
        self.tt_entry_gen = vec![0; size];
        self.tt_mask = (size - 1) as u32;
        self.tt_bits = bits;
        self.tt_filled = 0;
    }

    /// Enable overflow-driven cache-tier TT growth (Titanium strategy). Starts the
    /// table small enough to live in L1, then jumps L1→L2→L3→18→22 as it fills, so
    /// at low node counts the whole TT stays hot in cache (the big win at short TC)
    /// and only grows when it genuinely overflows. `entry_bytes` = 25 (the 7 SoA
    /// arrays: 3×u32 key/anc + 2×i32 meta/score + 1×u8 rep ≈ 25 B/logical entry).
    pub fn enable_adaptive_tt(&mut self) {
        const ENTRY_BYTES: usize = 25;
        let (start, l2, l3) = crate::search::tt::cache_tier_bits(ENTRY_BYTES);
        self.tt_l2 = l2.max(start + 1);
        self.tt_l3 = l3.max(self.tt_l2 + 1);
        self.tt_d4 = 18.max(self.tt_l3);
        self.tt_d5 = 22.max(self.tt_d4);
        self.tt_max = 25.max(self.tt_d5);
        self.tt_adaptive = true;
        self.resize_tt(start); // start small — grows on overflow
    }

    /// Grow the TT to the next calibrated cache tier and rehash live entries.
    /// Always-replace on collision (matches the live store policy). Called from the
    /// store path when occupancy crosses 50%.
    fn tt_grow(&mut self) {
        let nb = if self.tt_bits < self.tt_l2 {
            self.tt_l2
        } else if self.tt_bits < self.tt_l3 {
            self.tt_l3
        } else if self.tt_bits < self.tt_d4 {
            self.tt_d4
        } else if self.tt_bits < self.tt_d5 {
            self.tt_d5
        } else {
            self.tt_bits + 1
        }
        .min(self.tt_max);
        if nb <= self.tt_bits {
            return;
        }
        let new_size = 1usize << nb;
        let new_mask = (new_size - 1) as u32;
        let mut k_hi = vec![0u32; new_size];
        let mut k_lo = vec![0u32; new_size];
        let mut meta = vec![0i32; new_size];
        let mut score = vec![0i32; new_size];
        let mut rep = vec![0u8; new_size];
        let mut a_lo = vec![0u32; new_size];
        let mut a_hi = vec![0u32; new_size];
        let mut e_gen = vec![0u8; new_size];
        let mut filled = 0usize;
        for i in 0..self.tt_meta.len() {
            if self.tt_meta[i] == 0 {
                continue;
            }
            let ni = (self.tt_key_lo[i] & new_mask) as usize;
            if meta[ni] == 0 {
                filled += 1;
            }
            k_hi[ni] = self.tt_key_hi[i];
            k_lo[ni] = self.tt_key_lo[i];
            meta[ni] = self.tt_meta[i];
            score[ni] = self.tt_score[i];
            rep[ni] = self.tt_rep[i];
            a_lo[ni] = self.tt_anc_lo[i];
            a_hi[ni] = self.tt_anc_hi[i];
            e_gen[ni] = self.tt_entry_gen[i];
        }
        self.tt_key_hi = k_hi;
        self.tt_key_lo = k_lo;
        self.tt_meta = meta;
        self.tt_score = score;
        self.tt_rep = rep;
        self.tt_anc_lo = a_lo;
        self.tt_anc_hi = a_hi;
        self.tt_entry_gen = e_gen;
        self.tt_mask = new_mask;
        self.tt_bits = nb;
        self.tt_filled = filled;
    }

    /// Advance the live game one ply, keeping TT/killers/history warm.
    /// Long-lived session path — the next `think` reuses prior analysis.
    pub fn apply_move(&mut self, m: i16) {
        self.g.make_move(m);
        if m >= 100 {
            self.cached_stamp = -1;
        }
        if self.pure_mode {
            // Faithful JS baseline: reset accumulator every move (no retention).
            self.np_b0 = -1;
        }
        // non-pure: do NOT reset np_b0/np_b1v — evaluate()'s bucket-aware diff handles any
        // accumulator transition (wall diff or full rebuild on bucket cross).
    }

    /// Replace the position outright (undo, new game) without clearing the
    /// TT — entries are hash-keyed, stale ones simply never match.
    pub fn set_position(&mut self, g: GameState) {
        self.g = g;
        self.position_changed();
    }

    /// Scale history table by a surprise-proportional factor.
    /// Called when the opponent played an unexpected move so stale tactical
    /// patterns from the abandoned search don't dominate the new root.
    /// For a correct prediction (|prior - current| ≈ 0) decay ≈ 1.0 (no-op).
    pub fn decay_history_by_surprise(&mut self, prior_score: i32) {
        let surprise = (prior_score - self.root_score).abs() as f32;
        let decay = 1.0 / (1.0 + surprise / 200.0);
        for h in self.history_tbl.iter_mut() {
            *h = (*h as f32 * decay) as i32;
        }
    }

    /// Advance the root by one ply (predicted opponent move) and adjust state
    /// for seamless continuation. For use after `go infinite` + `ponderhit`.
    pub fn migrate_root(&mut self, m: i16, prior_score: i32) {
        self.apply_move(m);
        self.decay_history_by_surprise(prior_score);
        if !self.pure_mode {
            self.tt_gen = self.tt_gen.wrapping_add(1);
        }
    }

    /// Static evaluation of the current position (no search) — primes the distance
    /// cache and forces an accumulator rebuild, then runs `evaluate()`. On mid-game
    /// positions this returns the pure HalfPW net output; used by the NNUE trainer
    /// parity harness to confirm the Python forward pass matches the engine.
    pub fn eval_position(&mut self) -> i32 {
        self.position_changed();
        self.refresh_dist(0);
        self.evaluate(0)
    }

    /// Enable Lague partial-iteration (keep the best fully-searched move from a
    /// time-aborted deepest iteration). Off by default; A/B-measured before adoption.
    pub fn set_partial_iter(&mut self, on: bool) {
        self.use_partial_iter = on;
    }

    /// Enter/exit ponder mode. While pondering, `think()` skips the tt_gen
    /// advance and history decay so all ponder chunks build on each other
    /// rather than aging their own work.  Call with `false` before the real
    /// think so it does the normal one-time decay and advances the generation.
    pub fn set_pondering(&mut self, on: bool) {
        self.is_pondering = on;
    }

    pub fn set_cat_lmr_fringe_pct(&mut self, pct: u16) {
        self.cat_lmr_fringe_pct = pct.min(crate::cat::CAT_V16_FRINGE_PCT_MAX);
    }

    pub fn set_cat_lmr_worker_profile(&mut self, worker_id: usize) {
        self.set_cat_lmr_fringe_pct(cat_v16_lmr_fringe_pct_for_worker(worker_id));
    }

    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    fn lazy_smp_width_percent(worker_id: usize) -> usize {
        LAZY_SMP_WIDTHS
            .get(worker_id)
            .copied()
            .unwrap_or(*LAZY_SMP_WIDTHS.last().expect("width schedule is non-empty"))
    }

    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    fn apply_think_start_state(&mut self) {
        if !self.pure_mode && !self.is_pondering {
            self.tt_gen = self.tt_gen.wrapping_add(1);
            for h in self.history_tbl.iter_mut() {
                *h >>= 1;
            }
        }
    }

    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    fn ordered_root_moves_snapshot(&mut self, depth: i32) -> Vec<i16> {
        self.refresh_dist(0);
        if self.bridge.is_some() {
            self.bridge = Some(TiBridge::from_game(&self.g));
        }
        let mut moves = [0i16; 160];
        let root_entry = self
            .shared_tt
            .as_ref()
            .and_then(|tt| tt.probe(self.g.hash_lo, self.g.hash_hi));
        let tt_move = root_entry
            .map(|entry| (entry.meta & 1023) as i16)
            .unwrap_or_else(|| {
                let idx = (self.g.hash_lo & self.tt_mask) as usize;
                let meta = self.tt_meta[idx];
                if meta != 0
                    && self.tt_key_hi[idx] == self.g.hash_hi
                    && self.tt_key_lo[idx] == self.g.hash_lo
                {
                    (meta & 1023) as i16
                } else {
                    0
                }
            });
        let n = self.gen_moves(0, depth.max(1), tt_move, &mut moves);
        self.order_moves(0, &mut moves[..n], tt_move, 0);
        moves[..n].to_vec()
    }

    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    fn gcd_usize(mut a: usize, mut b: usize) -> usize {
        while b != 0 {
            let r = a % b;
            a = b;
            b = r;
        }
        a
    }

    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    fn lazy_smp_profile_root_moves(
        root_moves: &[i16],
        worker_id: usize,
        allowed: usize,
    ) -> (Vec<i16>, Vec<usize>) {
        let len = root_moves.len();
        let allowed = allowed.min(len);
        if allowed == 0 {
            return (Vec::new(), Vec::new());
        }
        if worker_id == 0 || len <= 1 {
            return (
                root_moves[..allowed].to_vec(),
                (0..allowed).collect::<Vec<_>>(),
            );
        }

        let mut stride = worker_id.saturating_mul(2).saturating_add(1).max(3);
        while Self::gcd_usize(stride, len) != 1 {
            stride = stride.saturating_add(2);
        }
        let offset = worker_id.saturating_mul(37) % len;
        let mut seen = vec![false; len];
        let mut profiled = Vec::with_capacity(allowed);
        let mut original_indices = Vec::with_capacity(allowed);
        let mut cursor = offset;
        while profiled.len() < allowed {
            if !seen[cursor] {
                seen[cursor] = true;
                profiled.push(root_moves[cursor]);
                original_indices.push(cursor);
            }
            cursor = (cursor + stride) % len;
        }
        (profiled, original_indices)
    }

    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    fn fork_lazy_worker(&self, root: &GameState) -> Box<Self> {
        let mut worker = Self::new(root.clone());
        worker.history_tbl = self.history_tbl;
        worker.cm = self.cm;
        worker.killers = self.killers;
        worker.net = self.net;
        worker.ti_movegen = self.ti_movegen;
        worker.cat_walls = self.cat_walls;
        worker.cat_lmr_v16 = self.cat_lmr_v16;
        worker.cat_lmr_ceiling = self.cat_lmr_ceiling;
        worker.cat_lmr_fringe_pct = self.cat_lmr_fringe_pct;
        worker.dead_zone_prune = self.dead_zone_prune;
        worker.cheap_cert = self.cheap_cert;
        worker.cert_eval_leaves_only = self.cert_eval_leaves_only;
        worker.wall_ignore_cert_override = self.wall_ignore_cert_override;
        worker.eme = self.eme;
        worker.use_partial_iter = self.use_partial_iter;
        worker.pure_mode = self.pure_mode;
        worker.race_proof = self.race_proof;
        worker.tt_gen = self.tt_gen;
        worker.tt_mask = self.tt_mask;
        worker.tt_bits = self.tt_bits;
        worker.tt_adaptive = false;
        // Helpers search against the shared lazy-SMP TT only, so drop the full
        // local TT that Self::new allocated. At TT_BITS=20 that is ~26MB per
        // worker; 7 helpers previously allocated ~182MB of dead tables and
        // overflowed the wasm memory cap (the first threaded search aborted in
        // handle_alloc_error → bare `unreachable`). Local TT probes/stores are
        // gated on `shared_tt.is_none()`, so these 1-element vecs are never
        // indexed once install_lazy_smp_context() runs.
        worker.tt_key_hi = vec![0; 1];
        worker.tt_key_lo = vec![0; 1];
        worker.tt_meta = vec![0; 1];
        worker.tt_score = vec![0; 1];
        worker.tt_rep = vec![0; 1];
        worker.tt_anc_lo = vec![0; 1];
        worker.tt_anc_hi = vec![0; 1];
        worker.tt_entry_gen = vec![0; 1];
        if self.bridge.is_some() {
            worker.bridge = Some(TiBridge::from_game(root));
        }
        worker
    }

    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    fn install_lazy_smp_context(
        &mut self,
        worker_id: usize,
        shared_tt: Arc<SharedTitaniumTt>,
        runtime: Arc<LazySmpRuntime>,
        root_moves: Arc<Vec<i16>>,
        root_visit_map: Arc<Vec<usize>>,
        allowed: usize,
    ) {
        self.shared_tt = Some(shared_tt.clone());
        self.tt_mask = shared_tt.mask;
        self.tt_bits = shared_tt.bits;
        self.tt_adaptive = false;
        self.lazy_runtime = Some(runtime.clone());
        self.deadline = runtime.deadline;
        self.lazy_root_moves = Some(root_moves);
        self.lazy_root_visit_map = Some(root_visit_map);
        self.lazy_root_allowed = allowed;
        self.lazy_worker_id = worker_id;
        self.set_cat_lmr_worker_profile(worker_id);
        self.lazy_skip_setup = true;
        self.lazy_root_visits.clear();
    }

    /// Path-valid wall slot count for `ws[14]` (geometric; ignores wall budget / side).
    /// Path-valid = tentative placement keeps both players connected to goal (`pbff_wall_legal`).
    pub fn geometric_legal_wall_count(&mut self) -> u32 {
        if let Some(bridge) = self.bridge.as_mut() {
            return geometric_wall_len_cached(
                &mut bridge.geometric_walls,
                &mut bridge.board,
                &mut bridge.bfs,
                GeometricWallCacheRole::Eval,
                Some(&mut bridge.wall_cache_stats),
            ) as u32;
        }
        let mut board = Board::new();
        for i in 0..self.g.hist_len {
            let _ = board.make_move(move_id_to_board(self.g.hist_m[i]));
        }
        let mut scratch = BfsScratch::new();
        let mut cache = None;
        geometric_wall_len_cached(
            &mut cache,
            &mut board,
            &mut scratch,
            GeometricWallCacheRole::Eval,
            None,
        ) as u32
    }

    /// Wall-cache profiling counters (TiBridge path only).
    pub fn wall_cache_stats(&self) -> Option<GeometricWallCacheStats> {
        self.bridge.as_ref().map(|b| b.wall_cache_stats)
    }

    /// Dump the raw net inputs + the resulting eval as JSON. Lets the Python NNUE
    /// trainer verify its forward pass against the engine on the *inputs alone*,
    /// without reimplementing Quoridor rules/BFS in Python — and is the record
    /// format for training-data generation.
    ///
    /// `d0`/`d1` are the pawn shortest-path distances (scalars).
    /// Canonical field keys: `goal_inv_p0_field`, `pawn_fwd_p0_field`, `corridor_delta_p0_field`,
    /// `path_cross_p0_field` (and `_p1` variants). Legacy aliases `d0_field`, `player0_field`, …
    /// are duplicated in the JSON for old JSONL; trainer reads either via `rec_field()`.
    /// Same JSON as [`Self::eval_dump_json`] with packed-batch metadata prefix fields.
    pub fn eval_dump_json_packed(&mut self, row: u32) -> String {
        let body = self.eval_dump_json();
        format!(
            "{{\"row\":{row},\"ok\":true,\"feature_schema\":\"{FEATURE_SCHEMA}\",\"protocol\":\"eval-packed-v1\",{}",
            &body[1..]
        )
    }

    pub fn eval_dump_json(&mut self) -> String {
        self.position_changed();
        self.refresh_dist(0);
        let eval = self.evaluate(0);
        let d0_scalar = self.d0[self.dist0_idx][self.g.pawn[0]];
        let d1_scalar = self.d1[self.dist1_idx][self.g.pawn[1]];
        let bits = |arr: &[u8; 64]| {
            let mut s = String::new();
            for (i, b) in arr.iter().enumerate() {
                if i > 0 {
                    s.push(',');
                }
                s.push(if *b != 0 { '1' } else { '0' });
            }
            s
        };
        let field = |arr: &[u8; 81]| {
            let mut s = String::new();
            for (i, &v) in arr.iter().enumerate() {
                if i > 0 {
                    s.push(',');
                }
                s.push_str(&v.to_string());
            }
            s
        };
        let field16 = |arr: &[u16; 81]| {
            let mut s = String::new();
            for (i, &v) in arr.iter().enumerate() {
                if i > 0 {
                    s.push(',');
                }
                s.push_str(&v.to_string());
            }
            s
        };
        let d0f = self.d0[self.dist0_idx];
        let d1f = self.d1[self.dist1_idx];
        let mut p0_steps = [255u8; 81];
        let mut p1_steps = [255u8; 81];
        let mut delta0 = [255u8; 81];
        let mut delta1 = [255u8; 81];
        fill_ace_dist_from_pawn(&self.g, self.g.pawn[0], &mut p0_steps);
        fill_ace_dist_from_pawn(&self.g, self.g.pawn[1], &mut p1_steps);
        fill_corridor_delta(&p0_steps, &d0f, d0_scalar, &mut delta0);
        fill_corridor_delta(&p1_steps, &d1f, d1_scalar, &mut delta1);
        let mut cross0 = [0u8; 81];
        let mut cross1 = [0u8; 81];
        fill_path_crossing(&self.g, &p0_steps, &d0f, d0_scalar, &mut cross0);
        fill_path_crossing(&self.g, &p1_steps, &d1f, d1_scalar, &mut cross1);
        let mut choke0 = [0u8; 81];
        let mut choke1 = [0u8; 81];
        fill_choke_points(&self.g, &p0_steps, &d0f, d0_scalar, &mut choke0);
        fill_choke_points(&self.g, &p1_steps, &d1f, d1_scalar, &mut choke1);
        let mut contested = [0u8; 81];
        fill_contested(&delta0, &delta1, &mut contested);
        let mut route0 = [0u8; 81];
        let mut route1 = [0u8; 81];
        let mut flank0 = [0u8; 81];
        let mut flank1 = [0u8; 81];
        fill_sparse_route_masks(&self.g, self.g.pawn[0], &d0f, &mut route0, &mut flank0);
        fill_sparse_route_masks(&self.g, self.g.pawn[1], &d1f, &mut route1, &mut flank1);
        let (cat_best_p0, cat_best_p1, cat_heat) = {
            let mut bridge = TiBridge::from_game(&self.g);
            let cat = crate::cat::build::build_impact_heatmap(&bridge.board);
            let (best0, best1) = crate::cat::best_pawn_cat_heats(&bridge.board, &cat, &mut bridge.bfs);
            (best0, best1, cat.square_heat)
        };
        let legal_walls = self.geometric_legal_wall_count();
        let (cross_p0, cross_p1) = self.legal_path_crossing_counts_arr(&route0, &route1);
        let width_me = self.d0[self.dist0_idx]
            .iter()
            .filter(|&&d| d as i32 == d0_scalar as i32)
            .count();
        let width_opp = self.d1[self.dist1_idx]
            .iter()
            .filter(|&&d| d as i32 == d1_scalar as i32)
            .count();
        format!(
            "{{\"turn\":{},\"pawn0\":{},\"pawn1\":{},\"wl0\":{},\"wl1\":{},\
             \"d0\":{},\"d1\":{},\"legal_wall_count\":{},\"legal_path_cross_p0\":{},\"legal_path_cross_p1\":{},\
             \"cat_best_p0\":{},\"cat_best_p1\":{},\"cat_heat_field\":[{}],\
             \"corridor_width0\":{},\"corridor_width1\":{},\
             \"goal_inv_p0_field\":[{}],\"goal_inv_p1_field\":[{}],\
             \"pawn_fwd_p0_field\":[{}],\"pawn_fwd_p1_field\":[{}],\
             \"corridor_delta_p0_field\":[{}],\"corridor_delta_p1_field\":[{}],\
             \"path_cross_p0_field\":[{}],\"path_cross_p1_field\":[{}],\
             \"choke_p0_field\":[{}],\"choke_p1_field\":[{}],\
             \"contested_field\":[{}],\
             \"route_p0_field\":[{}],\"route_p1_field\":[{}],\
             \"route_flank_p0_field\":[{}],\"route_flank_p1_field\":[{}],\
             \"d0_field\":[{}],\"d1_field\":[{}],\
             \"player0_field\":[{}],\"player1_field\":[{}],\
             \"delta0_field\":[{}],\"delta1_field\":[{}],\
             \"cross0_field\":[{}],\"cross1_field\":[{}],\
             \"hw\":[{}],\"vw\":[{}],\"eval\":{}}}",
            self.g.turn,
            self.g.pawn[0],
            self.g.pawn[1],
            self.g.wl[0],
            self.g.wl[1],
            d0_scalar,
            d1_scalar,
            legal_walls,
            cross_p0,
            cross_p1,
            cat_best_p0,
            cat_best_p1,
            field16(&cat_heat),
            width_me,
            width_opp,
            field(&d0f),
            field(&d1f),
            field(&p0_steps),
            field(&p1_steps),
            field(&delta0),
            field(&delta1),
            field(&cross0),
            field(&cross1),
            field(&choke0),
            field(&choke1),
            field(&contested),
            field(&route0),
            field(&route1),
            field(&flank0),
            field(&flank1),
            field(&d0f),
            field(&d1f),
            field(&p0_steps),
            field(&p1_steps),
            field(&delta0),
            field(&delta1),
            field(&cross0),
            field(&cross1),
            bits(&self.g.hw),
            bits(&self.g.vw),
            eval
        )
    }

    fn position_changed(&mut self) {
        if self.bridge.is_some() {
            self.bridge = Some(TiBridge::from_game(&self.g));
        }
        self.cached_stamp = -1;
        self.np_b0 = -1; // force full accumulator rebuild (v10: no stamp gate)
        self.np_b1v = -1;
    }

    fn sync_stream_meta(
        &mut self,
        depth_log: &[AceDepthLogEntry],
        search_depth: i32,
        root_score: i32,
    ) {
        self.stream_depth_log.clear();
        self.stream_depth_log.extend_from_slice(depth_log);
        self.stream_search_depth = search_depth;
        self.stream_root_score = root_score;
    }

    /// Periodic + forced progress for website SSE (matches JS cumulative `search.nodes`).
    /// Periodic emits are throttled by node count AND wall time; forced emits
    /// (depth complete, root best-move change, deadline) always go out.
    fn emit_stream_progress(&mut self, force: bool) {
        if !self.stream_log {
            return;
        }
        let elapsed_ms = self.stream_t0.elapsed().as_millis() as u64;
        if !force {
            if self.nodes == 0 || self.nodes == self.stream_last_emit_nodes {
                return;
            }
            if (self.nodes & STREAM_EMIT_NODE_MASK) != 0 {
                return;
            }
            if elapsed_ms.saturating_sub(self.stream_last_emit_ms) < STREAM_EMIT_MIN_INTERVAL_MS {
                return;
            }
        }
        self.stream_last_emit_ms = elapsed_ms;
        self.stream_last_emit_nodes = self.nodes;
        self.refresh_dist(0);
        let white_dist = self.d0[self.dist0_idx][self.g.pawn[0]];
        let black_dist = self.d1[self.dist1_idx][self.g.pawn[1]];
        let elapsed_ms = self.stream_t0.elapsed().as_millis() as u64;
        emit_ace_progress(
            &self.stream_label,
            &self.stream_depth_log,
            self.stream_search_depth,
            self.nodes,
            self.stream_root_score,
            white_dist,
            black_dist,
            elapsed_ms,
            #[cfg(feature = "wasm")]
            self.wasm_progress.as_ref(),
        );
    }

    #[inline(always)]
    fn check_time(&mut self) -> Result<(), TimeUp> {
        #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
        if let Some(runtime) = self.lazy_runtime.as_ref() {
            if runtime.stop.load(Ordering::Relaxed) {
                return Err(TimeUp);
            }
        }
        if (self.nodes & 1023) == 0 {
            if Instant::now() > self.deadline {
                #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
                if let Some(runtime) = self.lazy_runtime.as_ref() {
                    runtime.stop.store(true, Ordering::Relaxed);
                }
                self.emit_stream_progress(true);
                return Err(TimeUp);
            }
            self.emit_stream_progress(false);
        }
        Ok(())
    }

    fn ace_time_fraction(last_score: i32) -> f64 {
        if last_score < -80 {
            0.92
        } else {
            0.85
        }
    }

    fn ace_over_time_budget(t0: Instant, time_ms: u64, last_score: i32) -> bool {
        let budget = time_ms as f64 * Self::ace_time_fraction(last_score);
        t0.elapsed().as_millis() as f64 > budget
    }

    /// Returns (score, route0_bits, route1_bits) so callers can reuse the route bitsets.
    fn route_feature_score(&self, nw: &Net) -> (f64, u128, u128) {
        crate::bench_instr::record(
            |b| &mut b.eval_route_features,
            || self.route_feature_score_inner(nw),
        )
    }

    fn route_feature_score_inner(&self, nw: &Net) -> (f64, u128, u128) {
        if !nw.route_active {
            return (0.0, 0, 0);
        }
        let d0f = &self.d0[self.dist0_idx];
        let d1f = &self.d1[self.dist1_idx];
        let masks = DirMasks::from_ace_game(&self.g);
        let route0 = shortest_route_bits(
            self.g.pawn[0],
            d0f[self.g.pawn[0]],
            &self.d0_layers[self.dist0_idx],
            masks,
        );
        let route1 = shortest_route_bits(
            self.g.pawn[1],
            d1f[self.g.pawn[1]],
            &self.d1_layers[self.dist1_idx],
            masks,
        );
        let near0 = expand_frontier(route0, masks) & !route0 & FLOOD_PLAYABLE;
        let near1 = expand_frontier(route1, masks) & !route1 & FLOOD_PLAYABLE;
        let (me_route, opp_route, me_near, opp_near) = if self.g.turn == 0 {
            (route0, route1, near0, near1)
        } else {
            (route1, route0, near1, near0)
        };
        let contested = (me_route | me_near) & (opp_route | opp_near);
        let canonical = |sq: usize| if self.g.turn == 0 { sq } else { NET_MIRC[sq] };
        let sum_bits = |mut bits: u128, weights: &[f64]| {
            let mut sum = 0.0;
            while bits != 0 {
                let bit = bits.trailing_zeros();
                bits &= bits - 1;
                if let Some(sq) = flood_sq_from_bit(bit) {
                    sum += weights[canonical(sq as usize)];
                }
            }
            sum
        };
        let score = sum_bits(me_route, &nw.route_me)
            + sum_bits(opp_route, &nw.route_opp)
            + sum_bits(me_near, &nw.route_near_me)
            + sum_bits(opp_near, &nw.route_near_opp)
            + sum_bits(contested, &nw.route_contested);
        (score, route0, route1)
    }

    /// Count legal walls from cache that cross the given route u128 bitset (flood-indexed).
    fn legal_path_crossing_counts_bits(&mut self, route0: u128, route1: u128) -> (u32, u32) {
        crate::bench_instr::record(
            |b| &mut b.eval_legal_wall_count,
            || self.legal_path_crossing_counts_bits_inner(route0, route1),
        )
    }

    fn legal_path_crossing_counts_bits_inner(&mut self, route0: u128, route1: u128) -> (u32, u32) {
        if let Some(bridge) = self.bridge.as_mut() {
            let _ = geometric_wall_len_cached(
                &mut bridge.geometric_walls,
                &mut bridge.board,
                &mut bridge.bfs,
                GeometricWallCacheRole::Eval,
                Some(&mut bridge.wall_cache_stats),
            );
            let cache = bridge.geometric_walls.as_ref().unwrap();
            return (
                wall_crossing_count(cache.wall_slice(), route0),
                wall_crossing_count(cache.wall_slice(), route1),
            );
        }
        let mut board = Board::new();
        for i in 0..self.g.hist_len {
            let _ = board.make_move(move_id_to_board(self.g.hist_m[i]));
        }
        let mut scratch = BfsScratch::new();
        let mut cache_opt: Option<GeometricWallCache> = None;
        geometric_wall_len_cached(
            &mut cache_opt,
            &mut board,
            &mut scratch,
            GeometricWallCacheRole::Eval,
            None,
        );
        if let Some(ref cache) = cache_opt {
            (
                wall_crossing_count(cache.wall_slice(), route0),
                wall_crossing_count(cache.wall_slice(), route1),
            )
        } else {
            (0, 0)
        }
    }

    /// Count legal walls crossing each player's path (for eval JSON). Uses [u8;81] route arrays.
    fn legal_path_crossing_counts_arr(
        &mut self,
        route0: &[u8; 81],
        route1: &[u8; 81],
    ) -> (u32, u32) {
        if let Some(bridge) = self.bridge.as_mut() {
            let _ = geometric_wall_len_cached(
                &mut bridge.geometric_walls,
                &mut bridge.board,
                &mut bridge.bfs,
                GeometricWallCacheRole::Eval,
                Some(&mut bridge.wall_cache_stats),
            );
            let cache = bridge.geometric_walls.as_ref().unwrap();
            return (
                wall_crossing_count_arr(cache.wall_slice(), route0),
                wall_crossing_count_arr(cache.wall_slice(), route1),
            );
        }
        let mut board = Board::new();
        for i in 0..self.g.hist_len {
            let _ = board.make_move(move_id_to_board(self.g.hist_m[i]));
        }
        let mut scratch = BfsScratch::new();
        let mut cache_opt: Option<GeometricWallCache> = None;
        geometric_wall_len_cached(
            &mut cache_opt,
            &mut board,
            &mut scratch,
            GeometricWallCacheRole::Eval,
            None,
        );
        if let Some(ref cache) = cache_opt {
            (
                wall_crossing_count_arr(cache.wall_slice(), route0),
                wall_crossing_count_arr(cache.wall_slice(), route1),
            )
        } else {
            (0, 0)
        }
    }

    fn refresh_dist(&mut self, ply: usize) {
        crate::bench_instr::record(|b| &mut b.refresh_dist, || self.refresh_dist_inner(ply))
    }

    fn refresh_dist_inner(&mut self, ply: usize) {
        let stamp = self.g.wall_stamp;
        if self.cached_stamp == stamp {
            return; // refs already valid for these walls
        }
        if self.cached_stamp == stamp - 1 && self.g.hist_len > 0 {
            // exactly one wall added since the cached config: slots hold its dists.
            // recompute a player's field only if the wall cuts a shortest-path edge
            // (|dist diff| === 1); equal-dist edges lie on no shortest path.
            let m = self.g.hist_m[self.g.hist_len - 1];
            if m >= 100 {
                let slot = (m % 100) as usize;
                let a = (slot >> 3) * 9 + (slot & 7);
                let (b2, c2, e2) = if m < 200 {
                    (a + 9, a + 1, a + 10) // hw: two vertical edges
                } else {
                    (a + 1, a + 9, a + 10) // vw: two horizontal edges
                };
                let refresh0 = {
                    let d0 = &self.d0[self.dist0_idx];
                    d0[a] != d0[b2] || d0[c2] != d0[e2]
                };
                let refresh1 = {
                    let d1 = &self.d1[self.dist1_idx];
                    d1[a] != d1[b2] || d1[c2] != d1[e2]
                };
                let masks = if refresh0 || refresh1 {
                    Some(DirMasks::from_ace_game(&self.g))
                } else {
                    None
                };
                if refresh0 {
                    self.dist0_idx = ply; // redirect first: never write an ancestor's array
                    fill_ace_dist_to_goal_with_masks(
                        0,
                        masks.expect("refresh masks"),
                        &mut self.d0[ply],
                    );
                    if self.net.route_active {
                        fill_distance_layers(&self.d0[ply], &mut self.d0_layers[ply]);
                    }
                }
                if refresh1 {
                    self.dist1_idx = ply;
                    fill_ace_dist_to_goal_with_masks(
                        1,
                        masks.expect("refresh masks"),
                        &mut self.d1[ply],
                    );
                    if self.net.route_active {
                        fill_distance_layers(&self.d1[ply], &mut self.d1_layers[ply]);
                    }
                }
                self.cached_stamp = stamp;
                return;
            }
        }
        self.dist0_idx = ply; // own arrays: ancestors stay intact
        self.dist1_idx = ply;
        let masks = DirMasks::from_ace_game(&self.g);
        crate::bench_instr::record(
            |b| &mut b.shortest_path,
            || {
                fill_ace_dist_to_goal_with_masks(0, masks, &mut self.d0[ply]);
                fill_ace_dist_to_goal_with_masks(1, masks, &mut self.d1[ply]);
                if self.net.route_active {
                    fill_distance_layers(&self.d0[ply], &mut self.d0_layers[ply]);
                    fill_distance_layers(&self.d1[ply], &mut self.d1_layers[ply]);
                }
            },
        );
        self.cached_stamp = stamp;
    }

    /// RaceProof: race table for the CURRENT wall config — LRU slot index, or
    /// `None` when the in-tree solve budget gates the build (JS `raceTbl`).
    /// Key = position hash with pawns and turn XORed out (wall config only).
    fn race_tbl(&mut self, force: bool) -> Option<usize> {
        let z = &ZOBRIST;
        let mut k_lo = self.g.hash_lo ^ z.pawn_lo[0][self.g.pawn[0]] ^ z.pawn_lo[1][self.g.pawn[1]];
        let mut k_hi = self.g.hash_hi ^ z.pawn_hi[0][self.g.pawn[0]] ^ z.pawn_hi[1][self.g.pawn[1]];
        if self.g.turn == 1 {
            k_lo ^= z.turn_lo;
            k_hi ^= z.turn_hi;
        }
        let li = self.rc_last;
        if li >= 0 && self.rc_key_lo[li as usize] == k_lo && self.rc_key_hi[li as usize] == k_hi {
            self.rc_hits += 1;
            return Some(li as usize);
        }
        if !force && self.rc_blocked && k_lo == self.rc_miss_lo && k_hi == self.rc_miss_hi {
            self.rc_budget_miss += 1;
            return None;
        }
        for i in 0..RC_SLOTS {
            if self.rc_tbl[i].is_some() && self.rc_key_lo[i] == k_lo && self.rc_key_hi[i] == k_hi {
                self.rc_last = i as i32;
                self.rc_tick += 1;
                self.rc_use[i] = self.rc_tick;
                self.rc_hits += 1;
                return Some(i);
            }
        }
        if !force {
            // in-tree miss: build only when cheap to amortize (ticket16 SPRT-kill lesson)
            if !self.rp_build_ok
                || self.rc_think_solves >= self.rc_count_cap
                || (self.rc_think_solve_ms + self.rc_build_ms) as f64 > self.rc_solve_cap
                || Instant::now() + Duration::from_millis(self.rc_build_ms) > self.deadline
            {
                self.rc_blocked = true;
                self.rc_miss_lo = k_lo;
                self.rc_miss_hi = k_hi;
                self.rc_budget_miss += 1;
                return None;
            }
            self.rc_think_solves += 1;
        }
        let mut slot = 0usize;
        let mut min_use = u64::MAX;
        for i in 0..RC_SLOTS {
            if self.rc_tbl[i].is_none() {
                slot = i;
                break;
            }
            if self.rc_use[i] < min_use {
                min_use = self.rc_use[i];
                slot = i;
            }
        }
        let mut tbl = self.rc_tbl[slot]
            .take()
            .unwrap_or_else(|| vec![0i16; RACE_STATES].into_boxed_slice());
        if self.race_scratch.is_none() {
            self.race_scratch = Some(Box::new(RaceScratch::new()));
        }
        let t0 = Instant::now();
        solve_race_config(
            &mut self.g,
            self.race_scratch.as_mut().expect("race scratch"),
            &mut tbl,
        );
        let dt0 = t0.elapsed().as_millis() as u64;
        self.rc_solve_ms += dt0;
        self.rc_think_solve_ms += dt0;
        let dt = dt0 + 1;
        if dt > self.rc_build_ms {
            self.rc_build_ms = dt.min(50); // conservative adaptive gate, capped
        }
        self.rc_tbl[slot] = Some(tbl);
        self.rc_key_lo[slot] = k_lo;
        self.rc_key_hi[slot] = k_hi;
        self.rc_tick += 1;
        self.rc_use[slot] = self.rc_tick;
        self.rc_last = slot as i32;
        self.rc_solves += 1;
        Some(slot)
    }

    /// Race-table value for the game's current state (helper around a slot).
    #[inline]
    fn race_value(&self, slot: usize) -> i16 {
        let idx = (self.g.pawn[0] * 81 + self.g.pawn[1]) * 2 + self.g.turn;
        self.rc_tbl[slot].as_ref().expect("race slot")[idx]
    }

    /// RaceProof(c): budget-capped static win certificate for side `s` at the
    /// current position (`certify_win.js` 'all' mode = sound). Memoized; gen13
    /// runs it in node AND browser (cf. `certify.rs`). 1:1 with JS `certWin`.
    ///
    /// Memo: `1` = proven (permanent, sound). A failure is stored as `-work`
    /// (work = certify nodes burned, else the budget); it answers `false` only
    /// for weaker-or-equal retries (`bud <= work`); a richer call (bigger
    /// budget / fresh deadline) re-runs instead of inheriting a starved failure.
    fn cert_win(&mut self, s: usize, budget: u64, deadline_ms: u64) -> bool {
        // Grafted fast path: hands-empty is a pure pawn race — Titanium's tempo
        // classifier resolves it exactly (deterministic in the common case, a tiny
        // forward race-minimax only when paths overlap within 1 tempo). Sound: with
        // no walls the win-certificate reduces to the race outcome, so this returns
        // the same verdict as `certify` at a fraction of the node cost.
        if self.g.wl[0] == 0 && self.g.wl[1] == 0 {
            use crate::titanium::cert_bridge::hands_empty_race_stm_wins;
            if let Some(stm_wins) = hands_empty_race_stm_wins(&mut self.g) {
                return if s == self.g.turn {
                    stm_wins
                } else {
                    !stm_wins
                };
            }
        }
        let key = (
            self.g.hash_lo,
            self.g.hash_hi,
            s,
            self.g.wl[0],
            self.g.wl[1],
        );
        let bud: i64 = if budget == 0 { 2500 } else { budget as i64 };
        let prior = self.cw_cache.get(&key).copied();
        if let Some(c) = prior {
            if c == 1 {
                return true; // proven: permanent (sound)
            }
            if bud <= -(c as i64) {
                return false; // weaker-or-equal retry of a recorded failure
            }
            // richer retry: fall through and re-run
        }
        self.cw_calls += 1;
        self.cw_think_calls += 1;
        let deadline = if deadline_ms > 0 {
            Some(Instant::now() + Duration::from_millis(deadline_ms))
        } else {
            None
        };
        let report = certify(
            &mut self.g,
            &CertifyOpts {
                budget: bud as u64,
                deadline,
                mode_pruned: false,
                slack: 2,
                side: Some(s),
                recommit: true,
            },
        );
        let res = report.proven == Some(s);
        let mut work = bud;
        if !res && report.nodes > 0 {
            work = report.nodes as i64; // deadline-starved: stamp only work done
        }
        if res {
            self.cw_proven += 1;
        }
        if self.cw_cache.len() > 16384 {
            self.cw_cache.clear();
        }
        if !res {
            if let Some(c) = prior {
                if -(c as i64) > work {
                    work = -(c as i64); // never weaken a recorded failure
                }
            }
        }
        self.cw_cache
            .insert(key, if res { 1 } else { -(work as i32) });
        res
    }

    /// Materialize the existing HalfPW child representation without computing
    /// route fields, legal-wall count, or the value projection. Probe/shadow only.
    fn current_hidden_features(&mut self) -> [f64; NET_H] {
        let nw = self.net;
        let b0 = NET_BKT[self.g.pawn[0]] as i32;
        let b1 = NET_BKT[NET_MIRC[self.g.pawn[1]]] as i32;
        if b0 != self.np_b0 || b1 != self.np_b1v {
            self.np_acc0.fill(0.0);
            self.np_acc1.fill(0.0);
            for s in 0..64 {
                if self.g.hw[s] != 0 {
                    let o0 = (b0 as usize * 128 + s) * NET_H;
                    let o1 = (b1 as usize * 128 + NET_MIRS[s]) * NET_H;
                    for j in 0..NET_H {
                        self.np_acc0[j] += nw.w1c[o0 + j];
                        self.np_acc1[j] += nw.w1c[o1 + j];
                    }
                }
                if self.g.vw[s] != 0 {
                    let o0 = (b0 as usize * 128 + 64 + s) * NET_H;
                    let o1 = (b1 as usize * 128 + 64 + NET_MIRS[s]) * NET_H;
                    for j in 0..NET_H {
                        self.np_acc0[j] += nw.w1c[o0 + j];
                        self.np_acc1[j] += nw.w1c[o1 + j];
                    }
                }
                self.np_hw[s] = self.g.hw[s];
                self.np_vw[s] = self.g.vw[s];
            }
            self.np_b0 = b0;
            self.np_b1v = b1;
        } else {
            for s in 0..64 {
                if self.g.hw[s] != self.np_hw[s] {
                    let sign = if self.g.hw[s] != 0 { 1.0 } else { -1.0 };
                    let o0 = (b0 as usize * 128 + s) * NET_H;
                    let o1 = (b1 as usize * 128 + NET_MIRS[s]) * NET_H;
                    for j in 0..NET_H {
                        self.np_acc0[j] += sign * nw.w1c[o0 + j];
                        self.np_acc1[j] += sign * nw.w1c[o1 + j];
                    }
                    self.np_hw[s] = self.g.hw[s];
                }
                if self.g.vw[s] != self.np_vw[s] {
                    let sign = if self.g.vw[s] != 0 { 1.0 } else { -1.0 };
                    let o0 = (b0 as usize * 128 + 64 + s) * NET_H;
                    let o1 = (b1 as usize * 128 + 64 + NET_MIRS[s]) * NET_H;
                    for j in 0..NET_H {
                        self.np_acc0[j] += sign * nw.w1c[o0 + j];
                        self.np_acc1[j] += sign * nw.w1c[o1 + j];
                    }
                    self.np_vw[s] = self.g.vw[s];
                }
            }
        }

        let mut hidden = [0.0; NET_H];
        if self.g.turn == 0 {
            let po = self.g.pawn[0] * NET_H;
            let px = self.g.pawn[1] * NET_H;
            for j in 0..NET_H {
                hidden[j] =
                    (nw.b1[j] + self.np_acc0[j] + nw.po[po + j] + nw.px[px + j]).clamp(0.0, 1.0);
            }
        } else {
            let po = NET_MIRC[self.g.pawn[1]] * NET_H;
            let px = NET_MIRC[self.g.pawn[0]] * NET_H;
            for j in 0..NET_H {
                hidden[j] =
                    (nw.b1[j] + self.np_acc1[j] + nw.po[po + j] + nw.px[px + j]).clamp(0.0, 1.0);
            }
        }
        hidden
    }

    /// Static/quiescence eval. `depth <= 0` = leaf (cert oracle eligible when gated).
    fn evaluate(&mut self, depth: i32) -> i32 {
        let _eval_timer = crate::bench_instr::OpTimer::start(|b| &mut b.evaluate);
        let me = self.g.turn;
        let opp = 1 - me;
        let d_me_u = if me == 0 {
            self.d0[self.dist0_idx][self.g.pawn[0]]
        } else {
            self.d1[self.dist1_idx][self.g.pawn[1]]
        };
        let d_opp_u = if opp == 0 {
            self.d0[self.dist0_idx][self.g.pawn[0]]
        } else {
            self.d1[self.dist1_idx][self.g.pawn[1]]
        };
        let w_me_i = self.g.wl[me];
        let w_opp_i = self.g.wl[opp];
        let d_me_i = d_me_u as i32;
        let d_opp_i = d_opp_u as i32;
        let mut hands_empty_loss_floor = false;
        if w_me_i == 0 && w_opp_i == 0 && (!self.cert_eval_leaves_only || depth <= 0) {
            // Service A (cheap_cert): SOUND fast race bound first. When the pawns'
            // shortest-path sets are separated the race is a pure tempo race and
            // the outcome is decided WITHOUT building the exact graph — the common
            // case. (The old `hands_empty_race` classifier here was unsound on
            // walled boards: it could false-certify an overlapping race; see
            // race.rs notes.) Only the overlapping minority (`Unknown`) falls
            // through to the exact retrograde table below.
            if self.cheap_cert {
                if self.race_scratch.is_none() {
                    self.race_scratch = Some(Box::new(RaceScratch::new()));
                }
                let bound = crate::bench_instr::record(
                    |b| &mut b.eval_race_bound,
                    || {
                        let scratch = self.race_scratch.as_mut().expect("race scratch");
                        race_outcome(&mut self.g, scratch)
                    },
                );
                match bound {
                    // Proven win: graded by own distance (faster = higher). The
                    // GUARANTEE is the bound; the magnitude is an ordering hint.
                    RaceBound::Lower(_) => return RACE_MATE - d_me_i.max(1),
                    // Proven loss: sign-correct loss floor (net eval grades it).
                    RaceBound::Upper(_) => hands_empty_loss_floor = true,
                    RaceBound::Exact(_) | RaceBound::Unknown => {}
                }
            }
            // Exact retrograde table (Service B) — for the overlapping minority and
            // for faithful (non-cheap_cert) race_proof engines. Never substitutes
            // BFS distance for loss depth (that mis-counts jump races and collapses
            // stubborn-loser plies). Budget-gated; built at most a few times/think.
            if !hands_empty_loss_floor && self.race_proof {
                if let Some(slot) = self.race_tbl(false) {
                    let rv = self.race_value(slot) as i32;
                    if rv > 0 {
                        return RACE_MATE - rv; // proven win in rv plies (faster = higher)
                    }
                    if rv < 0 {
                        return -(RACE_MATE + rv); // proven loss in -rv plies (slower = higher)
                    }
                    // rv==0 is unresolved by this table pass. Quoridor has no
                    // draws in our ruleset, so never score it as 0; fall through.
                }
            }
            if !hands_empty_loss_floor {
                // no table / unresolved: naive heuristic race
                if d_me_i <= d_opp_i {
                    return 3000 + (d_opp_i - d_me_i) * 50 - d_me_i;
                }
                return -3000 - (d_me_i - d_opp_i) * 50 + d_opp_i;
            }
        }

        let d_me = d_me_i as f64;
        let d_opp = d_opp_i as f64;
        let w_me = w_me_i as f64;
        let w_opp = w_opp_i as f64;
        let nw = self.net;
        let ws = &nw.ws;

        let mut out = crate::bench_instr::record(
            |b| &mut b.eval_misc_scalar,
            || {
                let pd = d_opp - d_me;
                let wd = w_me - w_opp;
                let mut out = ws[0]
                    + ws[1] * pd
                    + ws[2] * wd
                    + ws[3] * d_me
                    + ws[4] * d_opp
                    + ws[9] * pd * (w_me + w_opp) / 20.0
                    + ws[10] * wd * (d_me + d_opp) / 16.0;
                if w_opp_i == 0 {
                    out += ws[6];
                    if d_me <= d_opp {
                        out += ws[5];
                    }
                } else if w_me_i == 0 {
                    out += ws[8];
                    if d_opp <= d_me - 1.0 {
                        out += ws[7];
                    }
                }
                if d_opp <= 4.0 {
                    out += ws[11] * if w_me < 3.0 { w_me } else { 3.0 };
                }
                if d_me <= 4.0 {
                    out += ws[12] * if w_opp < 3.0 { w_opp } else { 3.0 };
                }
                out += ws[13] * pd * w_opp / 10.0;
                out
            },
        );
        let (route_score, route0_bits, route1_bits) = self.route_feature_score(nw);
        out += route_score;
        // CAT impact heatmap as a direct net input plane. Zeroed in legacy weights
        // (loader zero-pads) → `cat_active` false → NOT computed, so the live net is
        // byte-for-byte unaffected. Retrain-ready: a blob with learned `cat_heat`
        // weights activates it, giving the net the combined CAT signal alongside the
        // atomic route/near/contested planes (so it needn't reconstruct CAT itself).
        if nw.cat_active {
            if let Some(bridge) = self.bridge.as_ref() {
                let cat = crate::cat::build::build_impact_heatmap(&bridge.board);
                let canonical = |sq: usize| if me == 0 { sq } else { NET_MIRC[sq] };
                let mut cat_score = 0.0;
                for sq in 0..81usize {
                    let h = cat[sq];
                    if h != 0 {
                        cat_score += nw.cat_heat[canonical(sq)] * (f64::from(h) / 256.0);
                    }
                }
                out += cat_score;
            }
        }
        // ws[14]: unrealized wall action capacity (path-valid slots / 128). NOT corridor width.
        let legal_wall_norm = self.geometric_legal_wall_count() as f64 / 128.0;
        // ws[15]: opponent corridor width on their goal field (matches halfpw.py).
        let width_opp = if me == 0 {
            self.d1[self.dist1_idx]
                .iter()
                .filter(|&&d| d as i32 == d_opp_i)
                .count()
        } else {
            self.d0[self.dist0_idx]
                .iter()
                .filter(|&&d| d as i32 == d_opp_i)
                .count()
        } as f64;
        out += ws[14] * legal_wall_norm + ws[15] * width_opp;
        // ws[16]: legal walls crossing my path / 128; ws[17]: crossing opp path / 128
        let (route_me_bits, route_opp_bits) = if me == 0 {
            (route0_bits, route1_bits)
        } else {
            (route1_bits, route0_bits)
        };
        let (cross_me, cross_opp) = crate::bench_instr::record(
            |b| &mut b.eval_wall_cross,
            || self.legal_path_crossing_counts_bits(route_me_bits, route_opp_bits),
        );
        out += ws[16] * cross_me as f64 / 128.0 + ws[17] * cross_opp as f64 / 128.0;
        // CAT eval features (ws[18]/ws[19]) are DECOUPLED: computing them per leaf
        // (a full corridor-attention build + two legal-movegen passes in
        // best_pawn_cat_heats) was ~⅔ of total search time. CAT is being rebuilt
        // as a cheap BFF heatmap used for LMR move ordering only. Until that lands
        // and the net is retrained on CAT-free data, the two CAT inputs are 0 —
        // i.e. the live net runs as a non-CAT NNUE. Sanity check: if the engine
        // opens with a non-pawn move, the net needs retraining.
        // out += ws[18] * cat_me / 256.0 + ws[19] * cat_opp / 256.0;  // re-enable post-retrain

        let b0 = NET_BKT[self.g.pawn[0]] as i32;
        let b1 = NET_BKT[NET_MIRC[self.g.pawn[1]]] as i32;
        let _nnue_prep = crate::bench_instr::OpTimer::start(|b| &mut b.eval_nnue_prep);
        if b0 != self.np_b0 || b1 != self.np_b1v {
            crate::bench_instr::record(
                |b| &mut b.nnue_full_refresh,
                || {
                    self.np_acc0.fill(0.0);
                    self.np_acc1.fill(0.0);
                    for s in 0..64 {
                        if self.g.hw[s] != 0 {
                            let o = (b0 as usize * 128 + s) * NET_H;
                            for j in 0..NET_H {
                                self.np_acc0[j] += nw.w1c[o + j];
                            }
                            let o = (b1 as usize * 128 + NET_MIRS[s]) * NET_H;
                            for j in 0..NET_H {
                                self.np_acc1[j] += nw.w1c[o + j];
                            }
                        }
                        if self.g.vw[s] != 0 {
                            let o = (b0 as usize * 128 + 64 + s) * NET_H;
                            for j in 0..NET_H {
                                self.np_acc0[j] += nw.w1c[o + j];
                            }
                            let o = (b1 as usize * 128 + 64 + NET_MIRS[s]) * NET_H;
                            for j in 0..NET_H {
                                self.np_acc1[j] += nw.w1c[o + j];
                            }
                        }
                        self.np_hw[s] = self.g.hw[s];
                        self.np_vw[s] = self.g.vw[s];
                    }
                    self.np_b0 = b0;
                    self.np_b1v = b1;
                },
            );
        } else {
            crate::bench_instr::record(
                |b| &mut b.nnue_incr_update,
                || {
                    for s in 0..64 {
                        if self.g.hw[s] != self.np_hw[s] {
                            let sg = if self.g.hw[s] != 0 { 1.0 } else { -1.0 };
                            let o0 = (b0 as usize * 128 + s) * NET_H;
                            let o1 = (b1 as usize * 128 + NET_MIRS[s]) * NET_H;
                            for j in 0..NET_H {
                                self.np_acc0[j] += sg * nw.w1c[o0 + j];
                                self.np_acc1[j] += sg * nw.w1c[o1 + j];
                            }
                            self.np_hw[s] = self.g.hw[s];
                        }
                        if self.g.vw[s] != self.np_vw[s] {
                            let sg = if self.g.vw[s] != 0 { 1.0 } else { -1.0 };
                            let o0 = (b0 as usize * 128 + 64 + s) * NET_H;
                            let o1 = (b1 as usize * 128 + 64 + NET_MIRS[s]) * NET_H;
                            for j in 0..NET_H {
                                self.np_acc0[j] += sg * nw.w1c[o0 + j];
                                self.np_acc1[j] += sg * nw.w1c[o1 + j];
                            }
                            self.np_vw[s] = self.g.vw[s];
                        }
                    }
                },
            );
        }

        let mut hid = [0.0f64; NET_H];
        if me == 0 {
            for j in 0..NET_H {
                hid[j] = nw.b1[j] + self.np_acc0[j];
            }
            let o0 = self.g.pawn[0] * NET_H;
            for j in 0..NET_H {
                hid[j] += nw.po[o0 + j];
            }
            let o1 = self.g.pawn[1] * NET_H;
            for j in 0..NET_H {
                hid[j] += nw.px[o1 + j];
            }
        } else {
            for j in 0..NET_H {
                hid[j] = nw.b1[j] + self.np_acc1[j];
            }
            let o0 = NET_MIRC[self.g.pawn[1]] * NET_H;
            for j in 0..NET_H {
                hid[j] += nw.po[o0 + j];
            }
            let o1 = NET_MIRC[self.g.pawn[0]] * NET_H;
            for j in 0..NET_H {
                hid[j] += nw.px[o1 + j];
            }
        }
        crate::bench_instr::record(
            |b| &mut b.eval_nnue_infer,
            || {
                for j in 0..NET_H {
                    let a2 = hid[j].clamp(0.0, 1.0);
                    out += nw.w2[j] * a2 * 200.0;
                }
            },
        );
        // Integer centipawns (JS `out | 0` / halfpw `int(out)`).
        let mut ret = out as i32;
        // pathfix/RaceProof(c): certified-win floor (sound; lazy; memoized;
        // capped per think; plausibility filter dMe <= dOpp+1 — certificates
        // prove race-lead survival). gen13: RP_CERT always exists (certify_win.js
        // inlined), so unlike the v11 browser-parity port this floor is LIVE.
        // Grafted: leaf + both hands empty only (skip 1–2 wall cert in eval).
        if self.race_proof && w_me_i + w_opp_i > 0 {
            use crate::titanium::wall_ignore_cert::{
                cert_score_from_stm, try_wall_ignorance_loss_cert, wall_ignore_loss_cert_enabled,
                CertScratch,
            };
            let force =
                self.wall_ignore_cert_override.unwrap_or(false) || wall_ignore_loss_cert_enabled();
            if force {
                let mut wi_scratch = CertScratch::new();
                if let Some(verdict) =
                    try_wall_ignorance_loss_cert(&mut self.g, &mut wi_scratch, true)
                {
                    return cert_score_from_stm(&verdict, me);
                }
            }
        }
        let cert_ok = if self.cert_eval_leaves_only {
            depth <= 0 && w_me_i == 0 && w_opp_i == 0
        } else {
            w_me_i <= 2
        };
        if self.race_proof
            && cert_ok
            && ret < 2500
            && out > -700.0
            && out < 700.0
            && d_me_i <= d_opp_i + 1
        {
            let key = (
                self.g.hash_lo,
                self.g.hash_hi,
                me,
                self.g.wl[0],
                self.g.wl[1],
            );
            if (self.cw_think_calls < self.cw_cap || self.cw_cache.contains_key(&key))
                && self.cert_win(me, 1200, 0)
            {
                ret = 2500;
            }
        }
        if hands_empty_loss_floor {
            let band = ret.clamp(-CERT_BAND, CERT_BAND);
            return -CERT_WIN_SCORE + band;
        }
        ret
    }

    /// Race-root ordering: fastest win / slowest loss first; tie-break by net eval
    /// so we play the materially strongest move when plies-to-mate are equal.
    fn race_root_pick(&mut self, slot: usize, rv: i32) -> Option<(i16, i32, i32)> {
        let tbl = self.rc_tbl[slot].as_ref().expect("race slot").clone();
        let me = self.g.turn;
        let mut buf = [0i16; 16];
        let nm = self.g.gen_pawn_moves(&mut buf, 0);
        let mut best_m: i16 = -1;
        let mut best_v: i32 = 0;
        let mut best_key = i32::MIN;
        let mut best_eval = i32::MIN;
        for &c in &buf[..nm] {
            let cu = c as usize;
            let my_v = if (me == 0 && cu < 9) || (me == 1 && cu >= 72) {
                1
            } else {
                let v = tbl[if me == 0 {
                    (cu * 81 + self.g.pawn[1]) * 2 + 1
                } else {
                    (self.g.pawn[0] * 81 + cu) * 2
                }] as i32;
                if v == 0 {
                    continue;
                }
                if v > 0 {
                    -(v + 1)
                } else {
                    1 - v
                }
            };
            let key = if my_v > 0 {
                1_000_000 - my_v
            } else {
                -1_000_000 - my_v
            };
            self.g.make_move(c);
            self.refresh_dist(0);
            let d_me = if me == 0 {
                self.d0[self.dist0_idx][self.g.pawn[0]] as i32
            } else {
                self.d1[self.dist1_idx][self.g.pawn[1]] as i32
            };
            let d_opp = if me == 0 {
                self.d1[self.dist1_idx][self.g.pawn[1]] as i32
            } else {
                self.d0[self.dist0_idx][self.g.pawn[0]] as i32
            };
            let tie_eval = d_opp - d_me;
            self.g.unmake_move();
            self.cached_stamp = -1;
            if key > best_key || (key == best_key && tie_eval > best_eval) {
                best_key = key;
                best_eval = tie_eval;
                best_m = c;
                best_v = my_v;
            }
        }
        if best_m >= 0 && best_v == rv {
            Some((best_m, best_v, best_eval))
        } else {
            None
        }
    }

    fn gen_moves(&mut self, ply: usize, depth: i32, tt_move: i16, out: &mut [i16; 160]) -> usize {
        crate::bench_instr::record(
            |b| &mut b.gen_moves,
            || self.gen_moves_inner(ply, depth, tt_move, out),
        )
    }

    fn gen_moves_inner(
        &mut self,
        ply: usize,
        depth: i32,
        tt_move: i16,
        out: &mut [i16; 160],
    ) -> usize {
        let check_legal = ply == 0;
        // MoveGen+ : Titanium legal movegen at EVERY node (perft-parity search).
        // Fully legal walls — no lazy seal checks needed downstream, and inner
        // nodes can never search (or suggest via TT) a Titanium-illegal move.
        // The CAT hybrid keeps its own filtered path at inner nodes.
        if self.ti_movegen && (check_legal || (!self.cat_walls && !self.dead_zone_prune)) {
            return self
                .bridge
                .as_mut()
                .expect("ti movegen needs bridge")
                .gen_legal_ace(out);
        }
        let mut n = self.g.gen_pawn_moves(out, 0);
        if self.g.wl[self.g.turn] <= 0 {
            return n;
        }
        if self.cat_walls && !check_legal {
            return self.gen_walls_cat_filtered(depth, tt_move, out, n);
        }
        if self.dead_zone_prune && !check_legal {
            return self.gen_walls_deadzone_filtered(out, n);
        }
        for slot in 0..64 {
            if check_legal {
                if self.g.wall_legal(0, slot) {
                    out[n] = 100 + slot as i16;
                    n += 1;
                }
                if self.g.wall_legal(1, slot) {
                    out[n] = 200 + slot as i16;
                    n += 1;
                }
            } else {
                // lazy: geometry only; path-seal checked when the move is searched
                if self.g.wall_fits(0, slot) {
                    out[n] = 100 + slot as i16;
                    n += 1;
                }
                if self.g.wall_fits(1, slot) {
                    out[n] = 200 + slot as i16;
                    n += 1;
                }
            }
        }
        n
    }

    /// Hybrid wall generation: lazy geometry + CAT relevance filter.
    ///
    /// CAT (multi-route corridor heat) only above the leaf layer — depth-1 nodes
    /// dominate the tree and only need witness-path tactics, not breadth
    /// (mirrors `search::alphabeta`). The TT move always survives the filter.
    fn gen_walls_cat_filtered(
        &mut self,
        depth: i32,
        tt_move: i16,
        out: &mut [i16; 160],
        mut n: usize,
    ) -> usize {
        let me = self.g.turn;
        let our_dist = if me == 0 {
            self.d0[self.dist0_idx][self.g.pawn[0]]
        } else {
            self.d1[self.dist1_idx][self.g.pawn[1]]
        };
        let opp_dist = if me == 0 {
            self.d1[self.dist1_idx][self.g.pawn[1]]
        } else {
            self.d0[self.dist0_idx][self.g.pawn[0]]
        };
        let white_dist = if me == 0 { our_dist } else { opp_dist };
        let black_dist = if me == 0 { opp_dist } else { our_dist };
        let opp_player = if me == 0 { Player::Two } else { Player::One };

        let bridge = self.bridge.as_mut().expect("cat bridge");
        let cat = if depth >= 2 {
            bridge.bfs.build_corridor_attention(&bridge.board)
        } else {
            CorridorAttention::default()
        };
        let mut opp_path = [0u8; 81];
        let opp_path_len =
            get_shortest_path(&bridge.board, opp_player, &mut bridge.bfs, &mut opp_path);
        let reachable = bridge.bfs.both_reachable_mask(&bridge.board);
        let gap_zone = gap_play_zone_mask(reachable);
        let mut wall_candidates = [BoardMove::Pawn { row: 0, col: 0 }; 128];
        let mut wall_direct_heats = [0i32; 128];
        let mut wall_candidate_n = 0usize;

        for slot in 0..64 {
            for (wall_type, base) in [(0usize, 100i16), (1usize, 200i16)] {
                if !self.g.wall_fits(wall_type, slot) {
                    continue;
                }
                let m = base + slot as i16;
                let mv = move_id_to_board(m);
                wall_candidates[wall_candidate_n] = mv;
                wall_direct_heats[wall_candidate_n] = move_corridor_attention_with_path(
                    &mut bridge.board,
                    mv,
                    &cat,
                    white_dist,
                    black_dist,
                    &mut bridge.bfs,
                );
                wall_candidate_n += 1;
            }
        }

        for i in 0..wall_candidate_n {
            let mv = wall_candidates[i];
            let m = match mv {
                BoardMove::Wall {
                    row,
                    col,
                    orientation,
                } => {
                    let slot = i16::from(row) * 8 + i16::from(col);
                    match orientation {
                        WallOrientation::Horizontal => 100 + slot,
                        WallOrientation::Vertical => 200 + slot,
                    }
                }
                BoardMove::Pawn { .. } => continue,
            };
            let boosted_heat = move_corridor_attention_with_denial(
                &bridge.board,
                mv,
                &cat,
                &wall_candidates[..wall_candidate_n],
                &wall_direct_heats[..wall_candidate_n],
                wall_candidate_n,
            );
            let denied_hot_neighbor = boosted_heat > wall_direct_heats[i];
            let keep = m == tt_move
                || denied_hot_neighbor
                || wall_should_search(
                    mv,
                    &cat,
                    reachable,
                    gap_zone,
                    &mut bridge.board,
                    our_dist,
                    opp_dist,
                    &opp_path,
                    opp_path_len,
                    &mut bridge.bfs,
                );
            if keep {
                out[n] = m;
                n += 1;
            }
        }
        n
    }

    /// Wall generation with the SOUND dead-zone skip ONLY: emit every geometrically
    /// legal wall EXCEPT those whose every touched square is unreachable (a wall in
    /// a pure void). Those touch no pawn-reachable cell, block no path, and only
    /// burn inventory — never the best move, so pruning is NPS-only and can't cost
    /// Elo. A wall touching even one reachable square (incl. half-in-void) is kept.
    fn gen_walls_deadzone_filtered(&mut self, out: &mut [i16; 160], mut n: usize) -> usize {
        let bridge = self.bridge.as_mut().expect("dead-zone bridge");
        let reachable = bridge.bfs.both_reachable_mask(&bridge.board);
        for slot in 0..64 {
            for (wall_type, base) in [(0usize, 100i16), (1usize, 200i16)] {
                if !self.g.wall_fits(wall_type, slot) {
                    continue;
                }
                let m = base + slot as i16;
                if wall_in_dead_zone(move_id_to_board(m), reachable) {
                    continue;
                }
                out[n] = m;
                n += 1;
            }
        }
        n
    }

    fn order_moves(&self, ply: usize, moves: &mut [i16], tt_move: i16, cm_move: i16) {
        let dist_me = if self.g.turn == 0 {
            &self.d0[self.dist0_idx]
        } else {
            &self.d1[self.dist1_idx]
        };
        let k = &self.killers[ply];
        let n = moves.len();
        let mut sc = [0i32; 160];
        for i in 0..n {
            let m = moves[i];
            sc[i] = if m == tt_move {
                2_000_000_000
            } else if m < 100 {
                1_000_000 - dist_me[m as usize] as i32 * 1000
            } else if m == k[0] {
                900_000
            } else if m == cm_move {
                870_000
            } else if m == k[1] {
                850_000
            } else {
                self.history_tbl[m as usize]
            };
        }
        // stable insertion sort, descending — must match JS tie order exactly
        for a in 1..n {
            let mv = moves[a];
            let ms = sc[a];
            let mut b = a as isize - 1;
            while b >= 0 && sc[b as usize] < ms {
                moves[(b + 1) as usize] = moves[b as usize];
                sc[(b + 1) as usize] = sc[b as usize];
                b -= 1;
            }
            moves[(b + 1) as usize] = mv;
            sc[(b + 1) as usize] = ms;
        }
    }

    /// True when the current board hash already appeared in real game history
    /// (since the last wall — same rule as the in-search repetition cutoff).
    fn repeats_game_history(&self) -> bool {
        let lwp = self.g.last_wall_ply as isize;
        let mut gi = self.g.hist_len as isize * 2 - 4;
        while gi >= lwp * 2 {
            if self.g.hashes_u[gi as usize] == self.g.hash_lo
                && self.g.hashes_u[gi as usize + 1] == self.g.hash_hi
            {
                return true;
            }
            gi -= 2;
        }
        false
    }

    fn move_repeats_game_history(&mut self, m: i16) -> bool {
        self.g.make_move(m);
        let rep = self.repeats_game_history();
        self.g.unmake_move();
        rep
    }

    fn ab(
        &mut self,
        depth: i32,
        mut alpha: i32,
        beta: i32,
        ply: usize,
        allow_null: bool,
        prev_move: i16,
    ) -> Result<i32, TimeUp> {
        self.nodes += 1;
        #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
        if let Some(runtime) = self.lazy_runtime.as_ref() {
            runtime.global_nodes.fetch_add(1, Ordering::Relaxed);
        }
        self.check_time()?;
        self.sub_min[ply] = MAX_PLY as i32;
        let prev = 1 - self.g.turn;
        if (prev == 0 && self.g.pawn[0] < 9) || (prev == 1 && self.g.pawn[1] >= 72) {
            return Ok(-(MATE - ply as i32));
        }
        if ply >= MAX_PLY - 1 {
            // truncation-zero is unverified — taint ancestors (ZeroFence)
            self.sub_min[ply] = -1;
            self.sub_anc_lo[ply] = 0;
            self.sub_anc_hi[ply] = 0;
            return Ok(0);
        }
        self.path_lo[ply] = self.g.hash_lo;
        self.path_hi[ply] = self.g.hash_hi;
        if ply > 0 {
            // repetition: search line, then game history back to last wall
            for ri in (0..ply).rev() {
                if self.path_lo[ri] == self.g.hash_lo && self.path_hi[ri] == self.g.hash_hi {
                    // path-dependent zero: record the external dependency window
                    self.rep_path_c += 1;
                    if (ri as i32) < self.sub_min[ply] {
                        self.sub_min[ply] = ri as i32;
                        self.sub_anc_lo[ply] = self.g.hash_lo;
                        self.sub_anc_hi[ply] = self.g.hash_hi;
                    }
                    return Ok(0);
                }
            }
            let lwp = self.g.last_wall_ply as isize;
            let mut gi = self.g.hist_len as isize * 2 - 4;
            while gi >= lwp * 2 {
                if self.g.hashes_u[gi as usize] == self.g.hash_lo
                    && self.g.hashes_u[gi as usize + 1] == self.g.hash_hi
                {
                    // game-history rep: path-independent, no taint
                    self.rep_game_c += 1;
                    return Ok(0);
                }
                gi -= 2;
            }
        }

        self.refresh_dist(ply);
        let nd0 = self.dist0_idx; // restored on every unmake
        let nd1 = self.dist1_idx;
        let nst = self.cached_stamp;
        if depth <= 0 {
            return Ok(self.evaluate(depth));
        }

        // TT probe (typed, always-replace)
        let idx = (self.g.hash_lo & self.tt_mask) as usize;
        let mut tt_move: i16 = 0;
        #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
        let shared_entry = self
            .shared_tt
            .as_ref()
            .and_then(|tt| tt.probe(self.g.hash_lo, self.g.hash_hi));
        // Lazy SMP: when a shared TT is installed it is the ONLY TT — helper
        // workers carry no local TT (it would cost ~26MB each and blow the wasm
        // memory cap), so a shared miss must NOT fall back to the local arrays.
        #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
        let meta = match shared_entry {
            Some(entry) => entry.meta,
            None if self.shared_tt.is_some() => 0,
            None => self.tt_meta[idx],
        };
        #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
        let meta = self.tt_meta[idx];
        crate::bench_instr::bump(|b| &mut b.tt_probe);
        if meta != 0 && {
            #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
            {
                shared_entry.is_some()
                    || (self.tt_key_hi[idx] == self.g.hash_hi
                        && self.tt_key_lo[idx] == self.g.hash_lo)
            }
            #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
            {
                self.tt_key_hi[idx] == self.g.hash_hi && self.tt_key_lo[idx] == self.g.hash_lo
            }
        } {
            crate::bench_instr::bump(|b| &mut b.tt_hit);
            tt_move = (meta & 1023) as i16;
            let tdepth = meta >> 12;
            let tflag = (meta >> 10) & 3;
            if tdepth >= depth && ply > 0 {
                #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
                let mut es = match shared_entry {
                    Some(entry) => entry.score,
                    None => self.tt_score[idx],
                };
                #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
                let mut es = self.tt_score[idx]; // mate scores stored node-relative
                if es > MATE - 2 * MAX_PLY as i32 {
                    es -= ply as i32;
                } else if es < -(MATE - 2 * MAX_PLY as i32) {
                    es += ply as i32;
                }
                if (tflag == 0) || (tflag == 1 && es >= beta) || (tflag == 2 && es <= alpha) {
                    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
                    let tt_rep = match shared_entry {
                        Some(entry) => entry.rep,
                        None => self.tt_rep[idx],
                    };
                    #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
                    let tt_rep = self.tt_rep[idx];
                    if tt_rep == 0 {
                        crate::bench_instr::bump(|b| &mut b.tt_cutoff);
                        return Ok(es);
                    }
                    // tainted-zero entry: PLAIN ZeroFence ships with the anchor
                    // rescue disabled (`ghiAnchor=false` — the single min-ply
                    // anchor slot under-covers multi-dependency certificates),
                    // so a tainted entry never produces a score cutoff. The
                    // stored move is still used for ordering.
                    self.refused_cuts += 1;
                }
            }
        }

        // reverse futility: hopeless to fall below beta at shallow depth
        if depth <= 4 && beta > -2000 && beta < 2000 {
            let sev = self.evaluate(depth);
            if sev - 90 * depth >= beta {
                return Ok(sev);
            }
        }

        // null move
        if allow_null && depth >= 3 && ply > 0 {
            let ev = self.evaluate(depth);
            if ev >= beta {
                let z = &ZOBRIST;
                self.g.turn ^= 1;
                self.g.hash_lo ^= z.turn_lo;
                self.g.hash_hi ^= z.turn_hi;
                if let Some(bridge) = self.bridge.as_mut() {
                    // keep the mirrored board's side in sync (wall accounting)
                    bridge.board.side_to_move = bridge.board.side_to_move.opposite();
                }
                let res = self.ab(depth - 3, -beta, -beta + 1, ply + 1, false, 0);
                let z = &ZOBRIST;
                self.g.turn ^= 1;
                self.g.hash_lo ^= z.turn_lo;
                self.g.hash_hi ^= z.turn_hi;
                if let Some(bridge) = self.bridge.as_mut() {
                    bridge.board.side_to_move = bridge.board.side_to_move.opposite();
                }
                self.dist0_idx = nd0;
                self.dist1_idx = nd1;
                self.cached_stamp = nst;
                if self.sub_min[ply + 1] < self.sub_min[ply] {
                    self.sub_min[ply] = self.sub_min[ply + 1];
                    self.sub_anc_lo[ply] = self.sub_anc_lo[ply + 1];
                    self.sub_anc_hi[ply] = self.sub_anc_hi[ply + 1];
                }
                let ns = -res?;
                if ns >= beta && ns < MATE - 200 {
                    return Ok(beta);
                }
            }
        }

        let mut moves = [0i16; 160];
        let mut n = self.gen_moves(ply, depth, tt_move, &mut moves);
        if n == 0 {
            return Ok(self.evaluate(depth));
        }
        let cm_move = if prev_move > 0 {
            self.cm[prev_move as usize]
        } else {
            0
        };
        self.order_moves(ply, &mut moves[..n], tt_move, cm_move);
        #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
        if ply == 0 {
            if let Some(root_moves) = self.lazy_root_moves.as_ref() {
                let allowed = self
                    .lazy_root_allowed
                    .min(root_moves.len())
                    .min(moves.len());
                for (dst, src) in moves.iter_mut().zip(root_moves.iter()).take(allowed) {
                    *dst = *src;
                }
                n = allowed;
            }
        }

        let mut cat_heats = [0i32; 160];
        let mut cat_refs = CatHeatRefs::default();
        let cat_lmr_active = self.cat_lmr_v16 && depth >= 2 && n > 0;
        if cat_lmr_active {
            if let Some(bridge) = self.bridge.as_mut() {
                // Cheap BFF impact heatmap (bitboard path-set + flood) replaces both
                // the dense corridor build AND the per-move shortest-path recompute
                // (`move_corridor_attention_with_path` ran 2 BFS per wall move). A
                // move's impact is now a heatmap lookup via `wall_edge_heat`.
                let cat = crate::cat::build::build_impact_heatmap(&bridge.board);
                let mut buf = [BoardMove::Pawn { row: 0, col: 0 }; MAX_LEGAL_MOVES];
                for i in 0..n {
                    buf[i] = move_id_to_board(moves[i]);
                    // Wall inherits the hottest square it touches; pawn its destination.
                    cat_heats[i] = move_impact_heat(buf[i], &cat);
                }
                cat_refs = cat_heat_refs_from_scores(&buf[..n], n, &cat_heats[..n]);
            }
        }

        let mut best = i32::MIN; // JS -Infinity
        let mut best_move: i16 = 0;
        let mut flag = 2;

        for i in 0..n {
            let m = moves[i];
            // frontier LMP
            if depth <= 2
                && ply > 0
                && i >= 10
                && m >= 100
                && m != tt_move
                && self.history_tbl[m as usize] <= 0
                && best > -MATE + 200
            {
                continue;
            }
            // Seal check only needed for ACE's lazy pseudo-legal walls; with
            // MoveGen+ (Titanium legal gen at every node) all walls are legal.
            // The CAT and dead-zone paths both emit geometry-only (pseudo-legal)
            // walls, so they STILL need the seal check — only the pure ti_movegen
            // path (full legal gen) can skip it.
            let lazy_walls = !(self.ti_movegen && !self.cat_walls && !self.dead_zone_prune);
            if m >= 100 && ply > 0 && lazy_walls {
                let wt = if m < 200 { 0 } else { 1 };
                let slot = (m % 100) as usize;
                if self.g.wall_needs_path_check(wt, slot) {
                    self.g.set_wall_bits(wt, slot, true);
                    let paths_ok = self.g.has_path(0) && self.g.has_path(1);
                    self.g.set_wall_bits(wt, slot, false);
                    if !paths_ok {
                        continue; // sealing wall: pseudo-legal only
                    }
                }
            }
            #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
            if ply == 0 {
                let original_idx = self
                    .lazy_root_visit_map
                    .as_ref()
                    .and_then(|map| map.get(i).copied())
                    .unwrap_or(i);
                self.lazy_root_visits.push(original_idx);
            }
            let probe_parent_hash = if self.reduction_probe_enabled {
                Some((self.g.hash_lo, self.g.hash_hi))
            } else {
                None
            };
            crate::bench_instr::record(
                |b| &mut b.make_move,
                || {
                    self.g.make_move(m);
                    if let Some(bridge) = self.bridge.as_mut() {
                        bridge.push(m);
                    }
                },
            );
            let new_depth = depth - 1;
            let result = if self.eme
                && i > 0
                && i <= ACE_EME_TOP_MOVES
                && depth >= ACE_LMR_MIN_DEPTH
                && m >= 100
                && m != tt_move
            {
                // EME — extend only the top ordered walls (see ACE_EME_TOP_MOVES)
                let ext = ace_graduated_eme_extension(i, depth);
                let ed = new_depth + ext;
                self.ab(ed, -beta, -alpha, ply + 1, true, m).map(|s| -s)
            } else if i >= ACE_LMR_AFTER_MOVE
                && depth >= ACE_LMR_MIN_DEPTH
                && m >= 100
                && m != tt_move
            {
                // graduated LMR — v16 stacks CAT-scaled depth on top of ACE baseline
                let ace_red = ace_graduated_lmr_reduction(i, depth);
                let red = if cat_lmr_active {
                    let mv = move_id_to_board(m);
                    let cat_red = cat_v16_lmr_reduction_plies(
                        mv,
                        cat_heats[i],
                        cat_refs,
                        self.cat_lmr_ceiling,
                        self.cat_lmr_fringe_pct,
                        new_depth as u32,
                    ) as i32;
                    ace_red.max(cat_red)
                } else {
                    ace_red
                };
                if self.reduction_sidecar.is_some() {
                    let started = Instant::now();
                    let hidden = self.current_hidden_features();
                    let context = [
                        ((depth - 1).max(0) as f64 / 30.0).clamp(0.0, 1.0),
                        (i as f64 / 128.0).clamp(0.0, 1.0),
                        (red as f64 / 4.0).clamp(0.0, 1.0),
                        if m < 200 { 1.0 } else { 0.0 },
                        if m >= 200 { 1.0 } else { 0.0 },
                    ];
                    let sidecar = self.reduction_sidecar.as_ref().expect("checked above");
                    let probability = sidecar.predict(&hidden, &context);
                    self.reduction_shadow_stats.evaluations += 1;
                    self.reduction_shadow_stats.hypothetical_activations +=
                        u64::from(sidecar.would_activate(probability));
                    self.reduction_shadow_stats.inference_nanos +=
                        started.elapsed().as_nanos().min(u64::MAX as u128) as u64;
                }
                let probe_ordinal =
                    if self.reduction_probe_enabled && depth >= self.reduction_probe_min_depth {
                        let ordinal = self.reduction_probe_next;
                        self.reduction_probe_next += 1;
                        Some(ordinal)
                    } else {
                        None
                    };
                let extra_reduction = probe_ordinal
                    .is_some_and(|ordinal| self.reduction_probe_target == Some(ordinal));
                let rd = (new_depth - red - i32::from(extra_reduction)).max(0);
                let nodes_before = self.nodes;
                let mut verification_triggered = false;
                let pipeline_result = match self.ab(rd, -alpha - 1, -alpha, ply + 1, true, m) {
                    Ok(s) => {
                        let mut score = -s;
                        if score > alpha {
                            verification_triggered = true;
                            match self.ab(new_depth, -beta, -alpha, ply + 1, true, m) {
                                Ok(s2) => score = -s2,
                                Err(e) => {
                                    self.unwind_move(nd0, nd1, nst);
                                    return Err(e);
                                }
                            }
                        }
                        Ok(score)
                    }
                    Err(e) => Err(e),
                };
                if let (Some(ordinal), Ok(score), Some((parent_hash_lo, parent_hash_hi))) =
                    (probe_ordinal, pipeline_result.as_ref(), probe_parent_hash)
                {
                    let should_record = self.reduction_probe_events.len()
                        < self.reduction_probe_limit
                        && (self.reduction_probe_target.is_none()
                            || self.reduction_probe_target == Some(ordinal));
                    if should_record {
                        let hidden = self.current_hidden_features();
                        self.reduction_probe_events.push(ReductionProbeEvent {
                            ordinal,
                            parent_hash_lo,
                            parent_hash_hi,
                            child_hash_lo: self.g.hash_lo,
                            child_hash_hi: self.g.hash_hi,
                            mv: m,
                            depth,
                            ply,
                            alpha,
                            beta,
                            move_index: i,
                            base_reduction: red,
                            applied_extra_reduction: extra_reduction,
                            verification_triggered,
                            score: *score,
                            nodes: self.nodes.saturating_sub(nodes_before),
                            hidden,
                            total_legal_moves: n,
                            history_score: self.history_tbl[m as usize],
                        });
                    }
                }
                pipeline_result
            } else if i > 0 {
                match self.ab(new_depth, -alpha - 1, -alpha, ply + 1, true, m) {
                    Ok(s) => {
                        let mut score = -s;
                        if score > alpha && score < beta {
                            match self.ab(new_depth, -beta, -alpha, ply + 1, true, m) {
                                Ok(s2) => score = -s2,
                                Err(e) => {
                                    self.unwind_move(nd0, nd1, nst);
                                    return Err(e);
                                }
                            }
                        }
                        Ok(score)
                    }
                    Err(e) => Err(e),
                }
            } else {
                self.ab(new_depth, -beta, -alpha, ply + 1, true, m)
                    .map(|s| -s)
            };
            self.unwind_move(nd0, nd1, nst);
            if self.sub_min[ply + 1] < self.sub_min[ply] {
                self.sub_min[ply] = self.sub_min[ply + 1];
                self.sub_anc_lo[ply] = self.sub_anc_lo[ply + 1];
                self.sub_anc_hi[ply] = self.sub_anc_hi[ply + 1];
            }
            let score = result?;

            // RaceProof(b): best non-wall root alternative
            if ply == 0 && m < 100 && score > self.root_pawn_score {
                self.root_pawn_score = score;
                self.root_pawn_best = m;
            }

            let prefer_non_repeat = ply == 0
                && score == best
                && best_move != 0
                && self.move_repeats_game_history(best_move)
                && !self.move_repeats_game_history(m);

            if score > best || prefer_non_repeat {
                best = score;
                best_move = m;
                if score > alpha || prefer_non_repeat {
                    alpha = score;
                    flag = 0;
                    if ply == 0 {
                        self.root_best = m;
                        self.root_score = score;
                        // New best move at root → push an info-card update now
                        // (forced; bypasses the periodic throttle).
                        if self.stream_last_best != m {
                            self.stream_last_best = m;
                            self.stream_root_score = score;
                            self.emit_stream_progress(true);
                        }
                    }
                    if alpha >= beta {
                        flag = 1;
                        if m >= 100 {
                            if self.killers[ply][0] != m {
                                self.killers[ply][1] = self.killers[ply][0];
                                self.killers[ply][0] = m;
                            }
                            self.history_tbl[m as usize] += depth * depth;
                            if self.history_tbl[m as usize] > 100_000_000 {
                                for h in self.history_tbl.iter_mut() {
                                    *h >>= 1;
                                }
                            }
                        }
                        if prev_move > 0 {
                            self.cm[prev_move as usize] = m;
                        }
                        break;
                    }
                }
            }
        }

        if best == i32::MIN {
            return Ok(self.evaluate(depth)); // all pseudo-legal moves were sealing walls
        }
        let mut ts = best; // store mate scores node-relative
        if ts > MATE - 2 * MAX_PLY as i32 {
            ts += ply as i32;
        } else if ts < -(MATE - 2 * MAX_PLY as i32) {
            ts -= ply as i32;
        }
        // ZeroFence-A store: claim leans on an external (path-dependent) rep-0
        let mut sf = flag;
        let mut rb = 0u8;
        if self.sub_min[ply] < ply as i32 {
            if best > 0 {
                if sf == 0 {
                    sf = 1;
                    self.dg_el += 1;
                } else if sf == 2 {
                    rb = 1;
                }
            } else if best < 0 {
                if sf == 0 {
                    sf = 2;
                    self.dg_eu += 1;
                } else if sf == 1 {
                    rb = 1;
                }
            } else {
                rb = 1;
            }
            if rb != 0 {
                self.rb1_stores += 1;
            }
        }
        // Depth-preferred replacement (gen-aware when pure_mode=false).
        // Recompute idx: a child may have grown the TT (adaptive path) after our probe.
        let idx = (self.g.hash_lo & self.tt_mask) as usize;
        #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
        if let Some(shared) = self.shared_tt.as_ref() {
            crate::bench_instr::bump(|b| &mut b.tt_store);
            shared.store(
                self.g.hash_lo,
                self.g.hash_hi,
                self.tt_gen,
                self.pure_mode,
                SharedTtEntry {
                    key_hi: self.g.hash_hi,
                    key_lo: self.g.hash_lo,
                    meta: best_move as i32 | (sf << 10) | (depth << 12),
                    score: ts,
                    rep: rb,
                    anc_lo: if rb != 0 { self.sub_anc_lo[ply] } else { 0 },
                    anc_hi: if rb != 0 { self.sub_anc_hi[ply] } else { 0 },
                    entry_gen: self.tt_gen,
                },
            );
        } else {
            let was_empty = self.tt_meta[idx] == 0;
            let stale_gen = !self.pure_mode && !was_empty && self.tt_entry_gen[idx] != self.tt_gen;
            let deeper = !was_empty && !stale_gen && depth >= (self.tt_meta[idx] >> 12);
            if was_empty || stale_gen || deeper {
                crate::bench_instr::bump(|b| &mut b.tt_store);
                self.tt_key_hi[idx] = self.g.hash_hi;
                self.tt_key_lo[idx] = self.g.hash_lo;
                self.tt_meta[idx] = best_move as i32 | (sf << 10) | (depth << 12);
                self.tt_score[idx] = ts;
                self.tt_rep[idx] = rb;
                self.tt_entry_gen[idx] = self.tt_gen;
                if rb != 0 {
                    self.tt_anc_lo[idx] = self.sub_anc_lo[ply];
                    self.tt_anc_hi[idx] = self.sub_anc_hi[ply];
                }
                // Overflow-driven cache-tier growth (idx is dead after this — safe to grow).
                if was_empty {
                    self.tt_filled += 1;
                    if self.tt_adaptive
                        && self.tt_bits < self.tt_max
                        && self.tt_filled.saturating_mul(2) >= (1usize << self.tt_bits)
                    {
                        self.tt_grow();
                    }
                }
            }
        }
        #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
        {
            let was_empty = self.tt_meta[idx] == 0;
            let stale_gen = !self.pure_mode && !was_empty && self.tt_entry_gen[idx] != self.tt_gen;
            let deeper = !was_empty && !stale_gen && depth >= (self.tt_meta[idx] >> 12);
            if was_empty || stale_gen || deeper {
                crate::bench_instr::bump(|b| &mut b.tt_store);
                self.tt_key_hi[idx] = self.g.hash_hi;
                self.tt_key_lo[idx] = self.g.hash_lo;
                self.tt_meta[idx] = best_move as i32 | (sf << 10) | (depth << 12);
                self.tt_score[idx] = ts;
                self.tt_rep[idx] = rb;
                self.tt_entry_gen[idx] = self.tt_gen;
                if rb != 0 {
                    self.tt_anc_lo[idx] = self.sub_anc_lo[ply];
                    self.tt_anc_hi[idx] = self.sub_anc_hi[ply];
                }
                if was_empty {
                    self.tt_filled += 1;
                    if self.tt_adaptive
                        && self.tt_bits < self.tt_max
                        && self.tt_filled.saturating_mul(2) >= (1usize << self.tt_bits)
                    {
                        self.tt_grow();
                    }
                }
            }
        }
        Ok(best)
    }

    /// Restore after a time abort mid-move (JS `finally` semantics).
    fn unwind_move(&mut self, nd0: usize, nd1: usize, nst: i32) {
        crate::bench_instr::record(
            |b| &mut b.unmake_move,
            || {
                self.g.unmake_move();
                if let Some(bridge) = self.bridge.as_mut() {
                    bridge.pop();
                }
            },
        );
        self.dist0_idx = nd0;
        self.dist1_idx = nd1;
        self.cached_stamp = nst;
    }

    /// Entry: pathfix/RaceProof(a) — exact race endgame at ROOT. Cheap-cert
    /// engines resolve the no-wall race with the path-aware classifier plus
    /// tiny forward minimax only for volatile child states; faithful modes keep
    /// the old full race table.
    pub fn think(
        &mut self,
        time_ms: u64,
        max_depth: i32,
        full: bool,
        log: bool,
        engine_label: &str,
    ) -> ThinkResult {
        let mut stop_reason: &'static str = "unknown";
        if self.cheap_cert
            && !self.race_proof
            && self.g.wl[0] == 0
            && self.g.wl[1] == 0
            && self.g.pawn[0] >= 9
            && self.g.pawn[1] < 72
        {
            let rt0 = Instant::now();
            let me = self.g.turn;
            let mut buf = [0i16; 16];
            let nm = self.g.gen_pawn_moves(&mut buf, 0);
            let mut best_m: i16 = -1;
            let mut best_v: i32 = i32::MIN;
            let mut best_key = i32::MIN;

            for &m in &buf[..nm] {
                self.g.make_move(m);
                let moved = 1 - self.g.turn;
                let immediate_win =
                    (moved == 0 && self.g.pawn[0] < 9) || (moved == 1 && self.g.pawn[1] >= 72);

                let my_v = if immediate_win {
                    1
                } else {
                    use crate::titanium::cert_bridge::hands_empty_race_stm_wins;
                    let child_stm_wins = hands_empty_race_stm_wins(&mut self.g).unwrap_or(false);
                    let mut d_me = [255u8; 81];
                    let mut d_opp = [255u8; 81];
                    self.g.compute_dist(me, &mut d_me);
                    self.g.compute_dist(1 - me, &mut d_opp);
                    if child_stm_wins {
                        -(1 + d_opp[self.g.pawn[1 - me]] as i32)
                    } else {
                        1 + d_me[self.g.pawn[me]] as i32
                    }
                };
                self.g.unmake_move();

                // Prefer any forced win over any forced loss; among wins, take
                // the fastest race. Among losses, delay as long as possible.
                let key = if my_v > 0 {
                    1_000_000 - my_v
                } else {
                    -1_000_000 - my_v
                };
                if key > best_key {
                    best_key = key;
                    best_m = m;
                    best_v = my_v;
                }
            }

            if best_m >= 0 {
                self.rp_root_solves += 1;
                self.refresh_dist(0);
                let rk = best_v.abs();
                let score = if best_v > 0 {
                    RACE_MATE - rk
                } else {
                    -(RACE_MATE - rk)
                };
                if log {
                    emit_ace_progress(
                        engine_label,
                        &[],
                        99,
                        nm as u64,
                        score,
                        self.d0[self.dist0_idx][self.g.pawn[0]],
                        self.d1[self.dist1_idx][self.g.pawn[1]],
                        rt0.elapsed().as_millis() as u64,
                        #[cfg(feature = "wasm")]
                        self.wasm_progress.as_ref(),
                    );
                }
                return ThinkResult {
                    mv: best_m,
                    score,
                    depth: 99,
                    nodes: nm as u64,
                    main_thread_nodes: nm as u64,
                    helper_nodes: Vec::new(),
                    total_nodes: nm as u64,
                    main_completed_depth: 99,
                    helper_completed_depths: Vec::new(),
                    root_widths: Vec::new(),
                    root_visits: Vec::new(),
                    ms: rt0.elapsed().as_millis() as u64,
                    white_dist: self.d0[self.dist0_idx][self.g.pawn[0]],
                    black_dist: self.d1[self.dist1_idx][self.g.pawn[1]],
                    depth_log: Vec::new(),
                    stop_reason: "cheap_cert_root_race",
                };
            }
        }

        if self.race_proof
            && self.g.wl[0] == 0
            && self.g.wl[1] == 0
            && self.g.pawn[0] >= 9
            && self.g.pawn[1] < 72
        {
            let rt0 = Instant::now();
            // root-level: always allowed to build (force=true; deadline not set yet)
            let rv = self.race_tbl(true).map_or(0, |s| self.race_value(s)) as i32;
            if rv != 0 {
                let slot = self.rc_last as usize;
                let nm = self.g.gen_pawn_moves(&mut [0i16; 16], 0);
                if let Some((best_m, _best_v, _)) = self.race_root_pick(slot, rv) {
                    self.rp_root_solves += 1;
                    let rk = rv.abs();
                    self.refresh_dist(0);
                    return ThinkResult {
                        mv: best_m,
                        score: if rv > 0 {
                            RACE_MATE - rk
                        } else {
                            -(RACE_MATE - rk)
                        },
                        depth: 99,
                        nodes: nm as u64,
                        main_thread_nodes: nm as u64,
                        helper_nodes: Vec::new(),
                        total_nodes: nm as u64,
                        main_completed_depth: 99,
                        helper_completed_depths: Vec::new(),
                        root_widths: Vec::new(),
                        root_visits: Vec::new(),
                        ms: rt0.elapsed().as_millis() as u64,
                        white_dist: self.d0[self.dist0_idx][self.g.pawn[0]],
                        black_dist: self.d1[self.dist1_idx][self.g.pawn[1]],
                        depth_log: Vec::new(),
                        stop_reason: "race_proof_root_table",
                    };
                }
            }
        }
        self.think_search(
            time_ms,
            max_depth,
            full,
            log,
            engine_label,
            &mut stop_reason,
        )
    }

    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    pub fn think_with_threads(
        &mut self,
        time_ms: u64,
        max_depth: i32,
        full: bool,
        log: bool,
        engine_label: &str,
        threads: usize,
    ) -> ThinkResult {
        if threads <= 1 {
            self.shared_tt = None;
            self.lazy_runtime = None;
            self.lazy_root_moves = None;
            self.lazy_root_visit_map = None;
            return self.think(time_ms, max_depth, full, log, engine_label);
        }
        let threads = threads.min(LAZY_SMP_MAX_THREADS);
        self.think_lazy_smp(time_ms, max_depth, full, log, engine_label, threads)
    }

    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    fn lazy_smp_helper_partial<'a>(
        main_result: &ThinkResult,
        helper_results: &'a [(usize, ThinkResult, Vec<usize>)],
        root_moves_raw: &[i16],
    ) -> Option<&'a ThinkResult> {
        if main_result.depth > 0 && main_result.mv != super::TITANIUM_NO_MOVE {
            return None;
        }
        helper_results
            .iter()
            .map(|(_, result, _)| result)
            .filter(|result| {
                result.depth > 0
                    && result.mv != super::TITANIUM_NO_MOVE
                    && root_moves_raw.contains(&result.mv)
            })
            .max_by_key(|result| (result.depth, result.nodes))
    }

    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    fn think_lazy_smp(
        &mut self,
        time_ms: u64,
        max_depth: i32,
        full: bool,
        log: bool,
        engine_label: &str,
        threads: usize,
    ) -> ThinkResult {
        if self.g.winner() >= 0 {
            return self.think(time_ms, max_depth, full, log, engine_label);
        }

        // Parallel search uses a fixed shared allocation. Live adaptive growth is
        // intentionally disabled because resizing a TT while other workers probe
        // it would invalidate the shared slots.
        if self.shared_tt.is_none() && self.tt_bits < TT_BITS {
            self.resize_tt(TT_BITS);
        }
        self.tt_adaptive = false;
        self.apply_think_start_state();

        let depth_limit = if max_depth > 0 { max_depth } else { 30 };
        let root_moves_raw = self.ordered_root_moves_snapshot(depth_limit);
        if root_moves_raw.is_empty() {
            return self.think(time_ms, max_depth, full, log, engine_label);
        }
        let root_position = self.g.clone();
        let shared_tt = self
            .shared_tt
            .clone()
            .unwrap_or_else(|| Arc::new(SharedTitaniumTt::from_search(self)));
        let deadline = Instant::now() + Duration::from_millis(time_ms.max(1));
        let runtime = Arc::new(LazySmpRuntime::new(deadline));
        let plans: Vec<WorkerPlan> = (0..threads)
            .map(|worker_id| WorkerPlan {
                worker_id,
                root_move_count: root_moves_raw.len(),
                root_width_percent: Self::lazy_smp_width_percent(worker_id),
            })
            .collect();

        #[cfg(not(target_arch = "wasm32"))]
        let mut helper_results: Vec<(usize, ThinkResult, Vec<usize>)> = Vec::new();
        let main_allowed = plans[0].allowed_root_moves();
        let (main_root_moves, main_visit_map) =
            Self::lazy_smp_profile_root_moves(&root_moves_raw, 0, main_allowed);
        self.install_lazy_smp_context(
            0,
            shared_tt.clone(),
            runtime.clone(),
            Arc::new(main_root_moves),
            Arc::new(main_visit_map),
            main_allowed,
        );

        let helper_workers: Vec<(WorkerPlan, Box<TitaniumSearch>)> = plans
            .iter()
            .copied()
            .skip(1)
            .map(|plan| {
                let mut worker = self.fork_lazy_worker(&root_position);
                let allowed = plan.allowed_root_moves();
                let (profiled_root_moves, visit_map) =
                    Self::lazy_smp_profile_root_moves(&root_moves_raw, plan.worker_id, allowed);
                worker.install_lazy_smp_context(
                    plan.worker_id,
                    shared_tt.clone(),
                    runtime.clone(),
                    Arc::new(profiled_root_moves),
                    Arc::new(visit_map),
                    allowed,
                );
                (plan, worker)
            })
            .collect();

        #[cfg(not(target_arch = "wasm32"))]
        let mut main_result = std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(threads.saturating_sub(1));
            for (plan, mut worker) in helper_workers {
                handles.push(scope.spawn(move || {
                    let mut stop_reason = "unknown";
                    let result = worker.think_search(
                        time_ms,
                        max_depth,
                        full,
                        false,
                        engine_label,
                        &mut stop_reason,
                    );
                    (plan.worker_id, result, worker.lazy_root_visits)
                }));
            }

            let mut stop_reason = "unknown";
            let main_result = self.think_search(
                time_ms,
                max_depth,
                full,
                log,
                engine_label,
                &mut stop_reason,
            );
            runtime.stop.store(true, Ordering::Relaxed);

            for handle in handles {
                if let Ok(result) = handle.join() {
                    helper_results.push(result);
                }
            }
            main_result
        });

        #[cfg(all(target_arch = "wasm32", feature = "wasm-threads"))]
        let helper_results_shared =
            Arc::new(Mutex::new(Vec::with_capacity(threads.saturating_sub(1))));
        // Lazy SMP on wasm: dispatch helper searches with fire-and-forget
        // `rayon::spawn` onto the wasm-bindgen-rayon pool, run the main search on
        // this (seat) worker thread, then wait on an atomic completion latch.
        //
        // We deliberately AVOID `rayon::scope`/`join`: their join blocks the
        // external (non-pool) seat-worker thread on a Condvar that does not wake
        // under wasm-bindgen-rayon, which deadlocked every threaded search
        // The shared `stop` flag makes helpers return promptly once main
        // finishes, so the latch spin-wait below is brief.
        #[cfg(all(target_arch = "wasm32", feature = "wasm-threads"))]
        let mut main_result = {
            let pending = Arc::new(AtomicUsize::new(helper_workers.len()));
            let engine_label_owned = engine_label.to_string();
            for (plan, mut worker) in helper_workers {
                let helper_results_shared = helper_results_shared.clone();
                let pending = pending.clone();
                let engine_label = engine_label_owned.clone();
                rayon::spawn(move || {
                    crate::wasm::note_helper_start();
                    let mut stop_reason = "unknown";
                    let result = worker.think_search(
                        time_ms,
                        max_depth,
                        full,
                        false,
                        &engine_label,
                        &mut stop_reason,
                    );
                    helper_results_shared.lock().expect("helper results").push((
                        plan.worker_id,
                        result,
                        worker.lazy_root_visits,
                    ));
                    pending.fetch_sub(1, Ordering::Release);
                });
            }

            let mut stop_reason = "unknown";
            let main_result = self.think_search(
                time_ms,
                max_depth,
                full,
                log,
                engine_label,
                &mut stop_reason,
            );
            runtime.stop.store(true, Ordering::Relaxed);
            // Helpers observe `stop` in check_time() and return within ~1k nodes.
            // The seat worker is a Web Worker (not the UI thread), so a brief spin
            // here is safe. Bounded by a generous fallback deadline so a wedged
            // helper can never hang the move forever.
            let latch_deadline = Instant::now() + Duration::from_millis(time_ms.max(1) + 2000);
            while pending.load(Ordering::Acquire) > 0 && Instant::now() < latch_deadline {
                std::hint::spin_loop();
            }
            main_result
        };

        // Drain the collected helper results by locking — NOT `Arc::try_unwrap`,
        // which races: a helper decrements `pending` (its last statement) before
        // its captured `Arc` clone drops, so once the latch sees `pending == 0`
        // the clones may still be alive and `try_unwrap` would fail, silently
        // discarding every helper result. Locking is correct regardless of how
        // many `Arc` clones remain; `pending == 0` already guarantees all pushes
        // completed.
        #[cfg(all(target_arch = "wasm32", feature = "wasm-threads"))]
        let mut helper_results = {
            let mut guard = helper_results_shared.lock().expect("helper results");
            std::mem::take(&mut *guard)
        };
        helper_results.sort_by_key(|(worker_id, _, _)| *worker_id);

        let main_completed_depth = main_result.depth;
        let main_nodes = main_result.nodes;
        if let Some(helper) =
            Self::lazy_smp_helper_partial(&main_result, &helper_results, &root_moves_raw)
        {
            main_result.mv = helper.mv;
            main_result.score = helper.score;
            main_result.depth = helper.depth;
            main_result.ms = main_result.ms.max(helper.ms);
            main_result.white_dist = helper.white_dist;
            main_result.black_dist = helper.black_dist;
            main_result.depth_log = helper.depth_log.clone();
            main_result.stop_reason = "lazy_smp_helper_partial";
        }

        let helper_nodes: Vec<u64> = helper_results.iter().map(|(_, r, _)| r.nodes).collect();
        let helper_depths: Vec<i32> = helper_results.iter().map(|(_, r, _)| r.depth).collect();
        let mut root_visits = vec![self.lazy_root_visits.clone()];
        root_visits.extend(helper_results.iter().map(|(_, _, visits)| visits.clone()));
        let total_nodes = main_nodes + helper_nodes.iter().copied().sum::<u64>();
        main_result.main_thread_nodes = main_nodes;
        main_result.helper_nodes = helper_nodes;
        main_result.total_nodes = total_nodes;
        main_result.nodes = total_nodes;
        main_result.main_completed_depth = main_completed_depth;
        main_result.helper_completed_depths = helper_depths;
        main_result.root_widths = plans;
        main_result.root_visits = root_visits;
        main_result
    }

    /// Iterative deepening within `time_ms`. `full` disables the easy-move stop.
    fn think_search(
        &mut self,
        time_ms: u64,
        max_depth: i32,
        full: bool,
        log: bool,
        engine_label: &str,
        stop_reason: &mut &'static str,
    ) -> ThinkResult {
        let t0 = Instant::now();
        crate::bench_instr::begin_search();
        // pathfix/RaceProof(b): reserve the commitment gate's worst-case cost
        // out of the search deadline when the gate can fire — it runs after
        // the search loop and its raceTbl(force=true) call ignores deadline.
        let mut gate_reserve_ms = 0u64;
        if self.race_proof && !self.cheap_cert && self.g.wl[self.g.turn] == 1 {
            let cap = (0.3 * time_ms as f64) as u64;
            gate_reserve_ms = self
                .rc_build_ms
                .max(25)
                .max((time_ms as f64 * 0.15) as u64)
                .min(cap);
        }
        // Each thread derives its deadline from its OWN monotonic clock. Under
        // wasm, `web_time::Instant` is backed by per-Worker `performance.now()`
        // origins, so a deadline created on the main thread is meaningless to a
        // rayon helper thread — it would never time out and the scope join would
        // hang. Cross-thread early-exit is handled by `LazySmpRuntime::stop`
        // (checked in `check_time`), NOT by a shared Instant. (Native clocks are
        // cross-thread comparable, so per-thread t0 is equivalent there ±µs.)
        self.deadline = t0 + Duration::from_millis(time_ms.saturating_sub(gate_reserve_ms));
        self.nodes = 0;
        self.root_best = super::TITANIUM_NO_MOVE;
        self.root_score = 0;
        #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
        let skip_setup = self.lazy_skip_setup;
        #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
        {
            self.lazy_skip_setup = false;
        }
        #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
        let skip_setup = false;
        if !skip_setup {
            // Advance TT generation and decay history once at think start.
            // Lazy SMP does this before forking workers so every worker stores
            // into the same generation and starts from the same ordered root.
            #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
            self.apply_think_start_state();
            #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
            if !self.pure_mode && !self.is_pondering {
                self.tt_gen = self.tt_gen.wrapping_add(1);
                for h in self.history_tbl.iter_mut() {
                    *h >>= 1;
                }
            }
        }
        // RaceProof per-think solve budgets + caps
        self.rc_think_solve_ms = 0;
        self.rc_solve_cap = time_ms as f64 * 0.25;
        self.rc_blocked = false;
        self.rc_think_solves = 0;
        self.rp_root_empty = self.race_proof && self.g.wl[0] == 0 && self.g.wl[1] == 0;
        self.rp_build_ok = false;
        self.stream_log = log;
        self.stream_label = engine_label.to_string();
        self.stream_t0 = t0;
        self.stream_root_score = 0;
        self.stream_search_depth = 0;
        self.stream_depth_log.clear();
        self.stream_last_emit_nodes = 0;
        self.stream_last_emit_ms = 0;
        self.stream_last_best = super::TITANIUM_NO_MOVE;
        // Re-sync the mirrored Titanium board from the authoritative ACE game.
        // Kills any drift left over from a previous search (e.g. an unbalanced
        // push/pop on time-abort) before it can poison this move's root list.
        if self.bridge.is_some() {
            self.bridge = Some(TiBridge::from_game(&self.g));
        }
        let mut last_best: i16 = super::TITANIUM_NO_MOVE;
        let mut last_score = 0;
        let mut last_depth = 0;
        let mut stable = 0;
        // RaceProof(b); -1 sentinel — pawn-move id 0 (a1) is legal
        let mut last_pawn_best: i16 = -1;
        let mut last_pawn_score: i32 = i32::MIN;
        let mut depth_log: Vec<AceDepthLogEntry> = Vec::new();
        let max_depth = if max_depth > 0 { max_depth } else { 30 };

        // Dynamic iterative-deepening startup: probe the TT for the root position.
        // If the prior think (or pondering) left a deep exact entry, skip the
        // shallow iterations we already know the answer to and resume from near
        // that depth. last_score is seeded from the TT so aspiration windows are
        // correctly centred on the first iteration we actually run.
        // Disabled in pure_mode (faithful JS baseline).
        let start_depth = if !self.pure_mode {
            let ridx = (self.g.hash_lo & self.tt_mask) as usize;
            #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
            let root_entry = self
                .shared_tt
                .as_ref()
                .and_then(|tt| tt.probe(self.g.hash_lo, self.g.hash_hi));
            // Lazy SMP shared TT is authoritative; no local fallback (see ab()).
            #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
            let rmeta = match root_entry {
                Some(entry) => entry.meta,
                None if self.shared_tt.is_some() => 0,
                None => self.tt_meta[ridx],
            };
            #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
            let rmeta = self.tt_meta[ridx];
            if rmeta != 0 && {
                #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
                {
                    root_entry.is_some()
                        || (self.tt_key_hi[ridx] == self.g.hash_hi
                            && self.tt_key_lo[ridx] == self.g.hash_lo)
                }
                #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
                {
                    self.tt_key_hi[ridx] == self.g.hash_hi && self.tt_key_lo[ridx] == self.g.hash_lo
                }
            } {
                let tt_depth = rmeta >> 12;
                let tt_flag = (rmeta >> 10) & 3;
                if tt_depth >= 4 && tt_flag == 0 {
                    // Exact score: safe to use as aspiration seed and skip iterations.
                    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
                    {
                        last_score = match root_entry {
                            Some(entry) => entry.score,
                            None => self.tt_score[ridx],
                        };
                    }
                    #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
                    {
                        last_score = self.tt_score[ridx];
                    }
                    (tt_depth - 2).max(1)
                } else {
                    1
                }
            } else {
                1
            }
        } else {
            1
        };

        for d in start_depth..=max_depth {
            if !full && d > 1 && Self::ace_over_time_budget(t0, time_ms, last_score) {
                *stop_reason = "ace_over_time_budget_before_depth";
                break;
            }
            if Instant::now() >= self.deadline {
                *stop_reason = "deadline_before_depth";
                break;
            }
            // RaceProof: in-tree solves only when cheap to amortize
            self.rp_build_ok = self.rp_root_empty || d >= 6;
            self.root_pawn_best = -1;
            self.root_pawn_score = i32::MIN;
            self.stream_root_score = last_score;
            self.stream_search_depth = d;
            let nodes_at_depth = self.nodes;
            let result = if d >= 4 && last_score > -2000 && last_score < 2000 {
                // aspiration
                let mut lo = last_score - 75;
                let mut hi = last_score + 75;
                loop {
                    match self.ab(d, lo, hi, 0, true, 0) {
                        Ok(sc) => {
                            if sc <= lo {
                                lo = -INF;
                            } else if sc >= hi {
                                hi = INF;
                            } else {
                                break Ok(sc);
                            }
                        }
                        Err(e) => break Err(e),
                    }
                }
            } else {
                self.ab(d, -INF, INF, 0, true, 0)
            };
            match result {
                Ok(sc) => {
                    stable = if self.root_best == last_best {
                        stable + 1
                    } else {
                        0
                    };
                    last_best = self.root_best;
                    last_score = sc;
                    last_depth = d;
                    if self.root_pawn_best >= 0 {
                        // RaceProof(b)
                        last_pawn_best = self.root_pawn_best;
                        last_pawn_score = self.root_pawn_score;
                    }
                    let elapsed_ms = t0.elapsed().as_millis() as u64;
                    let pv = if last_best >= 0 {
                        super::move_id_to_algebraic(last_best)
                    } else {
                        String::new()
                    };
                    depth_log.push(AceDepthLogEntry {
                        depth: d,
                        score: last_score,
                        nodes: self.nodes,
                        elapsed_ms,
                        marginal_nodes: self.nodes.saturating_sub(nodes_at_depth),
                        pv,
                    });
                    if log {
                        self.sync_stream_meta(&depth_log, d, last_score);
                        self.emit_stream_progress(true);
                    }
                    if sc > MATE - 200 || sc < -(MATE - 200) {
                        *stop_reason = "forced_mate_or_loss";
                        break; // forced result
                    }
                    // v8 easy-move stop (acev8_engine.js)
                    if !full
                        && d >= 9
                        && stable >= 3
                        && last_score > -120
                        && t0.elapsed().as_millis() as u64 > time_ms * 3 / 10
                    {
                        *stop_reason = "easy_move_stable";
                        break;
                    }
                }
                Err(TimeUp) => {
                    // Lague partial-iteration: the aborted depth-`d` iteration
                    // still searched its best-ordered root moves to full depth.
                    // `root_best` only updates after a root move's search FULLY
                    // completes (the `?` on an aborted child returns first), so it
                    // holds the best completed move — adopt it instead of falling
                    // back to depth d-1. On a pure fail-low (no alpha-raise this
                    // iteration) root_best/root_score still equal the prior depth's
                    // values, so this is a no-op exactly in the unsafe case.
                    if self.use_partial_iter && self.root_best >= 0 {
                        last_best = self.root_best;
                        last_score = self.root_score;
                        last_depth = d;
                        if self.root_pawn_best >= 0 {
                            last_pawn_best = self.root_pawn_best;
                            last_pawn_score = self.root_pawn_score;
                        }
                    }
                    *stop_reason = "time_up";
                    break; // state already restored by unwinding unmakes
                }
            }
            if !full && Self::ace_over_time_budget(t0, time_ms, last_score) {
                *stop_reason = "ace_over_time_budget_after_depth";
                break;
            }
        }
        if *stop_reason == "unknown" {
            *stop_reason = "max_depth_completed";
        }

        // ---------- pathfix/RaceProof(b): last-wall commitment gate (DEMOTE, never forbid) ----------
        // About to commit our FINAL wall: demote it below the best non-wall
        // root alternative unless the post-wall position is PROVEN won/
        // not-lost for us. When the wall empties both hands, the k=0 race
        // oracle decides (verdict <= 0 for the opponent = we are not lost).
        // gen13: otherwise use the inlined certifier with REFUTATION semantics
        // — demote ONLY on positive evidence the wall LOSES (a certificate that
        // the OPPONENT, stm after our wall, wins). The v11 browser port kept
        // the wall here unconditionally (RP_CERT was null); gen13's certify_win
        // inlining makes this branch live. Proven-mate walls and positions
        // without a pawn alternative are kept. Worst-case gate cost was
        // reserved out of the search deadline up front (gate_reserve_ms).
        if self.race_proof
            && last_best >= 100
            && self.g.wl[self.g.turn] == 1
            && last_pawn_best >= 0
            && last_score < MATE - 200
            && last_pawn_score > -(MATE - 200)
        {
            self.g.make_move(last_best);
            let rp_ok = if self.g.wl[0] == 0 && self.g.wl[1] == 0 {
                use crate::titanium::cert_bridge::hands_empty_race_stm_wins;
                match hands_empty_race_stm_wins(&mut self.g) {
                    Some(opp_wins) => !opp_wins,
                    None => true, // unknown ⇒ do not demote without proof
                }
            } else if self.cert_eval_leaves_only {
                // Walls remain: search + EME cover tempo; skip recursive certify here.
                true
            } else {
                // gen13 refutation: demote only if the opponent's win is certified.
                let deadline_ms = 25u64.max(time_ms * 15 / 100);
                !self.cert_win(self.g.turn, 60_000, deadline_ms)
            };
            self.g.unmake_move();
            self.cached_stamp = -1;
            if !rp_ok {
                self.rp_demotions += 1;
                last_best = last_pawn_best;
                last_score = last_pawn_score;
            }
        }

        // Bridge desync detector: whenever control is back at the root the
        // mirrored board's undo stack MUST be empty. If not, a make/unmake
        // path leaked a frame (this is how "illegal move" crashes happen) —
        // log it loudly and rebuild from the authoritative game.
        if let Some(bridge) = self.bridge.as_ref() {
            if !bridge.undo_stack.is_empty() {
                eprintln!(
                    "info string ace bridge DESYNC: {} unpopped frames after search — rebuilding",
                    bridge.undo_stack.len()
                );
                self.bridge = Some(TiBridge::from_game(&self.g));
            }
        }

        // Root legality guard: never emit a move the true position rejects.
        // Regenerates the legal root list from clean state; if the searched
        // best move is not in it, substitute the best legal alternative.
        self.refresh_dist(0);
        let mut legal = [0i16; 160];
        let nlegal = self.gen_moves(0, 1, last_best, &mut legal);
        let root_ok = nlegal > 0 && last_best >= 0 && legal[..nlegal].contains(&last_best);
        if !root_ok {
            if last_best >= 0 && nlegal > 0 {
                eprintln!(
                    "info string ace root guard: searched best {} is illegal in true position — substituting",
                    super::move_id_to_algebraic(last_best)
                );
            }
            if nlegal > 0 {
                self.order_moves(0, &mut legal[..nlegal], 0, 0);
                last_best = legal[0];
            } else {
                last_best = super::TITANIUM_NO_MOVE;
            }
        }

        self.refresh_dist(0);
        let white_dist = self.d0[self.dist0_idx][self.g.pawn[0]];
        let black_dist = self.d1[self.dist1_idx][self.g.pawn[1]];
        let ms = t0.elapsed().as_millis() as u64;

        if log {
            self.sync_stream_meta(&depth_log, last_depth, last_score);
            self.emit_stream_progress(true);
        }

        if std::env::var_os("TITANIUM_WALL_CACHE_STATS").is_some() {
            if let Some(s) = self.wall_cache_stats() {
                eprintln!(
                    "info string wall_cache hits_eval={} misses_eval={} hits_movegen={} misses_movegen={} wall_gen_calls={}",
                    s.hits_eval,
                    s.misses_eval,
                    s.hits_movegen,
                    s.misses_movegen,
                    s.wall_generation_calls
                );
            }
        }

        crate::bench_instr::set_stop_reason(stop_reason);
        crate::bench_instr::end_search(self.nodes);

        ThinkResult {
            mv: last_best,
            score: last_score,
            depth: last_depth,
            nodes: self.nodes,
            main_thread_nodes: self.nodes,
            helper_nodes: Vec::new(),
            total_nodes: self.nodes,
            main_completed_depth: last_depth,
            helper_completed_depths: Vec::new(),
            root_widths: Vec::new(),
            root_visits: Vec::new(),
            ms,
            white_dist,
            black_dist,
            depth_log,
            stop_reason: *stop_reason,
        }
    }
}
