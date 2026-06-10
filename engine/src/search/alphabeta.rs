//! Iterative-deepening αβ with aspiration windows, LMR, and TT.

use std::io::Write;
use std::time::Instant;

use crate::cat::constants::DIST_PENALTY;
use crate::cat::prune::{
    self, collect_search_moves, get_shortest_path, is_tactical_move, move_corridor_attention,
    move_immediate_gain, order_moves, our_path_gain, path_distance,
};
use crate::cat::CorridorAttention;
use crate::core::board::{Board, Move, Player, WallOrientation};
use crate::movegen::{generate_legal_moves_slice, MAX_LEGAL_MOVES};
use crate::opening::book::BookHint;
use crate::path::BfsScratch;
use crate::search::lmr_profile::{
    apply_depth_feedback, build_lmr_table, compute_stage_t, EvalZoneState, LmrProfile,
    MateStopReason, MateZoneState,
};
use crate::util::grid::is_goal;
use crate::util::perft::format_move;

const MATE: i32 = 20_000;
const MATE_WINDOW: i32 = 500;
/// Default iterative-deepening ceiling (u8-friendly 256). Same as recursion stack cap.
/// Do not lower this — use `SearchConfig::max_id_depth` only for explicit depth-capped runs.
pub const DEFAULT_MAX_ID_DEPTH: u32 = 256;
const MAX_PLY: u32 = DEFAULT_MAX_ID_DEPTH;
const CM_PER_SQUARE: i32 = 100;
const MAX_EVAL: i32 = 10_000;
const WALL_INVENTORY_CM: i32 = 12;
const PAWN_PROGRESS_CM: i32 = 6;
const RACE_LEAD_CM: i32 = 15;
const LOW_WALL_TRAP_CM: i32 = 18;

const LMR_MIN_DEPTH: u32 = 2;
/// Max walls expanded at the root when `stage_t` is low — rest are CAT-ranked out.
const ROOT_WALL_CAP_OPENING: usize = 26;
const ROOT_WALL_CAP_MID: usize = 38;
const ASPIRATION_DELTA: i32 = 200;
// Futility margin per depth ply in centi-squares.
// At depth 1 we allow 2.5 squares slack, at depth 2 we allow 5.0 — beyond that no futility.
const FUTILITY_MARGIN: [i32; 3] = [0, 250, 500];
const SEARCH_TT_BITS: usize = 20;
const SEARCH_TT_SIZE: usize = 1 << SEARCH_TT_BITS;
const SEARCH_TT_MASK: usize = SEARCH_TT_SIZE - 1;

pub const DEFAULT_TIME_MS: u64 = 10_000;
pub const DEFAULT_MAX_NODES: u64 = 2_000_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TtBound {
    Exact,
    Lower,
    Upper,
}

#[derive(Clone, Copy, Default)]
struct SearchTtEntry {
    key: u64,
    depth: i8,
    score: i32,
    bound: u8,
    best: u32,
}

#[derive(Default)]
struct SearchTt {
    entries: Vec<SearchTtEntry>,
}

impl SearchTt {
    fn new() -> Self {
        Self {
            entries: vec![SearchTtEntry::default(); SEARCH_TT_SIZE],
        }
    }

    fn probe(&self, key: u64) -> Option<SearchTtEntry> {
        let e = &self.entries[key as usize & SEARCH_TT_MASK];
        if e.key == key {
            Some(*e)
        } else {
            None
        }
    }

    fn store(&mut self, key: u64, depth: i8, score: i32, bound: TtBound, best: u32) {
        let slot = &mut self.entries[key as usize & SEARCH_TT_MASK];
        if slot.key != 0 && slot.key != key && slot.depth > depth {
            return;
        }
        *slot = SearchTtEntry {
            key,
            depth,
            score,
            bound: bound as u8,
            best,
        };
    }
}

#[derive(Debug, Clone)]
pub struct DepthLogEntry {
    pub depth: u32,
    pub score: i32,
    pub nodes: u64,
    /// Wall time from search start through completion of this depth (ms).
    pub elapsed_ms: u64,
    /// Nodes spent on this depth iteration only.
    pub marginal_nodes: u64,
    /// Principal variation from this depth (algebraic, space-separated).
    pub pv: String,
}

/// Per-root-move diagnostic snapshot captured at the last completed search depth.
#[derive(Debug, Clone)]
pub struct RootMoveInfo {
    /// Algebraic notation of the move.
    pub mv: String,
    /// Score from the root side's perspective (after clamp_unproven_mate).
    pub score: i32,
    /// White shortest-path distance after this move.
    pub white_dist_after: u8,
    /// Black shortest-path distance after this move.
    pub black_dist_after: u8,
    /// Immediate path gain (>0 means the move shortens our path or lengthens theirs).
    pub gain: i32,
    /// Mate distance when score is a mate score.
    pub mate_distance: Option<u32>,
    /// True if it's a pawn move, false for a wall.
    pub is_pawn: bool,
}

#[derive(Debug, Clone)]
pub struct SearchReport {
    pub best_move: Move,
    pub search_depth: u32,
    pub nodes: u64,
    pub root_score: i32,
    pub white_dist: u8,
    pub black_dist: u8,
    pub aspiration_fails: u32,
    pub lmr_re_searches: u32,
    pub mate_extensions: u32,
    pub pv_mate_failures: u32,
    pub depth_log: Vec<DepthLogEntry>,
    pub elapsed_ms: u64,
    /// All root move candidates from the last completed depth, ordered by search order.
    pub root_moves: Vec<RootMoveInfo>,
}

#[derive(Debug, Clone, Copy)]
pub struct SearchConfig {
    pub time_ms: u64,
    pub max_nodes: u64,
    pub log: bool,
    /// Opening book move + centimeter bias — biases ordering/aspiration only.
    pub book_hint: Option<BookHint>,
    /// ID stops here unless time, nodes, or proven mate stop first.
    /// Defaults to [`DEFAULT_MAX_ID_DEPTH`] (256). Set lower only for explicit
    /// depth-capped searches (unit tests, perft-style benches).
    pub max_id_depth: u32,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            time_ms: DEFAULT_TIME_MS,
            max_nodes: DEFAULT_MAX_NODES,
            log: false,
            book_hint: None,
            max_id_depth: DEFAULT_MAX_ID_DEPTH,
        }
    }
}

#[inline]
fn effective_max_id_depth(config: &SearchConfig) -> u32 {
    config.max_id_depth.clamp(1, MAX_PLY)
}

struct SearchState<'a> {
    config: SearchConfig,
    tt: &'a mut SearchTt,
    bfs: &'a mut BfsScratch,
    nodes: u64,
    deadline: Instant,
    aspiration_fails: u32,
    lmr_re_searches: u32,
    mate_extensions: u32,
    pv_mate_failures: u32,
    depth_log: Vec<DepthLogEntry>,
    log: bool,
    pv_move: Move,
    search_depth: u32,
    book_hint: Option<BookHint>,
    our_root_dist: u8,
    /// Stockfish-style LMR table: lmr_table[depth][moves_searched] = plies to reduce.
    /// Formula: floor(0.5 + ln(depth) * ln(moves_searched) / 2.25)
    lmr_table: [[u32; 64]; 64],
    /// Per-branch forcing extension budget.
    /// Each branch can contribute at most this many extension plies before
    /// extensions stop firing — prevents depth from never decreasing.
    /// Saved before and restored after each node's subtree so sibling
    /// branches each get an independent cap.
    extensions_budget: u32,
    /// Root candidate diagnostics — rebuilt on every ply-0 negamax call.
    /// After the search loop, contains all root moves from the last completed depth.
    root_moves: Vec<RootMoveInfo>,
    /// Best secondary resistance score seen at root (used for tiebreaking equal scores).
    root_best_resistance: i32,
    /// Immediate path gain of the current root best move (tiebreak: never trade
    /// a path-gaining move for a tempo-wasting one at equal score).
    root_best_gain: i32,
    /// Position hashes on the current search path (for repetition detection).
    rep_keys: [u64; MAX_PLY as usize],
    rep_len: u32,
    started: Instant,
    lmr_profile: LmrProfile,
    mate_zone: MateZoneState,
    eval_zone: EvalZoneState,
    white_dist: u8,
    black_dist: u8,
    last_iter_asp_fails: u32,
    last_iter_score_delta: i32,
}

/// Score a repeated position: draw when racing evenly; penalize shuffles when behind.
fn repetition_score(state: &mut SearchState<'_>, board: &Board) -> i32 {
    let side = board.side();
    let stm = eval_stm(board, side, state.bfs);
    let our = state
        .bfs
        .shortest_distance(board, side)
        .unwrap_or(DIST_PENALTY) as i32;
    let opp = state
        .bfs
        .shortest_distance(board, side.opposite())
        .unwrap_or(DIST_PENALTY) as i32;
    let race_margin = opp - our;
    if race_margin <= -3 {
        return stm.saturating_sub(800);
    }
    if race_margin < 0 {
        return stm.saturating_sub(300);
    }
    if race_margin == 0 {
        return 0;
    }
    // Slightly ahead — repetition still wastes tempo.
    stm.saturating_sub(80).min(50)
}

impl SearchState<'_> {
    fn should_stop(&self) -> bool {
        self.nodes >= self.config.max_nodes || Instant::now() >= self.deadline
    }

    fn remaining_budget_ms(&self) -> u64 {
        self.deadline
            .saturating_duration_since(Instant::now())
            .as_millis() as u64
    }

    fn fraction_elapsed(&self) -> f32 {
        let total = self.config.time_ms.max(1) as f32;
        let elapsed = self
            .started
            .elapsed()
            .as_millis()
            .min(self.config.time_ms as u128) as f32;
        (elapsed / total).clamp(0.0, 1.0)
    }

    fn bump_nodes(&mut self) -> bool {
        self.nodes += 1;
        // Check more frequently to respect limits better
        self.nodes % 1024 == 0 && self.should_stop()
    }
}

fn root_cat_heat_stats(
    board: &Board,
    moves: &[Move],
    n: usize,
    cat: &CorridorAttention,
) -> (u16, u16) {
    let mut heats = Vec::with_capacity(n);
    for mv in &moves[..n] {
        heats.push(move_corridor_attention(board, *mv, cat).max(0) as u16);
    }
    if heats.is_empty() {
        return (0, 0);
    }
    heats.sort_by(|a, b| b.cmp(a));
    let max = heats[0];
    let p75_idx = (heats.len() * 3 / 4).min(heats.len() - 1);
    (max, heats[p75_idx])
}

fn update_lmr_profile_for_depth(
    state: &mut SearchState<'_>,
    board: &mut Board,
    depth: u32,
    prev_score: i32,
) {
    let root_side = board.side();
    let opp_root_dist = if root_side == Player::One {
        state.black_dist
    } else {
        state.white_dist
    };
    let min_dist = state.our_root_dist.min(opp_root_dist);
    let endgame_race = min_dist <= 4;
    let in_mate_refine = state.mate_zone.refine_ceiling.is_some()
        || (is_mate_score(prev_score) && mate_distance(prev_score).is_some());

    if in_mate_refine {
        state.lmr_profile = LmrProfile::mate_refine();
    } else if depth == 1 {
        state.lmr_profile = LmrProfile::first_iteration();
        state.lmr_profile.apply_time_budget(state.config.time_ms);
        state
            .lmr_profile
            .apply_pierce_schedule(state.fraction_elapsed());
    } else {
        let mut buf = [Move::Pawn { row: 1, col: 4 }; MAX_LEGAL_MOVES];
        let n = generate_legal_moves_slice(board, &mut buf, state.bfs);
        let cat = state.bfs.build_corridor_attention(board);
        let (cat_max, cat_p75) = root_cat_heat_stats(board, &buf, n, &cat);
        let stage_t = compute_stage_t(board, state.our_root_dist, opp_root_dist, cat_max, cat_p75);
        state.lmr_profile = LmrProfile::from_stage(stage_t, endgame_race, false);
        state.lmr_profile.apply_time_budget(state.config.time_ms);

        let log = &state.depth_log;
        let (marginal, prev_marginal) = if log.len() >= 2 {
            (
                log[log.len() - 1].marginal_nodes,
                log[log.len() - 2].marginal_nodes,
            )
        } else if log.len() == 1 {
            (log[0].marginal_nodes, 0)
        } else {
            (0, 0)
        };
        let completed = log.last().map(|e| e.depth).unwrap_or(0);
        let fraction = state.fraction_elapsed();
        apply_depth_feedback(
            &mut state.lmr_profile,
            completed,
            marginal,
            prev_marginal,
            fraction,
            state.last_iter_score_delta,
            state.last_iter_asp_fails,
        );
        state.lmr_profile.apply_pierce_schedule(fraction);
    }

    state.lmr_table = build_lmr_table(state.lmr_profile.aggression);

    if state.log && std::env::var("TITANIUM_LOG").is_ok() {
        let p = state.lmr_profile;
        let time_p = crate::search::lmr_profile::time_pressure_from_ms(state.config.time_ms);
        let pierce =
            crate::search::lmr_profile::LmrProfile::pierce_strength(state.fraction_elapsed());
        eprintln!(
            "info lmr_profile depth={} t={:.2} aggression={:.2} after={} slope={:.3} hot_pct={} cold={} floor={} push_cap={} time_p={:.2} pierce={:.2} remaining_ms={}",
            depth,
            p.stage_t,
            p.aggression,
            p.lmr_after_move,
            p.cat_heat_lmr_slope,
            p.hot_ratio_pct,
            p.cold_cm,
            p.depth_balance_floor,
            p.depth_push_marginal_cap,
            time_p,
            pierce,
            state.remaining_budget_ms()
        );
    }
}

fn mate_stop_label(reason: MateStopReason) -> &'static str {
    match reason {
        MateStopReason::RefineCeiling => "refine_ceiling",
        MateStopReason::MateSpin => "mate_spin",
        MateStopReason::ForcedOutcome => "forced_outcome",
    }
}

fn is_mate_score(score: i32) -> bool {
    score > MATE - MATE_WINDOW || score < -MATE + MATE_WINDOW
}

/// Plies until mate for the side that benefits from `score` (Stockfish-style MATE - d).
fn mate_distance(score: i32) -> Option<u32> {
    if score > MATE - MATE_WINDOW {
        Some((MATE - score).max(0) as u32)
    } else if score < -MATE + MATE_WINDOW {
        Some((MATE + score).max(0) as u32)
    } else {
        None
    }
}

/// Mate is proven only if remaining search depth covers the claimed mate distance.
fn mate_proven(score: i32, remaining_depth: u32) -> bool {
    match mate_distance(score) {
        // A zero-distance mate score is not a real child result in this search
        // (immediate wins score as MATE - 1). Treat exact MATE as an unproven
        // horizon artifact unless PV verification later reaches a terminal.
        Some(d) => d > 0 && d <= remaining_depth,
        None => true,
    }
}

/// Replace horizon mate claims with static eval — never trust `#` without depth proof.
fn clamp_unproven_mate(score: i32, remaining_depth: u32, fallback: i32) -> i32 {
    if mate_proven(score, remaining_depth) {
        return score;
    }
    if score > MAX_EVAL {
        return fallback.clamp(-MAX_EVAL, MAX_EVAL);
    }
    if score < -MAX_EVAL {
        return fallback.clamp(-MAX_EVAL, MAX_EVAL);
    }
    score
}

fn score_to_tt(score: i32, ply: u32) -> i32 {
    if score > MATE - MATE_WINDOW {
        score.saturating_add(ply as i32)
    } else if score < -MATE + MATE_WINDOW {
        score.saturating_sub(ply as i32)
    } else {
        score
    }
}

fn score_from_tt(score: i32, ply: u32) -> i32 {
    if score > MATE - MATE_WINDOW {
        score.saturating_sub(ply as i32)
    } else if score < -MATE + MATE_WINDOW {
        score.saturating_add(ply as i32)
    } else {
        score
    }
}

fn pack_move(mv: Move) -> u32 {
    match mv {
        Move::Pawn { row, col } => 1 | (u32::from(row) << 8) | (u32::from(col) << 16),
        Move::Wall {
            row,
            col,
            orientation,
        } => {
            let o = match orientation {
                WallOrientation::Horizontal => 0u32,
                WallOrientation::Vertical => 1,
            };
            2 | (u32::from(row) << 8) | (u32::from(col) << 16) | (o << 24)
        }
    }
}

fn unpack_move(packed: u32) -> Option<Move> {
    match packed & 0xFF {
        0 => None,
        1 => Some(Move::Pawn {
            row: ((packed >> 8) & 0xFF) as u8,
            col: ((packed >> 16) & 0xFF) as u8,
        }),
        2 => Some(Move::Wall {
            row: ((packed >> 8) & 0xFF) as u8,
            col: ((packed >> 16) & 0xFF) as u8,
            orientation: if (packed >> 24) & 1 == 0 {
                WallOrientation::Horizontal
            } else {
                WallOrientation::Vertical
            },
        }),
        _ => None,
    }
}

fn distance_cm(d: Option<u8>) -> i32 {
    i32::from(d.unwrap_or(DIST_PENALTY)) * CM_PER_SQUARE
}

fn goal_progress(player: Player, row: u8) -> i32 {
    match player {
        Player::One => i32::from(row),
        Player::Two => i32::from(8 - row),
    }
}

fn pawn_mobility(board: &Board, player: Player) -> i32 {
    // Pawn moves only — the old version ran FULL legal movegen (BFS-validating
    // ~40 wall placements) per eval call just to count ≤5 pawn moves.
    let mut buf = [Move::Pawn { row: 0, col: 0 }; 8];
    crate::movegen::legal::generate_pawn_moves_for(board, player, &mut buf) as i32
}

/// Static eval in centi-squares: 100 cm == one shortest-path step.
///
/// CAT intentionally does not feed this function.  CAT is a search-ordering
/// signal; eval stays on stable board features so TT scores remain meaningful.
fn eval_stm(board: &Board, stm: Player, bfs: &mut BfsScratch) -> i32 {
    let opp = stm.opposite();
    let our_steps = bfs.shortest_distance(board, stm).unwrap_or(DIST_PENALTY);
    let opp_steps = bfs.shortest_distance(board, opp).unwrap_or(DIST_PENALTY);
    let our = distance_cm(Some(our_steps));
    let their = distance_cm(Some(opp_steps));
    let distance_score = their - our;

    let wall_hoard_cm = if our_steps > opp_steps && opp_steps <= 4 {
        // Opponent is close to goal and ahead — hoarding walls is suicidal.
        WALL_INVENTORY_CM / 4
    } else {
        WALL_INVENTORY_CM
    };
    let wall_score = (i32::from(board.walls_remaining[stm as usize])
        - i32::from(board.walls_remaining[opp as usize]))
        * wall_hoard_cm;

    let (our_row, _) = board.pawn(stm);
    let (opp_row, _) = board.pawn(opp);
    let progress_score =
        (goal_progress(stm, our_row) - goal_progress(opp, opp_row)) * PAWN_PROGRESS_CM;

    let race_score = if our < their {
        RACE_LEAD_CM
    } else if our > their {
        -RACE_LEAD_CM
    } else {
        0
    };

    // Urgency bonus: when either player is very close to the goal (≤3 steps),
    // the race outcome is nearly decided.  Scale up the distance advantage to
    // reward resistance in endgame sprint positions.  The multiplier is chosen
    // so that a 1-step advantage when opp is 2 away is worth ~1 extra square
    // (100 cm) on top of the normal distance term — enough to override quiet
    // wall-hoarding but small enough not to break horizon eval.
    // Gate on opp_steps ≤ 3 so startpos and mid-game eval are unchanged.
    let urgency_score = if opp_steps <= 3 {
        let lead = i32::from(opp_steps) - i32::from(our_steps);
        // Each step of lead is worth an extra 50 cm when opp is close.
        lead * 50
    } else if our_steps <= 3 {
        // We're close — reward being ahead in the sprint.
        let lead = i32::from(opp_steps) - i32::from(our_steps);
        lead * 30
    } else {
        0
    };

    let boxed_or_urgent =
        our_steps >= opp_steps.saturating_add(2) || our_steps <= 4 || opp_steps <= 4;
    let mobility_score = if boxed_or_urgent {
        let our_mobility = pawn_mobility(board, stm);
        let opp_mobility = pawn_mobility(board, opp);
        (our_mobility - opp_mobility) * 8
    } else {
        0
    };

    let trap_score = if our_steps >= opp_steps.saturating_add(3) {
        -20 * i32::from(our_steps.saturating_sub(opp_steps))
    } else if opp_steps >= our_steps.saturating_add(3) {
        20 * i32::from(opp_steps.saturating_sub(our_steps))
    } else {
        0
    };

    let our_walls = board.walls_remaining[stm as usize];
    let opp_walls = board.walls_remaining[opp as usize];
    let corridor_flex_score = if our_walls <= 2 || opp_walls <= 2 {
        let our_bottlenecks = i32::from(bfs.corridor_bottleneck_count(board, stm));
        let opp_bottlenecks = i32::from(bfs.corridor_bottleneck_count(board, opp));
        let own_risk = if our_walls <= 2 && opp_walls > our_walls {
            our_bottlenecks * LOW_WALL_TRAP_CM
        } else {
            our_bottlenecks * (LOW_WALL_TRAP_CM / 3)
        };
        let opp_risk = if opp_walls <= 2 && our_walls > opp_walls {
            opp_bottlenecks * LOW_WALL_TRAP_CM
        } else {
            opp_bottlenecks * (LOW_WALL_TRAP_CM / 3)
        };
        opp_risk - own_risk
    } else {
        0
    };

    (distance_score
        + wall_score
        + progress_score
        + race_score
        + urgency_score
        + mobility_score
        + trap_score
        + corridor_flex_score)
        .clamp(-MAX_EVAL, MAX_EVAL)
}

fn terminal_score(ply: u32) -> i32 {
    -MATE + ply as i32
}

/// Keep every pawn; retain only the hottest `max_walls` walls by CAT edge heat.
fn cap_root_wall_moves(buf: &mut [Move], n: &mut usize, cat: &CorridorAttention, max_walls: usize) {
    if *n == 0 {
        return;
    }
    let mut ranked = [(0usize, 0u16); MAX_LEGAL_MOVES];
    let mut wall_count = 0usize;
    for i in 0..*n {
        if let Move::Wall {
            row,
            col,
            orientation,
        } = buf[i]
        {
            ranked[wall_count] = (i, cat.wall_edge_heat(row, col, orientation));
            wall_count += 1;
        }
    }
    if wall_count <= max_walls {
        return;
    }
    ranked[..wall_count].sort_by(|a, b| b.1.cmp(&a.1));
    let mut keep = [false; MAX_LEGAL_MOVES];
    for &(i, _) in &ranked[..max_walls] {
        keep[i] = true;
    }
    let mut out = 0usize;
    for i in 0..*n {
        if matches!(buf[i], Move::Pawn { .. }) || keep[i] {
            buf[out] = buf[i];
            out += 1;
        }
    }
    *n = out;
}

/// Leaf score: stand-pat eval plus at most one ply of **forward pawn** pushes.
/// Wall quiescence was removed (too expensive, fought LMR); this fixes the
/// odd/even depth oscillation (0 / −1.21 / 0 / …) in symmetric pawn races.
fn leaf_eval(
    state: &mut SearchState<'_>,
    board: &mut Board,
    mut alpha: i32,
    beta: i32,
    ply: u32,
) -> i32 {
    if state.bump_nodes() {
        return alpha;
    }

    if board.is_terminal().is_some() {
        return terminal_score(ply);
    }

    let stand_pat = eval_stm(board, board.side(), state.bfs);
    if stand_pat >= beta {
        return beta;
    }
    if stand_pat > alpha {
        alpha = stand_pat;
    }

    let stm = board.side();
    let our_dist = state
        .bfs
        .shortest_distance(board, stm)
        .unwrap_or(DIST_PENALTY);

    let mut buf = [Move::Pawn { row: 0, col: 0 }; 8];
    let n = crate::movegen::legal::generate_pawn_moves_for(board, stm, &mut buf) as usize;

    for i in 0..n {
        let mv = buf[i];
        let extends = match mv {
            Move::Pawn { row, .. } => {
                our_path_gain(board, mv, our_dist, state.bfs) > 0 || is_goal(stm, row)
            }
            _ => false,
        };
        if !extends {
            continue;
        }

        let undo = board.make_move(mv);
        let score = -eval_stm(board, board.side(), state.bfs);
        board.unmake_move(undo);

        if state.should_stop() {
            break;
        }
        if score > alpha {
            alpha = score;
        }
        if alpha >= beta {
            break;
        }
    }

    alpha
}

fn make_null_move(board: &mut Board) -> u64 {
    let old_hash = board.hash;
    crate::core::zobrist::xor_side(&mut board.hash);
    board.side_to_move = board.side_to_move.opposite();
    if board.side_to_move == Player::One {
        board.move_number += 1;
    }
    old_hash
}

fn unmake_null_move(board: &mut Board, old_hash: u64) {
    if board.side_to_move == Player::One {
        board.move_number -= 1;
    }
    board.side_to_move = board.side_to_move.opposite();
    board.hash = old_hash;
}

fn search_child(
    state: &mut SearchState<'_>,
    board: &mut Board,
    depth: u32,
    alpha: i32,
    beta: i32,
    ply: u32,
) -> i32 {
    let mut score = -negamax(state, board, depth, -beta, -alpha, ply + 1);

    // Mate extension: if the child returns a mate claim that the remaining depth
    // cannot prove, keep extending (up to 3 extra plies) until either the claim
    // is proven or we run out of budget.  This ensures forcing wins are never
    // truncated at the horizon.
    //
    // Must run BEFORE the unproven-mate clamp — the old order clamped the mate
    // away first, so this loop could never fire (mateExtensions was always 0).
    let mut extra = 0u32;
    while extra < 3 {
        let Some(d) = mate_distance(score) else { break };
        let proven_depth = depth + extra;
        if d <= proven_depth {
            break; // mate is already covered by the depth we searched
        }
        if proven_depth + 1 > MAX_PLY {
            break;
        }
        state.mate_extensions += 1;
        extra += 1;
        score = -negamax(state, board, proven_depth + 1, -beta, -alpha, ply + 1);
    }

    // Fallback eval only when an unproven mate claim survived the extensions —
    // eval_stm costs two BFS and this path runs for EVERY child node.
    if is_mate_score(score) && !mate_proven(score, depth + extra) {
        let fallback = eval_stm(board, board.side().opposite(), state.bfs);
        score = clamp_unproven_mate(score, depth + extra, fallback);
    }

    score
}

fn negamax(
    state: &mut SearchState<'_>,
    board: &mut Board,
    depth: u32,
    alpha: i32,
    beta: i32,
    ply: u32,
) -> i32 {
    if state.bump_nodes() {
        return alpha;
    }

    if board.is_terminal().is_some() {
        return terminal_score(ply);
    }

    if ply >= MAX_PLY {
        return eval_stm(board, board.side(), state.bfs);
    }

    let hash = board.hash;
    if ply >= 2 {
        for i in 0..state.rep_len as usize {
            if state.rep_keys[i] == hash {
                return repetition_score(state, board);
            }
        }
    }

    let rep_idx = state.rep_len as usize;
    if rep_idx < MAX_PLY as usize {
        state.rep_keys[rep_idx] = hash;
    }
    state.rep_len += 1;
    let score = negamax_inner(state, board, depth, alpha, beta, ply);
    state.rep_len -= 1;
    score
}

fn negamax_inner(
    state: &mut SearchState<'_>,
    board: &mut Board,
    depth: u32,
    mut alpha: i32,
    beta: i32,
    ply: u32,
) -> i32 {
    let hash = board.hash;
    let mut tt_best = None;
    if let Some(entry) = state.tt.probe(hash) {
        tt_best = unpack_move(entry.best);
        if i32::from(entry.depth) >= depth as i32 {
            let score = score_from_tt(entry.score, ply);
            let bound = match entry.bound {
                0 => TtBound::Exact,
                1 => TtBound::Lower,
                _ => TtBound::Upper,
            };
            // Lazy fallback eval — only mate scores ever need clamping.
            let corrected = if is_mate_score(score) && !mate_proven(score, depth) {
                clamp_unproven_mate(score, depth, eval_stm(board, board.side(), state.bfs))
            } else {
                score
            };
            match bound {
                TtBound::Exact => return corrected,
                TtBound::Lower if corrected >= beta => return corrected,
                TtBound::Upper if corrected <= alpha => return corrected,
                _ => {}
            }
        }
    }

    if depth == 0 {
        return leaf_eval(state, board, alpha, beta, ply);
    }

    // ── Static eval (shared by NMP and futility) ────────────────────────────
    let static_eval = eval_stm(board, board.side(), state.bfs);

    // Null Move Pruning (NMP)
    // Use R=3 at depth ≥ 5 for deeper cuts; R=2 otherwise.
    if depth >= 3 {
        if static_eval >= beta {
            let r = if depth >= 5 { 3u32 } else { 2u32 };
            let reduced_depth = depth.saturating_sub(1 + r);
            let old_hash = make_null_move(board);
            let score = -negamax(state, board, reduced_depth, -beta, -beta + 1, ply + 1);
            unmake_null_move(board, old_hash);
            if score >= beta {
                return beta;
            }
        }
    }

    // Futility pruning at depth 1–2: if we are well below alpha even with a
    // generous margin, skip all non-tactical moves — they cannot raise alpha.
    // Only at non-root nodes (ply > 0) to never prune root choices.
    let futility_depth = depth as usize;
    let apply_futility =
        ply > 0 && futility_depth <= 2 && !is_mate_score(static_eval) && !is_mate_score(alpha);

    // Witness path + dists once per node — shared by filtering, ordering, LMR.
    let mut opp_path = [0u8; 81];
    let opp_path_len = get_shortest_path(board, board.side().opposite(), state.bfs, &mut opp_path);
    let opp_dist_pre = path_distance(board.side().opposite(), &opp_path, opp_path_len);
    let our_dist_pre = state
        .bfs
        .shortest_distance(board, board.side())
        .unwrap_or(DIST_PENALTY);

    // CAT (multi-route corridor heat) only above the leaf layer: depth-1 nodes
    // dominate the tree and only need witness-path tactics, not breadth.
    let cat = if depth >= 2 {
        state.bfs.build_corridor_attention(board)
    } else {
        crate::cat::CorridorAttention::default()
    };

    let mut buf = [Move::Pawn { row: 0, col: 0 }; MAX_LEGAL_MOVES];
    let mut n = collect_search_moves(
        board,
        &mut buf,
        state.bfs,
        &cat,
        &opp_path,
        opp_path_len,
        our_dist_pre,
        opp_dist_pre,
        false,
        true,
    );
    if n == 0 {
        return eval_stm(board, board.side(), state.bfs);
    }

    let mut scores = [0i32; MAX_LEGAL_MOVES];
    order_moves(
        board,
        &mut buf,
        n,
        tt_best,
        state.book_hint,
        &mut scores,
        our_dist_pre,
        opp_dist_pre,
        &opp_path,
        opp_path_len,
        state.bfs,
        &cat,
    );

    // ── Forcing extension ─────────────────────────────────────────────────
    // Extend by 1 ply when the position is near-forcing:
    //   (a) STM has ≤ 1 legal pawn move — line is essentially forced.
    //   (b) Either player is ≤ 2 steps from the goal — race outcome is near.
    //
    // CRITICAL safety: child_depth = (depth-1)+1 = depth, so WITHOUT a budget
    // cap depth would NEVER decrease → stack overflow. The budget is saved here
    // and restored after the move loop, giving each branch an independent cap
    // of MAX_EXTENSIONS_PER_BRANCH plies. When the budget hits 0 the extension
    // doesn't fire and depth decreases normally.
    let forcing_extension: u32 = if ply > 0 && depth > 1 && state.extensions_budget > 0 {
        let pawn_count = buf[..n]
            .iter()
            .filter(|m| matches!(m, Move::Pawn { .. }))
            .count();
        // ≤1: truly decided sprints only. The old ≤2 cutoff fired in every
        // ordinary race and blew the tree up (depth never decreased for whole
        // subtrees in normal midgames).
        let near_goal = our_dist_pre <= 1 || opp_dist_pre <= 1;
        if pawn_count <= 1 || near_goal {
            1
        } else {
            0
        }
    } else {
        0
    };
    // Save budget now; decrement for this whole subtree; restore after the loop.
    let budget_before_subtree = state.extensions_budget;
    if forcing_extension > 0 {
        state.extensions_budget = state.extensions_budget.saturating_sub(1);
    }

    let mut best_score = -MATE;
    let mut best_mv = buf[0];
    let mut best_packed = pack_move(best_mv);
    let mut moves_searched = 0usize;
    let original_alpha = alpha;

    let mut cat_max = 0u16;
    for j in 0..n {
        let cm = move_corridor_attention(board, buf[j], &cat).max(0) as u16;
        cat_max = cat_max.max(cm);
    }

    // At root, clear diagnostics so only the current depth's data is retained.
    if ply == 0 {
        state.root_moves.clear();
        state.root_best_resistance = i32::MIN;
        state.root_best_gain = i32::MIN;
    }

    let profile = state.lmr_profile;

    if ply == 0 && depth >= 3 {
        let cap = profile.root_wall_cap().min(if profile.stage_t < 0.40 {
            ROOT_WALL_CAP_OPENING
        } else {
            ROOT_WALL_CAP_MID
        });
        cap_root_wall_moves(&mut buf, &mut n, &cat, cap);
    }

    for i in 0..n {
        let mv = buf[i];

        // ── Tactical classification ───────────────────────────────────────────
        // Compute this once per move — used by both futility and LMR.
        // A move is tactical if it:
        //   (a) Shortens our BFS distance to goal (pawn), or
        //   (b) Disturbs the opponent's shortest path (wall).
        // Tactical moves are NEVER reduced or pruned.
        let cat_cm = move_corridor_attention(board, mv, &cat);
        let heat_ratio_hot = cat_max > 0
            && (cat_cm.max(0) as u32) * 100 >= (cat_max as u32) * u32::from(profile.hot_ratio_pct);
        let corridor_relevant = cat_cm >= i32::from(profile.cold_cm);
        let full_depth_slots = profile.move_window.max(profile.lmr_after_move);
        let is_tactical = if moves_searched == 0
            || depth < LMR_MIN_DEPTH
            || (ply > 0 && moves_searched < full_depth_slots)
            || heat_ratio_hot
        {
            true
        } else if matches!(mv, Move::Wall { .. })
            && !prune::wall_intersects_path(mv, &opp_path, opp_path_len)
        {
            // A wall off the opponent's witness path cannot lengthen their
            // BFS distance — non-tactical without spending a make + BFS.
            false
        } else {
            is_tactical_move(board, mv, our_dist_pre, opp_dist_pre, state.bfs)
        };

        // ── Futility pruning ──────────────────────────────────────────────────
        // At depth 1-2, if the static eval is already so far below alpha that
        // even a large margin cannot recover, skip quiet moves entirely.
        if apply_futility && !is_tactical && !corridor_relevant {
            let margin = FUTILITY_MARGIN[futility_depth];
            if static_eval + margin <= alpha {
                moves_searched += 1;
                continue;
            }
        }

        // [LMR_BLOCK_START]
        // Adaptive LMR — profile rebuilt each ID depth. Root: only move 1 is
        // always full-depth; cold root walls get reduced like internal nodes.
        let reduction = if (ply == 0 && moves_searched == 0)
            || (ply > 0 && is_tactical)
            || depth < LMR_MIN_DEPTH
            || heat_ratio_hot
        {
            0u32
        } else {
            let d = (depth as usize).min(63);
            let m = (i + 1).min(63);
            let base_r = state.lmr_table[d][m];
            let gap = cat_max.saturating_sub(cat_cm.max(0) as u16);
            let cat_extra = (gap as f32 * profile.cat_heat_lmr_slope).round() as u32;
            // Extra reduction for pure quiet walls (not intersecting opp path at all).
            let wall_extra = if matches!(mv, Move::Wall { .. }) && cat_cm == 0 {
                4u32
            } else if matches!(mv, Move::Wall { .. })
                && !prune::wall_intersects_path(mv, &opp_path, opp_path_len)
                && !corridor_relevant
            {
                3u32
            } else if cat_cm < i32::from(profile.cold_cm) {
                if profile.stage_t < 0.35 {
                    3u32
                } else {
                    1u32
                }
            } else {
                0u32
            };
            (base_r + wall_extra + cat_extra).min(depth.saturating_sub(1))
        };
        // [LMR_BLOCK_END]

        let undo = board.make_move(mv);
        // child_depth: one ply below current, plus any forcing extension so that
        // near-forced positions are searched one ply deeper throughout the subtree.
        let child_depth = (depth - 1) + forcing_extension;
        let score = if moves_searched == 0 {
            search_child(state, board, child_depth, alpha, beta, ply)
        } else {
            let reduced = child_depth.saturating_sub(reduction);
            let mut s = if reduced == 0 {
                -leaf_eval(state, board, -beta, -alpha, ply + 1)
            } else {
                search_child(state, board, reduced, alpha, alpha + 1, ply)
            };
            if s > alpha && (reduction > 0 || s < beta) {
                if reduction > 0 {
                    state.lmr_re_searches += 1;
                }
                s = search_child(state, board, child_depth, alpha, beta, ply);
            }
            s
        };

        // Capture post-move distances while board is still in post-move state.
        let (root_w_dist, root_b_dist) = if ply == 0 {
            let wd = state
                .bfs
                .shortest_distance(board, Player::One)
                .unwrap_or(DIST_PENALTY);
            let bd = state
                .bfs
                .shortest_distance(board, Player::Two)
                .unwrap_or(DIST_PENALTY);
            (wd, bd)
        } else {
            (0u8, 0u8)
        };

        board.unmake_move(undo);

        // Stop check BEFORE consuming the score: an aborted child returns its
        // own alpha, which negates into -alpha/-beta window artifacts here — a
        // fake fail-high could otherwise be recorded as the best move.
        if state.should_stop() {
            break;
        }

        // ── Root diagnostics & resistance tiebreaking ──────────────────────────
        // Board is now back to pre-move state; compute gain and resistance here.
        // Resistance = opp_dist_after - our_dist_after (higher is better for us).
        let (root_gain, root_resistance) = if ply == 0 {
            let gain = move_immediate_gain(board, mv, our_dist_pre, opp_dist_pre, state.bfs);
            let stm = board.side();
            let (our_after, opp_after) = if stm == Player::One {
                (root_w_dist, root_b_dist)
            } else {
                (root_b_dist, root_w_dist)
            };
            let resistance = i32::from(opp_after) - i32::from(our_after);
            (gain, resistance)
        } else {
            (0i32, 0i32)
        };

        if ply == 0 {
            state.root_moves.push(RootMoveInfo {
                mv: crate::util::perft::format_move(mv),
                score,
                white_dist_after: root_w_dist,
                black_dist_after: root_b_dist,
                gain: root_gain,
                mate_distance: mate_distance(score),
                is_pawn: matches!(mv, Move::Pawn { .. }),
            });
        }

        moves_searched += 1;

        // best_score always tracks the true maximum (αβ/TT correctness).
        // At root, exact score ties break on immediate path gain first (never
        // trade a path-gaining move for a tempo-wasting wall — that was the
        // "weird passive wall" bug), then on race resistance.
        if score > best_score {
            best_score = score;
            best_mv = mv;
            best_packed = pack_move(best_mv);
            if ply == 0 {
                state.root_best_gain = root_gain;
                state.root_best_resistance = root_resistance;
            }
        } else if ply == 0 && score == best_score {
            let better_gain = root_gain > state.root_best_gain;
            let better_resistance =
                root_gain == state.root_best_gain && root_resistance > state.root_best_resistance;
            if better_gain || better_resistance {
                best_mv = mv;
                best_packed = pack_move(best_mv);
                state.root_best_gain = root_gain;
                state.root_best_resistance = root_resistance;
            }
        }
        if score > alpha {
            alpha = score;
        }
        if alpha >= beta {
            break;
        }
    }

    // Belt-and-suspenders: root_moves must never contain a strictly higher
    // score than the move we picked (ties are resolved by gain/resistance above).
    if ply == 0 {
        if let Some(entry) = state.root_moves.iter().max_by_key(|r| r.score) {
            if entry.score > best_score {
                if let Some(&mv) = buf[..n]
                    .iter()
                    .find(|m| crate::util::perft::format_move(**m) == entry.mv)
                {
                    best_score = entry.score;
                    best_mv = mv;
                    best_packed = pack_move(best_mv);
                }
            }
        }
    }

    // Restore extension budget so sibling branches each get an independent cap.
    state.extensions_budget = budget_before_subtree;

    let bound = if best_score <= original_alpha {
        TtBound::Upper
    } else if best_score >= beta {
        TtBound::Lower
    } else {
        TtBound::Exact
    };
    if is_mate_score(best_score) && !mate_proven(best_score, depth) {
        let stand_pat = eval_stm(board, board.side(), state.bfs);
        best_score = clamp_unproven_mate(best_score, depth, stand_pat);
    }

    state.tt.store(
        hash,
        depth.min(i8::MAX as u32) as i8, // extensions can push depth past 127 — never wrap negative
        score_to_tt(best_score, ply),
        bound,
        best_packed,
    );

    if ply == 0 {
        state.pv_move = best_mv;
    }

    best_score
}

/// Walk TT PV — if root claims mate, line must reach a real terminal within distance.
fn extract_pv_algebraic(board: &Board, tt: &SearchTt, max_ply: u32) -> String {
    use std::collections::HashSet;

    let mut copy = board.clone();
    let mut parts = Vec::new();
    let mut seen = HashSet::with_capacity(max_ply as usize + 1);
    for _ in 0..max_ply {
        if copy.is_terminal().is_some() {
            break;
        }
        if !seen.insert(copy.hash) {
            break;
        }
        let Some(entry) = tt.probe(copy.hash) else {
            break;
        };
        let Some(mv) = unpack_move(entry.best) else {
            break;
        };
        parts.push(format_move(mv));
        let _ = copy.make_move(mv);
    }
    parts.join(" ")
}

fn verify_pv_mate(board: &Board, tt: &SearchTt, claimed_score: i32) -> bool {
    let Some(m_dist) = mate_distance(claimed_score) else {
        return true;
    };

    let mut copy = board.clone();
    let mut plies = 0u32;
    while plies < m_dist.saturating_add(2) && plies < MAX_PLY {
        if copy.is_terminal().is_some() {
            return true;
        }
        let Some(entry) = tt.probe(copy.hash) else {
            break;
        };
        let Some(mv) = unpack_move(entry.best) else {
            break;
        };
        let _ = copy.make_move(mv);
        plies += 1;
    }

    copy.is_terminal().is_some()
}

fn corrected_root_score(
    board: &Board,
    tt: &SearchTt,
    claimed: i32,
    depth: u32,
    bfs: &mut BfsScratch,
) -> i32 {
    if !is_mate_score(claimed) {
        return claimed;
    }
    if let Some(d) = mate_distance(claimed) {
        if d > 0 && depth >= d {
            return claimed;
        }
    }
    if verify_pv_mate(board, tt, claimed) {
        return claimed;
    }
    eval_stm(board, board.side(), bfs)
}

fn find_immediate_win(moves: &[Move], stm: Player) -> Option<Move> {
    for &mv in moves {
        if let Move::Pawn { row, col: _ } = mv {
            if is_goal(stm, row) {
                return Some(mv);
            }
        }
    }
    None
}

/// Stockfish-style early exit when the outcome is already decided at this depth.
///
/// Stops once mate is PV-verified or search depth covers the claimed mate distance.
/// Applies to both winning and losing mates: at depth ≥ d the root has been
/// searched deeply enough to compare delaying defenses; continuing to d150+ only
/// re-walks the same forced PV (~8k nodes/depth) without changing the answer.
fn should_stop_forced_outcome(
    verified: i32,
    depth: u32,
    board: &Board,
    tt: &SearchTt,
    our_root_dist: u8,
) -> bool {
    if is_mate_score(verified) {
        if verify_pv_mate(board, tt, verified) {
            return true;
        }
        if let Some(d) = mate_distance(verified) {
            if d > 0 && depth >= d {
                return true;
            }
        }
    }
    // One step from goal — shallow search is enough to pick the winning pawn step.
    our_root_dist == 1 && depth >= 2
}

fn log_depth(state: &SearchState<'_>, depth: u32, score: i32) {
    if !state.log {
        return;
    }
    let display = if is_mate_score(score) {
        if score > 0 {
            format!("#+{}", MATE - score)
        } else {
            format!("#-{}", MATE + score)
        }
    } else {
        score.to_string()
    };
    eprintln!(
        "info depth {} score {} nodes {} asp {} lmr {}",
        depth, display, state.nodes, state.aspiration_fails, state.lmr_re_searches
    );
    flush_search_log();
}

fn format_depth_log_json(depth_log: &[DepthLogEntry]) -> String {
    let mut depth_json = String::new();
    for (i, e) in depth_log.iter().enumerate() {
        if i > 0 {
            depth_json.push(',');
        }
        let pv = e.pv.replace('\\', "\\\\").replace('"', "\\\"");
        depth_json.push_str(&format!(
            "{{\"depth\":{},\"score\":{},\"nodes\":{},\"elapsedMs\":{},\"marginalNodes\":{},\"pv\":\"{}\"}}",
            e.depth, e.score, e.nodes, e.elapsed_ms, e.marginal_nodes, pv
        ));
    }
    depth_json
}

fn flush_search_log() {
    let _ = std::io::stderr().flush();
}

fn emit_search_progress(
    state: &SearchState,
    config: &SearchConfig,
    white_dist: u8,
    black_dist: u8,
) {
    if !config.log {
        return;
    }
    let depth_json = format_depth_log_json(&state.depth_log);
    let root_score = state.depth_log.last().map(|e| e.score).unwrap_or(0);
    eprintln!(
        "info json {{\"stoppedBy\":\"minimax\",\"searchDepth\":{},\"nodes\":{},\"rootScore\":{},\"whiteDist\":{},\"blackDist\":{},\"depthLog\":[{}]}}",
        state.search_depth,
        state.nodes,
        root_score,
        white_dist,
        black_dist,
        depth_json
    );
    flush_search_log();
}

fn emit_json_report(report: &SearchReport, log: bool) {
    if !log {
        return;
    }
    let depth_json = format_depth_log_json(&report.depth_log);
    let mut root_json = String::new();
    for (i, r) in report.root_moves.iter().enumerate() {
        if i > 0 {
            root_json.push(',');
        }
        root_json.push_str(&format!(
            "{{\"move\":\"{}\",\"score\":{},\"mateDistance\":{},\"whiteDist\":{},\"blackDist\":{},\"gain\":{},\"kind\":\"{}\"}}",
            r.mv,
            r.score,
            r.mate_distance
                .map(|d| d.to_string())
                .unwrap_or_else(|| "null".to_owned()),
            r.white_dist_after,
            r.black_dist_after,
            r.gain,
            if r.is_pawn { "pawn" } else { "wall" }
        ));
    }
    eprintln!(
        "info json {{\"stoppedBy\":\"minimax\",\"searchDepth\":{},\"nodes\":{},\"rootScore\":{},\"whiteDist\":{},\"blackDist\":{},\"aspirationFails\":{},\"lmrReSearches\":{},\"mateExtensions\":{},\"pvMateFailures\":{},\"elapsedMs\":{},\"depthLog\":[{}],\"rootMoves\":[{}]}}",
        report.search_depth,
        report.nodes,
        report.root_score,
        report.white_dist,
        report.black_dist,
        report.aspiration_fails,
        report.lmr_re_searches,
        report.mate_extensions,
        report.pv_mate_failures,
        report.elapsed_ms,
        depth_json,
        root_json
    );
}

/// Full-strength search from `board` — returns best move + diagnostics.
pub fn search_best_move(board: &mut Board, config: SearchConfig) -> Option<SearchReport> {
    let mut bfs = BfsScratch::new();
    let mut buf = [Move::Pawn { row: 0, col: 0 }; MAX_LEGAL_MOVES];
    let n = generate_legal_moves_slice(board, &mut buf, &mut bfs);
    if n == 0 {
        return None;
    }
    if n == 1 {
        let white_dist = bfs
            .shortest_distance(board, Player::One)
            .unwrap_or(DIST_PENALTY);
        let black_dist = bfs
            .shortest_distance(board, Player::Two)
            .unwrap_or(DIST_PENALTY);
        return Some(SearchReport {
            best_move: buf[0],
            search_depth: 0,
            nodes: 1,
            root_score: eval_stm(board, board.side(), &mut bfs),
            white_dist,
            black_dist,
            aspiration_fails: 0,
            lmr_re_searches: 0,
            mate_extensions: 0,
            pv_mate_failures: 0,
            depth_log: Vec::new(),
            elapsed_ms: 0,
            root_moves: Vec::new(),
        });
    }

    let root_side = board.side();
    if let Some(win_mv) = find_immediate_win(&buf[..n], root_side) {
        let white_dist = bfs
            .shortest_distance(board, Player::One)
            .unwrap_or(DIST_PENALTY);
        let black_dist = bfs
            .shortest_distance(board, Player::Two)
            .unwrap_or(DIST_PENALTY);
        return Some(SearchReport {
            best_move: win_mv,
            search_depth: 1,
            nodes: 1,
            root_score: MATE - 1,
            white_dist,
            black_dist,
            aspiration_fails: 0,
            lmr_re_searches: 0,
            mate_extensions: 0,
            pv_mate_failures: 0,
            depth_log: Vec::new(),
            elapsed_ms: 0,
            root_moves: Vec::new(),
        });
    }

    let started = Instant::now();
    let deadline = started + std::time::Duration::from_millis(config.time_ms);
    let mut tt = SearchTt::new();

    let white_dist = bfs
        .shortest_distance(board, Player::One)
        .unwrap_or(DIST_PENALTY);
    let black_dist = bfs
        .shortest_distance(board, Player::Two)
        .unwrap_or(DIST_PENALTY);

    let our_root_dist = bfs
        .shortest_distance(board, root_side)
        .unwrap_or(DIST_PENALTY);
    let mut pv_move = buf[0];
    if let Some(hint) = config.book_hint {
        if buf[..n].contains(&hint.mv) {
            pv_move = hint.mv;
        }
    }

    let mut state = SearchState {
        config,
        tt: &mut tt,
        bfs: &mut bfs,
        nodes: 0,
        deadline,
        aspiration_fails: 0,
        lmr_re_searches: 0,
        mate_extensions: 0,
        pv_mate_failures: 0,
        depth_log: Vec::new(),
        log: config.log,
        pv_move,
        search_depth: 0,
        book_hint: config.book_hint,
        our_root_dist,
        lmr_table: build_lmr_table(LmrProfile::first_iteration().aggression),
        extensions_budget: 4,
        root_moves: Vec::new(),
        root_best_resistance: i32::MIN,
        root_best_gain: i32::MIN,
        rep_keys: [0; MAX_PLY as usize],
        rep_len: 0,
        started,
        lmr_profile: LmrProfile::first_iteration(),
        mate_zone: MateZoneState::default(),
        eval_zone: EvalZoneState::default(),
        white_dist,
        black_dist,
        last_iter_asp_fails: 0,
        last_iter_score_delta: i32::MAX,
    };

    let static_eval = eval_stm(board, root_side, state.bfs);
    let mut prev_score = if let Some(hint) = config.book_hint {
        // Soft aspiration toward book PV — mined lines (priority≥150) get stronger pull.
        let book_pull = if hint.priority >= 150 {
            80 + i32::from(hint.stm_bias) / 2
        } else {
            i32::from(hint.stm_bias) / 4
        };
        static_eval.saturating_add(book_pull)
    } else {
        static_eval
    };
    let mut best_mv = pv_move;
    let mut completed_depth = 0u32;
    // Root diagnostics snapshot from the last *completed* depth — state.root_moves
    // may hold a partially-searched (aborted) iteration when the loop exits.
    let mut committed_root_moves: Vec<RootMoveInfo> = Vec::new();

    let max_id_depth = effective_max_id_depth(&config);
    let mut depth = 1u32;
    loop {
        if state.should_stop() {
            break;
        }

        update_lmr_profile_for_depth(&mut state, board, depth, prev_score);
        let nodes_at_depth_start = state.nodes;

        let asp_start_fails = state.aspiration_fails;
        let delta = ASPIRATION_DELTA + depth as i32 * 3;
        let mut alpha = prev_score.saturating_sub(delta);
        let mut beta = prev_score.saturating_add(delta);
        let score = loop {
            let s = negamax(&mut state, board, depth, alpha, beta, 0);
            if s <= alpha && !is_mate_score(s) {
                state.aspiration_fails += 1;
                alpha = -MAX_EVAL;
                if state.aspiration_fails > asp_start_fails + 3 {
                    break negamax(&mut state, board, depth, -MAX_EVAL, MAX_EVAL, 0);
                }
                continue;
            }
            if s >= beta && !is_mate_score(s) {
                state.aspiration_fails += 1;
                beta = MAX_EVAL;
                if state.aspiration_fails > asp_start_fails + 3 {
                    break negamax(&mut state, board, depth, -MAX_EVAL, MAX_EVAL, 0);
                }
                continue;
            }
            break s;
        };

        // Iteration aborted mid-search (time/nodes): subtree scores are window
        // artifacts. Keep the previous completed depth's answer instead of
        // committing a partially-searched, possibly garbage best move.
        if state.should_stop() && completed_depth > 0 {
            break;
        }

        let verified = corrected_root_score(board, state.tt, score, depth, state.bfs);
        if is_mate_score(score) && !is_mate_score(verified) {
            state.pv_mate_failures += 1;
            if state.log {
                eprintln!(
                    "info pv reject depth {} claimed_mate dist {:?} -> eval {}",
                    depth,
                    mate_distance(score),
                    verified
                );
            }
        }

        prev_score = verified;
        best_mv = state.pv_move;
        completed_depth = depth;
        state.search_depth = depth;
        committed_root_moves.clone_from(&state.root_moves);

        let pv = extract_pv_algebraic(board, state.tt, depth);
        let elapsed_ms = started.elapsed().as_millis() as u64;
        let marginal_nodes = state.nodes.saturating_sub(nodes_at_depth_start);
        let prev_log_score = state.depth_log.last().map(|e| e.score);
        state.last_iter_score_delta = prev_log_score
            .map(|s| (verified - s).abs())
            .unwrap_or(i32::MAX);
        state.last_iter_asp_fails = state.aspiration_fails.saturating_sub(asp_start_fails);
        state.depth_log.push(DepthLogEntry {
            depth,
            score: verified,
            nodes: state.nodes,
            elapsed_ms,
            marginal_nodes,
            pv,
        });
        log_depth(&state, depth, verified);
        emit_search_progress(&state, &config, white_dist, black_dist);

        let mate_proven_at_depth = verify_pv_mate(board, state.tt, verified)
            || mate_distance(verified).is_some_and(|d| d > 0 && depth >= d);
        if let Some(reason) = state.mate_zone.update_after_depth(
            verified,
            depth,
            marginal_nodes,
            mate_proven_at_depth,
            verify_pv_mate(board, state.tt, verified),
        ) {
            if state.log {
                eprintln!(
                    "info mate stop {} at depth {} dist {:?}",
                    mate_stop_label(reason),
                    depth,
                    mate_distance(verified)
                );
            }
            break;
        }

        if state
            .eval_zone
            .update_after_depth(verified, depth, marginal_nodes)
        {
            if state.log {
                eprintln!(
                    "info eval stop eval_spin at depth {} score {}",
                    depth, verified
                );
            }
            break;
        }

        if state.our_root_dist == 1 && depth >= 2 {
            if state.log {
                eprintln!("info forced outcome at depth {}, stopping search", depth);
            }
            break;
        }

        if state.should_stop() {
            break;
        }

        if depth >= max_id_depth {
            if state.log {
                eprintln!("info max id depth {}, stopping search", max_id_depth);
            }
            break;
        }

        depth = depth.saturating_add(1);
    }

    let elapsed_ms = started.elapsed().as_millis() as u64;
    let report = SearchReport {
        best_move: best_mv,
        search_depth: completed_depth,
        nodes: state.nodes,
        root_score: prev_score,
        white_dist,
        black_dist,
        aspiration_fails: state.aspiration_fails,
        lmr_re_searches: state.lmr_re_searches,
        mate_extensions: state.mate_extensions,
        pv_mate_failures: state.pv_mate_failures,
        depth_log: state.depth_log,
        elapsed_ms,
        root_moves: committed_root_moves,
    };
    emit_json_report(&report, config.log);
    Some(report)
}

/// CLI helper — algebraic best move after full search.
pub fn genmove_algebraic(board: &mut Board, config: SearchConfig) -> Option<String> {
    search_best_move(board, config).map(|r| format_move(r.best_move))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::board::{Board, Player};
    use crate::util::perft::format_move;

    #[test]
    fn startpos_eval_is_bounded() {
        let board = Board::new();
        let mut bfs = BfsScratch::new();
        let score = eval_stm(&board, Player::One, &mut bfs);
        assert!(score.abs() <= MAX_EVAL);
        assert_eq!(score, 0);
    }

    #[test]
    fn eval_uses_centisquares_for_path_distance() {
        let mut board = Board::new();
        board.apply_algebraic("e2");

        let mut bfs = BfsScratch::new();
        let p1_score = eval_stm(&board, Player::One, &mut bfs);
        let p2_score = eval_stm(&board, Player::Two, &mut bfs);

        assert_eq!(p1_score, CM_PER_SQUARE + PAWN_PROGRESS_CM + RACE_LEAD_CM);
        assert_eq!(p2_score, -p1_score);
    }

    #[test]
    fn eval_charges_spent_walls_without_cat() {
        let mut board = Board::new();
        board.apply_algebraic("e2");
        board.apply_algebraic("e8");
        board.apply_algebraic("a1h");

        let mut bfs = BfsScratch::new();
        let p1_score = eval_stm(&board, Player::One, &mut bfs);
        let p2_score = eval_stm(&board, Player::Two, &mut bfs);

        assert_eq!(board.walls_remaining[Player::One as usize], 9);
        assert_eq!(board.walls_remaining[Player::Two as usize], 10);
        assert_eq!(p1_score, -WALL_INVENTORY_CM);
        assert_eq!(p2_score, WALL_INVENTORY_CM);
    }

    #[test]
    fn unproven_mate_clamped_to_eval() {
        let fallback = 12;
        let fake_mate = MATE - 8;
        assert_eq!(clamp_unproven_mate(fake_mate, 3, fallback), fallback);
        assert_eq!(clamp_unproven_mate(fake_mate, 10, fallback), fake_mate);
    }

    #[test]
    fn default_max_id_depth_is_256() {
        assert_eq!(DEFAULT_MAX_ID_DEPTH, 256);
        assert_eq!(MAX_PLY, DEFAULT_MAX_ID_DEPTH);
        assert_eq!(SearchConfig::default().max_id_depth, DEFAULT_MAX_ID_DEPTH);
    }

    #[test]
    fn forced_outcome_stops_when_mate_proven_at_depth() {
        let board = Board::new();
        let tt = SearchTt::new();

        // Shallow depth: mate claim not yet proven — keep comparing root moves.
        assert!(!should_stop_forced_outcome(-MATE + 8, 5, &board, &tt, 8));
        assert!(!should_stop_forced_outcome(MATE - 8, 5, &board, &tt, 8));
        // Depth covers mate distance — stop (win or loss).
        assert!(should_stop_forced_outcome(-MATE + 8, 8, &board, &tt, 8));
        assert!(should_stop_forced_outcome(MATE - 8, 8, &board, &tt, 8));
    }

    #[test]
    fn lost_endgame_does_not_run_absurd_id_depth() {
        let moves = [
            "e2", "d2v", "c2h", "e2h", "e1h", "e8", "f2", "e7", "g2", "g2h", "b1v", "h8h", "h2",
            "f8h", "i2", "d8h", "i3", "e6", "h3", "e5", "g3", "g3h", "a2h", "e3h", "e4v", "e6",
            "e6v", "d7v", "h1h", "d5v", "h3", "e7", "i3", "e8", "i4", "f8", "i5", "g8", "h5", "g7",
            "h6", "h7", "h8", "h6", "h6v", "h5", "h4v", "h6", "g7h", "g6", "g8", "g7", "f8", "f7",
            "e8", "f8", "e7", "g8", "e6", "h8", "e5", "i8", "e4",
        ];
        let mut board = Board::new();
        for m in moves {
            board.apply_algebraic(m);
        }
        assert_eq!(board.side(), Player::Two);

        let config = SearchConfig {
            time_ms: 10_000,
            max_nodes: 2_000_000_000,
            log: false,
            book_hint: None,
            ..SearchConfig::default()
        };
        let report = search_best_move(&mut board, config).expect("report");
        assert!(
            report.search_depth < 40,
            "lost endgame should stop after mate is proven, got depth {} score {}",
            report.search_depth,
            report.root_score
        );
    }

    #[test]
    fn funnel_position_avoids_tempo_waste() {
        let moves = [
            "e2", "e8", "e3", "e7", "e4", "e6", "d1h", "e6h", "d4", "c6h", "d5", "a6h", "e5", "e5v",
        ];
        let mut board = Board::new();
        for m in moves {
            board.apply_algebraic(m);
        }
        assert_eq!(board.side(), Player::One);

        let mut bfs = BfsScratch::new();
        let our_dist = bfs
            .shortest_distance(&board, Player::One)
            .unwrap_or(DIST_PENALTY);
        let opp_dist = bfs
            .shortest_distance(&board, Player::Two)
            .unwrap_or(DIST_PENALTY);

        let config = SearchConfig {
            time_ms: 3000,
            max_nodes: 2_000_000,
            log: false,
            book_hint: None,
            ..SearchConfig::default()
        };
        let report = search_best_move(&mut board, config).expect("report");
        let gain = move_immediate_gain(&mut board, report.best_move, our_dist, opp_dist, &mut bfs);
        assert!(
            gain > 0,
            "expected move that shortens our path or lengthens theirs, got {} (score {})",
            format_move(report.best_move),
            report.root_score
        );
    }

    #[test]
    fn root_move_matches_best_scored_candidate_when_behind_with_walls() {
        // Regression: LMR at root + time stop left optimistic wall scores in root_moves
        // while bestmove stayed on the first-searched pawn (e9).
        let moves = [
            "e2", "e8", "e3", "e7", "e4", "e7h", "e5", "c7h", "d4h", "g8h", "e6h", "d7", "b4h",
            "d6", "f5", "f5v", "f4h", "c6", "b5v", "a8h", "a5h", "c8h", "f7v", "c7", "e5", "b7",
            "h6h", "b8", "c6v", "c8", "h3h", "d8", "d5", "e8", "c5",
        ];
        let mut board = Board::new();
        for m in moves {
            board.apply_algebraic(m);
        }
        assert_eq!(board.side(), Player::Two);

        let config = SearchConfig {
            time_ms: 10_000,
            max_nodes: 2_000_000_000,
            log: false,
            book_hint: None,
            ..SearchConfig::default()
        };
        let report = search_best_move(&mut board, config).expect("report");
        let best_root = report
            .root_moves
            .iter()
            .max_by_key(|r| r.score)
            .expect("root moves");
        let played = report
            .root_moves
            .iter()
            .find(|r| r.mv == crate::util::perft::format_move(report.best_move))
            .expect("played move must be among root candidates");
        // Exact ties are legal (resistance tiebreak picks among them) — the
        // played move's SCORE must match the best candidate score.
        assert_eq!(
            played.score, best_root.score,
            "played {} (score {}) but best root candidate is {} (score {})",
            played.mv, played.score, best_root.mv, best_root.score
        );
    }

    fn replay(moves: &[&str]) -> Board {
        let mut board = Board::new();
        for m in moves {
            board.apply_algebraic(m);
        }
        board
    }

    const POS_TQ1_LOST_PLY69: &[&str] = &[
        "e2", "d2v", "e4v", "e2h", "f2", "f1v", "e2", "d1h", "e8h", "b1h", "f2", "a8h", "f8v",
        "d9", "f1", "c9", "e1", "c8", "d1", "c8h", "c1", "f6v", "b1", "c7", "a1", "b7", "b6h",
        "f5h", "a7v", "d4v", "c6v", "c7", "c7h", "b7", "f3v", "b8", "a2", "c8", "b2", "d8", "c2",
        "e8", "d2", "e7", "d3", "e6", "d4", "e5", "d5", "e4", "d6", "e3", "g2h", "f3", "e6", "f4",
        "e5", "f5", "e4", "g5", "e3", "g4", "f3", "h4", "f4", "h3", "h3v", "h4", "f5",
    ];

    fn pos_tq1_lost_ply73() -> Board {
        let mut board = replay(POS_TQ1_LOST_PLY69);
        for m in ["h5", "g5", "i5"] {
            board.apply_algebraic(m);
        }
        board
    }

    #[test]
    fn lost_mate_tq1_ply69_stops_before_absurd_depth() {
        let mut board = replay(POS_TQ1_LOST_PLY69);
        let config = SearchConfig {
            time_ms: 10_000,
            max_nodes: 2_000_000_000,
            log: false,
            book_hint: None,
            ..SearchConfig::default()
        };
        let report = search_best_move(&mut board, config).expect("report");
        assert!(
            report.search_depth < 40,
            "must not spin to absurd depth, got {} score {}",
            report.search_depth,
            report.root_score
        );
        if let Some(dist) = mate_distance(report.root_score) {
            assert!(
                report.search_depth <= dist + 10,
                "ply69 lost mate should stop near mate_dist+slack, got depth {} dist {}",
                report.search_depth,
                dist
            );
        }
    }

    #[test]
    fn lost_mate_tq1_ply73_stops_before_absurd_depth() {
        let mut board = pos_tq1_lost_ply73();
        let config = SearchConfig {
            time_ms: 10_000,
            max_nodes: 2_000_000_000,
            log: false,
            book_hint: None,
            ..SearchConfig::default()
        };
        let report = search_best_move(&mut board, config).expect("report");
        assert!(
            report.search_depth < 40,
            "ply73 must not spin, got depth {} score {}",
            report.search_depth,
            report.root_score
        );
        if let Some(dist) = mate_distance(report.root_score) {
            assert!(
                report.search_depth <= dist + 6,
                "ply73 lost mate should stop near mate_dist+slack, got depth {} dist {}",
                report.search_depth,
                dist
            );
        }
    }

    #[test]
    fn after_e2_depth_log_does_not_oscillate_zero_and_negative() {
        let mut board = Board::new();
        board.apply_algebraic("e2");
        assert_eq!(board.side(), Player::Two);

        let config = SearchConfig {
            time_ms: 500,
            max_nodes: 500_000,
            log: false,
            book_hint: None,
            ..SearchConfig::default()
        };
        let report = search_best_move(&mut board, config).expect("report");
        assert!(report.search_depth >= 4, "depth {}", report.search_depth);

        let d1 = report
            .depth_log
            .iter()
            .find(|e| e.depth == 1)
            .map(|e| e.score)
            .expect("d1");
        assert!(
            d1 < -50,
            "d1 must see white edge after e8 (pawn leaf); got {d1}"
        );
    }

    #[test]
    fn opening_startpos_reaches_reasonable_depth() {
        let mut board = Board::new();
        let config = SearchConfig {
            time_ms: 10_000,
            max_nodes: 2_000_000_000,
            log: false,
            book_hint: None,
            ..SearchConfig::default()
        };
        let report = search_best_move(&mut board, config).expect("report");
        let min_depth = if cfg!(debug_assertions) { 3 } else { 4 };
        assert!(
            report.search_depth >= min_depth,
            "opening should reach depth >= {min_depth} in 10s, got {}",
            report.search_depth
        );
    }

    #[test]
    fn startpos_search_no_false_mate_at_shallow_depth() {
        let mut board = Board::new();
        let config = SearchConfig {
            time_ms: 500,
            max_nodes: 500_000,
            log: false,
            book_hint: None,
            ..SearchConfig::default()
        };
        let report = search_best_move(&mut board, config).expect("report");
        assert!(
            !is_mate_score(report.root_score),
            "root score should not be mate from startpos: {}",
            report.root_score
        );
        for entry in &report.depth_log {
            assert!(
                !is_mate_score(entry.score),
                "depth {} false mate {}",
                entry.depth,
                entry.score
            );
        }
    }
}
