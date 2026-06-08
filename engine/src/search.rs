//! Iterative-deepening αβ with aspiration windows, LMR, quiescence, and TT.

use std::time::Instant;

use crate::board::{Board, Move, Player, WallOrientation};
use crate::grid::{is_goal, square_index, unpack_square};
use crate::moves::{generate_legal_moves_slice, MAX_LEGAL_MOVES};
use crate::path::BfsScratch;
use crate::perft::format_move;

const MATE: i32 = 20_000;
const MATE_WINDOW: i32 = 500;
const MAX_PLY: u32 = 64;
const DIST_PENALTY: u8 = 255;
const MAX_EVAL: i32 = 500;

const LMR_MIN_DEPTH: u32 = 2;
// Full-depth moves before LMR kicks in — 4 protects the critical 4th move
// (e.g. the best reply wall when opp has 3 pawn options).
const LMR_AFTER_MOVE: usize = 4;
const ASPIRATION_DELTA: i32 = 20;
const MAX_QDEPTH: u32 = 10;
// Futility margin per depth ply (centipawns equivalent in our eval units).
// At depth 1 we allow 25 slack, at depth 2 we allow 50 — beyond that no futility.
const FUTILITY_MARGIN: [i32; 3] = [0, 25, 50];
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
}

#[derive(Debug, Clone, Copy)]
pub struct SearchConfig {
    pub time_ms: u64,
    pub max_nodes: u64,
    pub log: bool,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            time_ms: DEFAULT_TIME_MS,
            max_nodes: DEFAULT_MAX_NODES,
            log: false,
        }
    }
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
    /// Stockfish-style LMR table: lmr_table[depth][moves_searched] = plies to reduce.
    /// Formula: floor(0.5 + ln(depth) * ln(moves_searched) / 2.25)
    lmr_table: [[u32; 64]; 64],
    /// Per-branch forcing extension budget.
    /// Each branch can contribute at most this many extension plies before
    /// extensions stop firing — prevents depth from never decreasing.
    /// Saved before and restored after each node's subtree so sibling
    /// branches each get an independent cap.
    extensions_budget: u32,
}

impl SearchState<'_> {
    fn should_stop(&self) -> bool {
        self.nodes >= self.config.max_nodes || Instant::now() >= self.deadline
    }

    fn bump_nodes(&mut self) -> bool {
        self.nodes += 1;
        self.nodes % 4096 == 0 && self.should_stop()
    }
}

/// Precompute LMR reduction table.
/// Formula: floor(0.5 + ln(depth) * ln(moves_searched) / 2.25)
/// Lower base (0.5) and higher divisor (2.25) → gentler reductions that
/// still grow with depth and move count, but protect early moves better.
/// The cap of depth/2 ensures we never burn more than half our remaining budget.
/// At depth 12, move 5  → 1 ply reduced.
/// At depth 12, move 15 → 3 plies reduced.
/// At depth 12, move 40 → 4 plies reduced.
fn build_lmr_table() -> [[u32; 64]; 64] {
    let mut table = [[0u32; 64]; 64];
    for depth in 1usize..64 {
        for mv_count in 1usize..64 {
            let r = 0.5 + (depth as f64).ln() * (mv_count as f64).ln() / 2.25;
            // Cap at depth/2 — never reduce by more than half the remaining budget.
            let cap = (depth / 2) as u32;
            table[depth][mv_count] = (r.max(0.0) as u32).min(cap);
        }
    }
    table
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
        Some(d) => d <= remaining_depth,
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



/// Path distance + wall stock — bounded so horizon leaves cannot look like mate.
fn eval_stm(board: &Board, stm: Player, bfs: &mut BfsScratch) -> i32 {
    let us = stm;
    let opp = stm.opposite();
    let our = i32::from(bfs.shortest_distance(board, us).unwrap_or(DIST_PENALTY));
    let opp_d = i32::from(bfs.shortest_distance(board, opp).unwrap_or(DIST_PENALTY));
    let our_walls = i32::from(board.walls_remaining[us as usize]);
    let opp_walls = i32::from(board.walls_remaining[opp as usize]);
    let wall_term = (our_walls - opp_walls) * 2;
    (opp_d - our + wall_term).clamp(-MAX_EVAL, MAX_EVAL)
}

fn terminal_score(ply: u32) -> i32 {
    -MATE + ply as i32
}


fn wall_blocks_path_step(mv: Move, sq1: u8, sq2: u8) -> bool {
    let Move::Wall {
        row,
        col,
        orientation,
    } = mv
    else {
        return false;
    };
    let (r1, c1) = unpack_square(sq1);
    let (r2, c2) = unpack_square(sq2);
    match orientation {
        WallOrientation::Horizontal => {
            if c1 == c2 && r1.abs_diff(r2) == 1 {
                let min_r = r1.min(r2);
                min_r == row && (c1 == col || c1 == col + 1)
            } else {
                false
            }
        }
        WallOrientation::Vertical => {
            if r1 == r2 && c1.abs_diff(c2) == 1 {
                let min_c = c1.min(c2);
                min_c == col && (r1 == row || r1 == row + 1)
            } else {
                false
            }
        }
    }
}

fn wall_intersects_path(mv: Move, path: &[u8], len: usize) -> bool {
    if len <= 1 {
        return false;
    }
    for i in 0..(len - 1) {
        if wall_blocks_path_step(mv, path[i], path[i + 1]) {
            return true;
        }
    }
    false
}

fn get_shortest_path(
    board: &Board,
    player: Player,
    bfs: &mut BfsScratch,
    path_out: &mut [u8; 81],
) -> usize {
    let mut next_out = [u8::MAX; 81];
    bfs.fill_next_toward_goal(board, player, &mut next_out);

    let (pr, pc) = board.pawn(player);
    let mut current = square_index(pr, pc);
    let mut len = 0;
    while current != u8::MAX {
        path_out[len] = current;
        len += 1;
        if len >= 81 {
            break;
        }
        current = next_out[current as usize];
    }
    len
}

fn path_distance(player: Player, path: &[u8], len: usize) -> u8 {
    if len == 0 {
        return DIST_PENALTY;
    }
    let last_sq = path[len - 1];
    let (r, _) = unpack_square(last_sq);
    if is_goal(player, r) {
        (len - 1) as u8
    } else {
        DIST_PENALTY
    }
}

fn wall_disturbs_path(board: &mut Board, mv: Move, opp_dist: u8, bfs: &mut BfsScratch) -> bool {
    let Move::Wall { .. } = mv else {
        return false;
    };
    let opp = board.side().opposite();
    let undo = board.make_move(mv);
    let new_opp = bfs.shortest_distance(board, opp).unwrap_or(DIST_PENALTY);
    board.unmake_move(undo);
    new_opp > opp_dist
}

/// A pawn move is tactical if it actually shortens our BFS distance to the goal.
/// This is stricter than row-progress: a sideways detour can be "forward" by row
/// but not reduce the path length at all.
/// We avoid a full BFS by using the pre-computed `our_dist` and checking that the
/// new square is strictly closer to the goal row than the current position.
/// For final-rank moves (dist == 1), any pawn move to the goal row is always tactical.
fn is_tactical_pawn(board: &Board, mv: Move, our_dist: u8) -> bool {
    let Move::Pawn { row, .. } = mv else {
        return false;
    };
    let stm = board.side();
    let goal = if stm == Player::One { 8u8 } else { 0u8 };
    // Distance from destination square to goal row.
    let dest_dist_to_goal = row.abs_diff(goal);
    // Tactical: the new square is strictly closer to the goal than our current distance.
    // our_dist is pawn-steps; dest_dist_to_goal is row-distance (lower bound on pawn-steps).
    // This correctly handles both straight advances and jump-overs.
    dest_dist_to_goal < our_dist
}

fn is_tactical_move(
    board: &mut Board,
    mv: Move,
    our_dist: u8,
    opp_dist: u8,
    opp_path: &[u8],
    opp_path_len: usize,
    bfs: &mut BfsScratch,
) -> bool {
    match mv {
        Move::Pawn { .. } => is_tactical_pawn(board, mv, our_dist),
        Move::Wall { .. } => {
            if wall_intersects_path(mv, opp_path, opp_path_len) {
                wall_disturbs_path(board, mv, opp_dist, bfs)
            } else {
                false
            }
        }
    }
}

fn wall_proximity_score(mv: Move, opp_goal: u8) -> i32 {
    let Move::Wall { row, .. } = mv else { return 0 };
    // Higher = closer to opponent's goal = more likely to block them.
    80i32 - i32::from(row.abs_diff(opp_goal))
}

fn collect_moves(
    board: &mut Board,
    buf: &mut [Move],
    bfs: &mut BfsScratch,
    tactical_only: bool,
    // Maximum number of quiet (non-path-disturbing) walls to include.
    // 0 = same as old prune_quiet_walls=true but depth-scaled.
    // usize::MAX = include all quiet walls (too slow at root).
    max_quiet_walls: usize,
    allow_walls: bool,
) -> usize {
    let mut scratch = [Move::Pawn { row: 0, col: 0 }; MAX_LEGAL_MOVES];
    let full = generate_legal_moves_slice(board, &mut scratch, bfs);
    if full == 0 {
        return 0;
    }

    let mut opp_path = [0u8; 81];
    let mut opp_path_len = 0;
    let mut opp_dist = DIST_PENALTY;
    if allow_walls {
        opp_path_len = get_shortest_path(board, board.side().opposite(), bfs, &mut opp_path);
        opp_dist = path_distance(board.side().opposite(), &opp_path, opp_path_len);
    }
    let our_dist = bfs.shortest_distance(board, board.side()).unwrap_or(DIST_PENALTY);

    let stm = board.side();
    let opp_goal: u8 = if stm.opposite() == Player::One { 8 } else { 0 };
    let mut n = 0usize;

    // First pass: all pawns + path-disturbing walls.
    for i in 0..full {
        let mv = scratch[i];
        match mv {
            Move::Pawn { .. } => {
                if tactical_only && !is_tactical_pawn(board, mv, our_dist) {
                    continue;
                }
                buf[n] = mv;
                n += 1;
            }
            Move::Wall { .. } => {
                if !allow_walls {
                    continue;
                }
                let disturbs = if wall_intersects_path(mv, &opp_path, opp_path_len) {
                    wall_disturbs_path(board, mv, opp_dist, bfs)
                } else {
                    false
                };
                if tactical_only && !disturbs {
                    continue;
                }
                if disturbs {
                    buf[n] = mv;
                    n += 1;
                }
            }
        }
    }

    // Second pass: top quiet walls by proximity to opponent goal, up to cap.
    if allow_walls && !tactical_only && max_quiet_walls > 0 {
        // Collect quiet wall candidates with their proximity score.
        let mut quiet: [(i32, Move); MAX_LEGAL_MOVES] =
            [(0, Move::Pawn { row: 0, col: 0 }); MAX_LEGAL_MOVES];
        let mut q_count = 0usize;
        for i in 0..full {
            let mv = scratch[i];
            if !matches!(mv, Move::Wall { .. }) {
                continue;
            }
            // Skip walls already added as path-disturbing.
            let blocks_path = wall_intersects_path(mv, &opp_path, opp_path_len);
            let disturbs = if blocks_path {
                wall_disturbs_path(board, mv, opp_dist, bfs)
            } else {
                false
            };
            if disturbs {
                continue;
            }
            let score = wall_proximity_score(mv, opp_goal);
            quiet[q_count] = (score, mv);
            q_count += 1;
        }
        // Partial sort: extract top max_quiet_walls by score (selection sort, cheap for small cap).
        let take = max_quiet_walls.min(q_count);
        for slot in 0..take {
            let mut best_idx = slot;
            for j in (slot + 1)..q_count {
                if quiet[j].0 > quiet[best_idx].0 {
                    best_idx = j;
                }
            }
            quiet.swap(slot, best_idx);
            buf[n] = quiet[slot].1;
            n += 1;
        }
    }

    if n == 0 && !tactical_only {
        buf[..full].copy_from_slice(&scratch[..full]);
        return full;
    }
    if n == 0 && tactical_only {
        for i in 0..full {
            if matches!(scratch[i], Move::Pawn { .. }) {
                buf[n] = scratch[i];
                n += 1;
            }
        }
    }
    n
}

fn move_order_score(
    board: &Board,
    mv: Move,
    tt_best: Option<Move>,
    opp_path: &[u8],
    opp_path_len: usize,
    _opp_dist: u8,
    bfs: &mut BfsScratch,
) -> i32 {
    if tt_best == Some(mv) {
        return 10_000;
    }

    let stm = board.side();
    let base_our = bfs.shortest_distance(board, stm).unwrap_or(DIST_PENALTY);
    match mv {
        Move::Pawn { row, .. } => {
            let goal = if stm == Player::One { 8 } else { 0 };
            let progress = i32::from(base_our) - i32::from(row.abs_diff(goal));
            500 + progress * 10
        }
        Move::Wall { row, .. } => {
            let opp_goal = if stm.opposite() == Player::One { 8 } else { 0 };
            // row_bonus: 0..80 (highest near opponent goal)
            let row_bonus = 80i32 - i32::from(row.abs_diff(opp_goal)) * 8;
            let stock = i32::from(board.walls_remaining[stm as usize]);

            // Non-mutating fast check for path intersection
            let disturbs = wall_intersects_path(mv, opp_path, opp_path_len);

            if disturbs {
                // Tactical blocking wall: high priority
                600 + row_bonus + stock
            } else {
                // Quiet wall: low priority
                200 + row_bonus + stock
            }
        }
    }
}

fn order_moves(
    board: &Board,
    moves: &mut [Move],
    n: usize,
    tt_best: Option<Move>,
    scores: &mut [i32; MAX_LEGAL_MOVES],
    opp_path: &[u8],
    opp_path_len: usize,
    opp_dist: u8,
    bfs: &mut BfsScratch,
) {
    for i in 0..n {
        scores[i] = move_order_score(board, moves[i], tt_best, opp_path, opp_path_len, opp_dist, bfs);
    }
    let mut order: [usize; MAX_LEGAL_MOVES] = core::array::from_fn(|i| i);
    order[..n].sort_unstable_by(|&a, &b| scores[b].cmp(&scores[a]));
    let mut tmp = [Move::Pawn { row: 0, col: 0 }; MAX_LEGAL_MOVES];
    tmp[..n].copy_from_slice(&moves[..n]);
    for i in 0..n {
        moves[i] = tmp[order[i]];
    }
}

fn quiescence(
    state: &mut SearchState<'_>,
    board: &mut Board,
    mut alpha: i32,
    beta: i32,
    ply: u32,
    qdepth: u32,
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
    if qdepth == 0 {
        return alpha;
    }

    let mut buf = [Move::Pawn { row: 0, col: 0 }; MAX_LEGAL_MOVES];
    let n = collect_moves(board, &mut buf, state.bfs, true, 0, false);
    if n == 0 {
        return alpha;
    }

    let mut scores = [0i32; MAX_LEGAL_MOVES];
    order_moves(
        board,
        &mut buf,
        n,
        None,
        &mut scores,
        &[],
        0,
        DIST_PENALTY,
        state.bfs,
    );

    for i in 0..n {
        let mv = buf[i];
        let undo = board.make_move(mv);
        let mut score = -quiescence(state, board, -beta, -alpha, ply + 1, qdepth - 1);
        let fallback = eval_stm(board, board.side().opposite(), state.bfs);
        score = clamp_unproven_mate(score, qdepth.saturating_sub(1), fallback);
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

    let stand = eval_stm(board, board.side(), state.bfs);
    clamp_unproven_mate(alpha, qdepth, stand)
}

fn make_null_move(board: &mut Board) -> u64 {
    let old_hash = board.hash;
    crate::zobrist::xor_side(&mut board.hash);
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
    let fallback = eval_stm(board, board.side().opposite(), state.bfs);
    score = clamp_unproven_mate(score, depth, fallback);

    // Mate extension: if the child returns a mate claim that the remaining depth
    // cannot prove, keep extending (up to 3 extra plies) until either the claim
    // is proven or we run out of budget.  This ensures forcing wins are never
    // truncated at the horizon.
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
        score = clamp_unproven_mate(score, proven_depth + 1, fallback);
    }

    score
}

fn negamax(
    state: &mut SearchState<'_>,
    board: &mut Board,
    depth: u32,
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

    // Hard ply ceiling — prevents runaway extensions from overflowing the stack.
    if ply >= MAX_PLY {
        return eval_stm(board, board.side(), state.bfs);
    }

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
            let corrected = clamp_unproven_mate(
                score,
                depth,
                eval_stm(board, board.side(), state.bfs),
            );
            match bound {
                TtBound::Exact => return corrected,
                TtBound::Lower if corrected >= beta => return corrected,
                TtBound::Upper if corrected <= alpha => return corrected,
                _ => {}
            }
        }
    }

    if depth == 0 {
        return quiescence(state, board, alpha, beta, ply, MAX_QDEPTH);
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
    let apply_futility = ply > 0
        && futility_depth <= 2
        && !is_mate_score(static_eval)
        && !is_mate_score(alpha);

    // Quiet-wall cap: root keeps 3/6/10; sub-nodes are very tight to control
    // branching factor. Path-disturbing walls are NEVER capped (added separately).
    // Target effective branching factor ~7-8 to reach depth 12 in 10 seconds.
    let quiet_wall_cap = if ply == 0 {
        match depth {
            0..=2 => 3usize,
            3..=4 => 6usize,
            _     => 10usize,
        }
    } else {
        // Sub-nodes: keep very lean — only 1 strategic quiet wall at high depth.
        // Path-blocking walls are already captured in the tactical pass.
        match depth {
            0..=2 => 0usize,
            3..=4 => 1usize,
            _     => 2usize,
        }
    };
    let mut buf = [Move::Pawn { row: 0, col: 0 }; MAX_LEGAL_MOVES];
    let n = collect_moves(board, &mut buf, state.bfs, false, quiet_wall_cap, true);
    if n == 0 {
        return eval_stm(board, board.side(), state.bfs);
    }

    let mut opp_path = [0u8; 81];
    let opp_path_len = get_shortest_path(board, board.side().opposite(), state.bfs, &mut opp_path);
    let opp_dist_pre = path_distance(board.side().opposite(), &opp_path, opp_path_len);
    let our_dist_pre = state.bfs.shortest_distance(board, board.side()).unwrap_or(DIST_PENALTY);

    let mut scores = [0i32; MAX_LEGAL_MOVES];
    order_moves(
        board,
        &mut buf,
        n,
        tt_best,
        &mut scores,
        &opp_path,
        opp_path_len,
        opp_dist_pre,
        state.bfs,
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
    const MAX_EXTENSIONS_PER_BRANCH: u32 = 4;
    let forcing_extension: u32 = if ply > 0
        && depth > 1
        && state.extensions_budget > 0
    {
        let pawn_count = buf[..n]
            .iter()
            .filter(|m| matches!(m, Move::Pawn { .. }))
            .count();
        let near_goal = our_dist_pre <= 2 || opp_dist_pre <= 2;
        if pawn_count <= 1 || near_goal { 1 } else { 0 }
    } else {
        0
    };
    // Save budget now; decrement for this whole subtree; restore after the loop.
    let budget_before_subtree = state.extensions_budget;
    if forcing_extension > 0 {
        state.extensions_budget = state.extensions_budget.saturating_sub(1);
    }
    let _ = MAX_EXTENSIONS_PER_BRANCH; // used in init only

    let mut best_score = -MATE;
    let mut best_mv = buf[0];
    let mut best_packed = pack_move(best_mv);
    let mut moves_searched = 0usize;
    let original_alpha = alpha;

    for i in 0..n {
        let mv = buf[i];

        // ── Tactical classification ───────────────────────────────────────────
        // Compute this once per move — used by both futility and LMR.
        // A move is tactical if it:
        //   (a) Shortens our BFS distance to goal (pawn), or
        //   (b) Disturbs the opponent's shortest path (wall).
        // Tactical moves are NEVER reduced or pruned.
        let is_tactical = if moves_searched == 0
            || depth < LMR_MIN_DEPTH
            || moves_searched < LMR_AFTER_MOVE
        {
            // We treat early moves as tactical by definition (no reduction either way).
            true
        } else {
            is_tactical_move(
                board,
                mv,
                our_dist_pre,
                opp_dist_pre,
                &opp_path,
                opp_path_len,
                state.bfs,
            )
        };

        // ── Futility pruning ──────────────────────────────────────────────────
        // At depth 1-2, if the static eval is already so far below alpha that
        // even a large margin cannot recover, skip quiet moves entirely.
        if apply_futility && !is_tactical {
            let margin = FUTILITY_MARGIN[futility_depth];
            if static_eval + margin <= alpha {
                moves_searched += 1;
                continue;
            }
        }

        // ── LMR reduction ─────────────────────────────────────────────────────
        // Formula: floor(0.5 + ln(depth) * ln(moves_searched) / 2.25)
        // Cap: depth/2 — never burn more than half the remaining budget.
        // Quiet walls (no path intersection at all) get +1 extra reduction ply
        // because a wall that touches no path is very unlikely to change the eval.
        let reduction = if is_tactical {
            0u32
        } else {
            let d = (depth as usize).min(63);
            let m = moves_searched.min(63);
            let base_r = state.lmr_table[d][m];
            // Extra reduction for pure quiet walls (not intersecting opp path at all).
            let extra = if matches!(mv, Move::Wall { .. })
                && !wall_intersects_path(mv, &opp_path, opp_path_len)
            {
                1u32
            } else {
                0u32
            };
            (base_r + extra).min(depth.saturating_sub(1))
        };

        let undo = board.make_move(mv);
        // child_depth: one ply below current, plus any forcing extension so that
        // near-forced positions are searched one ply deeper throughout the subtree.
        let child_depth = (depth - 1) + forcing_extension;
        let score = if moves_searched == 0 {
            search_child(state, board, child_depth, alpha, beta, ply)
        } else {
            let reduced = child_depth.saturating_sub(reduction);
            let mut s = if reduced == 0 {
                -quiescence(state, board, -alpha - 1, -alpha, ply + 1, MAX_QDEPTH)
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
        board.unmake_move(undo);

        if state.should_stop() {
            break;
        }

        moves_searched += 1;
        if score > best_score {
            best_score = score;
            best_mv = mv;
            best_packed = pack_move(best_mv);
        }
        if score > alpha {
            alpha = score;
        }
        if alpha >= beta {
            break;
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
    let stand_pat = eval_stm(board, board.side(), state.bfs);
    best_score = clamp_unproven_mate(best_score, depth, stand_pat);

    state.tt.store(
        hash,
        depth as i8,
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

fn corrected_root_score(board: &Board, tt: &SearchTt, claimed: i32, bfs: &mut BfsScratch) -> i32 {
    if !is_mate_score(claimed) {
        return claimed;
    }
    if verify_pv_mate(board, tt, claimed) {
        return claimed;
    }
    eval_stm(board, board.side(), bfs)
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
}

fn emit_json_report(report: &SearchReport, log: bool) {
    if !log {
        return;
    }
    let mut depth_json = String::new();
    for (i, e) in report.depth_log.iter().enumerate() {
        if i > 0 {
            depth_json.push(',');
        }
        depth_json.push_str(&format!(
            "{{\"depth\":{},\"score\":{},\"nodes\":{}}}",
            e.depth, e.score, e.nodes
        ));
    }
    eprintln!(
        "info json {{\"searchDepth\":{},\"nodes\":{},\"rootScore\":{},\"whiteDist\":{},\"blackDist\":{},\"aspirationFails\":{},\"lmrReSearches\":{},\"mateExtensions\":{},\"pvMateFailures\":{},\"elapsedMs\":{},\"depthLog\":[{}]}}",
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
        depth_json
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
        let white_dist = bfs.shortest_distance(board, Player::One).unwrap_or(DIST_PENALTY);
        let black_dist = bfs.shortest_distance(board, Player::Two).unwrap_or(DIST_PENALTY);
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
        });
    }

    let started = Instant::now();
    let deadline = started + std::time::Duration::from_millis(config.time_ms);
    let mut tt = SearchTt::new();

    let white_dist = bfs.shortest_distance(board, Player::One).unwrap_or(DIST_PENALTY);
    let black_dist = bfs.shortest_distance(board, Player::Two).unwrap_or(DIST_PENALTY);

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
        pv_move: buf[0],
        search_depth: 0,
        lmr_table: build_lmr_table(),
        extensions_budget: 4,
    };

    let root_side = board.side();
    let mut prev_score = eval_stm(board, root_side, state.bfs);
    let mut best_mv = buf[0];
    let mut completed_depth = 0u32;

    for depth in 1u32..=64 {
        if state.should_stop() {
            break;
        }

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

        let verified = corrected_root_score(board, state.tt, score, state.bfs);
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

        state.depth_log.push(DepthLogEntry {
            depth,
            score: verified,
            nodes: state.nodes,
        });
        log_depth(&state, depth, verified);

        if state.should_stop() {
            break;
        }
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
    use crate::board::Board;

    #[test]
    fn startpos_eval_is_bounded() {
        let board = Board::new();
        let mut bfs = BfsScratch::new();
        let score = eval_stm(&board, Player::One, &mut bfs);
        assert!(score.abs() <= MAX_EVAL);
        assert_eq!(score, 0);
    }

    #[test]
    fn unproven_mate_clamped_to_eval() {
        let fallback = 12;
        let fake_mate = MATE - 8;
        assert_eq!(clamp_unproven_mate(fake_mate, 3, fallback), fallback);
        assert_eq!(clamp_unproven_mate(fake_mate, 10, fallback), fake_mate);
    }

    #[test]
    fn startpos_search_no_false_mate_at_shallow_depth() {
        let mut board = Board::new();
        let config = SearchConfig {
            time_ms: 500,
            max_nodes: 500_000,
            log: false,
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
