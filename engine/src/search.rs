//! Iterative-deepening αβ with aspiration windows, LMR, quiescence, and TT.

use std::time::Instant;

use crate::board::{Board, Move, Player, WallOrientation};
use crate::grid::{has_wall, is_goal, square_index, unpack_square, wall_touch_squares};
use crate::moves::{generate_legal_moves_slice, MAX_LEGAL_MOVES};
use crate::opening::BookHint;
use crate::path::{BfsScratch, CorridorAttention};
use crate::perft::format_move;

const MATE: i32 = 20_000;
const MATE_WINDOW: i32 = 500;
const MAX_PLY: u32 = 64;
const DIST_PENALTY: u8 = 255;
const CM_PER_SQUARE: i32 = 100;
const MAX_EVAL: i32 = 10_000;
const WALL_INVENTORY_CM: i32 = 12;
const PAWN_PROGRESS_CM: i32 = 6;
const RACE_LEAD_CM: i32 = 15;
const LOW_WALL_TRAP_CM: i32 = 18;
/// Wasted turn: opponent gets to improve on reply.
const TEMPO_PENALTY: i32 = -10;
// Corridor attention thresholds — calibrated against the focused route heat range.
// Combined two-player maxima: on-path=400, delta=1=200, delta=2≈100, delta=3≈60, floor=20.
// HOT  (≥160): on-path for ≥1 player, or delta=1 for both  → never reduce/prune.
// COLD (<60):  delta≥3 for both players, or near floor      → apply extra LMR reduction.
const CAT_HOT_CM: u16 = 160;
const CAT_COLD_CM: u16 = 60;
/// Small ordering bonus for half-protruding corridor walls (centi-units, not eval).
const WALL_SHAPE_PROTRUSION_CM: i32 = 60;
/// Slightly weaker bonus for prophylactic blocks one step along the chain.
const WALL_SHAPE_PREVENT_CM: i32 = 50;

const LMR_MIN_DEPTH: u32 = 2;
// Full-depth moves before LMR kicks in — 4 protects the critical 4th move
// (e.g. the best reply wall when opp has 3 pawn options).
const LMR_AFTER_MOVE: usize = 4;
const ASPIRATION_DELTA: i32 = 200;
const MAX_QDEPTH: u32 = 10;
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
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            time_ms: DEFAULT_TIME_MS,
            max_nodes: DEFAULT_MAX_NODES,
            log: false,
            book_hint: None,
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
}

impl SearchState<'_> {
    fn should_stop(&self) -> bool {
        self.nodes >= self.config.max_nodes || Instant::now() >= self.deadline
    }

    fn bump_nodes(&mut self) -> bool {
        self.nodes += 1;
        // Check more frequently to respect limits better
        self.nodes % 1024 == 0 && self.should_stop()
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

fn pawn_mobility(board: &Board, player: Player, bfs: &mut BfsScratch) -> i32 {
    let mut copy = board.clone();
    copy.side_to_move = player;
    let mut buf = [Move::Pawn { row: 0, col: 0 }; MAX_LEGAL_MOVES];
    let n = generate_legal_moves_slice(&mut copy, &mut buf, bfs);
    buf[..n]
        .iter()
        .filter(|mv| matches!(mv, Move::Pawn { .. }))
        .count() as i32
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

    let wall_score = (i32::from(board.walls_remaining[stm as usize])
        - i32::from(board.walls_remaining[opp as usize]))
        * WALL_INVENTORY_CM;

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
        let our_mobility = pawn_mobility(board, stm, bfs);
        let opp_mobility = pawn_mobility(board, opp, bfs);
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

fn opp_path_gain(board: &mut Board, mv: Move, opp_dist: u8, bfs: &mut BfsScratch) -> i32 {
    let Move::Wall { .. } = mv else {
        return 0;
    };
    let opp = board.side().opposite();
    let undo = board.make_move(mv);
    let new_opp = bfs.shortest_distance(board, opp).unwrap_or(DIST_PENALTY);
    board.unmake_move(undo);
    i32::from(new_opp.saturating_sub(opp_dist))
}

fn our_path_gain(board: &mut Board, mv: Move, our_dist: u8, bfs: &mut BfsScratch) -> i32 {
    let Move::Pawn { .. } = mv else {
        return 0;
    };
    let us = board.side();
    let undo = board.make_move(mv);
    let new_our = bfs.shortest_distance(board, us).unwrap_or(DIST_PENALTY);
    board.unmake_move(undo);
    i32::from(our_dist.saturating_sub(new_our))
}

fn move_immediate_gain(
    board: &mut Board,
    mv: Move,
    our_dist: u8,
    opp_dist: u8,
    bfs: &mut BfsScratch,
) -> i32 {
    match mv {
        Move::Pawn { .. } => {
            let g = our_path_gain(board, mv, our_dist, bfs);
            if g > 0 {
                g
            } else {
                TEMPO_PENALTY
            }
        }
        Move::Wall { .. } => {
            let g = opp_path_gain(board, mv, opp_dist, bfs);
            if g > 0 {
                g
            } else {
                TEMPO_PENALTY
            }
        }
    }
}

fn is_tactical_move(
    board: &mut Board,
    mv: Move,
    our_dist: u8,
    opp_dist: u8,
    bfs: &mut BfsScratch,
) -> bool {
    match mv {
        Move::Pawn { .. } => our_path_gain(board, mv, our_dist, bfs) > 0,
        Move::Wall { .. } => opp_path_gain(board, mv, opp_dist, bfs) > 0,
    }
}

#[inline]
fn wall_coord_in_bounds(row: u8, col: u8) -> bool {
    row <= 7 && col <= 7
}

/// Perpendicular wall at the junction of two aligned same-orientation chain segments.
fn is_half_protruding_wall(board: &Board, row: u8, col: u8, orientation: WallOrientation) -> bool {
    if !wall_coord_in_bounds(row, col) || has_wall(board, row, col, orientation) {
        return false;
    }
    match orientation {
        WallOrientation::Vertical => {
            for hr in [row, row.saturating_add(1)] {
                if hr > 7 {
                    continue;
                }
                if col > 0
                    && has_wall(board, hr, col - 1, WallOrientation::Horizontal)
                    && has_wall(board, hr, col, WallOrientation::Horizontal)
                {
                    return true;
                }
            }
        }
        WallOrientation::Horizontal => {
            if row == 0 {
                return false;
            }
            for vc in [col, col.saturating_add(1)] {
                if vc > 7 {
                    continue;
                }
                if has_wall(board, row - 1, vc, WallOrientation::Vertical)
                    && has_wall(board, row, vc, WallOrientation::Vertical)
                {
                    return true;
                }
            }
        }
    }
    false
}

/// Shifted one step along the chain from a would-be half-protrusion site.
fn prevents_half_protruding_wall(
    board: &Board,
    row: u8,
    col: u8,
    orientation: WallOrientation,
) -> bool {
    if is_half_protruding_wall(board, row, col, orientation) {
        return false;
    }
    if !wall_coord_in_bounds(row, col) {
        return false;
    }

    let check_shift = |sr: u8, sc: u8, so: WallOrientation, dr: i8, dc: i8| -> bool {
        if so != orientation {
            return false;
        }
        let br = sr as i8 + dr;
        let bc = sc as i8 + dc;
        if br >= 0 && br <= 7 && bc >= 0 && bc <= 7 {
            br as u8 == row && bc as u8 == col
        } else {
            false
        }
    };

    for hr in 0..=7u8 {
        for hc in 1..=7u8 {
            if has_wall(board, hr, hc - 1, WallOrientation::Horizontal)
                && has_wall(board, hr, hc, WallOrientation::Horizontal)
            {
                if check_shift(hr, hc, WallOrientation::Vertical, 0, -1)
                    || check_shift(hr, hc, WallOrientation::Vertical, 0, 1)
                {
                    return true;
                }
            }
        }
    }
    for hc in 0..=7u8 {
        for vr in 1..=7u8 {
            if has_wall(board, vr - 1, hc, WallOrientation::Vertical)
                && has_wall(board, vr, hc, WallOrientation::Vertical)
            {
                if check_shift(vr, hc, WallOrientation::Horizontal, -1, 0)
                    || check_shift(vr, hc, WallOrientation::Horizontal, 1, 0)
                    || check_shift(vr, hc, WallOrientation::Horizontal, -1, -1)
                    || check_shift(vr, hc, WallOrientation::Horizontal, 1, -1)
                {
                    return true;
                }
            }
        }
    }
    false
}

fn wall_shape_local_heat(
    cat: &CorridorAttention,
    row: u8,
    col: u8,
    orientation: WallOrientation,
) -> u16 {
    let edge = cat.wall_edge_heat(row, col, orientation);
    let touch = wall_touch_squares(row, col, orientation)
        .iter()
        .map(|&(r, c)| cat.square_heat(r, c))
        .max()
        .unwrap_or(0);
    edge.max(touch)
}

fn wall_shape_attention_bonus(board: &Board, mv: Move, cat: &CorridorAttention) -> i32 {
    let Move::Wall {
        row,
        col,
        orientation,
    } = mv
    else {
        return 0;
    };
    if wall_shape_local_heat(cat, row, col, orientation) < CAT_COLD_CM {
        return 0;
    }
    if is_half_protruding_wall(board, row, col, orientation) {
        WALL_SHAPE_PROTRUSION_CM
    } else if prevents_half_protruding_wall(board, row, col, orientation) {
        WALL_SHAPE_PREVENT_CM
    } else {
        0
    }
}

fn wall_shape_relevant(board: &Board, mv: Move, cat: &CorridorAttention) -> bool {
    wall_shape_attention_bonus(board, mv, cat) > 0
}

fn wall_in_dead_zone(mv: Move, reachable: u128) -> bool {
    let Move::Wall {
        row,
        col,
        orientation,
    } = mv
    else {
        return false;
    };
    for (r, c) in wall_touch_squares(row, col, orientation) {
        if reachable & (1u128 << square_index(r, c)) != 0 {
            return false;
        }
    }
    true
}

/// Whether a wall is worth searching.
///
/// Prune fully enclosed T-walls with zero corridor heat and no race effect.
/// Keep half-protruding corridor walls and mouth blocks that deny opponent
/// useful protruding placements (hot edge or hot touched square).
fn wall_should_search(
    mv: Move,
    cat: &CorridorAttention,
    reachable: u128,
    board: &mut Board,
    opp_dist: u8,
    opp_path: &[u8],
    opp_path_len: usize,
    bfs: &mut BfsScratch,
) -> bool {
    if wall_in_dead_zone(mv, reachable) {
        return false;
    }
    let Move::Wall {
        row,
        col,
        orientation,
    } = mv
    else {
        return false;
    };

    if cat.wall_edge_heat(row, col, orientation) >= CAT_COLD_CM {
        return true;
    }
    if opp_path_gain(board, mv, opp_dist, bfs) > 0 {
        return true;
    }
    if wall_intersects_path(mv, opp_path, opp_path_len) {
        return true;
    }
    // Hot touched square: denies opponent a half-protruding anchor on the corridor.
    for (r, c) in wall_touch_squares(row, col, orientation) {
        if cat.square_heat(r, c) >= CAT_COLD_CM {
            return true;
        }
    }
    if wall_shape_relevant(board, mv, cat) {
        return true;
    }
    false
}

fn collect_moves(
    board: &mut Board,
    buf: &mut [Move],
    bfs: &mut BfsScratch,
    tactical_only: bool,
    allow_walls: bool,
) -> usize {
    let mut scratch = [Move::Pawn { row: 0, col: 0 }; MAX_LEGAL_MOVES];
    let full = generate_legal_moves_slice(board, &mut scratch, bfs);
    if full == 0 {
        return 0;
    }

    let mut opp_dist = DIST_PENALTY;
    let mut opp_path = [0u8; 81];
    let mut opp_path_len = 0usize;
    let cat = if allow_walls {
        let opp = board.side().opposite();
        opp_dist = bfs.shortest_distance(board, opp).unwrap_or(DIST_PENALTY);
        opp_path_len = get_shortest_path(board, opp, bfs, &mut opp_path);
        bfs.build_corridor_attention(board)
    } else {
        CorridorAttention::default()
    };
    let our_dist = bfs
        .shortest_distance(board, board.side())
        .unwrap_or(DIST_PENALTY);
    let reachable = bfs.both_reachable_mask(board);

    let mut n = 0usize;

    for i in 0..full {
        let mv = scratch[i];
        match mv {
            Move::Pawn { .. } => {
                if tactical_only && our_path_gain(board, mv, our_dist, bfs) <= 0 {
                    continue;
                }
                buf[n] = mv;
                n += 1;
            }
            Move::Wall { .. } => {
                if !allow_walls {
                    continue;
                }
                if !wall_should_search(
                    mv,
                    &cat,
                    reachable,
                    board,
                    opp_dist,
                    &opp_path,
                    opp_path_len,
                    bfs,
                ) {
                    continue;
                }
                buf[n] = mv;
                n += 1;
            }
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

fn cat_score_for_move(mv: Move, cat: &CorridorAttention) -> i32 {
    match mv {
        Move::Pawn { row, col } => i32::from(cat.square_heat(row, col)),
        Move::Wall {
            row,
            col,
            orientation,
        } => i32::from(cat.wall_edge_heat(row, col, orientation)),
    }
}

fn move_corridor_attention(board: &Board, mv: Move, cat: &CorridorAttention) -> i32 {
    cat_score_for_move(mv, cat) + wall_shape_attention_bonus(board, mv, cat)
}

fn move_order_score(
    board: &mut Board,
    mv: Move,
    tt_best: Option<Move>,
    book_hint: Option<BookHint>,
    our_dist: u8,
    opp_dist: u8,
    bfs: &mut BfsScratch,
    cat: &CorridorAttention,
) -> i32 {
    if tt_best == Some(mv) {
        return 10_000;
    }
    if let Some(hint) = book_hint {
        if hint.mv == mv {
            let bias = i32::from(hint.stm_bias) / 2;
            return 12_000 + i32::from(hint.priority) + bias;
        }
    }
    let gain = move_immediate_gain(board, mv, our_dist, opp_dist, bfs);
    if gain > 0 {
        1000 + gain * 100
    } else {
        move_corridor_attention(board, mv, cat) + TEMPO_PENALTY
    }
}

fn order_moves(
    board: &mut Board,
    moves: &mut [Move],
    n: usize,
    tt_best: Option<Move>,
    book_hint: Option<BookHint>,
    scores: &mut [i32; MAX_LEGAL_MOVES],
    our_dist: u8,
    opp_dist: u8,
    bfs: &mut BfsScratch,
    cat: &CorridorAttention,
) {
    for i in 0..n {
        scores[i] = move_order_score(
            board, moves[i], tt_best, book_hint, our_dist, opp_dist, bfs, cat,
        );
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
    let n = collect_moves(board, &mut buf, state.bfs, true, true);
    if n == 0 {
        return alpha;
    }

    let our_dist = state
        .bfs
        .shortest_distance(board, board.side())
        .unwrap_or(DIST_PENALTY);
    let opp_dist = state
        .bfs
        .shortest_distance(board, board.side().opposite())
        .unwrap_or(DIST_PENALTY);

    let cat = state.bfs.build_corridor_attention(board);
    let mut scores = [0i32; MAX_LEGAL_MOVES];
    order_moves(
        board,
        &mut buf,
        n,
        None,
        state.book_hint,
        &mut scores,
        our_dist,
        opp_dist,
        state.bfs,
        &cat,
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
            let corrected =
                clamp_unproven_mate(score, depth, eval_stm(board, board.side(), state.bfs));
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
    let apply_futility =
        ply > 0 && futility_depth <= 2 && !is_mate_score(static_eval) && !is_mate_score(alpha);

    let mut buf = [Move::Pawn { row: 0, col: 0 }; MAX_LEGAL_MOVES];
    let n = collect_moves(board, &mut buf, state.bfs, false, true);
    if n == 0 {
        return eval_stm(board, board.side(), state.bfs);
    }

    let mut opp_path = [0u8; 81];
    let opp_path_len = get_shortest_path(board, board.side().opposite(), state.bfs, &mut opp_path);
    let opp_dist_pre = path_distance(board.side().opposite(), &opp_path, opp_path_len);
    let our_dist_pre = state
        .bfs
        .shortest_distance(board, board.side())
        .unwrap_or(DIST_PENALTY);

    let cat = state.bfs.build_corridor_attention(board);
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
    const MAX_EXTENSIONS_PER_BRANCH: u32 = 4;
    let forcing_extension: u32 = if ply > 0 && depth > 1 && state.extensions_budget > 0 {
        let pawn_count = buf[..n]
            .iter()
            .filter(|m| matches!(m, Move::Pawn { .. }))
            .count();
        let near_goal = our_dist_pre <= 2 || opp_dist_pre <= 2;
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
    let _ = MAX_EXTENSIONS_PER_BRANCH; // used in init only

    let mut best_score = -MATE;
    let mut best_mv = buf[0];
    let mut best_packed = pack_move(best_mv);
    let mut moves_searched = 0usize;
    let original_alpha = alpha;

    // At root, clear diagnostics so only the current depth's data is retained.
    if ply == 0 {
        state.root_moves.clear();
        state.root_best_resistance = i32::MIN;
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
        let corridor_relevant = cat_cm >= i32::from(CAT_COLD_CM);
        let is_tactical = if moves_searched == 0
            || depth < LMR_MIN_DEPTH
            || moves_searched < LMR_AFTER_MOVE
            || cat_cm >= i32::from(CAT_HOT_CM)
        {
            // We treat early moves as tactical by definition (no reduction either way).
            true
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
            let extra = if matches!(mv, Move::Wall { .. }) && cat_cm == 0 {
                2u32
            } else if matches!(mv, Move::Wall { .. })
                && !wall_intersects_path(mv, &opp_path, opp_path_len)
                && !corridor_relevant
            {
                1u32
            } else if cat_cm < i32::from(CAT_COLD_CM) {
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

        // ── Root diagnostics & resistance tiebreaking ──────────────────────────
        // Board is now back to pre-move state; compute gain and resistance here.
        // Resistance = opp_dist_after - our_dist_after (higher is better for us).
        const ROOT_TIEBREAK_BAND: i32 = 15;
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
                mv: crate::perft::format_move(mv),
                score,
                white_dist_after: root_w_dist,
                black_dist_after: root_b_dist,
                gain: root_gain,
                mate_distance: mate_distance(score),
                is_pawn: matches!(mv, Move::Pawn { .. }),
            });
        }

        if state.should_stop() {
            break;
        }

        moves_searched += 1;

        // At root: use resistance only as a secondary tiebreaker for near-equal
        // non-mate losing scores. Mate losses are already ordered correctly by
        // the score itself: -19984 is better than -19998 because it delays mate.
        let is_better = if ply == 0 {
            if score > best_score {
                state.root_best_resistance = root_resistance;
                true
            } else if score < 0
                && best_score < 0
                && !is_mate_score(score)
                && !is_mate_score(best_score)
                && score >= best_score - ROOT_TIEBREAK_BAND
                && root_resistance > state.root_best_resistance
            {
                state.root_best_resistance = root_resistance;
                true
            } else {
                false
            }
        } else {
            score > best_score
        };

        if is_better {
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

/// Stockfish-style early exit when the outcome is already decided.
///
/// IMPORTANT: Only stop early on a *winning* (positive) mate.
/// If the score is a losing mate (verified < 0), keep searching — there may be
/// a root move that delays the loss longer, and stopping now means the engine
/// surrenders to the first losing line it finds rather than the most resilient one.
fn should_stop_forced_outcome(
    verified: i32,
    depth: u32,
    board: &Board,
    tt: &SearchTt,
    our_root_dist: u8,
) -> bool {
    if verified > 0 && is_mate_score(verified) {
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
        "info json {{\"searchDepth\":{},\"nodes\":{},\"rootScore\":{},\"whiteDist\":{},\"blackDist\":{},\"aspirationFails\":{},\"lmrReSearches\":{},\"mateExtensions\":{},\"pvMateFailures\":{},\"elapsedMs\":{},\"depthLog\":[{}],\"rootMoves\":[{}]}}",
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
        lmr_table: build_lmr_table(),
        extensions_budget: 4,
        root_moves: Vec::new(),
        root_best_resistance: i32::MIN,
    };

    let static_eval = eval_stm(board, root_side, state.bfs);
    let mut prev_score = if let Some(hint) = config.book_hint {
        static_eval.saturating_add(i32::from(hint.stm_bias) / 2)
    } else {
        static_eval
    };
    let mut best_mv = pv_move;
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

        state.depth_log.push(DepthLogEntry {
            depth,
            score: verified,
            nodes: state.nodes,
        });
        log_depth(&state, depth, verified);

        if should_stop_forced_outcome(verified, depth, board, state.tt, state.our_root_dist) {
            if state.log {
                eprintln!("info forced outcome at depth {}, stopping search", depth);
            }
            break;
        }

        if state.should_stop() {
            break;
        }
    }

    // If the search didn't go deep enough to be trusted, fall back to the book move.
    // Shallow searches (especially depth 1 trapped in quiescence) can be misled by
    // tactical noise and override good opening theory.
    const BOOK_TRUST_MIN_DEPTH: u32 = 3;
    if let Some(hint) = config.book_hint {
        if completed_depth < BOOK_TRUST_MIN_DEPTH
            && hint.priority >= 100
            && buf[..n].contains(&hint.mv)
        {
            if config.log {
                eprintln!(
                    "info book fallback: depth {} < {}, using book hint {}",
                    completed_depth,
                    BOOK_TRUST_MIN_DEPTH,
                    format_move(hint.mv)
                );
            }
            best_mv = hint.mv;
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
        root_moves: state.root_moves,
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
    use crate::board::{Board, Player};
    use crate::perft::format_move;

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
    fn root_does_not_stop_on_losing_mate() {
        let board = Board::new();
        let tt = SearchTt::new();

        assert!(!should_stop_forced_outcome(-MATE + 8, 12, &board, &tt, 8));
        assert!(should_stop_forced_outcome(MATE - 8, 12, &board, &tt, 8));
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
    fn wall_search_prunes_enclosed_t_keeps_corridor_blocks() {
        use crate::grid::set_wall;

        let board = Board::new();
        let mut bfs = BfsScratch::new();
        let cat = bfs.build_corridor_attention(&board);
        let reachable = bfs.both_reachable_mask(&board);
        let opp_dist = bfs
            .shortest_distance(&board, Player::Two)
            .unwrap_or(DIST_PENALTY);
        let mut opp_path = [0u8; 81];
        let opp_path_len = get_shortest_path(&board, Player::Two, &mut bfs, &mut opp_path);

        let passive_corner = Move::Wall {
            row: 0,
            col: 0,
            orientation: WallOrientation::Horizontal,
        };
        let mut passive_board = board.clone();
        assert!(
            !wall_should_search(
                passive_corner,
                &cat,
                reachable,
                &mut passive_board,
                opp_dist,
                &opp_path,
                opp_path_len,
                &mut bfs,
            ),
            "passive corner T-wall should be pruned"
        );

        let corridor_wall = Move::Wall {
            row: 3,
            col: 4,
            orientation: WallOrientation::Horizontal,
        };
        let mut corridor_board = board.clone();
        assert!(
            wall_should_search(
                corridor_wall,
                &cat,
                reachable,
                &mut corridor_board,
                opp_dist,
                &opp_path,
                opp_path_len,
                &mut bfs,
            ),
            "central corridor wall should stay searchable"
        );

        // Mouth block: enclosure with a hot corridor square on the boundary.
        let mut pocket = Board::new();
        for &(row, col, orient) in &[
            (1, 0, WallOrientation::Vertical),
            (1, 1, WallOrientation::Horizontal),
            (2, 0, WallOrientation::Horizontal),
        ] {
            set_wall(&mut pocket, row, col, orient, true);
        }
        let cat_pocket = bfs.build_corridor_attention(&pocket);
        let reachable_pocket = bfs.both_reachable_mask(&pocket);
        let opp_dist_pocket = bfs
            .shortest_distance(&pocket, Player::Two)
            .unwrap_or(DIST_PENALTY);
        let mut opp_path_pocket = [0u8; 81];
        let opp_path_len_pocket =
            get_shortest_path(&pocket, Player::Two, &mut bfs, &mut opp_path_pocket);

        let inner_t = Move::Wall {
            row: 0,
            col: 0,
            orientation: WallOrientation::Vertical,
        };
        let mut inner_board = pocket.clone();
        assert!(
            !wall_should_search(
                inner_t,
                &cat_pocket,
                reachable_pocket,
                &mut inner_board,
                opp_dist_pocket,
                &opp_path_pocket,
                opp_path_len_pocket,
                &mut bfs,
            ),
            "fully buried inner T-wall should be pruned"
        );

        let mouth_block = Move::Wall {
            row: 1,
            col: 2,
            orientation: WallOrientation::Vertical,
        };
        if cat_pocket.wall_edge_heat(1, 2, WallOrientation::Vertical) >= CAT_COLD_CM
            || wall_touch_squares(1, 2, WallOrientation::Vertical)
                .iter()
                .any(|&(r, c)| cat_pocket.square_heat(r, c) >= CAT_COLD_CM)
        {
            let mut mouth_board = pocket.clone();
            assert!(
                wall_should_search(
                    mouth_block,
                    &cat_pocket,
                    reachable_pocket,
                    &mut mouth_board,
                    opp_dist_pocket,
                    &opp_path_pocket,
                    opp_path_len_pocket,
                    &mut bfs,
                ),
                "mouth block on hot corridor should stay searchable"
            );
        }
    }

    #[test]
    fn half_protruding_wall_detects_perpendicular_junction() {
        use crate::grid::set_wall;

        let mut board = Board::new();
        set_wall(&mut board, 3, 3, WallOrientation::Horizontal, true);
        set_wall(&mut board, 3, 4, WallOrientation::Horizontal, true);
        assert!(is_half_protruding_wall(
            &board,
            3,
            4,
            WallOrientation::Vertical
        ));
        assert!(!is_half_protruding_wall(
            &board,
            3,
            4,
            WallOrientation::Horizontal
        ));
    }

    #[test]
    fn half_protruding_wall_detects_vertical_chain_junction() {
        use crate::grid::set_wall;

        let mut board = Board::new();
        set_wall(&mut board, 2, 4, WallOrientation::Vertical, true);
        set_wall(&mut board, 3, 4, WallOrientation::Vertical, true);
        assert!(is_half_protruding_wall(
            &board,
            3,
            4,
            WallOrientation::Horizontal
        ));
    }

    #[test]
    fn half_protruding_wall_detects_vertical_chain_other_endpoint() {
        use crate::grid::set_wall;

        let mut board = Board::new();
        set_wall(&mut board, 2, 5, WallOrientation::Vertical, true);
        set_wall(&mut board, 3, 5, WallOrientation::Vertical, true);
        assert!(is_half_protruding_wall(
            &board,
            3,
            4,
            WallOrientation::Horizontal
        ));
    }

    #[test]
    fn prevents_half_protruding_wall_shifted_from_other_endpoint() {
        use crate::grid::set_wall;

        let mut board = Board::new();
        set_wall(&mut board, 2, 5, WallOrientation::Vertical, true);
        set_wall(&mut board, 3, 5, WallOrientation::Vertical, true);
        assert!(prevents_half_protruding_wall(
            &board,
            2,
            4,
            WallOrientation::Horizontal
        ));
        assert!(prevents_half_protruding_wall(
            &board,
            4,
            4,
            WallOrientation::Horizontal
        ));
    }

    #[test]
    fn prevents_half_protruding_wall_shifted_along_chain() {
        use crate::grid::set_wall;

        let mut board = Board::new();
        set_wall(&mut board, 3, 3, WallOrientation::Horizontal, true);
        set_wall(&mut board, 3, 4, WallOrientation::Horizontal, true);

        assert!(prevents_half_protruding_wall(
            &board,
            3,
            3,
            WallOrientation::Vertical
        ));
        assert!(prevents_half_protruding_wall(
            &board,
            3,
            5,
            WallOrientation::Vertical
        ));
        assert!(!prevents_half_protruding_wall(
            &board,
            3,
            4,
            WallOrientation::Vertical
        ));
    }

    #[test]
    fn wall_shape_bonus_gated_by_local_cat_heat() {
        use crate::grid::set_wall;

        let mut board = Board::new();
        set_wall(&mut board, 3, 3, WallOrientation::Horizontal, true);
        set_wall(&mut board, 3, 4, WallOrientation::Horizontal, true);
        let cold_cat = CorridorAttention::default();
        let protrusion = Move::Wall {
            row: 3,
            col: 4,
            orientation: WallOrientation::Vertical,
        };
        assert_eq!(
            wall_shape_attention_bonus(&board, protrusion, &cold_cat),
            0,
            "cold CAT should not revive unrelated shape bonus"
        );

        let mut bfs = BfsScratch::new();
        let hot_cat = bfs.build_corridor_attention(&board);
        assert!(
            wall_shape_attention_bonus(&board, protrusion, &hot_cat) >= WALL_SHAPE_PROTRUSION_CM,
            "central corridor should pick up half-protrusion bonus"
        );
    }

    #[test]
    fn wall_search_keeps_shape_relevant_protrusion() {
        use crate::grid::set_wall;

        let mut board = Board::new();
        set_wall(&mut board, 3, 3, WallOrientation::Horizontal, true);
        set_wall(&mut board, 3, 4, WallOrientation::Horizontal, true);

        let mut bfs = BfsScratch::new();
        let cat = bfs.build_corridor_attention(&board);
        let reachable = bfs.both_reachable_mask(&board);
        let opp_dist = bfs
            .shortest_distance(&board, Player::Two)
            .unwrap_or(DIST_PENALTY);
        let mut opp_path = [0u8; 81];
        let opp_path_len = get_shortest_path(&board, Player::Two, &mut bfs, &mut opp_path);

        let protrusion = Move::Wall {
            row: 3,
            col: 4,
            orientation: WallOrientation::Vertical,
        };
        let mut search_board = board.clone();
        assert!(
            wall_should_search(
                protrusion,
                &cat,
                reachable,
                &mut search_board,
                opp_dist,
                &opp_path,
                opp_path_len,
                &mut bfs,
            ),
            "half-protruding corridor wall should stay searchable"
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
