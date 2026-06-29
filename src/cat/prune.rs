//! CAT-backed move pruning and ordering — does **not** generate moves.
//!
//! Legal moves always come from `moves::generate_legal_moves_slice`. This module
//! filters them using BFS shortest-path data and multi-route corridor heat for
//! both players, then feeds αβ / MCTS with a smaller, tactically relevant set.

use crate::cat::attention::CorridorAttention;
use crate::cat::constants::{CAT_COLD_CM, CAT_HOT_CM, DIST_PENALTY};
use crate::core::board::{Board, Move, Player, WallOrientation};
use crate::movegen::{generate_legal_moves_slice, MAX_LEGAL_MOVES};
use crate::opening::book::BookHint;
use crate::path::BfsScratch;
use crate::util::grid::{has_wall, is_goal, square_index, unpack_square, wall_touch_squares};
/// Wasted turn: opponent gets to improve on reply.
pub const TEMPO_PENALTY: i32 = -10;
const WALL_CROSS_GAP_CM: i32 = 40;
const WALL_CROSS_BLOCK_CM: i32 = 35;
const WALL_LOCAL_DENIAL_SLOTS: usize = 6;
const WALL_DENIAL_BOOST_NUM: i32 = 3;
const WALL_DENIAL_BOOST_DEN: i32 = 2;

pub fn wall_blocks_path_step(mv: Move, sq1: u8, sq2: u8) -> bool {
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

pub fn wall_intersects_path(mv: Move, path: &[u8], len: usize) -> bool {
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

pub fn get_shortest_path(
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

pub fn path_distance(player: Player, path: &[u8], len: usize) -> u8 {
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

/// Opponent path gain and our path loss from a wall — one make/unmake, two BFS.
pub fn wall_race_swing(
    board: &mut Board,
    mv: Move,
    our_dist: u8,
    opp_dist: u8,
    bfs: &mut BfsScratch,
) -> (i32, i32) {
    let Move::Wall { .. } = mv else {
        return (0, 0);
    };
    let us = board.side();
    let opp = us.opposite();
    let undo = board.make_move(mv);
    let our_after = bfs.shortest_distance(board, us).unwrap_or(DIST_PENALTY);
    let opp_after = bfs.shortest_distance(board, opp).unwrap_or(DIST_PENALTY);
    board.unmake_move(undo);
    let opp_gain = i32::from(opp_after.saturating_sub(opp_dist));
    let our_loss = i32::from(our_after.saturating_sub(our_dist));
    (opp_gain, our_loss)
}

/// Net race swing from playing a wall: opponent path lengthening minus our path lengthening.
pub fn wall_net_race(
    board: &mut Board,
    mv: Move,
    our_dist: u8,
    opp_dist: u8,
    bfs: &mut BfsScratch,
) -> i32 {
    let (opp_gain, our_loss) = wall_race_swing(board, mv, our_dist, opp_dist, bfs);
    opp_gain - our_loss
}

pub fn min_wall_net_race(our_dist: u8, opp_dist: u8) -> i32 {
    if our_dist > opp_dist {
        // Losing the race — any wall that lengthens the opponent counts.
        1
    } else if our_dist == opp_dist {
        // Tied — need a stronger swing to spend a wall.
        2
    } else {
        1
    }
}

pub fn opp_path_gain(board: &mut Board, mv: Move, opp_dist: u8, bfs: &mut BfsScratch) -> i32 {
    let Move::Wall { .. } = mv else {
        return 0;
    };
    let opp = board.side().opposite();
    let undo = board.make_move(mv);
    let new_opp = bfs.shortest_distance(board, opp).unwrap_or(DIST_PENALTY);
    board.unmake_move(undo);
    i32::from(new_opp.saturating_sub(opp_dist))
}

pub fn our_path_gain(board: &mut Board, mv: Move, our_dist: u8, bfs: &mut BfsScratch) -> i32 {
    let Move::Pawn { .. } = mv else {
        return 0;
    };
    let us = board.side();
    let undo = board.make_move(mv);
    let new_our = bfs.shortest_distance(board, us).unwrap_or(DIST_PENALTY);
    board.unmake_move(undo);
    i32::from(our_dist.saturating_sub(new_our))
}

pub fn move_immediate_gain(
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

pub fn is_tactical_move(
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

fn is_cross_gap_wall(board: &Board, row: u8, col: u8, orientation: WallOrientation) -> bool {
    if !wall_coord_in_bounds(row, col) || has_wall(board, row, col, orientation) {
        return false;
    }
    match orientation {
        WallOrientation::Horizontal => {
            row >= 1
                && row <= 6
                && has_wall(board, row - 1, col, WallOrientation::Vertical)
                && has_wall(board, row + 1, col, WallOrientation::Vertical)
        }
        WallOrientation::Vertical => {
            col >= 1
                && col <= 6
                && has_wall(board, row, col - 1, WallOrientation::Horizontal)
                && has_wall(board, row, col + 1, WallOrientation::Horizontal)
        }
    }
}

fn blocks_cross_gap_wall(board: &Board, row: u8, col: u8, orientation: WallOrientation) -> bool {
    if is_cross_gap_wall(board, row, col, orientation) || !wall_coord_in_bounds(row, col) {
        return false;
    }
    match orientation {
        WallOrientation::Horizontal => {
            for dc in [-1i8, 1i8] {
                let gap_col = col as i8 + dc;
                if !(1..=6).contains(&gap_col) {
                    continue;
                }
                let gc = gap_col as u8;
                if row >= 1
                    && row <= 6
                    && has_wall(board, row - 1, gc, WallOrientation::Vertical)
                    && has_wall(board, row + 1, gc, WallOrientation::Vertical)
                {
                    return true;
                }
            }
        }
        WallOrientation::Vertical => {
            for dr in [-1i8, 1i8] {
                let gap_row = row as i8 + dr;
                if !(1..=6).contains(&gap_row) {
                    continue;
                }
                let gr = gap_row as u8;
                if col >= 1
                    && col <= 6
                    && has_wall(board, gr, col - 1, WallOrientation::Horizontal)
                    && has_wall(board, gr, col + 1, WallOrientation::Horizontal)
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

pub fn wall_slot_index(mv: Move) -> Option<usize> {
    match mv {
        Move::Wall {
            row,
            col,
            orientation,
        } if row < 8 && col < 8 => {
            let base = match orientation {
                WallOrientation::Horizontal => 0,
                WallOrientation::Vertical => 64,
            };
            Some(base + row as usize * 8 + col as usize)
        }
        _ => None,
    }
}

fn push_wall_slot(
    row: i16,
    col: i16,
    orientation: WallOrientation,
    out: &mut [usize; WALL_LOCAL_DENIAL_SLOTS],
    n: &mut usize,
) {
    if !(0..=7).contains(&row) || !(0..=7).contains(&col) {
        return;
    }
    let mv = Move::Wall {
        row: row as u8,
        col: col as u8,
        orientation,
    };
    let Some(idx) = wall_slot_index(mv) else {
        return;
    };
    if out[..*n].contains(&idx) {
        return;
    }
    out[*n] = idx;
    *n += 1;
}

/// Wall slots made physically illegal by placing `mv`: same/cross slot plus
/// adjacent same-orientation slots. This is intentionally local and does not
/// recurse into child move generation.
pub fn locally_invalidated_wall_slots(
    mv: Move,
    out: &mut [usize; WALL_LOCAL_DENIAL_SLOTS],
) -> usize {
    let Move::Wall {
        row,
        col,
        orientation,
    } = mv
    else {
        return 0;
    };
    let mut n = 0usize;
    let row = row as i16;
    let col = col as i16;
    let other = match orientation {
        WallOrientation::Horizontal => WallOrientation::Vertical,
        WallOrientation::Vertical => WallOrientation::Horizontal,
    };
    push_wall_slot(row, col, orientation, out, &mut n);
    push_wall_slot(row, col, other, out, &mut n);
    match orientation {
        WallOrientation::Horizontal => {
            push_wall_slot(row, col - 1, orientation, out, &mut n);
            push_wall_slot(row, col + 1, orientation, out, &mut n);
        }
        WallOrientation::Vertical => {
            push_wall_slot(row - 1, col, orientation, out, &mut n);
            push_wall_slot(row + 1, col, orientation, out, &mut n);
        }
    }
    n
}

pub fn legal_neighbor_denial_heat(
    mv: Move,
    candidates: &[Move],
    direct_heats: &[i32],
    n: usize,
) -> i32 {
    let Some(self_slot) = wall_slot_index(mv) else {
        return 0;
    };
    let mut local = [usize::MAX; WALL_LOCAL_DENIAL_SLOTS];
    let local_n = locally_invalidated_wall_slots(mv, &mut local);
    let mut best = 0i32;
    for i in 0..n.min(candidates.len()).min(direct_heats.len()) {
        let Some(slot) = wall_slot_index(candidates[i]) else {
            continue;
        };
        if slot == self_slot || !local[..local_n].contains(&slot) {
            continue;
        }
        let heat = direct_heats[i].max(0);
        if heat >= i32::from(CAT_HOT_CM) {
            let boosted = heat.saturating_mul(WALL_DENIAL_BOOST_NUM) / WALL_DENIAL_BOOST_DEN;
            best = best.max(boosted);
        }
    }
    best
}

pub fn wall_shape_attention_bonus(board: &Board, mv: Move, cat: &CorridorAttention) -> i32 {
    let Move::Wall {
        row,
        col,
        orientation,
    } = mv
    else {
        return 0;
    };
    if wall_shape_local_heat(cat, row, col, orientation) < CAT_HOT_CM {
        return 0;
    }
    if is_cross_gap_wall(board, row, col, orientation) {
        WALL_CROSS_GAP_CM
    } else if blocks_cross_gap_wall(board, row, col, orientation) {
        WALL_CROSS_BLOCK_CM
    } else {
        0
    }
}

/// Live squares orthogonally adjacent to sealed-off (unreachable) territory.
pub fn corridor_mouth_mask(reachable: u128) -> u128 {
    let mut mouths = 0u128;
    for sq in 0u8..81 {
        if reachable & (1u128 << sq) == 0 {
            continue;
        }
        let (r, c) = unpack_square(sq);
        for (dr, dc) in [(-1i8, 0), (1, 0), (0, -1), (0, 1)] {
            let nr = r as i16 + dr as i16;
            let nc = c as i16 + dc as i16;
            // 0..=8 — board edges are NOT sealed territory. With 0..=9 every
            // bottom/right edge square became a phantom "mouth".
            if !(0..=8).contains(&nr) || !(0..=8).contains(&nc) {
                continue;
            }
            let neighbor = square_index(nr as u8, nc as u8);
            if reachable & (1u128 << neighbor) == 0 {
                mouths |= 1u128 << sq;
                break;
            }
        }
    }
    mouths
}

/// Mouth squares, their reachable ring, and adjacent sealed cells (the gap slot itself).
pub fn gap_play_zone_mask(reachable: u128) -> u128 {
    let mouths = corridor_mouth_mask(reachable);
    let mut zone = mouths;
    for sq in 0u8..81 {
        if mouths & (1u128 << sq) == 0 {
            continue;
        }
        let (r, c) = unpack_square(sq);
        for (dr, dc) in [(-1i8, 0), (1, 0), (0, -1), (0, 1)] {
            let nr = r as i16 + dr as i16;
            let nc = c as i16 + dc as i16;
            if !(0..=8).contains(&nr) || !(0..=8).contains(&nc) {
                continue;
            }
            // Include both live ring and the sealed gap cell — half-walls and cross-gap H/V land here.
            zone |= 1u128 << square_index(nr as u8, nc as u8);
        }
    }
    zone
}

/// Touches sealed (unreachable) territory that is not part of the gap mouth play zone.
fn wall_probes_sealed_interior(mv: Move, reachable: u128, gap_zone: u128) -> bool {
    let Move::Wall {
        row,
        col,
        orientation,
    } = mv
    else {
        return false;
    };
    for (r, c) in wall_touch_squares(row, col, orientation) {
        let sq = square_index(r, c);
        if reachable & (1u128 << sq) == 0 && gap_zone & (1u128 << sq) == 0 {
            return true;
        }
    }
    false
}

fn wall_touches_gap_zone(mv: Move, gap_zone: u128) -> bool {
    let Move::Wall {
        row,
        col,
        orientation,
    } = mv
    else {
        return false;
    };
    for (r, c) in wall_touch_squares(row, col, orientation) {
        if gap_zone & (1u128 << square_index(r, c)) != 0 {
            return true;
        }
    }
    false
}

/// SOUND "useless wall" test: true iff EVERY square the wall touches is
/// unreachable (outside both pawns' reachable region). Such a wall can never be
/// adjacent to any pawn, so it can block no path — placing it only wastes
/// inventory, which is never an advantage → it can never be the best move, so
/// pruning it is NPS-only and cannot cost Elo.
///
/// Exclusion is built in: a wall touching even ONE reachable square — including a
/// half-in-void / half-in-playable wall — returns `false` (kept), preserving the
/// tactical half-wall placement.
pub fn wall_in_dead_zone(mv: Move, reachable: u128) -> bool {
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

/// Whether a wall can affect either player's reasonable routes to goal.
pub fn wall_should_search(
    mv: Move,
    cat: &CorridorAttention,
    reachable: u128,
    gap_zone: u128,
    board: &mut Board,
    _our_dist: u8,
    _opp_dist: u8,
    opp_path: &[u8],
    opp_path_len: usize,
    _bfs: &mut BfsScratch,
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
    // Gap geometry: H through V|V (or flank block beside it) can seal/open the pocket.
    // If the wall is not in a dead zone it touches live territory — always search it.
    if is_cross_gap_wall(board, row, col, orientation)
        || blocks_cross_gap_wall(board, row, col, orientation)
    {
        return true;
    }
    // Any wall touching the playable/sealed mouth, gap slot, or immediate ring.
    if gap_zone != 0 && wall_touches_gap_zone(mv, gap_zone) {
        return true;
    }
    // Wall reaches into sealed void away from the gap — no tactical value.
    if gap_zone != 0 && wall_probes_sealed_interior(mv, reachable, gap_zone) {
        return false;
    }
    // Fast exact hit: wall blocks a step of the opponent's current shortest path.
    if wall_intersects_path(mv, opp_path, opp_path_len) {
        return true;
    }
    // CAT v3 multi-route check: does the wall cut an edge on a HOT corridor
    // (exact shortest routes / contested lanes) of either player? This is the
    // anti-tunnel-vision signal — a single witness path (CAT v2) misses
    // equal-length reroutes. CAT is built once per node: no extra BFS per wall.
    // HOT (not COLD) keeps the move list tight — the delta-2/3 fringe admitted
    // nearly every wall on the board and exploded the tree.
    cat.wall_edge_heat(row, col, orientation) >= CAT_HOT_CM
}

/// Hard skip — dead void or sealed interior away from gap; never searched (not LMR).
pub fn wall_completely_skipped(mv: Move, board: &Board, reachable: u128, gap_zone: u128) -> bool {
    if wall_in_dead_zone(mv, reachable) {
        return true;
    }
    let Move::Wall {
        row,
        col,
        orientation,
    } = mv
    else {
        return false;
    };
    if is_cross_gap_wall(board, row, col, orientation)
        || blocks_cross_gap_wall(board, row, col, orientation)
    {
        return false;
    }
    if gap_zone != 0 && wall_touches_gap_zone(mv, gap_zone) {
        return false;
    }
    gap_zone != 0 && wall_probes_sealed_interior(mv, reachable, gap_zone)
}

/// Filter legal moves for search — never generates moves, only prunes `moves` output.
/// `cat` is the caller's corridor attention and `opp_path` the caller's witness
/// shortest path — both computed once per node and shared with move ordering.
#[allow(clippy::too_many_arguments)]
pub fn collect_search_moves(
    board: &mut Board,
    buf: &mut [Move],
    bfs: &mut BfsScratch,
    cat: &CorridorAttention,
    opp_path: &[u8; 81],
    opp_path_len: usize,
    our_dist: u8,
    opp_dist: u8,
    tactical_only: bool,
    allow_walls: bool,
) -> usize {
    let mut scratch = [Move::Pawn { row: 0, col: 0 }; MAX_LEGAL_MOVES];
    let full = generate_legal_moves_slice(board, &mut scratch, bfs);
    if full == 0 {
        return 0;
    }

    let reachable = bfs.both_reachable_mask(board);
    let gap_zone = if allow_walls {
        gap_play_zone_mask(reachable)
    } else {
        0
    };

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
                // Quiescence (tactical_only): walls are "noisy" only when they
                // actually lengthen the opponent's shortest path — the Quoridor
                // analog of a capture. Quiet walls must stand pat, not extend.
                if tactical_only {
                    // Cheap witness-path gate first; BFS only for the few that touch it.
                    if !wall_intersects_path(mv, opp_path, opp_path_len)
                        || opp_path_gain(board, mv, opp_dist, bfs) <= 0
                    {
                        continue;
                    }
                } else if !wall_should_search(
                    mv,
                    cat,
                    reachable,
                    gap_zone,
                    board,
                    our_dist,
                    opp_dist,
                    opp_path,
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

    // Main search must never be left without moves — fall back to full legality.
    // Quiescence (tactical_only) returns 0 instead: a position with no noisy
    // moves is quiet by definition and the caller stands pat on static eval.
    if n == 0 && !tactical_only {
        buf[..full].copy_from_slice(&scratch[..full]);
        return full;
    }
    n
}

/// Absolute CAT corridor floor — same threshold as `wall_should_search` / CAT viz.
#[inline]
pub fn is_cat_hot_corridor(cat_cm: i32) -> bool {
    cat_cm >= i32::from(CAT_HOT_CM)
}

/// Normalized heat at this node: 0 = at/below `cold_cm`, 1 = `cat_max`.
/// A move at 200 cm vs 100 cm with max 250 scales ~11× in fraction (proportional to hotspot).
#[inline]
pub fn cat_heat_fraction(cat_cm: i32, cat_max: u16, cold_cm: u16) -> f32 {
    if cat_max <= cold_cm {
        return if cat_cm > i32::from(cold_cm) {
            1.0
        } else {
            0.0
        };
    }
    let h = cat_cm.max(0) as f32;
    let cold = cold_cm as f32;
    let max_h = cat_max as f32;
    ((h - cold) / (max_h - cold)).clamp(0.0, 1.0)
}

/// Full-depth LMR when heat fraction reaches the profile hot-ratio gate.
#[inline]
pub fn cat_heat_skips_lmr(cat_cm: i32, cat_max: u16, cold_cm: u16, hot_ratio_pct: u16) -> bool {
    if cat_max == 0 {
        return is_cat_hot_corridor(cat_cm);
    }
    cat_heat_fraction(cat_cm, cat_max, cold_cm) * 100.0 >= hot_ratio_pct as f32
}

/// Skip LMR when move heat fraction clears the hot-ratio gate (replaces flat cm cutoff).
#[inline]
pub fn is_lmr_heat_hot(cat_cm: i32, cat_max: u16, cold_cm: u16, hot_ratio_pct: u16) -> bool {
    cat_heat_skips_lmr(cat_cm, cat_max, cold_cm, hot_ratio_pct)
}

/// Per-node CAT ceilings — walls scale against wall hotspots, not the sprint pawn.
#[derive(Debug, Clone, Copy, Default)]
pub struct CatHeatRefs {
    pub all: u16,
    pub walls: u16,
    pub pawns: u16,
}

pub fn cat_heat_refs(
    buf: &[Move],
    n: usize,
    board: &Board,
    cat: &CorridorAttention,
) -> CatHeatRefs {
    let mut refs = CatHeatRefs::default();
    for i in 0..n {
        let cm = move_corridor_attention(board, buf[i], cat).max(0) as u16;
        refs.all = refs.all.max(cm);
        match buf[i] {
            Move::Wall { .. } => refs.walls = refs.walls.max(cm),
            Move::Pawn { .. } => refs.pawns = refs.pawns.max(cm),
        }
    }
    refs
}

#[inline]
pub fn cat_heat_ref_max(mv: Move, refs: CatHeatRefs) -> u16 {
    match mv {
        Move::Wall { .. } => refs.walls,
        Move::Pawn { .. } => refs.all,
    }
}

pub fn cat_heat_refs_from_scores(buf: &[Move], n: usize, cat_heats: &[i32]) -> CatHeatRefs {
    let mut refs = CatHeatRefs::default();
    for i in 0..n.min(buf.len()).min(cat_heats.len()) {
        let cm = cat_heats[i].max(0) as u16;
        refs.all = refs.all.max(cm);
        match buf[i] {
            Move::Wall { .. } => refs.walls = refs.walls.max(cm),
            Move::Pawn { .. } => refs.pawns = refs.pawns.max(cm),
        }
    }
    refs
}

/// Target child plies from CAT heat — cold fringe caps at 1–2, hotspots keep full depth.
pub fn cat_heat_child_depth(
    cat_cm: i32,
    cat_ref_max: u16,
    cold_cm: u16,
    child_depth_full: u32,
) -> u32 {
    if child_depth_full == 0 {
        return 0;
    }
    if cat_ref_max == 0 || cat_cm <= 0 {
        return 1.min(child_depth_full);
    }
    let heat_t = cat_heat_fraction(cat_cm, cat_ref_max, cold_cm);
    // Steep curve — 245cm peak keeps ~full depth; 98cm fringe gets 1–2 plies (not flat 4% nodes).
    let mut used = (heat_t.powf(2.35) * child_depth_full as f32)
        .round()
        .max(1.0) as u32;
    if heat_t < 0.08 {
        used = 1;
    } else if heat_t < 0.18 {
        used = used.min(2);
    } else if heat_t < 0.35 {
        used = used.min((child_depth_full / 3).max(2));
    }
    used.min(child_depth_full)
}

/// Default CAT attention ceiling for Titanium v16 LMR (override via `TITANIUM_CAT_LMR_CEILING`).
pub const CAT_V16_LMR_CEILING_DEFAULT: u16 = 800;
pub const CAT_V16_LMR_CEILINGS: [u16; 3] = [500, 800, 1000];
/// Fringe cutoff: moves below this fraction of the effective position maximum search at child depth 1.
pub const CAT_V16_FRINGE_PCT_DEFAULT: u16 = 5;
pub const CAT_V16_FRINGE_PCT_STEP_PER_WORKER: u16 = 10;
pub const CAT_V16_FRINGE_PCT_MAX: u16 = 70;

#[inline]
pub fn cat_v16_lmr_fringe_pct_for_worker(worker_id: usize) -> u16 {
    if worker_id == 0 {
        CAT_V16_FRINGE_PCT_DEFAULT
    } else {
        (worker_id as u16)
            .saturating_mul(CAT_V16_FRINGE_PCT_STEP_PER_WORKER)
            .min(CAT_V16_FRINGE_PCT_MAX)
    }
}

/// Parse v16 LMR ceiling from env (`500`, `800`, or `1000`); defaults to [`CAT_V16_LMR_CEILING_DEFAULT`].
pub fn cat_v16_lmr_ceiling_from_env() -> u16 {
    std::env::var("TITANIUM_CAT_LMR_CEILING")
        .ok()
        .and_then(|s| s.parse::<u16>().ok())
        .filter(|v| CAT_V16_LMR_CEILINGS.contains(v))
        .unwrap_or(CAT_V16_LMR_CEILING_DEFAULT)
}

/// Titanium v16 late-move reduction from CAT heat.
///
/// Normalizes against the hottest same-kind move in the position, capped by the
/// selected upper bound (`ceiling` in 500/800/1000 cm). Moves at or colder than
/// `fringe_pct` of that effective maximum search at child depth 1; warmer moves
/// keep a proportional fraction of the remaining depth.
pub fn cat_v16_lmr_reduction_plies(
    mv: Move,
    cat_cm: i32,
    refs: CatHeatRefs,
    ceiling: u16,
    fringe_pct: u16,
    child_depth_full: u32,
) -> u32 {
    if child_depth_full <= 1 {
        return 0;
    }
    let cat_ref = cat_heat_ref_max(mv, refs).min(ceiling);
    if cat_ref == 0 {
        return child_depth_full.saturating_sub(1);
    }
    let fringe_pct = fringe_pct.min(100);
    let threshold = (u32::from(cat_ref) * u32::from(fringe_pct)).div_ceil(100) as i32;
    if cat_cm <= threshold {
        return child_depth_full.saturating_sub(1);
    }
    let reduction =
        cat_heat_depth_reduction(cat_cm, cat_ref, threshold.max(0) as u16, child_depth_full);
    reduction.min(child_depth_full.saturating_sub(2))
}

/// CAT as a *continuous modifier* on top of base LMR — the v16 design.
///
/// Returns the EXTRA reduction to add to the conventional (move-index) LMR, not
/// a standalone reduction. `r_cat = max_extra · u^γ` where `u = 1 − cat_norm`
/// and `cat_norm = cat_cm / cat_ref` (clamped 0..1). High-impact moves get ~0
/// extra (searched near full); only genuinely low-impact moves get heavily
/// reduced. CAT answers "how dangerous is it to under-search this move?", never
/// "is it good?" — strength is decided by eval + the re-search recovery.
///
/// `fringe_pct` carries the lazy-SMP per-worker aggressiveness (main worker
/// gentle, helpers bolder) so search diversity is preserved without the old
/// hard tail-threshold.
pub fn cat_v16_lmr_extra_plies(
    mv: Move,
    cat_cm: i32,
    refs: CatHeatRefs,
    ceiling: u16,
    fringe_pct: u16,
    child_depth_full: u32,
    max_extra: f64,
) -> u32 {
    if child_depth_full <= 1 || max_extra <= 0.0 {
        return 0;
    }
    const GAMMA: f64 = 2.0;
    let cat_ref = cat_heat_ref_max(mv, refs).min(ceiling);
    let cat_norm = if cat_ref == 0 {
        0.0 // no corridor structure for this player → treat as low-impact
    } else {
        (cat_cm.max(0) as f64 / f64::from(cat_ref)).clamp(0.0, 1.0)
    };
    let unimportance = 1.0 - cat_norm;
    // Per-worker weight centred on the main worker: worker 0 (~5%) ≈ 1.0 so the
    // main search prunes meaningfully and the LMR-vision slider has a clear
    // effect; helpers scale up to ~3.0 for lazy-SMP diversity.
    let cat_weight = (0.85 + f64::from(fringe_pct.min(100)) / 33.0).clamp(0.5, 3.0);
    let extra = (max_extra * unimportance.powf(GAMMA) * cat_weight).round();
    (extra.max(0.0) as u32).min(child_depth_full.saturating_sub(1))
}

/// CAT-shaped LMR plies from heat fraction — scales child depth like the heatmap (245 ≫ 98).
pub fn cat_heat_depth_reduction(
    cat_cm: i32,
    cat_ref_max: u16,
    cold_cm: u16,
    child_depth_full: u32,
) -> u32 {
    let used = cat_heat_child_depth(cat_cm, cat_ref_max, cold_cm, child_depth_full);
    child_depth_full.saturating_sub(used)
}

/// CAT-proportional depth — no hard move-count cap; late-index table only stacks on cold moves.
pub fn cat_lmr_total_reduction(
    mv: Move,
    cat_cm: i32,
    refs: CatHeatRefs,
    cold_cm: u16,
    depth: u32,
    base_r: u32,
    opp_path: &[u8],
    opp_path_len: usize,
    corridor_relevant: bool,
) -> u32 {
    let child_full = depth.saturating_sub(1);
    let cat_ref = cat_heat_ref_max(mv, refs);
    let heat_t = if cat_ref > 0 && cat_cm > 0 {
        cat_heat_fraction(cat_cm, cat_ref, cold_cm)
    } else {
        0.0
    };
    let mut reduction = cat_heat_depth_reduction(cat_cm, cat_ref, cold_cm, child_full);
    // Stockfish LMR table only bites on cold late moves — never flatten CAT hotspots.
    if heat_t < 0.22 {
        reduction = reduction.saturating_add(base_r);
    }
    if matches!(mv, Move::Wall { .. }) && cat_cm == 0 {
        reduction = reduction.saturating_add(2);
    } else if matches!(mv, Move::Wall { .. })
        && !wall_intersects_path(mv, opp_path, opp_path_len)
        && !corridor_relevant
        && heat_t < 0.35
    {
        reduction = reduction.saturating_add(1);
    }
    let min_used = 1u32;
    let max_reduction = child_full.saturating_sub(min_used);
    reduction.min(max_reduction).min(depth.saturating_sub(1))
}

/// Quiet-corridor ordering boost — hotter CAT slots sort before colder ones.
#[inline]
pub fn cat_corridor_order_boost(cat_cm: i32, cat_max: u16, cold_cm: u16) -> i32 {
    (cat_heat_fraction(cat_cm, cat_max, cold_cm) * 16_000.0).round() as i32
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

/// Combined corridor heat for LMR / futility (higher = more likely to matter).
/// Peak CAT heat (cm) over legal pawn moves for each player — NNUE `cat_best_p0/p1`.
pub fn best_pawn_cat_heats(
    board: &Board,
    cat: &CorridorAttention,
    bfs: &mut BfsScratch,
) -> (u16, u16) {
    let mut best = [0i32; 2];
    for (pi, player) in [Player::One, Player::Two].into_iter().enumerate() {
        let mut b = board.clone();
        b.side_to_move = player;
        let mut buf = [Move::Pawn { row: 0, col: 0 }; MAX_LEGAL_MOVES];
        let n = generate_legal_moves_slice(&mut b, &mut buf, bfs);
        for mv in &buf[..n] {
            if matches!(mv, Move::Pawn { .. }) {
                best[pi] = best[pi].max(move_corridor_attention(&b, *mv, cat));
            }
        }
    }
    (
        best[0].clamp(0, u16::MAX as i32) as u16,
        best[1].clamp(0, u16::MAX as i32) as u16,
    )
}

pub fn move_corridor_attention(board: &Board, mv: Move, cat: &CorridorAttention) -> i32 {
    cat_score_for_move(mv, cat) + wall_shape_attention_bonus(board, mv, cat)
}

/// Simple move impact from the BFF heatmap (v16 LMR ordering): a pawn move
/// inherits its destination square's heat; a wall inherits the HIGHEST heat of
/// the squares it touches. No per-move pathfinding, no defensive special-casing —
/// just "how hot is the part of the board this move acts on".
pub fn move_impact_heat(mv: Move, cat: &CorridorAttention) -> i32 {
    match mv {
        Move::Pawn { row, col } => i32::from(cat.square_heat(row, col)),
        Move::Wall { row, col, .. } => {
            // A wall borders the 2×2 block of cells (row,col)..=(row+1,col+1).
            let h = |r: u8, c: u8| i32::from(cat.square_heat(r, c));
            h(row, col)
                .max(h(row + 1, col))
                .max(h(row, col + 1))
                .max(h(row + 1, col + 1))
        }
    }
}

pub fn wall_path_impact_attention(
    board: &mut Board,
    mv: Move,
    white_dist: u8,
    black_dist: u8,
    bfs: &mut BfsScratch,
) -> i32 {
    let Move::Wall { .. } = mv else {
        return 0;
    };
    let undo = board.make_move(mv);
    let white_after = bfs
        .shortest_distance(board, Player::One)
        .unwrap_or(DIST_PENALTY);
    let black_after = bfs
        .shortest_distance(board, Player::Two)
        .unwrap_or(DIST_PENALTY);
    board.unmake_move(undo);

    let white_gain = u32::from(white_after.saturating_sub(white_dist));
    let black_gain = u32::from(black_after.saturating_sub(black_dist));
    let total = white_gain + black_gain;
    if total == 0 {
        return 0;
    }
    let strongest = white_gain.max(black_gain);
    let affected_paths = u32::from(white_gain > 0) + u32::from(black_gain > 0);
    let shared_bonus = if affected_paths > 1 { 40 } else { 0 };
    (total * 120 + strongest * 50 + shared_bonus).min(i32::MAX as u32) as i32
}

pub fn move_corridor_attention_with_path(
    board: &mut Board,
    mv: Move,
    cat: &CorridorAttention,
    white_dist: u8,
    black_dist: u8,
    bfs: &mut BfsScratch,
) -> i32 {
    move_corridor_attention(board, mv, cat).max(wall_path_impact_attention(
        board, mv, white_dist, black_dist, bfs,
    ))
}

pub fn move_corridor_attention_with_denial(
    board: &Board,
    mv: Move,
    cat: &CorridorAttention,
    candidates: &[Move],
    direct_heats: &[i32],
    n: usize,
) -> i32 {
    let direct = move_corridor_attention(board, mv, cat);
    if matches!(mv, Move::Wall { .. }) {
        direct.max(legal_neighbor_denial_heat(mv, candidates, direct_heats, n))
    } else {
        direct
    }
}

/// Stockfish-style extras layered on top of tactical ordering.
#[derive(Clone, Copy, Default)]
pub struct OrderExtras {
    pub pv_move: Option<Move>,
    pub killers: [Option<Move>; 2],
}

#[inline]
fn apply_order_extras(base: i32, mv: Move, tt_best: Option<Move>, extras: &OrderExtras) -> i32 {
    let mut score = base;
    if tt_best == Some(mv) {
        return score;
    }
    if extras.pv_move == Some(mv) {
        score = score.max(9_500);
    }
    for killer in extras.killers {
        if killer == Some(mv) {
            score = score.max(8_500);
        }
    }
    score
}

pub fn move_order_score(
    board: &mut Board,
    mv: Move,
    tt_best: Option<Move>,
    book_hint: Option<BookHint>,
    our_dist: u8,
    opp_dist: u8,
    opp_path: &[u8],
    opp_path_len: usize,
    bfs: &mut BfsScratch,
    cat: &CorridorAttention,
    cat_max: u16,
    cold_cm: u16,
) -> i32 {
    if tt_best == Some(mv) {
        return 10_000;
    }
    if let Some(hint) = book_hint {
        if hint.mv == mv {
            // PV bias only — tactical race gains and TT still outrank theory.
            let bias = i32::from(hint.stm_bias) / 4;
            return 9_000 + i32::from(hint.priority) + bias;
        }
    }
    let behind = our_dist > opp_dist;
    let race_pressure = behind || opp_dist <= 4;

    if matches!(mv, Move::Wall { .. }) {
        // Witness-path gate: a wall off the opponent's current shortest path
        // has opp_gain = 0, so net ≤ 0 < min_net — score it without any BFS.
        // Only walls that actually cut the path pay one make + two BFS.
        if !wall_intersects_path(mv, opp_path, opp_path_len) {
            let attn = move_corridor_attention(board, mv, cat);
            // Proportional CAT hotspot boost — 200 cm sorts well above 100 cm at same node.
            return -20_000 + cat_corridor_order_boost(attn, cat_max, cold_cm);
        }
        let (opp_gain, our_loss) = wall_race_swing(board, mv, our_dist, opp_dist, bfs);
        let net = opp_gain - our_loss;
        let min_net = min_wall_net_race(our_dist, opp_dist);
        let attn = move_corridor_attention(board, mv, cat);
        if net < min_net {
            return -20_000 + cat_corridor_order_boost(attn, cat_max, cold_cm);
        }
        if race_pressure {
            return 15_000 + net * 120 + attn / 8;
        }
        return 12_000 + net * 80 + attn / 16;
    }

    let gain = our_path_gain(board, mv, our_dist, bfs);
    if our_dist >= opp_dist && gain > 0 {
        // Lateral / slow sprint while clearly losing the race is not a tactic.
        let closes_gap = gain >= 2 || our_dist.saturating_sub(1) <= opp_dist;
        if behind && !closes_gap {
            return 800 + gain * 40;
        }
        return 14_000 + gain * 100;
    }
    if gain > 0 {
        1000 + gain * 100
    } else {
        let attn = move_corridor_attention(board, mv, cat);
        TEMPO_PENALTY + cat_corridor_order_boost(attn, cat_max, cold_cm)
    }
}

/// Score band for treating root/order candidates as tied (symmetry interleave).
const ORDER_SCORE_TIE_BAND: i32 = 150;

#[inline]
fn move_col(mv: Move) -> u8 {
    match mv {
        Move::Pawn { col, .. } | Move::Wall { col, .. } => col,
    }
}

#[inline]
fn symmetry_side(col: u8) -> u8 {
    if col < 4 {
        0
    } else if col > 4 {
        2
    } else {
        1
    }
}

/// Mirror across the e-file — d↔f on a 9-wide board (cols 0..8).
pub fn mirror_move(mv: Move) -> Move {
    let mirrored = 8 - move_col(mv);
    match mv {
        Move::Pawn { row, .. } => Move::Pawn { row, col: mirrored },
        Move::Wall {
            row, orientation, ..
        } => Move::Wall {
            row,
            col: mirrored,
            orientation,
        },
    }
}

/// Within tied score bands, round-robin left / right / center so LMR does not
/// always feast on the d-file before the f-file.
fn rebalance_symmetric_order(moves: &mut [Move], scores: &mut [i32], n: usize) {
    if n <= 1 {
        return;
    }
    let mut out_moves = [Move::Pawn { row: 0, col: 0 }; MAX_LEGAL_MOVES];
    let mut out_scores = [0i32; MAX_LEGAL_MOVES];
    let mut out = 0usize;
    let mut start = 0usize;
    while start < n {
        let top = scores[start];
        let mut end = start + 1;
        while end < n && scores[end] >= top - ORDER_SCORE_TIE_BAND {
            end += 1;
        }
        let bucket_len = end - start;
        if bucket_len <= 1 {
            out_moves[out] = moves[start];
            out_scores[out] = scores[start];
            out += 1;
            start = end;
            continue;
        }

        let mut left = Vec::new();
        let mut center = Vec::new();
        let mut right = Vec::new();
        for idx in start..end {
            match symmetry_side(move_col(moves[idx])) {
                0 => left.push(idx),
                1 => center.push(idx),
                _ => right.push(idx),
            }
        }

        let mut merged = Vec::with_capacity(bucket_len);
        let max_lr = left.len().max(right.len());
        for k in 0..max_lr {
            if k < left.len() {
                merged.push(left[k]);
            }
            if k < right.len() {
                merged.push(right[k]);
            }
        }
        for &idx in &center {
            merged.push(idx);
        }

        for &idx in &merged {
            out_moves[out] = moves[idx];
            out_scores[out] = scores[idx];
            out += 1;
        }
        start = end;
    }
    moves[..n].copy_from_slice(&out_moves[..n]);
    scores[..n].copy_from_slice(&out_scores[..n]);
}

#[allow(clippy::too_many_arguments)]
pub fn order_moves(
    board: &mut Board,
    moves: &mut [Move],
    n: usize,
    tt_best: Option<Move>,
    book_hint: Option<BookHint>,
    scores: &mut [i32; MAX_LEGAL_MOVES],
    our_dist: u8,
    opp_dist: u8,
    opp_path: &[u8; 81],
    opp_path_len: usize,
    bfs: &mut BfsScratch,
    cat: &CorridorAttention,
    extras: &OrderExtras,
    history_bonus: impl Fn(Move) -> i32,
) {
    let cold_cm = CAT_COLD_CM;
    let refs = cat_heat_refs(moves, n, board, cat);
    for i in 0..n {
        let mv = moves[i];
        let cat_ref = cat_heat_ref_max(mv, refs);
        let base = move_order_score(
            board,
            mv,
            tt_best,
            book_hint,
            our_dist,
            opp_dist,
            opp_path,
            opp_path_len,
            bfs,
            cat,
            cat_ref,
            cold_cm,
        );
        let mut score = apply_order_extras(base, mv, tt_best, extras);
        score += history_bonus(mv);
        scores[i] = score;
    }
    let mut order: [usize; MAX_LEGAL_MOVES] = core::array::from_fn(|i| i);
    order[..n].sort_unstable_by(|&a, &b| scores[b].cmp(&scores[a]));
    let mut tmp = [Move::Pawn { row: 0, col: 0 }; MAX_LEGAL_MOVES];
    let mut tmp_scores = [0i32; MAX_LEGAL_MOVES];
    tmp[..n].copy_from_slice(&moves[..n]);
    for i in 0..n {
        moves[i] = tmp[order[i]];
        tmp_scores[i] = scores[order[i]];
    }
    scores[..n].copy_from_slice(&tmp_scores[..n]);
    rebalance_symmetric_order(moves, scores, n);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::board::{Board, Player};
    use crate::util::grid::set_wall;

    #[test]
    fn cat_heat_fraction_and_lmr_scale_proportionally() {
        let cold = 90u16;
        let max = 250u16;
        let f100 = cat_heat_fraction(100, max, cold);
        let f200 = cat_heat_fraction(200, max, cold);
        assert!(
            f200 > f100 * 5.0,
            "200cm should dominate 100cm: {f200} vs {f100}"
        );
        let r100 = cat_heat_depth_reduction(100, max, cold, 7);
        let r200 = cat_heat_depth_reduction(200, max, cold, 7);
        let r245 = cat_heat_depth_reduction(245, max, cold, 7);
        assert!(
            r100 > r200 && r200 > r245,
            "colder walls should lose more depth: 100={r100} 200={r200} 245={r245}"
        );
        let boost100 = cat_corridor_order_boost(100, max, cold);
        let boost200 = cat_corridor_order_boost(200, max, cold);
        assert!(boost200 > boost100 * 5);
    }

    #[test]
    fn cat_v16_fringe_caps_child_depth_at_one() {
        let refs = CatHeatRefs {
            all: 900,
            walls: 900,
            pawns: 400,
        };
        let cold_wall = Move::Wall {
            row: 0,
            col: 0,
            orientation: crate::core::board::WallOrientation::Horizontal,
        };
        let red = cat_v16_lmr_reduction_plies(cold_wall, 80, refs, 800, 10, 9);
        assert_eq!(red, 8, "80cm <= 10% of 800cm ceiling → depth-1 child");
        let warm = cat_v16_lmr_reduction_plies(cold_wall, 500, refs, 800, 10, 9);
        assert!(warm < 8, "warm corridor should keep more than 1 ply");
    }

    #[test]
    fn cat_v16_worker_fringe_schedule_caps_at_seventy_percent() {
        let schedule: Vec<u16> = (0..10).map(cat_v16_lmr_fringe_pct_for_worker).collect();
        assert_eq!(schedule, vec![5, 10, 20, 30, 40, 50, 60, 70, 70, 70]);
    }

    #[test]
    fn cat_v16_helper_fringe_is_more_aggressive_than_main() {
        let refs = CatHeatRefs {
            all: 800,
            walls: 800,
            pawns: 400,
        };
        let wall = Move::Wall {
            row: 0,
            col: 0,
            orientation: crate::core::board::WallOrientation::Horizontal,
        };
        let main_red = cat_v16_lmr_reduction_plies(wall, 200, refs, 800, 5, 9);
        let helper_red = cat_v16_lmr_reduction_plies(wall, 200, refs, 800, 30, 9);
        assert!(main_red < 8, "200cm is warm at the main 5% threshold");
        assert_eq!(
            helper_red, 8,
            "200cm is fringe for a helper using a 30% threshold"
        );
    }

    #[test]
    fn cat_v16_uses_wall_position_max_not_pawn_heat() {
        let refs = CatHeatRefs {
            all: 900,
            walls: 120,
            pawns: 900,
        };
        let wall = Move::Wall {
            row: 0,
            col: 0,
            orientation: crate::core::board::WallOrientation::Horizontal,
        };
        let warm_wall = cat_v16_lmr_reduction_plies(wall, 80, refs, 800, 10, 7);
        assert!(
            warm_wall < 6,
            "80cm is warm against 120cm wall max, even if pawn heat is 900cm"
        );
        let cold_wall = cat_v16_lmr_reduction_plies(wall, 12, refs, 800, 10, 7);
        assert_eq!(cold_wall, 6, "at 10% of wall max searches child depth 1");
    }

    #[test]
    fn symmetric_flank_pawns_interleave_in_tied_order_band() {
        use crate::movegen::generate_legal_moves_slice;
        use crate::util::perft::format_move;

        let mut board = Board::new();
        board.apply_algebraic("e2");
        board.apply_algebraic("e8");
        let mut bfs = BfsScratch::new();
        let mut scratch = [Move::Pawn { row: 0, col: 0 }; MAX_LEGAL_MOVES];
        let full = generate_legal_moves_slice(&mut board, &mut scratch, &mut bfs);
        let mut buf = [Move::Pawn { row: 0, col: 0 }; MAX_LEGAL_MOVES];
        let n = full.min(buf.len());
        buf[..n].copy_from_slice(&scratch[..n]);
        let mut scores = [0i32; MAX_LEGAL_MOVES];
        let cat = bfs.build_corridor_attention(&board);
        let mut opp_path = [0u8; 81];
        let opp_len = get_shortest_path(&board, board.side().opposite(), &mut bfs, &mut opp_path);
        let our_dist = bfs
            .shortest_distance(&board, board.side())
            .unwrap_or(DIST_PENALTY);
        let opp_dist = path_distance(board.side().opposite(), &opp_path, opp_len);
        order_moves(
            &mut board,
            &mut buf,
            n,
            None,
            None,
            &mut scores,
            our_dist,
            opp_dist,
            &opp_path,
            opp_len,
            &mut bfs,
            &cat,
            &OrderExtras::default(),
            |_| 0,
        );
        let d2 = buf[..n]
            .iter()
            .position(|&m| format_move(m) == "d2")
            .expect("d2 legal");
        let f2 = buf[..n]
            .iter()
            .position(|&m| format_move(m) == "f2")
            .expect("f2 legal");
        assert!(
            d2.abs_diff(f2) <= 2,
            "flank pawns should interleave (d2 @ {d2}, f2 @ {f2}); order: {}",
            buf[..n]
                .iter()
                .map(|m| format_move(*m))
                .collect::<Vec<_>>()
                .join(" ")
        );
    }

    #[test]
    fn blocks_cross_gap_detects_shifted_prevention() {
        let mut board = Board::new();
        set_wall(&mut board, 2, 4, WallOrientation::Vertical, true);
        set_wall(&mut board, 4, 4, WallOrientation::Vertical, true);
        assert!(blocks_cross_gap_wall(
            &board,
            3,
            3,
            WallOrientation::Horizontal
        ));
        assert!(blocks_cross_gap_wall(
            &board,
            3,
            5,
            WallOrientation::Horizontal
        ));
        assert!(!blocks_cross_gap_wall(
            &board,
            3,
            4,
            WallOrientation::Horizontal
        ));
    }

    #[test]
    fn wall_shape_bonus_only_for_hot_cross_gap() {
        let mut board = Board::new();
        set_wall(&mut board, 2, 4, WallOrientation::Vertical, true);
        set_wall(&mut board, 4, 4, WallOrientation::Vertical, true);
        let cold_cat = CorridorAttention::default();
        let cross = Move::Wall {
            row: 3,
            col: 4,
            orientation: WallOrientation::Horizontal,
        };
        assert_eq!(
            wall_shape_attention_bonus(&board, cross, &cold_cat),
            0,
            "cold CAT should not revive unrelated shape bonus"
        );

        let mut bfs = BfsScratch::new();
        let hot_cat = bfs.build_corridor_attention(&board);
        assert!(
            wall_shape_attention_bonus(&board, cross, &hot_cat) >= WALL_CROSS_GAP_CM,
            "hot corridor cross-gap should get a tiny ordering nudge"
        );
    }

    #[test]
    fn cross_gap_wall_detects_perpendicular_through_gap() {
        let mut board = Board::new();
        set_wall(&mut board, 2, 4, WallOrientation::Vertical, true);
        set_wall(&mut board, 4, 4, WallOrientation::Vertical, true);
        assert!(is_cross_gap_wall(&board, 3, 4, WallOrientation::Horizontal));
        assert!(!is_cross_gap_wall(&board, 3, 4, WallOrientation::Vertical));
    }

    #[test]
    fn cross_gap_ignores_adjacent_chain_t_junction() {
        let mut board = Board::new();
        set_wall(&mut board, 2, 4, WallOrientation::Vertical, true);
        set_wall(&mut board, 3, 4, WallOrientation::Vertical, true);
        assert!(!is_cross_gap_wall(
            &board,
            3,
            4,
            WallOrientation::Horizontal
        ));
    }

    #[test]
    fn left_chain_keeps_gap_tactics() {
        let mut board = Board::new();
        // Vertical chain on c|d (col 2) with deliberate gaps between segments.
        for row in [0u8, 2, 4, 6] {
            set_wall(&mut board, row, 2, WallOrientation::Vertical, true);
        }
        board.pawns = [(3, 0), (5, 0)]; // a4, a6 — left pocket
        board.hash = crate::core::zobrist::hash_board(&board);

        let mut bfs = BfsScratch::new();
        let cat = bfs.build_corridor_attention(&board);
        let reachable = bfs.both_reachable_mask(&board);
        let our_dist = bfs
            .shortest_distance(&board, Player::One)
            .unwrap_or(DIST_PENALTY);
        let opp_dist = bfs
            .shortest_distance(&board, Player::Two)
            .unwrap_or(DIST_PENALTY);
        let mut opp_path = [0u8; 81];
        let opp_path_len = get_shortest_path(&board, Player::Two, &mut bfs, &mut opp_path);
        let gap_zone = gap_play_zone_mask(reachable);

        let cross_gap = Move::Wall {
            row: 3,
            col: 2,
            orientation: WallOrientation::Horizontal,
        };
        let mut cross_board = board.clone();
        assert!(
            is_cross_gap_wall(&cross_board, 3, 2, WallOrientation::Horizontal),
            "H through V|gap|V should be detected"
        );
        assert!(
            wall_should_search(
                cross_gap,
                &cat,
                reachable,
                gap_zone,
                &mut cross_board,
                our_dist,
                opp_dist,
                &opp_path,
                opp_path_len,
                &mut bfs,
            ),
            "horizontal through chain gap must stay searchable"
        );

        let flank_block = Move::Wall {
            row: 3,
            col: 3,
            orientation: WallOrientation::Horizontal,
        };
        let mut flank_board = board.clone();
        assert!(
            blocks_cross_gap_wall(&flank_board, 3, 3, WallOrientation::Horizontal),
            "shifted block beside gap should be detected"
        );
        assert!(
            wall_should_search(
                flank_block,
                &cat,
                reachable,
                gap_zone,
                &mut flank_board,
                our_dist,
                opp_dist,
                &opp_path,
                opp_path_len,
                &mut bfs,
            ),
            "flank block preventing half-protrusion into void must stay searchable"
        );
    }

    #[test]
    fn gap_mouth_keeps_t_junction_tactics_prunes_deep_void() {
        let mut board = Board::new();
        // Three walls around a T mouth; fourth slot open at (3,4) horizontal.
        set_wall(&mut board, 2, 4, WallOrientation::Vertical, true);
        set_wall(&mut board, 3, 3, WallOrientation::Vertical, true);
        set_wall(&mut board, 3, 5, WallOrientation::Vertical, true);
        board.pawns = [(4, 4), (6, 4)];
        board.hash = crate::core::zobrist::hash_board(&board);

        let mut bfs = BfsScratch::new();
        let cat = bfs.build_corridor_attention(&board);
        let reachable = bfs.both_reachable_mask(&board);
        let gap_zone = gap_play_zone_mask(reachable);
        // No square is actually sealed here — gap zone must be empty. (The old
        // 0..=9 bounds bug made every board edge a phantom mouth.)
        assert_eq!(
            gap_zone, 0,
            "no sealed territory → gap play zone must be empty"
        );
        let our_dist = bfs
            .shortest_distance(&board, Player::One)
            .unwrap_or(DIST_PENALTY);
        let opp_dist = bfs
            .shortest_distance(&board, Player::Two)
            .unwrap_or(DIST_PENALTY);
        let mut opp_path = [0u8; 81];
        let opp_path_len = get_shortest_path(&board, Player::Two, &mut bfs, &mut opp_path);

        let mouth_wall = Move::Wall {
            row: 3,
            col: 4,
            orientation: WallOrientation::Horizontal,
        };
        let mut mouth_board = board.clone();
        assert!(
            wall_should_search(
                mouth_wall,
                &cat,
                reachable,
                gap_zone,
                &mut mouth_board,
                our_dist,
                opp_dist,
                &opp_path,
                opp_path_len,
                &mut bfs,
            ),
            "wall at T-junction gap mouth must stay searchable"
        );

        // Fully sealed pocket on the far left — interior wall does not touch gap mouth.
        let mut pocket = Board::new();
        for &(row, col, orient) in &[
            (1, 0, WallOrientation::Vertical),
            (1, 1, WallOrientation::Horizontal),
            (2, 0, WallOrientation::Horizontal),
        ] {
            set_wall(&mut pocket, row, col, orient, true);
        }
        let reachable_pocket = bfs.both_reachable_mask(&pocket);
        let gap_zone_pocket = gap_play_zone_mask(reachable_pocket);
        let inner = Move::Wall {
            row: 0,
            col: 0,
            orientation: WallOrientation::Vertical,
        };
        let mut inner_board = pocket.clone();
        assert!(
            !wall_should_search(
                inner,
                &CorridorAttention::default(),
                reachable_pocket,
                gap_zone_pocket,
                &mut inner_board,
                DIST_PENALTY,
                DIST_PENALTY,
                &opp_path,
                opp_path_len,
                &mut bfs,
            ),
            "interior sealed void wall away from gap mouth must be pruned"
        );
    }

    #[test]
    fn dead_zone_prunes_walls_in_unreachable_void() {
        let board = Board::new();
        let mut bfs = BfsScratch::new();
        let cat = CorridorAttention::default();
        let _reachable = bfs.both_reachable_mask(&board);
        let mut opp_path = [0u8; 81];
        let opp_path_len = get_shortest_path(&board, Player::Two, &mut bfs, &mut opp_path);

        // Fully buried inner wall — every touched square is outside both floods.
        let mut pocket = Board::new();
        for &(row, col, orient) in &[
            (1, 0, WallOrientation::Vertical),
            (1, 1, WallOrientation::Horizontal),
            (2, 0, WallOrientation::Horizontal),
        ] {
            set_wall(&mut pocket, row, col, orient, true);
        }
        let inner_t = Move::Wall {
            row: 0,
            col: 0,
            orientation: WallOrientation::Vertical,
        };
        let reachable_pocket = bfs.both_reachable_mask(&pocket);
        let gap_zone_pocket = gap_play_zone_mask(reachable_pocket);
        let mut pocket_board = pocket.clone();
        assert!(
            !wall_should_search(
                inner_t,
                &cat,
                reachable_pocket,
                gap_zone_pocket,
                &mut pocket_board,
                DIST_PENALTY,
                DIST_PENALTY,
                &opp_path,
                opp_path_len,
                &mut bfs,
            ),
            "walls in a sealed void cannot affect play on the live side of the chain"
        );
    }

    #[test]
    fn wall_search_prunes_enclosed_t_keeps_corridor_blocks() {
        let board = Board::new();
        let mut bfs = BfsScratch::new();
        let cat = bfs.build_corridor_attention(&board);
        let reachable = bfs.both_reachable_mask(&board);
        let our_dist = bfs
            .shortest_distance(&board, Player::One)
            .unwrap_or(DIST_PENALTY);
        let opp_dist = bfs
            .shortest_distance(&board, Player::Two)
            .unwrap_or(DIST_PENALTY);
        let mut opp_path = [0u8; 81];
        let opp_path_len = get_shortest_path(&board, Player::Two, &mut bfs, &mut opp_path);
        let gap_zone = gap_play_zone_mask(reachable);

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
                gap_zone,
                &mut passive_board,
                our_dist,
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
                gap_zone,
                &mut corridor_board,
                our_dist,
                opp_dist,
                &opp_path,
                opp_path_len,
                &mut bfs,
            ),
            "central corridor wall should stay searchable"
        );

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
        let gap_zone_pocket = gap_play_zone_mask(reachable_pocket);
        let our_dist_pocket = bfs
            .shortest_distance(&pocket, Player::One)
            .unwrap_or(DIST_PENALTY);
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
                gap_zone_pocket,
                &mut inner_board,
                our_dist_pocket,
                opp_dist_pocket,
                &opp_path_pocket,
                opp_path_len_pocket,
                &mut bfs,
            ),
            "fully buried inner T-wall should be pruned"
        );
    }

    #[test]
    fn useless_t_junction_gets_no_shape_bonus() {
        let mut board = Board::new();
        set_wall(&mut board, 0, 5, WallOrientation::Vertical, true);
        set_wall(&mut board, 1, 5, WallOrientation::Vertical, true);
        let mut bfs = BfsScratch::new();
        let cat = bfs.build_corridor_attention(&board);
        let t_junction = Move::Wall {
            row: 1,
            col: 5,
            orientation: WallOrientation::Horizontal,
        };
        assert_eq!(
            wall_shape_attention_bonus(&board, t_junction, &cat),
            0,
            "far-side T junction should not get shape attention"
        );
    }

    #[test]
    fn sprint_line_orders_wall_before_lateral_pawn() {
        use crate::core::board::Board;
        use crate::util::perft::format_move;

        let seq = ["e2", "e8", "d2", "e7", "d3", "e6", "d4", "e5", "c4", "e4"];
        let mut board = Board::new();
        for m in seq {
            board.apply_algebraic(m);
        }
        let mut bfs = BfsScratch::new();
        let our = bfs
            .shortest_distance(&board, Player::One)
            .unwrap_or(DIST_PENALTY);
        let opp = bfs
            .shortest_distance(&board, Player::Two)
            .unwrap_or(DIST_PENALTY);
        let cat = bfs.build_corridor_attention(&board);
        let mut opp_path = [0u8; 81];
        let opp_path_len = get_shortest_path(&board, Player::Two, &mut bfs, &mut opp_path);
        let mut buf = [Move::Pawn { row: 0, col: 0 }; MAX_LEGAL_MOVES];
        let n = collect_search_moves(
            &mut board,
            &mut buf,
            &mut bfs,
            &cat,
            &opp_path,
            opp_path_len,
            our,
            opp,
            false,
            true,
        );
        let mut scores = [0i32; MAX_LEGAL_MOVES];
        order_moves(
            &mut board,
            &mut buf,
            n,
            None,
            None,
            &mut scores,
            our,
            opp,
            &opp_path,
            opp_path_len,
            &mut bfs,
            &cat,
            &OrderExtras::default(),
            |_| 0,
        );
        let mut ranked: Vec<_> = (0..n).map(|i| (scores[i], format_move(buf[i]))).collect();
        ranked.sort_by(|a, b| b.0.cmp(&a.0));
        for (s, m) in ranked.iter().take(8) {
            eprintln!("  {s} {m}");
        }
        let top_has_wall = ranked
            .iter()
            .take(6)
            .any(|(_, m)| m.ends_with('h') || m.ends_with('v'));
        assert!(
            top_has_wall,
            "a blocking wall should rank in top 6, top={ranked:?}"
        );
    }

    #[test]
    fn sprint_line_includes_blocker_wall_when_behind() {
        use crate::core::board::Board;
        use crate::util::perft::format_move;

        let seq = ["e2", "e8", "d2", "e7", "d3", "e6", "d4", "e5", "c4", "e4"];
        let mut board = Board::new();
        for m in seq {
            board.apply_algebraic(m);
        }
        let mut bfs = BfsScratch::new();
        let our = bfs
            .shortest_distance(&board, Player::One)
            .unwrap_or(DIST_PENALTY);
        let opp = bfs
            .shortest_distance(&board, Player::Two)
            .unwrap_or(DIST_PENALTY);
        assert!(our > opp, "white should be behind in race, W{our} B{opp}");

        let cat = bfs.build_corridor_attention(&board);
        let mut opp_path = [0u8; 81];
        let opp_path_len = get_shortest_path(&board, Player::Two, &mut bfs, &mut opp_path);
        let mut buf = [Move::Pawn { row: 0, col: 0 }; MAX_LEGAL_MOVES];
        let n = collect_search_moves(
            &mut board,
            &mut buf,
            &mut bfs,
            &cat,
            &opp_path,
            opp_path_len,
            our,
            opp,
            false,
            true,
        );
        let walls: Vec<String> = buf[..n]
            .iter()
            .filter(|mv| matches!(mv, Move::Wall { .. }))
            .map(|&mv| format_move(mv))
            .collect();
        eprintln!("searchable walls ({n} total moves): {walls:?}");
        assert!(
            !walls.is_empty(),
            "must keep at least one blocking wall when losing the sprint"
        );
    }
}
