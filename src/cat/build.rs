//! Build CAT heat from BFS distance fields on the pawn grid.

use crate::cat::attention::CorridorAttention;
use crate::cat::constants::{
    BOTTLENECK_BONUS_CM, BOTTLENECK_CORRIDOR_DELTA, CAT_CORRIDOR_CM, MAX_RELEVANT_CORRIDOR_DELTA,
};
use crate::core::board::{Board, Player};
use crate::path::distance::{
    fill_dist_from_sq, fill_dist_layers_from_sq, fill_dist_layers_to_goal_row, fill_dist_to_goal_row,
    DistLayers,
};
use crate::path::masks::DirMasks;
use crate::path::BfsScratch;
use crate::util::grid::{flood_bit_sq, flood_sq_from_bit, square_index, FLOOD_PLAYABLE};

fn corridor_heat(delta: u16) -> u16 {
    if delta > MAX_RELEVANT_CORRIDOR_DELTA {
        return 0;
    }
    // Exact rounded values of `CAT_CORRIDOR_CM / (1 + delta·log2(delta+2))` for
    // delta 0..4 — kept as a LUT so the per-square hot loop never evaluates a
    // float `log2`. Bit-identical to the old formula:
    //   delta 0 → 200/1.0       = 200
    //   delta 1 → 200/(1+log2 3) = 77
    //   delta 2 → 200/(1+2·log2 4) = 40
    //   delta 3 → 200/(1+3·log2 5) = 25
    //   delta 4 → 200/(1+4·log2 6) = 18
    const HEAT_LUT: [u16; (MAX_RELEVANT_CORRIDOR_DELTA + 1) as usize] = [200, 77, 40, 25, 18];
    debug_assert_eq!(
        CAT_CORRIDOR_CM, 200,
        "HEAT_LUT computed for CAT_CORRIDOR_CM=200"
    );
    HEAT_LUT[delta as usize]
}

/// Centi-percent (45–100): gentle linear fade along the race — near-pawn squares
/// stay hottest, but mid-corridor wall zones keep meaningful heat for deeper play.
fn pawn_path_weight(dist_from: u8, shortest_to_goal: u8) -> u16 {
    if shortest_to_goal == 0 || shortest_to_goal == u8::MAX {
        return 100;
    }
    const MIN_WEIGHT: u16 = 45;
    const MAX_WEIGHT: u16 = 100;
    let from = u32::from(dist_from.min(shortest_to_goal));
    let total = u32::from(shortest_to_goal);
    let remaining = total.saturating_sub(from);
    MIN_WEIGHT + (u32::from(MAX_WEIGHT - MIN_WEIGHT) * remaining / total) as u16
}

fn neighbor_squares(sq: u8, masks: DirMasks, out: &mut [u8; 4]) -> usize {
    let bit = flood_bit_sq(sq);
    let mut n = 0usize;
    if masks.north & bit != 0 {
        out[n] = sq - 9;
        n += 1;
    }
    if masks.south & bit != 0 {
        out[n] = sq + 9;
        n += 1;
    }
    if masks.east & bit != 0 {
        out[n] = sq + 1;
        n += 1;
    }
    if masks.west & bit != 0 {
        out[n] = sq - 1;
        n += 1;
    }
    n
}

fn corridor_delta(
    sq: u8,
    dist_from_pawn: &[u8; 81],
    dist_to_goal: &[u8; 81],
    shortest_to_goal: u8,
) -> Option<u16> {
    let from = dist_from_pawn[sq as usize];
    let to = dist_to_goal[sq as usize];
    if from == u8::MAX || to == u8::MAX || shortest_to_goal == u8::MAX {
        return None;
    }
    Some((u16::from(from) + u16::from(to)).saturating_sub(u16::from(shortest_to_goal)))
}

/// `delta_arr[sq]` is the precomputed corridor delta (`u16::MAX` = off-path/None),
/// so the per-neighbor near-shortest test is an array read, not a recompute.
fn reasonable_forward_continuations(
    sq: u8,
    masks: DirMasks,
    dist_from_pawn: &[u8; 81],
    dist_to_goal: &[u8; 81],
    delta_arr: &[u16; 81],
) -> u8 {
    let from = dist_from_pawn[sq as usize];
    let to = dist_to_goal[sq as usize];
    if from == u8::MAX || to == 0 || to == u8::MAX {
        return 0;
    }
    let mut neighbors = [0u8; 4];
    let n = neighbor_squares(sq, masks, &mut neighbors);
    let mut count = 0u8;
    for &next in &neighbors[..n] {
        let next_from = dist_from_pawn[next as usize];
        let next_to = dist_to_goal[next as usize];
        // `u16::MAX` sentinel (None) is > MAX_RELEVANT, so it fails the bound naturally.
        if next_from == from.saturating_add(1)
            && next_to < to
            && delta_arr[next as usize] <= MAX_RELEVANT_CORRIDOR_DELTA
        {
            count = count.saturating_add(1);
        }
    }
    count
}

fn add_player_corridor_attention(
    board: &Board,
    player: Player,
    masks: DirMasks,
    out: &mut CorridorAttention,
    dist_from_pawn: &mut [u8; 81],
    dist_to_goal: &mut [u8; 81],
) {
    let (sr, sc) = board.pawn(player);
    let start = square_index(sr, sc);

    fill_dist_from_sq(start, masks, dist_from_pawn);
    fill_dist_to_goal_row(player, masks, dist_to_goal);

    let shortest_to_goal = dist_to_goal[start as usize];

    // Compute each square's corridor delta once (u16::MAX = off-path); the main
    // loop and the per-neighbor flex test both read it instead of recomputing.
    let mut delta_arr = [u16::MAX; 81];
    for sq in 0usize..81 {
        if let Some(d) = corridor_delta(sq as u8, dist_from_pawn, dist_to_goal, shortest_to_goal) {
            delta_arr[sq] = d;
        }
    }

    for sq in 0u8..81 {
        let idx = sq as usize;
        let delta = delta_arr[idx];
        let base = corridor_heat(delta);
        if base == 0 {
            continue;
        }

        let from = dist_from_pawn[idx];
        let weight = pawn_path_weight(from, shortest_to_goal);
        let heat = (u32::from(base) * u32::from(weight) / 100) as u16;
        if heat == 0 {
            continue;
        }

        let flex =
            reasonable_forward_continuations(sq, masks, dist_from_pawn, dist_to_goal, &delta_arr);
        out.square_heat[idx] = out.square_heat[idx].saturating_add(heat);
        out.route_flex[idx] = out.route_flex[idx].saturating_add(flex);
        if delta <= BOTTLENECK_CORRIDOR_DELTA && flex <= 1 && dist_to_goal[idx] > 0 {
            out.bottleneck_heat[idx] = out.bottleneck_heat[idx].saturating_add(BOTTLENECK_BONUS_CM);
        }
    }
}

pub fn build_player_corridor_attention(
    scratch: &mut BfsScratch,
    board: &Board,
    player: Player,
) -> CorridorAttention {
    let masks = DirMasks::from_board(board);
    let mut out = CorridorAttention::default();
    let (dist_from, dist_to) = scratch.dist_scratch_mut();
    add_player_corridor_attention(board, player, masks, &mut out, dist_from, dist_to);
    out
}

/// Per-square heat for the web overlay — max of each player's corridor signal.
///
/// Search sums both players into one `CorridorAttention` (contested corridors run hotter
/// in LMR). The board tint uses `max` so two overlapping paths do not paint the whole grid.
pub fn build_corridor_display_squares(_scratch: &mut BfsScratch, board: &Board) -> [u16; 81] {
    // CAT vision shows the SAME heatmap the v16 LMR ordering uses, so what you see
    // on the board is exactly what drives move reduction.
    build_impact_heatmap(board).square_heat
}

fn merge_corridor_max(a: &mut CorridorAttention, b: &CorridorAttention) {
    for i in 0..81 {
        a.square_heat[i] = a.square_heat[i].max(b.square_heat[i]);
        a.route_flex[i] = a.route_flex[i].max(b.route_flex[i]);
        a.bottleneck_heat[i] = a.bottleneck_heat[i].max(b.bottleneck_heat[i]);
    }
}

/// Build combined two-player corridor attention for search ordering.
///
/// Uses per-square **max** of each player's heat (same as the web overlay), not sum —
/// summing both races doubled fringe heat and qualified ~40 walls per node in open games.
pub fn build_corridor_attention(scratch: &mut BfsScratch, board: &Board) -> CorridorAttention {
    let masks = DirMasks::from_board(board);
    let mut white = CorridorAttention::default();
    let mut black = CorridorAttention::default();
    {
        let (dist_from, dist_to) = scratch.dist_scratch_mut();
        add_player_corridor_attention(board, Player::One, masks, &mut white, dist_from, dist_to);
    }
    {
        let (dist_from, dist_to) = scratch.dist_scratch_mut();
        add_player_corridor_attention(board, Player::Two, masks, &mut black, dist_from, dist_to);
    }
    let mut attention = white;
    merge_corridor_max(&mut attention, &black);
    attention
}

/// Count low-flex squares on exact/near-shortest corridors (caging heuristic).
pub fn corridor_bottleneck_count(scratch: &mut BfsScratch, board: &Board, player: Player) -> u8 {
    let masks = DirMasks::from_board(board);
    let (sr, sc) = board.pawn(player);
    let start = square_index(sr, sc);
    let (dist_from, dist_to) = scratch.dist_scratch_mut();
    fill_dist_from_sq(start, masks, dist_from);
    fill_dist_to_goal_row(player, masks, dist_to);
    let shortest_to_goal = dist_from[start as usize];
    if shortest_to_goal == u8::MAX {
        return 8;
    }

    let mut delta_arr = [u16::MAX; 81];
    for sq in 0usize..81 {
        if let Some(d) = corridor_delta(sq as u8, dist_from, dist_to, shortest_to_goal) {
            delta_arr[sq] = d;
        }
    }

    let mut bottlenecks = 0u8;
    for sq in 0u8..81 {
        let delta = delta_arr[sq as usize];
        if delta > BOTTLENECK_CORRIDOR_DELTA || dist_to[sq as usize] == 0 {
            continue;
        }
        let flex = reasonable_forward_continuations(sq, masks, dist_from, dist_to, &delta_arr);
        if flex <= 1 {
            bottlenecks = bottlenecks.saturating_add(1);
        }
    }
    bottlenecks.min(8)
}

// ---------------------------------------------------------------------------
// BFF impact heatmap (fast path for LMR move ordering)
//
// CAT is a cheap APPROXIMATION of move impact, not an exact field. This builds a
// per-square heatmap straight from bitboard distance LAYERS (no dense [u8;81]
// scatter, no per-cell 0..81 corridor/flex loops, no per-move shortest-path
// recompute): hottest on the near-shortest-path SET of both players (shared
// corridors run hottest), decaying outward by a binary flood. A wall/move's
// impact is then a heatmap lookup (see `CorridorAttention::wall_edge_heat`),
// which is what feeds the v16 CAT-LMR fringe cutoff.
// ---------------------------------------------------------------------------

/// Add `w` to `heat[sq]` for every set cell of `mask` (saturating). Bounded by the
/// set-bit count of one path-set / BFS layer (a few dozen), NOT all 81 — the
/// bitboard layers keep this an O(reached) scatter, unlike the old dense rebuild.
#[inline]
fn scatter_add(heat: &mut [u16; 81], mask: u128, w: u16) {
    if w == 0 {
        return;
    }
    let mut bits = mask & FLOOD_PLAYABLE;
    while bits != 0 {
        let fb = bits.trailing_zeros();
        bits &= bits - 1;
        if let Some(sq) = flood_sq_from_bit(fb) {
            let slot = &mut heat[sq as usize];
            *slot = slot.saturating_add(w);
        }
    }
}

/// One player's impact contribution: hottest on the near-shortest path SET (every
/// route within `MAX_RELEVANT_CORRIDOR_DELTA` of optimal, weighted by optimality),
/// decaying outward by a binary flood. Pure bitboard layers — no dense field.
fn add_player_impact_heat(board: &Board, player: Player, masks: DirMasks, heat: &mut [u16; 81]) {
    let (sr, sc) = board.pawn(player);
    let start = square_index(sr, sc);
    let mut from = DistLayers::default();
    let mut to = DistLayers::default();
    fill_dist_layers_from_sq(start, masks, &mut from);
    fill_dist_layers_to_goal_row(player, masks, &mut to);

    let start_bit = flood_bit_sq(start);
    let Some(shortest) = (0..to.depth).find(|&d| to.masks[d] & start_bit != 0) else {
        return; // goal unreachable (illegal position) — contribute nothing
    };
    let tol = MAX_RELEVANT_CORRIDOR_DELTA as usize;

    // Near-shortest path SET: cells where dist_from + dist_to <= shortest + tol
    // (every route within `tol` suboptimal moves). Each cell sits in exactly one
    // `from` layer i and one `to` layer j, so its (dist_from + dist_to) == i + j.
    // The δ≤tol band IS the tolerance halo — hottest on optimal routes, decaying
    // to ~0 by the 4th-suboptimal fringe (corridor_heat: 200,77,40,25,18). Off
    // the band → 0. Both players accumulate, so shared corridors run hottest.
    for i in 0..from.depth {
        let fi = from.masks[i];
        if fi == 0 {
            continue;
        }
        let jmax = (shortest + tol).saturating_sub(i).min(to.depth.saturating_sub(1));
        for j in 0..=jmax {
            let cells = fi & to.masks[j] & FLOOD_PLAYABLE;
            if cells == 0 {
                continue;
            }
            let delta = (i + j).saturating_sub(shortest).min(tol) as u16;
            scatter_add(heat, cells, corridor_heat(delta));
        }
    }
}

/// Cheap BFF impact heatmap — drop-in replacement for `build_corridor_attention`
/// in the v16 CAT-LMR ordering path. Both players accumulate so shared corridors
/// run hottest. Only `square_heat` is populated (route_flex / bottleneck stay 0;
/// `wall_edge_heat` degrades to the pure corridor signal, which is the intent).
pub fn build_impact_heatmap(board: &Board) -> CorridorAttention {
    let masks = DirMasks::from_board(board);
    let mut out = CorridorAttention::default();
    add_player_impact_heat(board, Player::One, masks, &mut out.square_heat);
    add_player_impact_heat(board, Player::Two, masks, &mut out.square_heat);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::board::WallOrientation;
    use crate::util::grid::set_wall;

    #[test]
    fn impact_heatmap_hot_on_shared_corridor_cold_in_corner() {
        let board = Board::new();
        let cat = build_impact_heatmap(&board);
        let center = cat.square_heat(4, 4); // e5 — both players' shortest corridor
        let corner = cat.square_heat(0, 0); // a1 — far off any near-shortest path
        assert!(center > 0, "center should be hot: {center}");
        assert!(
            center > corner.saturating_mul(2),
            "shared corridor {center} >> corner {corner}"
        );
    }

    #[test]
    fn impact_heatmap_wall_on_corridor_beats_wall_in_corner() {
        // A wall edge sitting on the central shared corridor must read hotter than
        // one tucked in the corner — the whole point of CAT for LMR.
        let board = Board::new();
        let cat = build_impact_heatmap(&board);
        let central = cat.wall_edge_heat(3, 3, WallOrientation::Horizontal);
        let corner = cat.wall_edge_heat(0, 0, WallOrientation::Horizontal);
        assert!(central > corner, "central wall {central} > corner wall {corner}");
    }

    #[test]
    fn center_hotter_than_corner_at_startpos() {
        let board = Board::new();
        let mut scratch = BfsScratch::new();
        let cat = build_corridor_attention(&mut scratch, &board);
        let center = cat.square_heat(4, 4);
        let corner = cat.square_heat(0, 0);
        // With the δ≤4 path-set tolerance the corner sits on a *4th-suboptimal*
        // route, so it carries minimal heat (corridor_heat(4)=18) rather than
        // exactly 0 — the invariant is that the central corridor runs far hotter.
        assert!(center > corner.saturating_mul(4), "center {center} ≫ corner {corner}");
    }

    #[test]
    fn e_file_heat_peaks_at_pawns_not_uniform() {
        let board = Board::new();
        let mut scratch = BfsScratch::new();
        let cat = build_corridor_attention(&mut scratch, &board);
        let white_pawn = cat.square_heat(0, 4);
        let center = cat.square_heat(4, 4);
        let black_pawn = cat.square_heat(8, 4);
        assert!(
            white_pawn > center,
            "e1 hotter than e5, {white_pawn} vs {center}"
        );
        assert!(
            black_pawn > center,
            "e9 hotter than e5, {black_pawn} vs {center}"
        );
        assert!(
            white_pawn >= 190,
            "pawn square near full corridor cm, got {white_pawn}"
        );
        assert!(black_pawn >= 190);
        assert!(
            center < white_pawn,
            "pawn still hottest, pawn={white_pawn} center={center}"
        );
        assert!(
            center > 100,
            "mid-race corridor stays warm enough for wall search, center={center}"
        );
    }

    #[test]
    fn open_board_corners_stay_cold_for_search() {
        let board = Board::new();
        let mut scratch = BfsScratch::new();
        let cat = build_corridor_attention(&mut scratch, &board);
        // δ≤4 tolerance: corners sit on a 4th-suboptimal route → minimal (not zero) heat.
        assert!(
            cat.square_heat(0, 0) <= corridor_heat(MAX_RELEVANT_CORRIDOR_DELTA),
            "corner stays minimal, got {}",
            cat.square_heat(0, 0)
        );
        assert_eq!(cat.square_heat(0, 0), cat.square_heat(8, 8));
        assert!(
            cat.square_heat(4, 4) < cat.square_heat(0, 4),
            "center must stay cooler than pawn lane"
        );
        assert!(
            cat.square_heat(0, 4) < 220,
            "pawn heat should not stack two players, got {}",
            cat.square_heat(0, 4)
        );
    }

    #[test]
    fn far_corridor_squares_cooler_than_near_pawn() {
        let board = Board::new();
        let mut scratch = BfsScratch::new();
        let cat = build_corridor_attention(&mut scratch, &board);
        assert!(cat.square_heat(0, 4) > cat.square_heat(2, 4));
        assert!(cat.square_heat(8, 4) > cat.square_heat(6, 4));
    }

    #[test]
    fn wall_heat_prefers_central_corridor() {
        let board = Board::new();
        let mut scratch = BfsScratch::new();
        let cat = build_corridor_attention(&mut scratch, &board);
        let central = cat.wall_edge_heat(3, 4, WallOrientation::Horizontal);
        let passive = cat.wall_edge_heat(0, 0, WallOrientation::Horizontal);
        assert!(central > passive);
        assert!(passive <= 50);
    }

    #[test]
    fn multiple_lanes_after_wall() {
        let mut board = Board::new();
        set_wall(&mut board, 3, 4, WallOrientation::Horizontal, true);
        let mut scratch = BfsScratch::new();
        let cat = build_corridor_attention(&mut scratch, &board);
        assert!(cat.square_heat(4, 3) > 0);
        assert!(cat.square_heat(4, 5) > 0);
    }
}
