//! Build CAT heat from BFS distance fields on the pawn grid.

use std::sync::atomic::{AtomicI32, Ordering};

use crate::cat::attention::CorridorAttention;
use crate::cat::constants::{
    BOTTLENECK_BONUS_CM, BOTTLENECK_CORRIDOR_DELTA, CAT_CORRIDOR_CM, DEFAULT_CAT_DISTANCE_BIAS_BP,
    MAX_IMPACT_HEAT_DELTA, MAX_RELEVANT_CORRIDOR_DELTA,
};
use crate::core::board::{Board, Player};
use crate::path::distance::{
    fill_dist_from_sq, fill_dist_layers_from_sq, fill_dist_layers_to_goal_row,
    fill_dist_to_goal_row, DistLayers,
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

/// Centi-percent (68–100): gentle linear fade along the race. The near-pawn
/// squares are still slightly hottest, but the deep corridor — where walls
/// actually decide the race — keeps most of its heat. The floor was raised
/// 45→68 (corridor +~50%) because the old curve over-focused on the pawn:
/// near-pawn squares are easy to walk around, mid/far corridor blocks are not.
fn pawn_path_weight(dist_from: u8, shortest_to_goal: u8) -> u16 {
    if shortest_to_goal == 0 || shortest_to_goal == u8::MAX {
        return 100;
    }
    const MIN_WEIGHT: u16 = 68;
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
/// Board square overlay: symmetric sum of both players' corridors so the display
/// shows hot areas for both sides regardless of who is to move. Uses the base
/// `build_impact_heatmap` (no STM-specific zeroing) so neither player's forward
/// corridor is erased by the other's rear-wipe.
pub fn build_corridor_display_squares(scratch: &mut BfsScratch, board: &Board) -> [u16; 81] {
    let _ = scratch;
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
// ---------------------------------------------------------------------------

static CAT_DISTANCE_BIAS_BP: AtomicI32 = AtomicI32::new(DEFAULT_CAT_DISTANCE_BIAS_BP as i32);

/// Visualization-only path tilt (basis points). CAT worker may call this; search worker does not.
pub fn set_cat_distance_bias_bp(bias: i16) {
    CAT_DISTANCE_BIAS_BP.store(i32::from(bias.clamp(-9_900, 9_900)), Ordering::Relaxed);
}

pub fn cat_distance_bias_bp() -> i16 {
    CAT_DISTANCE_BIAS_BP
        .load(Ordering::Relaxed)
        .clamp(-9_900, 9_900) as i16
}

pub fn default_cat_distance_bias_bp() -> i16 {
    DEFAULT_CAT_DISTANCE_BIAS_BP
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ImpactHeatPreset {
    Conservative,
    Aggressive,
}

const ACTIVE_IMPACT_HEAT_PRESET: ImpactHeatPreset = ImpactHeatPreset::Conservative;

fn impact_heat_for_preset(delta: usize, preset: ImpactHeatPreset) -> u16 {
    if delta > MAX_IMPACT_HEAT_DELTA {
        return 0;
    }
    const CONSERVATIVE: [u16; MAX_IMPACT_HEAT_DELTA + 1] = [200, 77, 40, 25, 18, 12, 8, 4, 2];
    const AGGRESSIVE: [u16; MAX_IMPACT_HEAT_DELTA + 1] = [200, 180, 160, 140, 100, 60, 30, 14, 6];
    match preset {
        ImpactHeatPreset::Conservative => CONSERVATIVE[delta],
        ImpactHeatPreset::Aggressive => AGGRESSIVE[delta],
    }
}

fn impact_heat(delta: usize) -> u16 {
    impact_heat_for_preset(delta, ACTIVE_IMPACT_HEAT_PRESET)
}

/// Goal-hot (+bias) / pawn-hot (−bias) tilt along the to-goal layer index `j`.
fn distance_bias_mult(j: usize, shortest: usize, bias_bp: i16) -> u16 {
    if shortest == 0 || bias_bp == 0 {
        return 100;
    }
    let magnitude = i32::from(bias_bp).abs().min(9_900);
    let j = j as i32;
    let shortest = shortest as i32;
    let reduction = if bias_bp > 0 {
        magnitude * j / shortest / 100
    } else {
        magnitude * (shortest - j) / shortest / 100
    };
    (100 - reduction).clamp(1, 100) as u16
}

/// Add `w` to `heat[sq]` for every set cell of `mask` (saturating).
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

/// One player's impact contribution via overlapping bitmask layers.
pub(crate) fn add_player_impact_heat_with_bias(
    board: &Board,
    player: Player,
    masks: DirMasks,
    heat: &mut [u16; 81],
    bias_bp: i16,
) {
    let (sr, sc) = board.pawn(player);
    let start = square_index(sr, sc);
    let mut from = DistLayers::default();
    let mut to = DistLayers::default();
    fill_dist_layers_from_sq(start, masks, &mut from);
    fill_dist_layers_to_goal_row(player, masks, &mut to);

    let start_bit = flood_bit_sq(start);
    let Some(shortest) = (0..to.depth).find(|&d| to.masks[d] & start_bit != 0) else {
        return;
    };
    let tol = MAX_IMPACT_HEAT_DELTA;

    for i in 0..from.depth {
        let fi = from.masks[i];
        if fi == 0 {
            continue;
        }
        let jmax = (shortest + tol)
            .saturating_sub(i)
            .min(shortest)
            .min(to.depth.saturating_sub(1));
        for j in 0..=jmax {
            // Pawn square is a path-set entry node, not corridor heat.
            let cells = fi & to.masks[j] & FLOOD_PLAYABLE & !start_bit;
            if cells == 0 {
                continue;
            }
            let delta = (i + j).saturating_sub(shortest);
            let base = impact_heat(delta);
            if base == 0 {
                continue;
            }
            let mult = distance_bias_mult(j, shortest, bias_bp);
            let w = (u32::from(base) * u32::from(mult) / 100) as u16;
            scatter_add(heat, cells, w);
        }
    }
}

fn add_player_impact_heat(board: &Board, player: Player, masks: DirMasks, heat: &mut [u16; 81]) {
    let bias_bp = cat_distance_bias_bp();
    add_player_impact_heat_with_bias(board, player, masks, heat, bias_bp);
}

fn zero_pawn_entry_and_rear(heat: &mut [u16; 81], board: &Board, player: Player, masks: DirMasks) {
    let (sr, sc) = board.pawn(player);
    let pawn_sq = square_index(sr, sc);
    heat[pawn_sq as usize] = 0;

    let mut dist_to_goal = [u8::MAX; 81];
    fill_dist_to_goal_row(player, masks, &mut dist_to_goal);
    let our_dist = dist_to_goal[pawn_sq as usize];
    if our_dist == u8::MAX {
        return;
    }
    for sq in 0usize..81 {
        let d = dist_to_goal[sq];
        if d != u8::MAX && d > our_dist {
            heat[sq] = 0;
        }
    }
}

/// Fast bitmask impact heatmap for v16 LMR ordering and web square overlay.
///
/// Each player's heat is built and rear-zeroed independently before summing.
/// Sequential add-then-zero on a shared array would let P2's rear-wipe erase
/// P1's forward corridor when the pawns are close to their respective goals.
pub fn build_impact_heatmap(board: &Board) -> CorridorAttention {
    let masks = DirMasks::from_board(board);

    let mut h1 = [0u16; 81];
    add_player_impact_heat(board, Player::One, masks, &mut h1);
    zero_pawn_entry_and_rear(&mut h1, board, Player::One, masks);

    let mut h2 = [0u16; 81];
    add_player_impact_heat(board, Player::Two, masks, &mut h2);
    zero_pawn_entry_and_rear(&mut h2, board, Player::Two, masks);

    let mut out = CorridorAttention::default();
    for i in 0..81 {
        out.square_heat[i] = h1[i].saturating_add(h2[i]);
    }
    for player in [Player::One, Player::Two] {
        let (r, c) = board.pawn(player);
        out.square_heat[square_index(r, c) as usize] = 0;
    }
    out
}

/// Race-aware variant: builds the symmetric heatmap then additionally zeros any
/// combined heat that is strictly behind the side-to-move's pawn (farther from
/// their goal than they currently are). The symmetric base gives the correct view
/// of both players' corridors without cross-player erasure; this extra pass lets
/// the search ignore walls that can never help the mover.
#[inline]
pub fn build_impact_heatmap_for_stm(board: &Board, bfs: &mut BfsScratch) -> CorridorAttention {
    let _ = bfs;
    let mut cat = build_impact_heatmap(board);
    let stm = board.side();
    let masks = DirMasks::from_board(board);
    zero_pawn_entry_and_rear(&mut cat.square_heat, board, stm, masks);
    cat
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
        assert!(
            central > corner,
            "central wall {central} > corner wall {corner}"
        );
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
        assert!(
            center > corner.saturating_mul(4),
            "center {center} ≫ corner {corner}"
        );
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

    #[test]
    fn impact_no_heat_behind_pawn_invariant() {
        let mut board = Board::new();
        for m in ["e2", "e8", "e3", "e7", "e4"] {
            board.apply_algebraic(m);
        }
        let player = Player::One;
        let masks = DirMasks::from_board(&board);
        let (sr, sc) = board.pawn(player);
        let start = square_index(sr, sc);
        let start_bit = flood_bit_sq(start);
        let mut to = DistLayers::default();
        fill_dist_layers_to_goal_row(player, masks, &mut to);
        let shortest = (0..to.depth)
            .find(|&d| to.masks[d] & start_bit != 0)
            .expect("pawn reaches goal");

        let mut heat = [0u16; 81];
        add_player_impact_heat_with_bias(&board, player, masks, &mut heat, 0);

        let mut found_behind = false;
        for sq in 0u8..81 {
            let bit = flood_bit_sq(sq);
            let dist = (0..to.depth)
                .find(|&d| to.masks[d] & bit != 0)
                .unwrap_or(usize::MAX);
            if dist > shortest {
                found_behind = true;
                assert_eq!(heat[sq as usize], 0, "sq {sq} behind pawn got heat");
            }
        }
        assert!(found_behind, "fixture must include behind-pawn squares");
    }

    #[test]
    fn pawn_entry_square_has_no_impact_heat() {
        let board = Board::new();
        let cat = build_impact_heatmap(&board);
        let w = board.pawn(Player::One);
        let b = board.pawn(Player::Two);
        assert_eq!(
            cat.square_heat(w.0, w.1),
            0,
            "white pawn entry must not be corridor heat"
        );
        assert_eq!(
            cat.square_heat(b.0, b.1),
            0,
            "black pawn entry must not be corridor heat"
        );
    }

    #[test]
    fn stm_impact_zeros_heat_behind_pawn_even_when_tied() {
        let mut board = Board::new();
        // e4 vs e6 → both 5 steps from their goal row (tied sprint)
        for m in ["e2", "e8", "e3", "e7", "e4", "e6"] {
            board.apply_algebraic(m);
        }
        let mut bfs = BfsScratch::new();
        let stm = board.side();
        let our_dist = bfs.shortest_distance(&board, stm).unwrap_or(255);
        let opp_dist = bfs.shortest_distance(&board, stm.opposite()).unwrap_or(255);
        assert_eq!(our_dist, opp_dist, "tied sprint fixture");

        let race = build_impact_heatmap_for_stm(&board, &mut bfs);
        let masks = DirMasks::from_board(&board);
        let mut dist_to_goal = [u8::MAX; 81];
        fill_dist_to_goal_row(stm, masks, &mut dist_to_goal);

        let mut found_behind = false;
        for sq in 0u8..81 {
            let d = dist_to_goal[sq as usize];
            if d != u8::MAX && d > our_dist {
                found_behind = true;
                assert_eq!(race.square_heat[sq as usize], 0, "sq {sq} behind pawn");
            }
        }
        assert!(found_behind);
    }

    #[test]
    fn winning_race_zeros_heat_behind_stm_pawn() {
        let mut board = Board::new();
        for m in ["e2", "e8", "e3", "e7", "e4", "e6", "e5", "d6"] {
            board.apply_algebraic(m);
        }
        let mut bfs = BfsScratch::new();
        let stm = board.side();
        let our_dist = bfs.shortest_distance(&board, stm).unwrap_or(255);
        let opp_dist = bfs.shortest_distance(&board, stm.opposite()).unwrap_or(255);
        assert!(our_dist < opp_dist, "fixture must be winning race");

        let raw = build_impact_heatmap(&board);
        let race = build_impact_heatmap_for_stm(&board, &mut bfs);
        let masks = DirMasks::from_board(&board);
        let mut dist_to_goal = [u8::MAX; 81];
        fill_dist_to_goal_row(stm, masks, &mut dist_to_goal);
        let (sr, sc) = board.pawn(stm);
        let pawn_sq = square_index(sr, sc);

        let mut found_behind = false;
        for sq in 0u8..81 {
            let d = dist_to_goal[sq as usize];
            if d != u8::MAX && d > our_dist {
                found_behind = true;
                if sq != pawn_sq {
                    assert_eq!(
                        race.square_heat[sq as usize], 0,
                        "sq {sq} behind winning pawn should be cold, raw={}",
                        raw.square_heat[sq as usize]
                    );
                }
            }
        }
        assert!(found_behind);
    }

    #[test]
    fn conservative_impact_heat_lut_is_exact() {
        let expected = [200, 77, 40, 25, 18, 12, 8, 4, 2];
        for (delta, &want) in expected.iter().enumerate() {
            assert_eq!(
                impact_heat_for_preset(delta, ImpactHeatPreset::Conservative),
                want,
                "delta {delta}"
            );
        }
        assert_eq!(impact_heat_for_preset(9, ImpactHeatPreset::Conservative), 0);
    }

    #[test]
    fn aggressive_impact_heat_lut_is_exact() {
        let expected = [200, 180, 160, 140, 100, 60, 30, 14, 6];
        for (delta, &want) in expected.iter().enumerate() {
            assert_eq!(
                impact_heat_for_preset(delta, ImpactHeatPreset::Aggressive),
                want,
                "delta {delta}"
            );
        }
        assert_eq!(impact_heat_for_preset(9, ImpactHeatPreset::Aggressive), 0);
    }

    #[test]
    fn impact_falloff_monotone_conservative() {
        let d0 = impact_heat_for_preset(0, ImpactHeatPreset::Conservative);
        let d4 = impact_heat_for_preset(4, ImpactHeatPreset::Conservative);
        let d8 = impact_heat_for_preset(8, ImpactHeatPreset::Conservative);
        let d9 = impact_heat_for_preset(9, ImpactHeatPreset::Conservative);
        assert!(d0 > d4);
        assert!(d4 > d8);
        assert!(d8 > d9);
        assert_eq!(d9, 0);
    }

    #[test]
    fn distance_bias_goal_hot_and_pawn_hot() {
        assert_eq!(distance_bias_mult(0, 8, 1500), 100);
        assert_eq!(distance_bias_mult(8, 8, 1500), 85);
        assert_eq!(distance_bias_mult(0, 8, -1500), 85);
        assert_eq!(distance_bias_mult(8, 8, -1500), 100);
    }
}
