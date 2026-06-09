//! Reachability — **bitwise flood fill** on the 9×9 pawn grid (uniform edge cost).
//!
//! Direction masks are built once per board snapshot; expansion is word-parallel
//! shifts on a **centered 11-wide u128 layout** (`grid::FLOOD_STRIDE`) so
//! east/west shifts land in side buffers instead of wrapping rows. CAT is
//! accumulated during the same level-BFS passes used for shortest-path distance.
//!
//! See `docs/video/PERFT-OPTIMIZATIONS.md` Layer 4 for timings and oracles.

use crate::board::{Board, Player, WallOrientation};
use crate::grid::{
    can_step, flood_bit_sq, flood_sq_from_bit, goal_row, pack_flood_mask, square_index,
    unpack_square, FLOOD_PLAYABLE, FLOOD_STRIDE,
};
use std::ops::Index;

/// Per-square attention scores for move ordering / LMR (centi-units, not eval).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CorridorAttention {
    /// Search heat for each square. Zero means unreachable or too far from any reasonable route.
    square_heat: [u16; 81],
    /// Number of reasonable forward continuations through this square, summed over both players.
    route_flex: [u8; 81],
    /// Extra pressure on low-flex near-shortest corridors.
    bottleneck_heat: [u16; 81],
}

impl Default for CorridorAttention {
    fn default() -> Self {
        Self {
            square_heat: [0; 81],
            route_flex: [0; 81],
            bottleneck_heat: [0; 81],
        }
    }
}

impl Index<usize> for CorridorAttention {
    type Output = u16;

    fn index(&self, index: usize) -> &Self::Output {
        &self.square_heat[index]
    }
}

impl CorridorAttention {
    pub fn square_heat(&self, row: u8, col: u8) -> u16 {
        self.square_heat[square_index(row, col) as usize]
    }

    pub fn route_flex(&self, row: u8, col: u8) -> u8 {
        self.route_flex[square_index(row, col) as usize]
    }

    pub fn wall_edge_heat(&self, row: u8, col: u8, orientation: WallOrientation) -> u16 {
        let edge_heat = |a: (u8, u8), b: (u8, u8)| -> u16 {
            let ai = square_index(a.0, a.1) as usize;
            let bi = square_index(b.0, b.1) as usize;
            let corridor = self.square_heat[ai].min(self.square_heat[bi]);
            if corridor == 0 {
                return 0;
            }
            let bottleneck = self.bottleneck_heat[ai].min(self.bottleneck_heat[bi]);
            corridor.saturating_add(bottleneck)
        };

        let (a, b) = match orientation {
            WallOrientation::Horizontal => (
                edge_heat((row, col), (row + 1, col)),
                edge_heat((row, col + 1), (row + 1, col + 1)),
            ),
            WallOrientation::Vertical => (
                edge_heat((row, col), (row, col + 1)),
                edge_heat((row + 1, col), (row + 1, col + 1)),
            ),
        };
        // A wall blocks two adjacent pawn edges.  The max edge is the main reason
        // to search it; the weaker edge adds local interpolation without making
        // unrelated walls hot.
        a.max(b).saturating_add(a.min(b) / 4)
    }
}

// ── Corridor attention constants ───────────────────────────────────────────────

/// Heat given to any square on a player's shortest path (delta = 0).
/// Combined two-player ceiling: `2 × CAT_CORRIDOR_CM = 400 cm`.
const CAT_CORRIDOR_CM: u16 = 200;

/// Exact and near-shortest corridors are considered search-relevant.
/// Larger detours are deliberately zero so attention does not bleed across the board.
const MAX_RELEVANT_CORRIDOR_DELTA: u16 = 3;
const BOTTLENECK_CORRIDOR_DELTA: u16 = 2;
const BOTTLENECK_BONUS_CM: u16 = 40;

/// Bit `sq` set iff a pawn on `sq` may step in that direction.
#[derive(Clone, Copy, Default)]
pub struct DirMasks {
    pub north: u128,
    pub south: u128,
    pub east: u128,
    pub west: u128,
}

impl DirMasks {
    pub fn from_board(board: &Board) -> Self {
        let mut m = Self::default();
        for r in 0..=8u8 {
            for c in 0..=8u8 {
                let sq = square_index(r, c);
                let bit = flood_bit_sq(sq);
                if can_step(board, r, c, -1, 0) {
                    m.north |= bit;
                }
                if can_step(board, r, c, 1, 0) {
                    m.south |= bit;
                }
                if can_step(board, r, c, 0, 1) {
                    m.east |= bit;
                }
                if can_step(board, r, c, 0, -1) {
                    m.west |= bit;
                }
            }
        }
        m
    }
}

#[inline]
fn goal_square_mask(player: Player) -> u128 {
    let grow = goal_row(player);
    let mut mask = 0u128;
    for c in 0..9u8 {
        mask |= flood_bit_sq(square_index(grow, c));
    }
    mask
}

/// Expand flood frontier in centered 11-wide layout (side buffers absorb E/W shifts).
#[inline]
fn expand_frontier(frontier: u128, masks: DirMasks) -> u128 {
    let north = (frontier & masks.north) >> FLOOD_STRIDE;
    let south = (frontier & masks.south) << FLOOD_STRIDE;
    let east = (frontier & masks.east) << 1;
    let west = (frontier & masks.west) >> 1;
    north | south | east | west
}

#[inline]
fn flood_fill_flood_bits(start_sq: u8, masks: DirMasks) -> u128 {
    let mut reached = flood_bit_sq(start_sq);
    let mut frontier = reached;
    while frontier != 0 {
        frontier = expand_frontier(frontier, masks) & !reached & FLOOD_PLAYABLE;
        reached |= frontier;
    }
    reached
}

#[inline]
#[cfg(test)]
fn flood_fill(start_sq: u8, masks: DirMasks) -> u128 {
    pack_flood_mask(flood_fill_flood_bits(start_sq, masks))
}

#[inline]
fn flood_to_goal(start_sq: u8, masks: DirMasks, goal_mask: u128) -> (bool, u128) {
    let mut reached = flood_bit_sq(start_sq);
    if reached & goal_mask != 0 {
        return (true, reached);
    }
    let mut frontier = reached;
    while frontier != 0 {
        frontier = expand_frontier(frontier, masks) & !reached & FLOOD_PLAYABLE;
        reached |= frontier;
        if frontier & goal_mask != 0 {
            return (true, reached);
        }
    }
    (false, reached)
}

// ── Corridor attention distance-field helpers ─────────────────────────────────

/// Fill `dist_from[sq]` with the BFS distance from `start` to every reachable square.
/// Unreachable squares keep `u8::MAX`.
fn fill_dist_from_sq(start: u8, masks: DirMasks, dist_from: &mut [u8; 81]) {
    dist_from.fill(u8::MAX);
    dist_from[start as usize] = 0;
    let mut reached = flood_bit_sq(start);
    let mut frontier = reached;
    let mut layer = 0u8;
    while frontier != 0 {
        layer += 1;
        let new = expand_frontier(frontier, masks) & !reached & FLOOD_PLAYABLE;
        if new == 0 {
            break;
        }
        let mut bits = new;
        while bits != 0 {
            let fb = bits.trailing_zeros();
            bits &= bits - 1;
            let sq = flood_sq_from_bit(fb).expect("playable flood bit");
            dist_from[sq as usize] = layer;
        }
        reached |= new;
        frontier = new;
    }
}

/// Fill `dist_to[sq]` with the BFS distance from `sq` to any goal-row cell for
/// `player`.  Since Quoridor movement is symmetric (walls block both directions),
/// this equals a forward BFS seeded from all nine goal squares.
/// Unreachable squares keep `u8::MAX`.
fn fill_dist_to_goal_row(player: Player, masks: DirMasks, dist_to: &mut [u8; 81]) {
    let grow = goal_row(player);
    dist_to.fill(u8::MAX);

    let mut reached = 0u128;
    for c in 0..9u8 {
        let sq = square_index(grow, c);
        dist_to[sq as usize] = 0;
        reached |= flood_bit_sq(sq);
    }

    let mut frontier = reached;
    let mut layer = 0u8;
    while frontier != 0 {
        layer += 1;
        let new = expand_frontier(frontier, masks) & !reached & FLOOD_PLAYABLE;
        if new == 0 {
            break;
        }
        let mut bits = new;
        while bits != 0 {
            let fb = bits.trailing_zeros();
            bits &= bits - 1;
            let sq = flood_sq_from_bit(fb).expect("playable flood bit");
            dist_to[sq as usize] = layer;
        }
        reached |= new;
        frontier = new;
    }
}

fn corridor_heat(delta: u16) -> u16 {
    if delta > MAX_RELEVANT_CORRIDOR_DELTA {
        return 0;
    }
    // Focused log-shaped decay: exact shortest corridors stay hot, delta 1-3
    // stays searchable, and anything farther is zero instead of polluting LMR.
    let denom = 1.0 + f32::from(delta) * f32::from(delta + 2).log2();
    (f32::from(CAT_CORRIDOR_CM) / denom).round().max(1.0) as u16
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

fn reasonable_forward_continuations(
    sq: u8,
    masks: DirMasks,
    dist_from_pawn: &[u8; 81],
    dist_to_goal: &[u8; 81],
    shortest_to_goal: u8,
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
        if next_from == from.saturating_add(1)
            && next_to < to
            && corridor_delta(next, dist_from_pawn, dist_to_goal, shortest_to_goal)
                .is_some_and(|d| d <= MAX_RELEVANT_CORRIDOR_DELTA)
        {
            count = count.saturating_add(1);
        }
    }
    count
}

/// Accumulate one player's corridor heat into `out`.
///
/// For each reachable square `sq`:
///   - `delta = dist_from[sq] + dist_to[sq] - shortest`
///   - `delta == 0`: square lies on at least one shortest path to any goal square.
///   - small delta: square lies on a near-shortest route.
///   - large delta or unreachable: zero heat, so dead walls can be pruned hard.
///
/// Squares not reachable from the pawn contribute nothing.
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

    let shortest_to_goal = dist_to_goal[start as usize]; // u8::MAX if pawn is trapped

    for sq in 0u8..81 {
        let Some(delta) = corridor_delta(sq, dist_from_pawn, dist_to_goal, shortest_to_goal) else {
            continue;
        };
        let heat = corridor_heat(delta);
        if heat == 0 {
            continue;
        }

        let idx = sq as usize;
        let flex = reasonable_forward_continuations(
            sq,
            masks,
            dist_from_pawn,
            dist_to_goal,
            shortest_to_goal,
        );
        out.square_heat[idx] = out.square_heat[idx].saturating_add(heat);
        out.route_flex[idx] = out.route_flex[idx].saturating_add(flex);
        if delta <= BOTTLENECK_CORRIDOR_DELTA && flex <= 1 && dist_to_goal[idx] > 0 {
            out.bottleneck_heat[idx] = out.bottleneck_heat[idx].saturating_add(BOTTLENECK_BONUS_CM);
        }
    }
}

/// Reused flood-fill scratch — pass through perft/move-gen hot loops.
#[derive(Clone)]
pub struct BfsScratch {
    visited: u128,
    queue: [u8; 81],
    /// Scratch for corridor forward BFS (distance from pawn to each square).
    dist_from_pawn: [u8; 81],
    /// Scratch for corridor reverse BFS (distance from each square to goal row).
    dist_to_goal: [u8; 81],
}

impl Default for BfsScratch {
    fn default() -> Self {
        Self::new()
    }
}

impl BfsScratch {
    pub fn new() -> Self {
        Self {
            visited: 0,
            queue: [0; 81],
            dist_from_pawn: [0; 81],
            dist_to_goal: [0; 81],
        }
    }

    /// Build corridor attention for search ordering, LMR, and futility.
    ///
    /// For each player, runs a forward BFS from its pawn and a backward BFS from
    /// all cells in its goal row. This considers every reachable winning square,
    /// not the first goal touched by BFS.
    ///
    /// ```text
    /// delta = dist_from_pawn[sq] + dist_to_goal[sq] - shortest
    /// heat  = logarithmic_decay(delta), for delta <= MAX_RELEVANT_CORRIDOR_DELTA
    /// ```
    ///
    /// Unreachable and far-detour squares stay at zero so dead walls can be
    /// reduced or pruned aggressively.
    pub fn build_corridor_attention(&mut self, board: &Board) -> CorridorAttention {
        let masks = DirMasks::from_board(board);
        let mut attention = CorridorAttention::default();
        add_player_corridor_attention(
            board,
            Player::One,
            masks,
            &mut attention,
            &mut self.dist_from_pawn,
            &mut self.dist_to_goal,
        );
        add_player_corridor_attention(
            board,
            Player::Two,
            masks,
            &mut attention,
            &mut self.dist_from_pawn,
            &mut self.dist_to_goal,
        );
        attention
    }

    /// Count low-flex squares on this player's exact/near-shortest corridors.
    /// Higher means the player is easier to cage if the opponent still has walls.
    pub fn corridor_bottleneck_count(&mut self, board: &Board, player: Player) -> u8 {
        let masks = DirMasks::from_board(board);
        let (sr, sc) = board.pawn(player);
        let start = square_index(sr, sc);
        fill_dist_from_sq(start, masks, &mut self.dist_from_pawn);
        fill_dist_to_goal_row(player, masks, &mut self.dist_to_goal);
        let shortest_to_goal = self.dist_to_goal[start as usize];
        if shortest_to_goal == u8::MAX {
            return 8;
        }

        let mut bottlenecks = 0u8;
        for sq in 0u8..81 {
            let Some(delta) = corridor_delta(
                sq,
                &self.dist_from_pawn,
                &self.dist_to_goal,
                shortest_to_goal,
            ) else {
                continue;
            };
            if delta > BOTTLENECK_CORRIDOR_DELTA || self.dist_to_goal[sq as usize] == 0 {
                continue;
            }
            let flex = reasonable_forward_continuations(
                sq,
                masks,
                &self.dist_from_pawn,
                &self.dist_to_goal,
                shortest_to_goal,
            );
            if flex <= 1 {
                bottlenecks = bottlenecks.saturating_add(1);
            }
        }
        bottlenecks.min(8)
    }

    /// Reachability only — bitwise flood to goal row.
    #[inline]
    pub fn can_reach_goal(&mut self, board: &Board, player: Player) -> bool {
        let masks = DirMasks::from_board(board);
        let (sr, sc) = board.pawn(player);
        let start = square_index(sr, sc);
        flood_to_goal(start, masks, goal_square_mask(player)).0
    }

    /// Both players must reach their goal. Reuses P1's component mask when P2 is inside it.
    #[inline]
    pub fn both_players_reach_goals(&mut self, board: &Board) -> bool {
        let masks = DirMasks::from_board(board);
        let (r1, c1) = board.pawn(Player::One);
        let start1 = square_index(r1, c1);
        let goal1 = goal_square_mask(Player::One);
        let (ok1, comp1) = flood_to_goal(start1, masks, goal1);
        if !ok1 {
            return false;
        }

        let (r2, c2) = board.pawn(Player::Two);
        let start2 = square_index(r2, c2);
        let goal2 = goal_square_mask(Player::Two);
        let start2_bit = flood_bit_sq(start2);

        if comp1 & start2_bit != 0 {
            return comp1 & goal2 != 0;
        }

        flood_to_goal(start2, masks, goal2).0
    }

    /// Bitwise flood from `player`'s pawn — sets bits in `mask`.
    pub fn fill_reachable(&mut self, board: &Board, player: Player, mask: &mut u128) {
        let masks = DirMasks::from_board(board);
        let (sr, sc) = board.pawn(player);
        let start = square_index(sr, sc);
        *mask |= pack_flood_mask(flood_fill_flood_bits(start, masks));
    }

    /// Union of squares reachable by either pawn.
    pub fn both_reachable_mask(&mut self, board: &Board) -> u128 {
        let mut mask = 0u128;
        self.fill_reachable(board, Player::One, &mut mask);
        self.fill_reachable(board, Player::Two, &mut mask);
        mask
    }

    /// Shortest pawn-step distance to any goal square on the goal row.
    pub fn shortest_distance(&mut self, board: &Board, player: Player) -> Option<u8> {
        let masks = DirMasks::from_board(board);
        let (sr, sc) = board.pawn(player);
        let start = square_index(sr, sc);
        let goal_mask = goal_square_mask(player);

        let mut reached = flood_bit_sq(start);
        if reached & goal_mask != 0 {
            return Some(0);
        }

        let mut frontier = reached;
        let mut d = 0u8;
        while frontier != 0 {
            d += 1;
            frontier = expand_frontier(frontier, masks) & !reached & FLOOD_PLAYABLE;
            if frontier & goal_mask != 0 {
                return Some(d);
            }
            reached |= frontier;
        }
        None
    }

    /// Backward BFS from all goal-row cells — next hop toward goal per square.
    pub fn fill_next_toward_goal(
        &mut self,
        board: &Board,
        player: Player,
        next_out: &mut [u8; 81],
    ) {
        next_out.fill(u8::MAX);
        let grow = goal_row(player);

        self.visited = 0;
        let mut head = 0usize;
        let mut tail = 0usize;

        for col in 0..9u8 {
            let sq = square_index(grow, col);
            let mask = 1u128 << sq;
            if self.visited & mask == 0 {
                self.visited |= mask;
                self.queue[tail] = sq;
                tail += 1;
            }
        }

        const NEIGHBORS: [(i8, i8); 4] = [(1, 0), (0, 1), (-1, 0), (0, -1)];

        while head < tail {
            let sq = self.queue[head];
            head += 1;
            let (r, c) = unpack_square(sq);

            for (dr, dc) in NEIGHBORS {
                let nr = r as i16 + dr as i16;
                let nc = c as i16 + dc as i16;
                if !(0..=8).contains(&nr) || !(0..=8).contains(&nc) {
                    continue;
                }
                let nr = nr as u8;
                let nc = nc as u8;
                if !can_step(board, nr, nc, -dr, -dc) {
                    continue;
                }
                let nsq = square_index(nr, nc);
                let mask = 1u128 << nsq;
                if self.visited & mask != 0 {
                    continue;
                }
                self.visited |= mask;
                next_out[nsq as usize] = sq;
                self.queue[tail] = nsq;
                tail += 1;
            }
        }
    }
}

#[inline]
pub fn can_reach_goal(board: &Board, player: Player) -> bool {
    BfsScratch::new().can_reach_goal(board, player)
}

pub fn shortest_distance(board: &Board, player: Player) -> Option<u8> {
    BfsScratch::new().shortest_distance(board, player)
}

#[inline]
pub fn both_players_reach_goals(board: &Board) -> bool {
    BfsScratch::new().both_players_reach_goals(board)
}

#[cfg(test)]
mod naive_reference {
    //! Queue BFS oracle — validates bitwise flood fill independent of DirMasks.

    use super::*;
    use crate::grid::is_goal;

    const NEIGHBORS: [(i8, i8); 4] = [(1, 0), (0, 1), (-1, 0), (0, -1)];

    pub fn flood_fill_naive(board: &Board, start: u8) -> u128 {
        let mut visited = 1u128 << start;
        let mut queue = [0u8; 81];
        let mut head = 0usize;
        let mut tail = 1usize;
        queue[0] = start;

        while head < tail {
            let sq = queue[head];
            head += 1;
            let (r, c) = unpack_square(sq);
            for (dr, dc) in NEIGHBORS {
                if !can_step(board, r, c, dr, dc) {
                    continue;
                }
                let nr = (r as i8 + dr) as u8;
                let nc = (c as i8 + dc) as u8;
                let nsq = square_index(nr, nc);
                let bit = 1u128 << nsq;
                if visited & bit != 0 {
                    continue;
                }
                visited |= bit;
                queue[tail] = nsq;
                tail += 1;
            }
        }
        visited
    }

    pub fn can_reach_goal_naive(board: &Board, player: Player) -> bool {
        let (sr, sc) = board.pawn(player);
        let start = square_index(sr, sc);
        let mut visited = 1u128 << start;
        let mut queue = [0u8; 81];
        let mut head = 0usize;
        let mut tail = 1usize;
        queue[0] = start;

        while head < tail {
            let sq = queue[head];
            head += 1;
            let (r, c) = unpack_square(sq);
            if is_goal(player, r) {
                return true;
            }
            for (dr, dc) in NEIGHBORS {
                if !can_step(board, r, c, dr, dc) {
                    continue;
                }
                let nr = (r as i8 + dr) as u8;
                let nc = (c as i8 + dc) as u8;
                let nsq = square_index(nr, nc);
                let bit = 1u128 << nsq;
                if visited & bit != 0 {
                    continue;
                }
                visited |= bit;
                queue[tail] = nsq;
                tail += 1;
            }
        }
        false
    }

    pub fn shortest_distance_naive(board: &Board, player: Player) -> Option<u8> {
        let (sr, sc) = board.pawn(player);
        let start = square_index(sr, sc);
        let mut visited = 1u128 << start;
        let mut queue = [0u8; 81];
        let mut depth = [0u8; 81];
        let mut head = 0usize;
        let mut tail = 1usize;
        queue[0] = start;
        depth[0] = 0;

        while head < tail {
            let sq = queue[head];
            let d = depth[head];
            head += 1;
            let (r, c) = unpack_square(sq);
            if is_goal(player, r) {
                return Some(d);
            }
            for (dr, dc) in NEIGHBORS {
                if !can_step(board, r, c, dr, dc) {
                    continue;
                }
                let nr = (r as i8 + dr) as u8;
                let nc = (c as i8 + dc) as u8;
                let nsq = square_index(nr, nc);
                let bit = 1u128 << nsq;
                if visited & bit != 0 {
                    continue;
                }
                visited |= bit;
                queue[tail] = nsq;
                depth[tail] = d + 1;
                tail += 1;
            }
        }
        None
    }

    pub fn both_players_reach_goals_naive(board: &Board) -> bool {
        can_reach_goal_naive(board, Player::One) && can_reach_goal_naive(board, Player::Two)
    }
}

#[cfg(test)]
mod tests {
    use super::naive_reference::{
        both_players_reach_goals_naive, can_reach_goal_naive, flood_fill_naive,
        shortest_distance_naive,
    };
    use super::*;
    use crate::board::WallOrientation;
    use crate::grid::set_wall;

    fn assert_bitwise_matches_naive(board: &Board) {
        let masks = DirMasks::from_board(board);
        let mut scratch = BfsScratch::new();

        for sq in 0u8..81 {
            let bitwise = flood_fill(sq, masks);
            let naive = flood_fill_naive(board, sq);
            assert_eq!(
                bitwise, naive,
                "reachable mismatch from sq {sq} on board {:?}",
                board
            );
        }

        for player in [Player::One, Player::Two] {
            assert_eq!(
                scratch.can_reach_goal(board, player),
                can_reach_goal_naive(board, player),
                "can_reach_goal mismatch for {player:?}"
            );
            assert_eq!(
                scratch.shortest_distance(board, player),
                shortest_distance_naive(board, player),
                "shortest_distance mismatch for {player:?}"
            );
        }

        assert_eq!(
            scratch.both_players_reach_goals(board),
            both_players_reach_goals_naive(board),
            "both_players_reach_goals mismatch"
        );

        let mut mask_bitwise = 0u128;
        scratch.fill_reachable(board, Player::One, &mut mask_bitwise);
        let mut mask_bitwise2 = 0u128;
        scratch.fill_reachable(board, Player::Two, &mut mask_bitwise2);
        let union_bitwise = mask_bitwise | mask_bitwise2;
        assert_eq!(union_bitwise, scratch.both_reachable_mask(board));
    }

    fn board_with_walls(walls: &[(u8, u8, WallOrientation)]) -> Board {
        let mut board = Board::new();
        for &(row, col, orientation) in walls {
            set_wall(&mut board, row, col, orientation, true);
        }
        board
    }

    #[test]
    fn dir_masks_agree_with_can_step_on_startpos() {
        let board = Board::new();
        let masks = DirMasks::from_board(&board);
        for r in 0..=8u8 {
            for c in 0..=8u8 {
                let sq = square_index(r, c);
                let bit = flood_bit_sq(sq);
                assert_eq!(
                    masks.north & bit != 0,
                    can_step(&board, r, c, -1, 0),
                    "north at ({r},{c})"
                );
                assert_eq!(
                    masks.south & bit != 0,
                    can_step(&board, r, c, 1, 0),
                    "south at ({r},{c})"
                );
                assert_eq!(
                    masks.east & bit != 0,
                    can_step(&board, r, c, 0, 1),
                    "east at ({r},{c})"
                );
                assert_eq!(
                    masks.west & bit != 0,
                    can_step(&board, r, c, 0, -1),
                    "west at ({r},{c})"
                );
            }
        }
    }

    #[test]
    fn bitwise_flood_matches_naive_queue_on_startpos() {
        assert_bitwise_matches_naive(&Board::new());
    }

    #[test]
    fn bitwise_flood_matches_naive_with_barrier() {
        let board = board_with_walls(&[
            (6, 0, WallOrientation::Horizontal),
            (6, 1, WallOrientation::Horizontal),
            (6, 2, WallOrientation::Horizontal),
            (6, 3, WallOrientation::Horizontal),
            (6, 4, WallOrientation::Horizontal),
            (6, 5, WallOrientation::Horizontal),
            (6, 6, WallOrientation::Horizontal),
            (6, 7, WallOrientation::Horizontal),
        ]);
        assert_bitwise_matches_naive(&board);
    }

    #[test]
    fn bitwise_flood_matches_naive_with_mixed_walls() {
        let board = board_with_walls(&[
            (3, 3, WallOrientation::Vertical),
            (4, 4, WallOrientation::Horizontal),
            (2, 6, WallOrientation::Vertical),
            (5, 1, WallOrientation::Horizontal),
            (7, 3, WallOrientation::Vertical),
        ]);
        assert_bitwise_matches_naive(&board);
    }

    #[test]
    fn bitwise_flood_matches_naive_perft_depth2_prefix() {
        // Replay first two plies (e2 e8) — exercises wall gen path checks on a real subtree.
        let mut board = Board::new();
        let _ = board.make_move(crate::board::Move::Pawn { row: 1, col: 4 });
        let _ = board.make_move(crate::board::Move::Pawn { row: 7, col: 4 });
        assert_bitwise_matches_naive(&board);
    }

    #[test]
    fn centered_layout_absorbs_east_shift_in_side_buffer() {
        use crate::grid::{flood_bit_index, FLOOD_COL_PAD, FLOOD_ROW_PAD, FLOOD_STRIDE};

        let board = Board::new();
        let masks = DirMasks::from_board(&board);

        // Playable grid fits inside u128 with side buffers (stride 11, max bit 108).
        assert!(flood_bit_index(8, 8) < 128);
        assert_eq!(FLOOD_PLAYABLE.count_ones(), 81);

        // Force east open at (0,8): shift lands in side buffer, never (1,0) on next row.
        let mut masks = masks;
        masks.east |= flood_bit_sq(square_index(0, 8));
        let frontier = flood_bit_sq(square_index(0, 8));
        let raw = expand_frontier(frontier, masks);
        assert_eq!(
            raw & flood_bit_sq(square_index(1, 0)),
            0,
            "must not reach (1,0) across rows"
        );
        let buffer_col = FLOOD_ROW_PAD * FLOOD_STRIDE + FLOOD_COL_PAD + 9;
        assert_ne!(
            raw & (1u128 << buffer_col),
            0,
            "east shift should land in side buffer"
        );

        assert_bitwise_matches_naive(&board);
    }

    #[test]
    fn start_position_reachable() {
        let board = Board::new();
        assert!(can_reach_goal(&board, Player::One));
        assert!(can_reach_goal(&board, Player::Two));
        assert_eq!(shortest_distance(&board, Player::One), Some(8));
        assert_eq!(shortest_distance(&board, Player::Two), Some(8));
    }

    #[test]
    fn full_barrier_blocks_p1() {
        let mut board = Board::new();
        for c in 0..8u8 {
            set_wall(&mut board, 6, c, WallOrientation::Horizontal, true);
        }
        assert!(!can_reach_goal(&board, Player::One));
    }

    #[test]
    fn scratch_matches_stack_bfs() {
        let board = Board::new();
        let mut scratch = BfsScratch::new();
        assert_eq!(scratch.shortest_distance(&board, Player::One), Some(8));
        assert!(scratch.both_players_reach_goals(&board));
    }

    #[test]
    fn both_reachable_mask_includes_both_pawns() {
        let board = Board::new();
        let mut scratch = BfsScratch::new();
        let mask = scratch.both_reachable_mask(&board);
        assert_ne!(mask & (1u128 << square_index(0, 4)), 0);
        assert_ne!(mask & (1u128 << square_index(8, 4)), 0);
    }

    #[test]
    fn bitwise_flood_reaches_goal_row() {
        let board = Board::new();
        let masks = DirMasks::from_board(&board);
        let start = square_index(0, 4);
        let reached = flood_fill(start, masks);
        let mut goal_row_mask = 0u128;
        for c in 0..9u8 {
            goal_row_mask |= 1u128 << square_index(8, c);
        }
        assert_ne!(reached & goal_row_mask, 0);
    }

    #[test]
    fn ishtar_component_reuse_same_board() {
        let board = Board::new();
        let mut scratch = BfsScratch::new();
        assert!(scratch.both_players_reach_goals(&board));
    }

    // ── Corridor attention tests ──────────────────────────────────────────────

    #[test]
    fn corridor_attention_startpos_center_hotter_than_corner() {
        let board = Board::new();
        let mut scratch = BfsScratch::new();
        let cat = scratch.build_corridor_attention(&board);
        // e5 (row 4, col 4) is on the shortest path for both players → max heat.
        let center = cat.square_heat(4, 4);
        // a1 (row 0, col 0) is too far from both shortest corridors → zero heat.
        let corner = cat.square_heat(0, 0);
        assert!(
            center > corner,
            "center={center} should be hotter than corner={corner}"
        );
        assert_eq!(corner, 0, "far detours should not bleed attention");
    }

    #[test]
    fn corridor_attention_on_path_squares_at_max_heat() {
        let board = Board::new();
        let mut scratch = BfsScratch::new();
        let cat = scratch.build_corridor_attention(&board);
        // At startpos the e-file (col 4) is the direct path for both pawns.
        // e1..e9 (rows 0..8, col 4) are all delta=0 for both → 200+200 = 400 cm.
        for row in 0u8..9 {
            let heat = cat.square_heat(row, 4);
            assert_eq!(
                heat, 400,
                "e-file row {row} should be 400 cm (on-path for both), got {heat}"
            );
        }
    }

    #[test]
    fn corridor_attention_near_path_decays_then_cuts_off() {
        let board = Board::new();
        let mut scratch = BfsScratch::new();
        let cat = scratch.build_corridor_attention(&board);
        let on_path = cat.square_heat(4, 4);
        let near_path = cat.square_heat(4, 3);
        let far_path = cat.square_heat(4, 0);
        assert!(
            near_path < on_path && near_path > 0,
            "near-path col3 row4 ({near_path}) should be nonzero and below on-path {on_path}"
        );
        assert_eq!(far_path, 0, "delta-4 detours should be cold, got {far_path}");
    }

    #[test]
    fn corridor_attention_dist_fields_match_naive_distances() {
        // Verify fill_dist_from_sq agrees with shortest_distance_naive.
        let board = Board::new();
        let masks = DirMasks::from_board(&board);
        let mut dist_from = [u8::MAX; 81];
        let start = square_index(0, 4); // e1
        fill_dist_from_sq(start, masks, &mut dist_from);

        // e9 (row 8, col 4) should be 8 steps from e1.
        assert_eq!(dist_from[square_index(8, 4) as usize], 8);
        // a1 (row 0, col 0) is 4 steps from e1.
        assert_eq!(dist_from[square_index(0, 0) as usize], 4);

        // Verify fill_dist_to_goal_row agrees with shortest_distance for P1.
        let mut dist_to = [u8::MAX; 81];
        fill_dist_to_goal_row(Player::One, masks, &mut dist_to);
        // e1 (start, row 0, col 4) is 8 steps from P1's goal row 8.
        assert_eq!(dist_to[square_index(0, 4) as usize], 8);
        // Goal row itself is 0 steps away.
        assert_eq!(dist_to[square_index(8, 4) as usize], 0);
    }

    #[test]
    fn corridor_attention_marks_multiple_equal_lanes() {
        let mut board = Board::new();
        set_wall(&mut board, 3, 4, WallOrientation::Horizontal, true);
        let mut scratch = BfsScratch::new();
        let cat = scratch.build_corridor_attention(&board);

        let left_lane = cat.square_heat(4, 3);
        let right_lane = cat.square_heat(4, 5);
        assert!(left_lane > 0, "left detour lane should be searchable");
        assert!(right_lane > 0, "right detour lane should be searchable");
    }

    #[test]
    fn corridor_wall_heat_prefers_shared_corridor_edges() {
        let board = Board::new();
        let mut scratch = BfsScratch::new();
        let cat = scratch.build_corridor_attention(&board);

        let central = cat.wall_edge_heat(3, 4, WallOrientation::Horizontal);
        let passive = cat.wall_edge_heat(0, 0, WallOrientation::Horizontal);
        assert!(
            central > passive,
            "central corridor wall {central} should outrank passive wall {passive}"
        );
        assert!(
            passive <= 50,
            "passive delta-3 wall should only have tiny heat, got {passive}"
        );
    }
}
