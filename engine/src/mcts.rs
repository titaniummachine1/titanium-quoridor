//! Gorisanson-style MCTS on the fast Rust board (make/unmake + BFS rollouts).
//!
//! When `use_cat_guidance` is enabled the tree uses corridor attention to bias
//! expansion child selection (weighted reservoir sampling) and the 20 %
//! random-move fraction of each rollout toward tactically hot squares.
//! CAT is computed once at the root and reused for the whole search —
//! cheap enough for the early game where walls haven't fragmented the board.

use std::time::Instant;

use rand::Rng;

use crate::board::{Board, Move, Player};
use crate::grid::{is_goal, square_index};
use crate::moves::{generate_legal_moves_slice, MAX_LEGAL_MOVES};
use crate::opening::BookHint;
use crate::path::{BfsScratch, CorridorAttention};
use crate::perft::format_move;

pub const DEFAULT_UCT: f64 = 0.2;
pub const DEFAULT_TIME_MS: u64 = 10_000;
pub const DEFAULT_MAX_SIMULATIONS: u64 = 2_000_000_000;

/// CAT score >= this threshold is considered "hot" during bridge rollouts.
const BRIDGE_HOT_CM: u16 = 160;

#[derive(Debug, Clone, Copy)]
pub struct MctsConfig {
    pub time_ms: u64,
    pub max_simulations: u64,
    pub uct: f64,
    pub log: bool,
    /// When true the search uses corridor heat to steer expansion and rollout
    /// decisions.  This is the "CAT-MCTS bridge" mode for early midgame.
    pub use_cat_guidance: bool,
    /// Opening book hint — biases expansion toward theory without skipping search.
    pub book_hint: Option<BookHint>,
}

impl Default for MctsConfig {
    fn default() -> Self {
        Self {
            time_ms: DEFAULT_TIME_MS,
            max_simulations: DEFAULT_MAX_SIMULATIONS,
            uct: DEFAULT_UCT,
            log: false,
            use_cat_guidance: false,
            book_hint: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MctsReport {
    pub best_move: Move,
    pub simulations: u64,
    pub elapsed_ms: u64,
    pub stopped_by: &'static str,
    pub white_dist: u8,
    pub black_dist: u8,
    pub root_win_rate: f64,
}

struct Node {
    mv: Move,
    parent: usize,
    children: Vec<usize>,
    wins: f64,
    visits: u64,
    terminal: bool,
}

struct MctsTree {
    nodes: Vec<Node>,
    uct: f64,
}

impl MctsTree {
    fn new(uct: f64) -> Self {
        Self {
            nodes: vec![Node {
                mv: Move::Pawn { row: 0, col: 0 },
                parent: usize::MAX,
                children: Vec::new(),
                wins: 0.0,
                visits: 0,
                terminal: false,
            }],
            uct,
        }
    }

    fn uct_score(&self, node: usize, parent_ln_visits: f64) -> f64 {
        let n = &self.nodes[node];
        if n.visits == 0 {
            return f64::INFINITY;
        }
        let exploit = n.wins / n.visits as f64;
        let explore = (self.uct * parent_ln_visits / n.visits as f64).sqrt();
        exploit + explore
    }

    fn best_child_uct(&self, node: usize) -> usize {
        let parent_ln_visits = (self.nodes[node].visits.max(1) as f64).ln();
        let mut best = self.nodes[node].children[0];
        let mut best_score = self.uct_score(best, parent_ln_visits);
        for &child in &self.nodes[node].children[1..] {
            let score = self.uct_score(child, parent_ln_visits);
            if score > best_score {
                best_score = score;
                best = child;
            }
        }
        best
    }

    fn random_unvisited_child(&self, node: usize, rng: &mut impl Rng) -> Option<usize> {
        let mut seen = 0usize;
        let mut picked = None;
        for &child in &self.nodes[node].children {
            if self.nodes[child].visits != 0 {
                continue;
            }
            seen += 1;
            // Reservoir sample 1 item uniformly without extra allocations/passes.
            if rng.gen_range(0..seen) == 0 {
                picked = Some(child);
            }
        }
        picked
    }

    fn best_child_sims(&self, node: usize) -> usize {
        let mut best = self.nodes[node].children[0];
        let mut best_sims = self.nodes[best].visits;
        for &child in &self.nodes[node].children[1..] {
            if self.nodes[child].visits > best_sims {
                best_sims = self.nodes[child].visits;
                best = child;
            }
        }
        best
    }

    fn add_child(&mut self, parent: usize, mv: Move) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(Node {
            mv,
            parent,
            children: Vec::new(),
            wins: 0.0,
            visits: 0,
            terminal: false,
        });
        self.nodes[parent].children.push(idx);
        idx
    }

    fn backprop(&mut self, mut node: usize, winner: Player, leaf_mover: Player) {
        let mut mover = leaf_mover;
        while node != usize::MAX {
            let n = &mut self.nodes[node];
            n.visits += 1;
            if winner == mover {
                n.wins += 1.0;
            }
            mover = mover.opposite();
            node = n.parent;
        }
    }

    /// Weighted reservoir sampling over unvisited children; each child's weight
    /// is its max CAT heat (floored at 1 so cold moves still get considered).
    /// P(select child i) = heat_i / sum(heats) — hot moves explored first.
    fn cat_weighted_unvisited_child(
        &self,
        node: usize,
        cat: &CorridorAttention,
        book_mv: Option<Move>,
        sprint_mv: Option<Move>,
        rng: &mut impl Rng,
    ) -> Option<usize> {
        let mut total_weight = 0u64;
        let mut picked = None;
        for &child in &self.nodes[node].children {
            if self.nodes[child].visits != 0 {
                continue;
            }
            let mv = self.nodes[child].mv;
            let heat = cat_heat_for_move(mv, cat);
            let mut weight = u64::from(heat).max(1);
            if sprint_mv == Some(mv) {
                weight = weight.saturating_mul(100);
            } else if book_mv == Some(mv) {
                weight = weight.saturating_mul(10);
            }
            total_weight += weight;
            // Accept this item with probability weight/total_weight (weighted reservoir).
            if rng.gen_range(0..total_weight) < weight {
                picked = Some(child);
            }
        }
        picked
    }
}

// ---------------------------------------------------------------------------
// CAT bridge helpers
// ---------------------------------------------------------------------------

/// Max CAT heat cm across all squares touched by `mv`.
fn cat_heat_for_move(mv: Move, cat: &CorridorAttention) -> u16 {
    match mv {
        Move::Pawn { row, col } => cat.square_heat(row, col),
        Move::Wall {
            row,
            col,
            orientation,
        } => cat.wall_edge_heat(row, col, orientation),
    }
}

/// During the 20 % random rollout fraction, prefer moves touching hot CAT
/// squares (>= BRIDGE_HOT_CM).  Falls back to uniform random if no hot move
/// exists or the 33 % cold-path lottery fires (keeps rollouts stochastic).
fn pick_hot_rollout_move(
    legal: &[Move; MAX_LEGAL_MOVES],
    n: usize,
    cat: &CorridorAttention,
    rng: &mut impl Rng,
) -> Move {
    // Collect hot moves into a small fixed-size scratch array (no heap alloc).
    const MAX_HOT: usize = 64;
    let mut hot = [Move::Pawn { row: 0, col: 0 }; MAX_HOT];
    let mut hot_n = 0usize;
    for i in 0..n {
        if cat_heat_for_move(legal[i], cat) >= BRIDGE_HOT_CM && hot_n < MAX_HOT {
            hot[hot_n] = legal[i];
            hot_n += 1;
        }
    }
    // 67 % chance to use a hot move when available; 33 % stays fully random
    // so the tree can still explore cold defensive ideas.
    if hot_n > 0 && rng.gen_range(0..3) < 2 {
        return hot[rng.gen_range(0..hot_n)];
    }
    legal[rng.gen_range(0..n)]
}

fn wall_disturbs_path(board: &mut Board, mv: Move, target: Player, bfs: &mut BfsScratch) -> bool {
    let before = bfs.shortest_distance(board, target).unwrap_or(255);
    let undo = board.make_move(mv);
    let after = bfs.shortest_distance(board, target).unwrap_or(255);
    board.unmake_move(undo);
    after > before
}

fn collect_shortest_pawn_moves(board: &mut Board, out: &mut [Move], bfs: &mut BfsScratch) -> usize {
    let stm = board.side();
    let base = bfs.shortest_distance(board, stm).unwrap_or(255);
    let mut legal = [Move::Pawn { row: 0, col: 0 }; MAX_LEGAL_MOVES];
    let n = generate_legal_moves_slice(board, &mut legal, bfs);

    let mut best = base;
    let mut out_n = 0usize;
    for i in 0..n {
        let mv = legal[i];
        let Move::Pawn { .. } = mv else { continue };
        if allows_opponent_double_advance(board, mv, bfs) {
            continue;
        }
        let undo = board.make_move(mv);
        let d = bfs.shortest_distance(board, stm).unwrap_or(255);
        board.unmake_move(undo);

        if d <= best {
            if d < best {
                best = d;
                out_n = 0;
            }
            out[out_n] = mv;
            out_n += 1;
        }
    }
    out_n
}

fn allows_opponent_double_advance(board: &mut Board, candidate: Move, bfs: &mut BfsScratch) -> bool {
    let opp = board.side().opposite();
    let opp_before = bfs.shortest_distance(board, opp).unwrap_or(255);
    let undo_candidate = board.make_move(candidate);

    let mut legal = [Move::Pawn { row: 0, col: 0 }; MAX_LEGAL_MOVES];
    let n = generate_legal_moves_slice(board, &mut legal, bfs);
    let mut gives_jump = false;
    for &reply in &legal[..n] {
        let Move::Pawn { .. } = reply else {
            continue;
        };
        let undo_reply = board.make_move(reply);
        let opp_after = bfs.shortest_distance(board, opp).unwrap_or(255);
        board.unmake_move(undo_reply);
        if opp_before.saturating_sub(opp_after) >= 2 {
            gives_jump = true;
            break;
        }
    }

    board.unmake_move(undo_candidate);
    gives_jump
}

fn wall_adds_detour(board: &mut Board, mv: Move, target: Player, bfs: &mut BfsScratch) -> u8 {
    let before = bfs.shortest_distance(board, target).unwrap_or(255);
    let undo = board.make_move(mv);
    let after = bfs.shortest_distance(board, target).unwrap_or(255);
    board.unmake_move(undo);
    after.saturating_sub(before)
}

fn expansion_moves_fixed(board: &mut Board, buf: &mut [Move], bfs: &mut BfsScratch) -> usize {
    let mut scratch = [Move::Pawn { row: 0, col: 0 }; MAX_LEGAL_MOVES];
    let full = generate_legal_moves_slice(board, &mut scratch, bfs);
    let stm = board.side();
    let opp = stm.opposite();
    let opp_no_walls = board.walls_remaining[opp as usize] == 0;
    let self_has_walls = board.walls_remaining[stm as usize] > 0;
    let mut n = 0usize;

    if opp_no_walls {
        n += collect_shortest_pawn_moves(board, &mut buf[n..], bfs);

        if self_has_walls {
            for i in 0..full {
                let mv = scratch[i];
                if let Move::Wall { .. } = mv {
                    if wall_disturbs_path(board, mv, opp, bfs) {
                        buf[n] = mv;
                        n += 1;
                    }
                }
            }
        }

        if n > 0 {
            return n;
        }
        buf[..full].copy_from_slice(&scratch[..full]);
        return full;
    }

    let our_dist = bfs.shortest_distance(board, stm).unwrap_or(255);
    let opp_dist = bfs.shortest_distance(board, opp).unwrap_or(255);
    // When we're behind or tied in the race, only include walls that add ≥2 steps
    // to the opponent. Sprinting is almost always better than passive walls.
    let race_deficit = our_dist.saturating_sub(opp_dist);
    let min_wall_detour: u8 = if race_deficit >= 2 { 2 } else { 1 };

    for i in 0..full {
        let mv = scratch[i];
        match mv {
            Move::Pawn { .. } => {
                buf[n] = mv;
                n += 1;
            }
            Move::Wall { .. } => {
                if !self_has_walls {
                    continue;
                }
                if wall_adds_detour(board, mv, opp, bfs) >= min_wall_detour {
                    buf[n] = mv;
                    n += 1;
                }
            }
        }
    }
    if n == 0 {
        buf[..full].copy_from_slice(&scratch[..full]);
        return full;
    }
    n
}

/// Gorisanson-style rollout: caches the full BFS next-step array per player and
/// only recomputes when a wall is placed (invalidating paths). Pawn moves just
/// index into the cached array — O(1) per step instead of O(BFS) per step.
///
/// When `cat_guidance` is `Some`, the 20 % random-move fraction is steered
/// toward tactically hot moves via `pick_hot_rollout_move`.
fn rollout_in_place(
    board: &mut Board,
    bfs: &mut BfsScratch,
    rng: &mut impl Rng,
    rollout_undo: &mut Vec<crate::board::Undo>,
    next_p1: &mut [u8; 81],
    next_p2: &mut [u8; 81],
    legal: &mut [Move; MAX_LEGAL_MOVES],
    cat_guidance: Option<&CorridorAttention>,
) -> Player {
    rollout_undo.clear();
    let mut p1_valid = false;
    let mut p2_valid = false;

    let mut steps = 0u32;
    let winner = loop {
        if let Some(w) = board.is_terminal() {
            break w;
        }
        if steps >= 200 {
            break board.side().opposite();
        }
        steps += 1;
        let stm = board.side();

        // Refresh cached paths when stale.
        if !p1_valid {
            bfs.fill_next_toward_goal(board, Player::One, next_p1);
            p1_valid = true;
        }
        if !p2_valid {
            bfs.fill_next_toward_goal(board, Player::Two, next_p2);
            p2_valid = true;
        }

        let next_arr = if stm == Player::One {
            &*next_p1
        } else {
            &*next_p2
        };
        let (pr, pc) = board.pawn(stm);
        let sq = square_index(pr, pc);
        let next_sq = next_arr[sq as usize];

        if rng.gen_range(0..10) < 8 && next_sq != u8::MAX {
            // Advance one step along shortest path — no BFS needed.
            let (nr, nc) = crate::grid::unpack_square(next_sq);
            let mv = Move::Pawn { row: nr, col: nc };
            rollout_undo.push(board.make_move(mv));
            // Pawn move doesn't invalidate wall-based paths.
        } else {
            // With 20 % probability pick a (CAT-biased) legal move.
            let n = generate_legal_moves_slice(board, legal, bfs);
            if n == 0 {
                break board.side().opposite();
            }
            let mv = if let Some(cat) = cat_guidance {
                pick_hot_rollout_move(legal, n, cat, rng)
            } else {
                legal[rng.gen_range(0..n)]
            };
            rollout_undo.push(board.make_move(mv));
            // Invalidate both caches if a wall was placed.
            if matches!(mv, Move::Wall { .. }) {
                p1_valid = false;
                p2_valid = false;
            }
        }
    };
    undo_all(board, rollout_undo);
    winner
}

fn undo_all(board: &mut Board, undo: &mut Vec<crate::board::Undo>) {
    while let Some(u) = undo.pop() {
        board.unmake_move(u);
    }
}

fn find_immediate_win(moves: &[Move], n: usize, stm: Player) -> Option<Move> {
    for i in 0..n {
        if let Move::Pawn { row, col: _ } = moves[i] {
            if is_goal(stm, row) {
                return Some(moves[i]);
            }
        }
    }
    None
}

fn should_stop_mcts_early(tree: &MctsTree, our_dist: u8, sims: u64) -> bool {
    if our_dist == 1 && sims >= 32 {
        return true;
    }
    if tree.nodes[0].children.is_empty() || sims < 100 {
        return false;
    }
    let best = tree.best_child_sims(0);
    let best_visits = tree.nodes[best].visits;
    if best_visits < 100 {
        return false;
    }
    let wr = tree.nodes[best].wins / best_visits as f64;
    if best_visits >= 300 && wr >= 0.98 {
        return true;
    }
    if best_visits >= 150 && wr >= 0.99 {
        return true;
    }
    if tree.nodes[0].children.len() == 1 && best_visits >= 500 && wr >= 0.95 {
        return true;
    }
    false
}

fn log_progress(config: &MctsConfig, sims: u64, elapsed_ms: u64, win_rate: f64) {
    if !config.log {
        return;
    }
    eprintln!(
        "info progress sims {} elapsed_ms {} winrate {:.3}",
        sims, elapsed_ms, win_rate
    );
}

/// Gorisanson-style MCTS — UCT selection, path-aware rollouts, max-sims best move.
///
/// If `config.use_cat_guidance` is true the search runs in "bridge" mode:
/// corridor attention is computed once at the root position, expansion picks
/// unvisited children weighted by corridor heat (hot corridors explored first),
/// and the random 20 % rollout fraction prefers tactically hot moves.
pub fn search_mcts(board: &mut Board, config: MctsConfig) -> Option<MctsReport> {
    let mut bfs = BfsScratch::new();

    let mut legal = [Move::Pawn { row: 0, col: 0 }; MAX_LEGAL_MOVES];
    let legal_n = generate_legal_moves_slice(board, &mut legal, &mut bfs);
    if legal_n == 0 {
        return None;
    }
    if legal_n == 1 {
        return Some(MctsReport {
            best_move: legal[0],
            simulations: 0,
            elapsed_ms: 0,
            stopped_by: "trivial",
            white_dist: bfs.shortest_distance(board, Player::One).unwrap_or(255),
            black_dist: bfs.shortest_distance(board, Player::Two).unwrap_or(255),
            root_win_rate: 1.0,
        });
    }

    let stm = board.side();
    if let Some(win_mv) = find_immediate_win(&legal, legal_n, stm) {
        return Some(MctsReport {
            best_move: win_mv,
            simulations: 0,
            elapsed_ms: 0,
            stopped_by: "win-in-1",
            white_dist: bfs.shortest_distance(board, Player::One).unwrap_or(255),
            black_dist: bfs.shortest_distance(board, Player::Two).unwrap_or(255),
            root_win_rate: 1.0,
        });
    }

    let book_mv = config.book_hint.map(|h| h.mv);

    // Find the sprint move: the pawn move that best advances us toward the goal.
    // This is given a huge weight in expansion so the tree always explores it first.
    let mut sprint_buf = [Move::Pawn { row: 0, col: 0 }; MAX_LEGAL_MOVES];
    let sprint_n = collect_shortest_pawn_moves(board, &mut sprint_buf, &mut bfs);
    let sprint_mv: Option<Move> = if sprint_n > 0 { Some(sprint_buf[0]) } else { None };

    // Build corridor attention once at the root when bridge mode is requested.
    // Using root attention throughout is a slight approximation (walls change it),
    // but in the early game with few walls the drift is minimal.
    let root_cat: Option<CorridorAttention> = if config.use_cat_guidance {
        Some(bfs.build_corridor_attention(board))
    } else {
        None
    };

    let white_dist = bfs.shortest_distance(board, Player::One).unwrap_or(255);
    let black_dist = bfs.shortest_distance(board, Player::Two).unwrap_or(255);
    let our_dist = bfs.shortest_distance(board, stm).unwrap_or(255);
    let started = Instant::now();
    let deadline = started + std::time::Duration::from_millis(config.time_ms);
    let mut tree = MctsTree::new(config.uct);
    let mut rng = rand::thread_rng();
    let mut sims = 0u64;
    let mut last_log = Instant::now();
    let batch = 32usize;
    // Reuse undo-path buffer across all simulations — no heap alloc per sim.
    let mut undo_path: Vec<crate::board::Undo> = Vec::with_capacity(64);
    // Reuse rollout undo buffer too; avoids clone per simulation.
    let mut rollout_undo: Vec<crate::board::Undo> = Vec::with_capacity(64);
    // Reuse rollout scratch buffers across simulations.
    let mut rollout_next_p1 = [u8::MAX; 81];
    let mut rollout_next_p2 = [u8::MAX; 81];
    let mut rollout_legal = [Move::Pawn { row: 0, col: 0 }; MAX_LEGAL_MOVES];
    // Reuse expansion move buffer.
    let mut expansion_buf = [Move::Pawn { row: 0, col: 0 }; MAX_LEGAL_MOVES];

    while sims < config.max_simulations && Instant::now() < deadline {
        for i in 0..batch {
            if sims >= config.max_simulations {
                break;
            }
            if (i & 7) == 0 && Instant::now() >= deadline {
                break;
            }

            undo_path.clear();
            let mut node = 0usize;
            let mut leaf_mover = board.side();

            loop {
                if board.is_terminal().is_some() {
                    tree.nodes[node].terminal = true;
                    break;
                }

                if tree.nodes[node].children.is_empty() {
                    let n = expansion_moves_fixed(board, &mut expansion_buf, &mut bfs);
                    if n == 0 {
                        tree.nodes[node].terminal = true;
                        break;
                    }
                    for k in 0..n {
                        tree.add_child(node, expansion_buf[k]);
                    }
                    // First child for a newly expanded node: random (standard)
                    // or CAT-weighted (bridge mode).
                    let child = if let Some(cat) = root_cat.as_ref() {
                        // pick_cat returns None only if all children are visited,
                        // which can't happen here (we just added unvisited children).
                        tree.cat_weighted_unvisited_child(node, cat, book_mv, sprint_mv, &mut rng)
                            .unwrap_or_else(|| tree.nodes[node].children[rng.gen_range(0..n)])
                    } else if let Some(sm) = sprint_mv.or(book_mv) {
                        let idx = expansion_buf[..n]
                            .iter()
                            .position(|&m| m == sm)
                            .unwrap_or_else(|| rng.gen_range(0..n));
                        tree.nodes[node].children[idx]
                    } else {
                        tree.nodes[node].children[rng.gen_range(0..n)]
                    };
                    let mv = tree.nodes[child].mv;
                    let mover = board.side();
                    undo_path.push(board.make_move(mv));
                    leaf_mover = mover;
                    node = child;
                    break;
                }

                // Existing node: prefer a CAT-weighted unvisited child in bridge mode.
                let unvisited = if let Some(cat) = root_cat.as_ref() {
                    tree.cat_weighted_unvisited_child(node, cat, book_mv, sprint_mv, &mut rng)
                } else {
                    tree.random_unvisited_child(node, &mut rng)
                };
                if let Some(child) = unvisited {
                    let mv = tree.nodes[child].mv;
                    let mover = board.side();
                    undo_path.push(board.make_move(mv));
                    leaf_mover = mover;
                    node = child;
                    break;
                }

                let child = tree.best_child_uct(node);
                let mv = tree.nodes[child].mv;
                let mover = board.side();
                undo_path.push(board.make_move(mv));
                leaf_mover = mover;
                node = child;
            }

            let winner = if let Some(w) = board.is_terminal() {
                w
            } else {
                rollout_in_place(
                    board,
                    &mut bfs,
                    &mut rng,
                    &mut rollout_undo,
                    &mut rollout_next_p1,
                    &mut rollout_next_p2,
                    &mut rollout_legal,
                    root_cat.as_ref(),
                )
            };

            tree.backprop(node, winner, leaf_mover);
            sims += 1;
            undo_all(board, &mut undo_path);
        }

        if should_stop_mcts_early(&tree, our_dist, sims) {
            break;
        }

        if last_log.elapsed().as_millis() >= 400 {
            let wr = if tree.nodes[0].visits > 0 {
                tree.nodes[0].wins / tree.nodes[0].visits as f64
            } else {
                0.5
            };
            log_progress(&config, sims, started.elapsed().as_millis() as u64, wr);
            last_log = Instant::now();
        }
    }

    let best_child = if tree.nodes[0].children.is_empty() {
        0
    } else {
        tree.best_child_sims(0)
    };
    let best_move = if best_child == 0 {
        legal[0]
    } else {
        tree.nodes[best_child].mv
    };

    let elapsed_ms = started.elapsed().as_millis() as u64;
    let stopped_by = if should_stop_mcts_early(&tree, our_dist, sims) {
        "forced"
    } else if config.use_cat_guidance {
        if sims >= config.max_simulations {
            "bridge-visits"
        } else {
            "bridge"
        }
    } else if sims >= config.max_simulations {
        "visits"
    } else {
        "time"
    };
    let root_win_rate = if tree.nodes[0].visits > 0 {
        tree.nodes[0].wins / tree.nodes[0].visits as f64
    } else {
        0.5
    };

    eprintln!(
        "info json {{\"stoppedBy\":\"{}\",\"simulations\":{},\"elapsedMs\":{},\"whiteDist\":{},\"blackDist\":{},\"rootWinRate\":{:.4}}}",
        stopped_by, sims, elapsed_ms, white_dist, black_dist, root_win_rate
    );

    Some(MctsReport {
        best_move,
        simulations: sims,
        elapsed_ms,
        stopped_by,
        white_dist,
        black_dist,
        root_win_rate,
    })
}

pub fn genmove_algebraic(board: &mut Board, config: MctsConfig) -> Option<String> {
    search_mcts(board, config).map(|r| format_move(r.best_move))
}
