//! Gorisanson-style MCTS on the fast Rust board (make/unmake + BFS rollouts).

use std::time::Instant;

use rand::Rng;

use crate::board::{Board, Move, Player};
use crate::moves::{generate_legal_moves_slice, MAX_LEGAL_MOVES};
use crate::path::BfsScratch;
use crate::perft::format_move;

pub const DEFAULT_UCT: f64 = 0.2;
pub const DEFAULT_TIME_MS: u64 = 10_000;
pub const DEFAULT_MAX_SIMULATIONS: u64 = 2_000_000_000;

#[derive(Debug, Clone, Copy)]
pub struct MctsConfig {
    pub time_ms: u64,
    pub max_simulations: u64,
    pub uct: f64,
    pub log: bool,
}

impl Default for MctsConfig {
    fn default() -> Self {
        Self {
            time_ms: DEFAULT_TIME_MS,
            max_simulations: DEFAULT_MAX_SIMULATIONS,
            uct: DEFAULT_UCT,
            log: false,
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

    fn uct_score(&self, node: usize, parent_visits: u64) -> f64 {
        let n = &self.nodes[node];
        if n.visits == 0 {
            return f64::INFINITY;
        }
        let exploit = n.wins / n.visits as f64;
        let explore = (self.uct * (parent_visits as f64).ln() / n.visits as f64).sqrt();
        exploit + explore
    }

    fn best_child_uct(&self, node: usize) -> usize {
        let parent_visits = self.nodes[node].visits.max(1);
        let mut best = self.nodes[node].children[0];
        let mut best_score = self.uct_score(best, parent_visits);
        for &child in &self.nodes[node].children[1..] {
            let score = self.uct_score(child, parent_visits);
            if score > best_score {
                best_score = score;
                best = child;
            }
        }
        best
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

fn expansion_moves_fixed(board: &mut Board, buf: &mut [Move], bfs: &mut BfsScratch) -> usize {
    let mut scratch = [Move::Pawn { row: 0, col: 0 }; MAX_LEGAL_MOVES];
    let full = generate_legal_moves_slice(board, &mut scratch, bfs);
    let stm = board.side();
    let opp = stm.opposite();
    let opp_no_walls = board.walls_remaining[opp as usize] == 0;
    let self_has_walls = board.walls_remaining[stm as usize] > 0;
    let mut n = 0usize;

    if opp_no_walls {
        // Match JS behavior: when opponent has no walls, race with shortest pawn
        // moves; only keep walls that disturb opponent's shortest path.
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
        // Safety fallback: never return empty if legal moves exist.
        buf[..full].copy_from_slice(&scratch[..full]);
        return full;
    }

    for i in 0..full {
        let mv = scratch[i];
        match mv {
            Move::Pawn { .. } => {
                buf[n] = mv;
                n += 1;
            }
            Move::Wall { .. } => {
                buf[n] = mv;
                n += 1;
            }
        }
    }
    if n == 0 {
        buf[..full].copy_from_slice(&scratch[..full]);
        return full;
    }
    n
}

fn try_opening_move(board: &mut Board, bfs: &mut BfsScratch) -> Option<Move> {
    if board.move_number > 2 {
        return None;
    }
    let stm = board.side();
    let (row, col) = board.pawn(stm);
    if col != 4 {
        return None;
    }
    let goal = if stm == Player::One { 8 } else { 0 };
    if row.abs_diff(goal) <= 1 {
        return None;
    }
    let mut buf = [Move::Pawn { row: 0, col: 0 }; MAX_LEGAL_MOVES];
    let n = generate_legal_moves_slice(board, &mut buf, bfs);
    for i in 0..n {
        if let Move::Pawn { row: nr, col: nc } = buf[i] {
            if nc == col && nr.abs_diff(goal) < row.abs_diff(goal) {
                return Some(buf[i]);
            }
        }
    }
    None
}

/// Gorisanson-style rollout: caches the full BFS next-step array per player and
/// only recomputes when a wall is placed (invalidating paths). Pawn moves just
/// index into the cached array — O(1) per step instead of O(BFS) per step.
fn rollout(board: &mut Board, bfs: &mut BfsScratch, rng: &mut impl Rng) -> Player {
    // Pre-compute next-step arrays for both players.
    let mut next_p1 = [u8::MAX; 81];
    let mut next_p2 = [u8::MAX; 81];
    let mut p1_valid = false;
    let mut p2_valid = false;

    let mut steps = 0u32;
    while board.is_terminal().is_none() && steps < 200 {
        steps += 1;
        let stm = board.side();

        // Refresh cached paths when stale.
        if !p1_valid {
            bfs.fill_next_toward_goal(board, Player::One, &mut next_p1);
            p1_valid = true;
        }
        if !p2_valid {
            bfs.fill_next_toward_goal(board, Player::Two, &mut next_p2);
            p2_valid = true;
        }

        let next_arr = if stm == Player::One { &next_p1 } else { &next_p2 };
        let (pr, pc) = board.pawn(stm);
        let sq = crate::grid::square_index(pr, pc);
        let next_sq = next_arr[sq as usize];

        if rng.gen::<f64>() < 0.7 && next_sq != u8::MAX {
            // Advance one step along shortest path — no BFS needed.
            let (nr, nc) = crate::grid::unpack_square(next_sq);
            let mv = Move::Pawn { row: nr, col: nc };
            let _ = board.make_move(mv);
            // Pawn move doesn't invalidate wall-based paths.
        } else {
            // With 30% probability pick a random legal move (may be a wall).
            let mut legal = [Move::Pawn { row: 0, col: 0 }; MAX_LEGAL_MOVES];
            let n = generate_legal_moves_slice(board, &mut legal, bfs);
            if n == 0 {
                break;
            }
            let mv = legal[rng.gen_range(0..n)];
            let _ = board.make_move(mv);
            // Invalidate both caches if a wall was placed.
            if matches!(mv, Move::Wall { .. }) {
                p1_valid = false;
                p2_valid = false;
            }
        }
    }
    board.is_terminal().unwrap_or(board.side().opposite())
}

fn undo_all(board: &mut Board, undo: &mut Vec<crate::board::Undo>) {
    while let Some(u) = undo.pop() {
        board.unmake_move(u);
    }
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
pub fn search_mcts(board: &mut Board, config: MctsConfig) -> Option<MctsReport> {
    let mut bfs = BfsScratch::new();

    if let Some(mv) = try_opening_move(board, &mut bfs) {
        let white_dist = bfs.shortest_distance(board, Player::One).unwrap_or(255);
        let black_dist = bfs.shortest_distance(board, Player::Two).unwrap_or(255);
        eprintln!(
            "info json {{\"stoppedBy\":\"opening\",\"simulations\":0,\"elapsedMs\":0,\"whiteDist\":{},\"blackDist\":{},\"rootWinRate\":1.0}}",
            white_dist, black_dist
        );
        return Some(MctsReport {
            best_move: mv,
            simulations: 0,
            elapsed_ms: 0,
            stopped_by: "opening",
            white_dist,
            black_dist,
            root_win_rate: 1.0,
        });
    }

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

    let white_dist = bfs.shortest_distance(board, Player::One).unwrap_or(255);
    let black_dist = bfs.shortest_distance(board, Player::Two).unwrap_or(255);
    let started = Instant::now();
    let deadline = started + std::time::Duration::from_millis(config.time_ms);
    let mut tree = MctsTree::new(config.uct);
    let mut rng = rand::thread_rng();
    let mut sims = 0u64;
    let mut last_log = Instant::now();
    let batch = 32usize;
    // Reuse undo-path buffer across all simulations — no heap alloc per sim.
    let mut undo_path: Vec<crate::board::Undo> = Vec::with_capacity(64);

    while sims < config.max_simulations && Instant::now() < deadline {
        for _ in 0..batch {
            if sims >= config.max_simulations || Instant::now() >= deadline {
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
                    let mut buf = [Move::Pawn { row: 0, col: 0 }; MAX_LEGAL_MOVES];
                    let n = expansion_moves_fixed(board, &mut buf, &mut bfs);
                    if n == 0 {
                        tree.nodes[node].terminal = true;
                        break;
                    }
                    for i in 0..n {
                        tree.add_child(node, buf[i]);
                    }
                    let child = tree.nodes[node].children[rng.gen_range(0..n)];
                    let mv = tree.nodes[child].mv;
                    let mover = board.side();
                    undo_path.push(board.make_move(mv));
                    leaf_mover = mover;
                    node = child;
                    break;
                }

                // Count unvisited children without allocating a Vec.
                let unvisited_count = tree.nodes[node]
                    .children
                    .iter()
                    .filter(|&&c| tree.nodes[c].visits == 0)
                    .count();
                if unvisited_count > 0 {
                    // Pick the k-th unvisited child in one extra pass.
                    let k = rng.gen_range(0..unvisited_count);
                    let child = tree.nodes[node]
                        .children
                        .iter()
                        .copied()
                        .filter(|&c| tree.nodes[c].visits == 0)
                        .nth(k)
                        .unwrap();
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
                let mut sim = board.clone();
                rollout(&mut sim, &mut bfs, &mut rng)
            };

            tree.backprop(node, winner, leaf_mover);
            sims += 1;
            undo_all(board, &mut undo_path);
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
    let stopped_by = if sims >= config.max_simulations {
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
