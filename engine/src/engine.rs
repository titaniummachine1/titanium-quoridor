//! `Engine` — coordinates perft today, αβ + Lazy SMP tomorrow.

use crate::board::Board;
use crate::context::{EngineLimits, SharedState, ThreadBenchResult, WorkerContext};
use crate::perft::{perft_fast_ctx, perft_iterative as perft_iterative_inner, perft_parallel_root};

/// Titanium entry point — perft now, search later on the same layout.
pub struct Engine {
    pub shared: SharedState,
    pub limits: EngineLimits,
    pool: Option<rayon::ThreadPool>,
}

impl Default for Engine {
    fn default() -> Self {
        Self::new()
    }
}

impl Engine {
    pub fn new() -> Self {
        Self {
            shared: SharedState::new(),
            limits: EngineLimits::default(),
            pool: None,
        }
    }

    pub fn with_threads(threads: usize) -> Self {
        let limits = EngineLimits::with_threads(threads);
        let pool = (limits.threads > 1).then(|| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(limits.threads)
                .build()
                .expect("rayon thread pool")
        });
        Self {
            shared: SharedState::new(),
            limits,
            pool,
        }
    }

    pub fn worker(&self) -> WorkerContext {
        WorkerContext::new()
    }

    /// Node count — root-parallel perft when `limits.threads > 1`.
    pub fn perft(&mut self, board: &Board, depth: u32) -> u64 {
        let mut copy = board.clone();
        self.perft_on_board(&mut copy, depth)
    }

    pub fn perft_on_board(&mut self, board: &mut Board, depth: u32) -> u64 {
        if self.limits.threads <= 1 {
            self.shared.clear_tt();
            let mut worker = self.worker();
            perft_fast_ctx(board, depth, Some(&mut self.shared), &mut worker)
        } else {
            let pool = self
                .pool
                .as_ref()
                .expect("parallel engine must have thread pool");
            perft_parallel_root(board, depth, pool)
        }
    }

    pub fn perft_iterative(&mut self, board: &mut Board, max_depth: u32) -> Vec<(u32, u64)> {
        if self.limits.threads <= 1 {
            perft_iterative_inner(board, max_depth, &mut self.shared)
        } else {
            let mut out = Vec::with_capacity(max_depth as usize + 1);
            for depth in 0..=max_depth {
                let nodes = if depth == 0 {
                    1
                } else {
                    self.perft_on_board(board, depth)
                };
                out.push((depth, nodes));
            }
            out
        }
    }

    /// Perft without TT — matches parallel subtree work for apples-to-apples benching.
    pub fn perft_no_tt(&mut self, board: &mut Board, depth: u32) -> u64 {
        let mut worker = self.worker();
        perft_fast_ctx(board, depth, None, &mut worker)
    }

    /// Run perft at `depth` with 1 thread vs `parallel_threads` — same nodes, compare wall time.
    /// Both paths disable TT so parallel workers are not penalized vs a cached single thread.
    pub fn bench_threads(board: &Board, depth: u32, parallel_threads: usize) -> ThreadBenchResult {
        let parallel_threads = parallel_threads.max(2);

        let mut copy = board.clone();
        let mut one = Engine::new();
        let start = std::time::Instant::now();
        let nodes = one.perft_no_tt(&mut copy, depth);
        let threads_one_secs = start.elapsed().as_secs_f64();

        let mut parallel = Engine::with_threads(parallel_threads);
        let start = std::time::Instant::now();
        let nodes_parallel = parallel.perft_on_board(&mut copy, depth);
        let threads_n_secs = start.elapsed().as_secs_f64();

        debug_assert_eq!(nodes, nodes_parallel);

        ThreadBenchResult {
            depth,
            nodes,
            threads_one_secs,
            threads_n_secs,
            threads_n: parallel_threads,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::perft::PERFT3_STARTPOS;

    #[test]
    fn default_limits_single_thread() {
        let engine = Engine::new();
        assert_eq!(engine.limits.threads, 1);
    }

    #[test]
    fn parallel_matches_single_depth3() {
        let board = Board::new();
        let mut single = Engine::new();
        let n1 = single.perft(&board, 3);

        let mut parallel = Engine::with_threads(4);
        let n4 = parallel.perft(&board, 3);

        assert_eq!(n1, PERFT3_STARTPOS);
        assert_eq!(n4, PERFT3_STARTPOS);
    }

    #[test]
    fn bench_threads_same_nodes() {
        let board = Board::new();
        let result = Engine::bench_threads(&board, 3, 4);
        assert_eq!(result.nodes, PERFT3_STARTPOS);
        assert_eq!(result.threads_n, 4);
    }
}
