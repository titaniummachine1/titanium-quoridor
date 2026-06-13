//! Perft (divide) — correctness + fast make/unmake + TT + iterative deepening driver.
//!
//! **Standard correctness depth:** 3 from startpos → **2_062_264** nodes.

/// Startpos perft(3) — agreed by scraped JS, gorisanson, and Titanium.
pub const PERFT3_STARTPOS: u64 = 2_062_264;

/// Startpos perft(4) — Ishtar / Canta oracle (2025).
pub const PERFT4_STARTPOS: u64 = 247_569_030;

/// Startpos perft(5) — cross-verified by Titanium and the reference engine.
/// Titanium reaches this in sub-12s (no TT, bulk leaf count), a public record.
pub const PERFT5_STARTPOS: u64 = 28_837_934_502;

/// Startpos perft(6) — cross-verified (a multi-hour full enumeration).
pub const PERFT6_STARTPOS: u64 = 3_257_436_276_501;

/// Max wall time for perft(4) in the ignored regression test.
pub const PERFT4_TEST_TIMEOUT_SECS: u64 = 10;

/// Per-depth timeout floor — each depth runs in its own thread; budget is `2×` previous depth.
pub const PERFT_DEPTH_TIMEOUT_FLOOR_MS: u64 = 250;

use crate::core::board::{Board, Move};
use crate::movegen::{
    generate_legal_moves_into, generate_legal_moves_slice, generate_legal_moves_slice_mode,
    generate_pawn_moves_slice_mode, PawnGenMode, MAX_LEGAL_MOVES,
};
use crate::path::BfsScratch;
use crate::search::context::{SharedState, WorkerContext};
use std::collections::BTreeMap;

/// Back-compat name — prefer [`WorkerContext`] + [`SharedState`] or [`crate::search::context::Engine`].
pub type PerftContext = WorkerContext;

pub fn perft_fast_ctx(
    board: &mut Board,
    depth: u32,
    mut shared: Option<&mut SharedState>,
    worker: &mut WorkerContext,
) -> u64 {
    perft_fast_mode_ctx(board, depth, PawnGenMode::default(), shared, worker)
}

/// TT perft with selectable pawn generator (production path uses TT + V11 walls).
pub fn perft_fast_mode_ctx(
    board: &mut Board,
    depth: u32,
    mode: PawnGenMode,
    mut shared: Option<&mut SharedState>,
    worker: &mut WorkerContext,
) -> u64 {
    if depth == 0 {
        return 1;
    }

    if let Some(shared) = shared.as_mut() {
        if let Some(nodes) = shared.tt.probe(board.hash, depth as u8) {
            return nodes;
        }
    }

    let mut move_buf = [Move::Pawn { row: 0, col: 0 }; MAX_LEGAL_MOVES];
    let move_count = generate_legal_moves_slice_mode(board, &mut move_buf, &mut worker.bfs, mode);

    // Bulk count: every depth-0 child is one node — no make/unmake needed.
    let nodes = if depth == 1 {
        move_count as u64
    } else {
        let mut nodes = 0u64;
        for i in 0..move_count {
            let mv = move_buf[i];
            let undo = board.make_move(mv);
            nodes += perft_fast_mode_ctx(board, depth - 1, mode, shared.as_deref_mut(), worker);
            board.unmake_move(undo);
        }
        nodes
    };

    if let Some(shared) = shared {
        shared.tt.store(board.hash, depth as u8, nodes);
    }
    nodes
}

/// Fast perft with TT and a pawn mode (use this to time O1 at depth 4 — not `perft_no_tt_mode`).
pub fn perft_fast_mode(board: &mut Board, depth: u32, mode: PawnGenMode) -> u64 {
    let mut shared = SharedState::new();
    let mut worker = WorkerContext::new();
    perft_fast_mode_ctx(board, depth, mode, Some(&mut shared), &mut worker)
}

/// Perft without TT — for timing pawn-generation variants fairly.
pub fn perft_no_tt_mode(board: &mut Board, depth: u32, mode: PawnGenMode) -> u64 {
    let mut scratch = BfsScratch::new();
    perft_no_tt_mode_ctx(board, depth, mode, &mut scratch)
}

pub fn perft_no_tt_mode_ctx(
    board: &mut Board,
    depth: u32,
    mode: PawnGenMode,
    scratch: &mut BfsScratch,
) -> u64 {
    if depth == 0 {
        return 1;
    }

    let mut move_buf = [Move::Pawn { row: 0, col: 0 }; MAX_LEGAL_MOVES];
    let move_count = generate_legal_moves_slice_mode(board, &mut move_buf, scratch, mode);
    if depth == 1 {
        return move_count as u64;
    }
    let mut nodes = 0u64;

    for i in 0..move_count {
        let mv = move_buf[i];
        let undo = board.make_move(mv);
        nodes += perft_no_tt_mode_ctx(board, depth - 1, mode, scratch);
        board.unmake_move(undo);
    }
    nodes
}

/// Pawn-only perft — no walls in tree; isolates pawn-generation cost (no TT, no wall BFS).
pub fn perft_pawn_only_mode(board: &mut Board, depth: u32, mode: PawnGenMode) -> u64 {
    let mut scratch = BfsScratch::new();
    perft_pawn_only_mode_ctx(board, depth, mode, &mut scratch)
}

pub fn perft_pawn_only_mode_ctx(
    board: &mut Board,
    depth: u32,
    mode: PawnGenMode,
    scratch: &mut BfsScratch,
) -> u64 {
    if depth == 0 {
        return 1;
    }

    let mut move_buf = [Move::Pawn { row: 0, col: 0 }; 8];
    let move_count = generate_pawn_moves_slice_mode(board, &mut move_buf, scratch, mode);
    if depth == 1 {
        return move_count as u64;
    }
    let mut nodes = 0u64;
    for i in 0..move_count {
        let mv = move_buf[i];
        let undo = board.make_move(mv);
        nodes += perft_pawn_only_mode_ctx(board, depth - 1, mode, scratch);
        board.unmake_move(undo);
    }
    nodes
}

/// Fast perft — single-threaded via a fresh [`SharedState`].
pub fn perft_fast(board: &mut Board, depth: u32) -> u64 {
    let mut shared = SharedState::new();
    let mut worker = WorkerContext::new();
    perft_fast_ctx(board, depth, Some(&mut shared), &mut worker)
}

/// Root-split parallel perft — experimental bench path when `threads > 1`.
/// Each root move runs in its own subtree with private TT (embarrassingly parallel).
#[cfg(feature = "parallel")]
pub fn perft_parallel_root(board: &Board, depth: u32, pool: &rayon::ThreadPool) -> u64 {
    if depth == 0 {
        return 1;
    }

    let mut probe = board.clone();
    let mut worker = WorkerContext::new();
    let mut move_buf = [Move::Pawn { row: 0, col: 0 }; MAX_LEGAL_MOVES];
    let move_count = generate_legal_moves_slice(&mut probe, &mut move_buf, &mut worker.bfs);
    let moves = &move_buf[..move_count];

    pool.install(|| {
        use rayon::prelude::*;
        moves
            .par_iter()
            .map(|&mv| {
                let mut child = board.clone();
                let mut worker = WorkerContext::new();
                let _undo = child.make_move(mv);
                // No TT per worker — avoids 131× heap alloc; subtrees are independent.
                perft_fast_ctx(&mut child, depth - 1, None, &mut worker)
            })
            .sum()
    })
}

pub fn perft_iterative(
    board: &mut Board,
    max_depth: u32,
    shared: &mut SharedState,
) -> Vec<(u32, u64)> {
    let mut out = Vec::with_capacity(max_depth as usize + 1);
    let mut worker = WorkerContext::new();
    for depth in 0..=max_depth {
        shared.clear_tt();
        let nodes = if depth == 0 {
            1
        } else {
            perft_fast_ctx(board, depth, Some(shared), &mut worker)
        };
        out.push((depth, nodes));
    }
    out
}

/// Legacy naive perft (clone) — kept for differential testing.
pub fn perft_naive(board: &Board, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }
    let mut probe = board.clone();
    let mut moves = Vec::new();
    let mut scratch = BfsScratch::new();
    generate_legal_moves_into(&mut probe, &mut moves, &mut scratch);
    let mut nodes = 0u64;
    for &mv in &moves {
        let mut next = board.clone();
        next.apply_move(mv);
        nodes += perft_naive(&next, depth - 1);
    }
    nodes
}

/// Default perft entry — single-thread [`crate::search::runtime::Engine`].
pub fn perft(board: &Board, depth: u32) -> u64 {
    crate::search::runtime::Engine::new().perft(board, depth)
}

pub fn perft_divide(board: &Board, depth: u32) -> (u64, BTreeMap<String, u64>) {
    let mut lines = BTreeMap::new();
    let mut moves = Vec::new();
    let mut copy = board.clone();
    let mut scratch = BfsScratch::new();
    generate_legal_moves_into(&mut copy, &mut moves, &mut scratch);
    let mut total = 0u64;

    for &mv in &moves {
        let label = format_move(mv);
        let undo = copy.make_move(mv);
        let nodes = perft(&copy, depth.saturating_sub(1));
        copy.unmake_move(undo);
        lines.insert(label, nodes);
        total += nodes;
    }
    (total, lines)
}

pub fn format_move(mv: Move) -> String {
    match mv {
        Move::Pawn { row, col } => Board::format_square(row, col),
        Move::Wall {
            row,
            col,
            orientation,
        } => {
            let suffix = match orientation {
                crate::core::board::WallOrientation::Horizontal => 'h',
                crate::core::board::WallOrientation::Vertical => 'v',
            };
            format!("{}{}{}", Board::column_char(col), row + 1, suffix)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::movegen::generate_legal_moves;
    use crate::search::runtime::Engine;
    use crate::util::clock::{Duration, Instant};
    use core_affinity::CoreId;
    use std::sync::mpsc;

    struct TimedPerftResult {
        nodes: u64,
        worker_elapsed: Duration,
        wall_elapsed: Duration,
        worker_core: usize,
        timer_core: usize,
    }

    const PERFT_ORACLE: [(u32, u64); 5] = [
        (0, 1),
        (1, 131),
        (2, 16_677),
        (3, PERFT3_STARTPOS),
        (4, PERFT4_STARTPOS),
    ];

    fn oracle_nodes(depth: u32) -> u64 {
        PERFT_ORACLE
            .iter()
            .find(|(d, _)| *d == depth)
            .map(|(_, n)| *n)
            .unwrap_or_else(|| panic!("no oracle for perft depth {depth}"))
    }

    fn core_ids() -> Vec<CoreId> {
        core_affinity::get_core_ids().unwrap_or_else(|| vec![CoreId { id: 0 }])
    }

    /// P-core-ish slot — last logical core is often an E-core on hybrid CPUs.
    fn perft_worker_core() -> CoreId {
        let ids = core_ids();
        if ids.len() >= 4 {
            ids[2]
        } else if ids.len() >= 2 {
            ids[1]
        } else {
            ids[0]
        }
    }

    fn perft_timer_core() -> CoreId {
        let ids = core_ids();
        if ids.len() >= 8 {
            *ids.last().unwrap_or(&ids[0])
        } else if ids.len() > 1 {
            ids[1]
        } else {
            ids[0]
        }
    }

    fn pin_current_to_core(core: CoreId) -> usize {
        let id = core.id;
        if !core_affinity::set_for_current(core) {
            eprintln!("warning: could not pin to core {id}");
        }
        id
    }

    fn run_perft_depth_timed(depth: u32, timeout: Duration) -> Result<TimedPerftResult, ()> {
        let (done_tx, done_rx) = mpsc::channel::<(u64, Duration)>();
        let (watch_tx, watch_rx) = mpsc::sync_channel::<Result<TimedPerftResult, ()>>(1);

        let worker_core = perft_worker_core();
        let timer_core = perft_timer_core();
        let worker_core_id = worker_core.id;
        let timer_core_id = timer_core.id;

        let worker_handle = std::thread::Builder::new()
            .name(format!("perft-d{depth}-worker"))
            .spawn(move || {
                pin_current_to_core(worker_core);
                let board = Board::new();
                let mut fast_board = board.clone();
                let t0 = Instant::now();
                let nodes = perft_fast(&mut fast_board, depth);
                let _ = done_tx.send((nodes, t0.elapsed()));
            })
            .expect("spawn perft worker");

        let _watcher_handle = std::thread::Builder::new()
            .name(format!("perft-d{depth}-timer"))
            .spawn(move || {
                pin_current_to_core(timer_core);
                let wall_start = Instant::now();
                let outcome = match done_rx.recv_timeout(timeout) {
                    Ok((nodes, worker_elapsed)) => Ok(TimedPerftResult {
                        nodes,
                        worker_elapsed,
                        wall_elapsed: wall_start.elapsed(),
                        worker_core: worker_core_id,
                        timer_core: timer_core_id,
                    }),
                    Err(mpsc::RecvTimeoutError::Timeout)
                    | Err(mpsc::RecvTimeoutError::Disconnected) => Err(()),
                };
                let _ = watch_tx.send(outcome);
            })
            .expect("spawn perft timer");

        let result = watch_rx.recv().expect("perft timer thread");

        match &result {
            Ok(_) => {
                worker_handle.join().ok();
            }
            Err(()) => {
                std::mem::forget(worker_handle);
            }
        }

        result
    }

    fn log_partial_perft(results: &[(u32, u64, Duration, Duration)]) {
        eprintln!("--- perft partial results ---");
        for (depth, nodes, worker_ms, wall_ms) in results {
            eprintln!(
                "perft {depth} {nodes} worker_ms={:.1} wall_ms={:.1}",
                worker_ms.as_secs_f64() * 1000.0,
                wall_ms.as_secs_f64() * 1000.0
            );
        }
    }

    fn fail_perft_timeout(
        depth: u32,
        budget: Duration,
        completed: &[(u32, u64, Duration, Duration)],
    ) {
        log_partial_perft(completed);
        eprintln!(
            "perft({depth}) TIMEOUT after {:.0}ms — aborting (worker left detached)",
            budget.as_secs_f64() * 1000.0
        );
        std::process::exit(1);
    }

    fn depth_timeout_budget(depth: u32, prev_elapsed: Duration) -> Duration {
        if depth == 4 {
            return Duration::from_secs(PERFT4_TEST_TIMEOUT_SECS);
        }
        let min = match depth {
            1 => Duration::from_millis(500),
            2 => Duration::from_millis(500),
            3 => Duration::from_secs(2),
            _ => Duration::from_millis(PERFT_DEPTH_TIMEOUT_FLOOR_MS),
        };
        prev_elapsed.saturating_mul(2).max(min)
    }

    #[test]
    fn perft_depth1_start() {
        let board = Board::new();
        assert_eq!(perft(&board, 1), generate_legal_moves(&board).len() as u64);
    }

    #[test]
    fn perft_depth0_is_one() {
        let board = Board::new();
        assert_eq!(perft(&board, 0), 1);
    }

    #[test]
    fn perft_depth2_smoke() {
        let board = Board::new();
        assert_eq!(perft(&board, 2), 16_677);
    }

    #[test]
    fn perft_depth3_matches_js_oracle() {
        let board = Board::new();
        assert_eq!(perft(&board, 3), PERFT3_STARTPOS);
    }

    #[test]
    fn fast_matches_naive_depth3() {
        let board = Board::new();
        let naive = perft_naive(&board, 3);
        let mut fast_board = board.clone();
        let fast = perft_fast(&mut fast_board, 3);
        assert_eq!(naive, fast);
        assert_eq!(fast, PERFT3_STARTPOS);
    }

    #[test]
    fn iterative_depth3_last() {
        let mut board = Board::new();
        let mut shared = SharedState::new();
        let lines = perft_iterative(&mut board, 3, &mut shared);
        assert_eq!(lines.last().map(|x| x.1), Some(PERFT3_STARTPOS));
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn parallel_root_depth3() {
        let board = Board::new();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        assert_eq!(perft_parallel_root(&board, 3, &pool), PERFT3_STARTPOS);
    }

    #[test]
    fn engine_iterative_depth3() {
        let mut board = Board::new();
        let mut engine = Engine::new();
        let lines = engine.perft_iterative(&mut board, 3);
        assert_eq!(lines.last().map(|x| x.1), Some(PERFT3_STARTPOS));
    }

    #[test]
    fn perft_o1_lookup_depth1_start() {
        let mut board = Board::new();
        assert_eq!(perft_no_tt_mode(&mut board, 1, PawnGenMode::O1Lookup), 131);
    }

    #[test]
    fn perft_o1_lookup_depth2_smoke() {
        let mut board = Board::new();
        assert_eq!(
            perft_no_tt_mode(&mut board, 2, PawnGenMode::O1Lookup),
            16_677
        );
    }

    #[test]
    fn perft_o1_lookup_depth3_matches_oracle() {
        let mut board = Board::new();
        assert_eq!(
            perft_no_tt_mode(&mut board, 3, PawnGenMode::O1Lookup),
            PERFT3_STARTPOS
        );
    }

    #[test]
    fn perft_o1_lookup_matches_scalar_depth3() {
        let mut board = Board::new();
        let scalar = perft_no_tt_mode(&mut board, 3, PawnGenMode::Scalar);
        let o1 = perft_no_tt_mode(&mut board, 3, PawnGenMode::O1Lookup);
        assert_eq!(scalar, o1);
        assert_eq!(o1, PERFT3_STARTPOS);
    }

    /// Full-tree correctness with TT (production-speed path ~3s @ d4 release).
    #[test]
    #[ignore = "slow; cargo test --release perft_o1_lookup_depth4 -- --ignored --nocapture"]
    fn perft_o1_lookup_depth4_matches_oracle() {
        let mut board = Board::new();
        let t0 = Instant::now();
        let nodes = perft_fast_mode(&mut board, 4, PawnGenMode::O1Lookup);
        eprintln!(
            "perft_o1+TT d4: {nodes} nodes in {:.2}s",
            t0.elapsed().as_secs_f64()
        );
        assert_eq!(nodes, PERFT4_STARTPOS);
    }

    /// O1 without TT — isolates pawn-gen cost only (~12s @ d4); not the production perft path.
    #[test]
    #[ignore = "slow; cargo test --release perft_o1_lookup_depth4_no_tt -- --ignored --nocapture"]
    fn perft_o1_lookup_depth4_no_tt() {
        let mut board = Board::new();
        let nodes = perft_no_tt_mode(&mut board, 4, PawnGenMode::O1Lookup);
        assert_eq!(nodes, PERFT4_STARTPOS);
    }

    /// Depths 1→4: worker on a P-core, timer on another; d4 budget 10s; `exit(1)` on timeout.
    #[test]
    #[ignore = "slow; release: cargo test --release perft_depth4 -- --ignored --nocapture"]
    fn perft_depth4_matches_oracle() {
        crate::movegen::prewarm(); // cold-start tables before the timed depths
        let mut completed: Vec<(u32, u64, Duration, Duration)> = Vec::with_capacity(4);
        let mut prev_elapsed = Duration::from_millis(PERFT_DEPTH_TIMEOUT_FLOOR_MS);

        for depth in 1..=4 {
            let budget = depth_timeout_budget(depth, prev_elapsed);
            match run_perft_depth_timed(depth, budget) {
                Ok(r) => {
                    eprintln!(
                        "perft {depth} {} worker_ms={:.1} wall_ms={:.1} budget_ms={:.0} \
                         cores worker={} timer={}",
                        r.nodes,
                        r.worker_elapsed.as_secs_f64() * 1000.0,
                        r.wall_elapsed.as_secs_f64() * 1000.0,
                        budget.as_secs_f64() * 1000.0,
                        r.worker_core,
                        r.timer_core,
                    );
                    assert_eq!(r.nodes, oracle_nodes(depth), "perft({depth}) node mismatch");
                    prev_elapsed = r.worker_elapsed;
                    completed.push((depth, r.nodes, r.worker_elapsed, r.wall_elapsed));
                }
                Err(()) => fail_perft_timeout(depth, budget, &completed),
            }
        }
    }
}
