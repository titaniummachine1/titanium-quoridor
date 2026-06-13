//! Flood-fill comparison: step-by-step `expand_wave` vs Kogge-Stone occluded fill.
//! Run: `cargo bench --bench flood_modes`
//! With PEXT-class build: `$env:RUSTFLAGS='-C target-cpu=native'; cargo bench --bench flood_modes`
//!
//! Builds a corpus of legal-ish boards (empty, light, heavy, snake mazes), asserts
//! the two floods agree on every one, then times each over many passes.

use std::time::Instant;

use titanium::core::board::{Board, Player, WallOrientation};
use titanium::path::parallel::{
    both_players_reach_goals_grids, both_players_reach_goals_grids_ks, pawn_bit, WallGrids,
};
use titanium::util::grid::{has_wall, set_wall};

struct Case {
    grids: WallGrids,
    p1: u128,
    p2: u128,
}

fn build_corpus() -> Vec<Case> {
    // Deterministic LCG — no rand dependency, mirrors parallel.rs tests.
    let mut state = 0x243F6A8885A308D3u64;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as u32
    };

    let mut cases = Vec::new();
    // Mix of densities so we sample short floods (open board, ~2 turns) and long
    // ones (heavy walls, snaking corridors, many turns).
    let densities = [0u32, 2, 4, 6, 8, 10, 14, 18];
    for _ in 0..600 {
        let mut board = Board::new();
        let target = densities[(next() as usize) % densities.len()];
        for _ in 0..target {
            let row = (next() % 8) as u8;
            let col = (next() % 8) as u8;
            let orientation = if next() & 1 == 0 {
                WallOrientation::Horizontal
            } else {
                WallOrientation::Vertical
            };
            if has_wall(&board, row, col, WallOrientation::Horizontal)
                || has_wall(&board, row, col, WallOrientation::Vertical)
            {
                continue;
            }
            set_wall(&mut board, row, col, orientation, true);
        }
        let p1 = ((next() % 9) as u8, (next() % 9) as u8);
        let mut p2 = ((next() % 9) as u8, (next() % 9) as u8);
        if p2 == p1 {
            p2 = ((p2.0 + 1) % 9, p2.1);
        }
        board.pawns[Player::One as usize] = p1;
        board.pawns[Player::Two as usize] = p2;
        cases.push(Case {
            grids: WallGrids::from_board(&board),
            p1: pawn_bit(p1.0, p1.1),
            p2: pawn_bit(p2.0, p2.1),
        });
    }
    cases
}

fn main() {
    titanium::movegen::prewarm(); // build cold-start tables before timing
    let corpus = build_corpus();

    // 1) Correctness: the two floods must agree on every position.
    let mut mismatches = 0usize;
    for c in &corpus {
        let a = both_players_reach_goals_grids(c.p1, c.p2, &c.grids);
        let b = both_players_reach_goals_grids_ks(c.p1, c.p2, &c.grids);
        if a != b {
            mismatches += 1;
        }
    }
    println!("corpus = {} positions, mismatches = {}", corpus.len(), mismatches);
    assert_eq!(mismatches, 0, "KS flood disagreed with step flood");

    const PASSES: u32 = 4000;

    // 2) Timing — accumulate a sink so nothing is optimised away.
    let t0 = Instant::now();
    let mut sink_step = 0u64;
    for _ in 0..PASSES {
        for c in &corpus {
            sink_step += both_players_reach_goals_grids(c.p1, c.p2, &c.grids) as u64;
        }
    }
    let step_secs = t0.elapsed().as_secs_f64();

    let t1 = Instant::now();
    let mut sink_ks = 0u64;
    for _ in 0..PASSES {
        for c in &corpus {
            sink_ks += both_players_reach_goals_grids_ks(c.p1, c.p2, &c.grids) as u64;
        }
    }
    let ks_secs = t1.elapsed().as_secs_f64();

    assert_eq!(sink_step, sink_ks, "sink mismatch (sanity)");

    let queries = (PASSES as u64) * (corpus.len() as u64);
    let step_nps = queries as f64 / step_secs;
    let ks_nps = queries as f64 / ks_secs;

    println!(
        "flood queries = {queries} ({PASSES} passes x {} positions)",
        corpus.len()
    );
    println!("| flood | seconds | queries/s | vs other |");
    println!("|-------|--------:|----------:|---------:|");
    println!(
        "| step expand_wave    | {step_secs:.3} | {:.0} | {:.3}x |",
        step_nps,
        step_secs / ks_secs.min(step_secs)
    );
    println!(
        "| kogge-stone occluded | {ks_secs:.3} | {:.0} | {:.3}x |",
        ks_nps,
        ks_secs / ks_secs.min(step_secs)
    );
    let speedup = step_secs / ks_secs;
    println!("\nKS is {speedup:.3}x the step flood ({:.1}% {}).",
        (speedup - 1.0).abs() * 100.0,
        if speedup > 1.0 { "faster" } else { "slower" });
}
