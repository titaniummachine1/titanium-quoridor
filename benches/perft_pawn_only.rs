//! Pawn-only perft — no walls; shipped vs research pawn gen only (~seconds, not minutes).
//! Run: `cargo run --release --bench perft_pawn_only`
//! Override depth: `$env:PAWN_ONLY_DEPTH='14'; cargo run --release --bench perft_pawn_only`

use std::env;
use std::time::Instant;

use titanium::{perft_pawn_only_mode, Board, PawnGenMode};

#[cfg(not(target_arch = "wasm32"))]
fn maybe_pin_core() {
    use core_affinity::CoreId;
    let core = env::var("TITANIUM_PIN_CORE")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .map(|id| CoreId { id })
        .or_else(|| {
            if env::var("TITANIUM_PIN_LAST").is_ok() {
                core_affinity::get_core_ids().and_then(|ids| ids.last().copied())
            } else {
                None
            }
        });
    if let Some(c) = core {
        if core_affinity::set_for_current(c) {
            eprintln!("pinned: logical core {}", c.id);
        }
    }
}

#[cfg(target_arch = "wasm32")]
fn maybe_pin_core() {}

fn main() {
    maybe_pin_core();
    titanium::movegen::prewarm(); // build cold-start tables before timing

    let depth: u32 = env::var("PAWN_ONLY_DEPTH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(12);

    let modes = [
        (PawnGenMode::ShiftCanStep, "shift_can_step (SHIPPED)"),
        (PawnGenMode::O1Lookup, "o1_full_lut"),
        (PawnGenMode::O1LeanLut, "o1_lean_lut (ek=0→shift, ek≠0→table)"),
    ];

    let oracle = {
        let mut board = Board::new();
        perft_pawn_only_mode(&mut board, depth, PawnGenMode::ShiftCanStep)
    };

    println!("pawn-only perft({depth}) startpos — pawns only, no walls/TT/BFS");
    println!("oracle nodes = {oracle}");
    println!("| mode | nodes | match | seconds | nps | vs fastest |");
    println!("|------|------:|-------|--------:|----:|-----------:|");

    let mut rows: Vec<(String, u64, bool, f64)> = Vec::new();

    for (mode, label) in modes {
        let mut board = Board::new();
        let t0 = Instant::now();
        let nodes = perft_pawn_only_mode(&mut board, depth, mode);
        let secs = t0.elapsed().as_secs_f64();
        rows.push((label.to_string(), nodes, nodes == oracle, secs));
    }

    let fastest = rows.iter().map(|r| r.3).fold(f64::INFINITY, f64::min);

    for (label, nodes, ok, secs) in &rows {
        let ratio = secs / fastest;
        let nps = (*nodes as f64 / secs) as u64;
        let mark = if *ok { "yes" } else { "NO" };
        println!("| {label} | {nodes} | {mark} | {secs:.3} | {nps} | {ratio:.3}x |",);
    }
}
