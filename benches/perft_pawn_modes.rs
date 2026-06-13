//! One-shot perft(4) timing table for pawn-generation modes (no TT).
//! Run: `cargo bench --bench perft_pawn_modes -- --noplot`

use std::time::Instant;

use titanium::{perft_no_tt_mode, Board, PawnGenMode, PERFT3_STARTPOS, PERFT4_STARTPOS};

fn main() {
    // Build the cold-start pawn tables BEFORE timing so the ~1-2s build never
    // lands inside a measured perft run.
    titanium::movegen::prewarm();
    const DEPTH: u32 = 4;
    let oracle = if DEPTH == 3 {
        PERFT3_STARTPOS
    } else {
        PERFT4_STARTPOS
    };

    let modes = [
        (PawnGenMode::Scalar, "scalar_can_step"),
        (PawnGenMode::ShiftCanStep, "shift_bit_can_step"),
        (
            PawnGenMode::BitboardFreshDirMasks,
            "bitboard_fresh_dirmasks",
        ),
        (
            PawnGenMode::BitboardCachedDirMasks,
            "bitboard_cached_dirmasks",
        ),
        (PawnGenMode::O1Lookup, "o1_full_lut"),
        (PawnGenMode::O1LeanLut, "o1_lean_lut"),
    ];

    println!("perft({DEPTH}) oracle={oracle} startpos — no TT, BFS mask cache shared");
    println!("| mode | nodes | correct | seconds | vs fastest |");
    println!("|------|------:|---------|--------:|-----------:|");

    let mut rows: Vec<(String, u64, bool, f64)> = Vec::new();

    for (mode, label) in modes {
        let mut board = Board::new();
        let t0 = Instant::now();
        let nodes = perft_no_tt_mode(&mut board, DEPTH, mode);
        let secs = t0.elapsed().as_secs_f64();
        let ok = nodes == oracle;
        rows.push((label.to_string(), nodes, ok, secs));
    }

    let fastest = rows.iter().map(|r| r.3).fold(f64::INFINITY, f64::min);

    for (label, nodes, ok, secs) in &rows {
        let ratio = secs / fastest;
        let mark = if *ok { "yes" } else { "NO" };
        println!("| {label} | {nodes} | {mark} | {secs:.3} | {ratio:.3}x |",);
    }
}
