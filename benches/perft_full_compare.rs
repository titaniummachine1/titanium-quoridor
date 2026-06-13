//! Full perft comparison: all modes × depths 1–5, with and without TT.
//! Run: `cargo bench --bench perft_full_compare`
//! (add `RUSTFLAGS="-C target-cpu=native"` for BMI2/PEXT path)

use std::time::Instant;
use titanium::{
    perft_fast_mode, perft_no_tt_mode, Board, PawnGenMode,
    PERFT3_STARTPOS, PERFT4_STARTPOS,
};
const PERFT5_STARTPOS: u64 = 28_837_934_502;

fn oracle(depth: u32) -> Option<u64> {
    match depth {
        1 => Some(131),
        2 => Some(16_677),
        3 => Some(PERFT3_STARTPOS),
        4 => Some(PERFT4_STARTPOS),
        5 => Some(PERFT5_STARTPOS),
        _ => None,
    }
}

fn check(n: u64, depth: u32) -> &'static str {
    match oracle(depth) {
        Some(o) if o == n => "✓",
        Some(_) => "WRONG",
        None => "?",
    }
}

fn main() {
    titanium::movegen::prewarm();

    let modes: &[(PawnGenMode, &str)] = &[
        (PawnGenMode::Scalar,       "scalar"),
        (PawnGenMode::ShiftCanStep, "shift"),
        (PawnGenMode::O1Lookup,     "o1-lut"),
    ];

    // ── No-TT table ──────────────────────────────────────────────────────────
    println!("## No-TT (raw movegen cost)");
    println!();
    print!("| depth |");
    for (_, name) in modes { print!(" {name} nodes | {name} s |"); }
    println!();
    print!("|-------|");
    for _ in modes { print!("----------:|------:|"); }
    println!();

    for depth in 1u32..=4 {
        print!("| d{depth} |");
        for (mode, _) in modes {
            let mut b = Board::new();
            let t0 = Instant::now();
            let n = perft_no_tt_mode(&mut b, depth, *mode);
            let s = t0.elapsed().as_secs_f64();
            let ok = check(n, depth);
            print!(" {n} {ok} | {s:.3} |");
        }
        println!();
    }
    // d5 no-TT: only O1 (others take minutes)
    {
        let depth = 5u32;
        print!("| d{depth} |");
        for (mode, _) in modes {
            if *mode == PawnGenMode::O1Lookup {
                let mut b = Board::new();
                let t0 = Instant::now();
                let n = perft_no_tt_mode(&mut b, depth, *mode);
                let s = t0.elapsed().as_secs_f64();
                let ok = check(n, depth);
                print!(" {n} {ok} | {s:.3} |");
            } else {
                print!(" (skipped — minutes) | — |");
            }
        }
        println!();
    }

    println!();

    // ── TT table ─────────────────────────────────────────────────────────────
    println!("## With TT (production path — O1 movegen, TT_BITS=22 by default)");
    println!();
    println!("| depth | nodes | correct | seconds | vs no-TT-O1 |");
    println!("|-------|------:|---------|--------:|------------:|");

    // Stash d4/d5 no-TT O1 times for speedup column.
    let no_tt_ref: &[(u32, f64)] = &{
        let mut v = Vec::new();
        for depth in [4u32, 5u32] {
            let mut b = Board::new();
            let t0 = Instant::now();
            perft_no_tt_mode(&mut b, depth, PawnGenMode::O1Lookup);
            v.push((depth, t0.elapsed().as_secs_f64()));
        }
        v
    };

    for depth in 1u32..=5 {
        let mut b = Board::new();
        let t0 = Instant::now();
        let n = perft_fast_mode(&mut b, depth, PawnGenMode::O1Lookup);
        let s = t0.elapsed().as_secs_f64();
        let ok = check(n, depth);
        let speedup = no_tt_ref.iter()
            .find(|(d, _)| *d == depth)
            .map(|(_, no_tt)| format!("{:.2}×", no_tt / s))
            .unwrap_or_else(|| "—".into());
        println!("| d{depth} | {n} | {ok} | {s:.3} | {speedup} |");
    }
}
