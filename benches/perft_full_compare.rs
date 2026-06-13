//! Full perft comparison: all modes × depths 1–5, with and without TT.
//! Run: `cargo bench --bench perft_full_compare`
//! (add `RUSTFLAGS="-C target-cpu=native"` for BMI2/PEXT path)
//!
//! Regression gate: any timed run that exceeds 20 seconds prints
//! "TIMEOUT — REGRESSION" and exits with code 1.

use std::sync::mpsc;
use std::time::{Duration, Instant};
use titanium::{perft_fast_mode, perft_no_tt_mode, Board, PawnGenMode, PERFT3_STARTPOS, PERFT4_STARTPOS};

const PERFT5_STARTPOS: u64 = 28_837_934_502;
const TIMEOUT: Duration = Duration::from_secs(20);

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

/// Run `f` in a thread; return `Some((result, elapsed))` or `None` on timeout.
fn timed<F>(f: F) -> Option<(u64, f64)>
where
    F: FnOnce() -> u64 + Send + 'static,
{
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let t0 = Instant::now();
        let n = f();
        let _ = tx.send((n, t0.elapsed().as_secs_f64()));
    });
    rx.recv_timeout(TIMEOUT).ok()
}

fn main() {
    titanium::movegen::prewarm();

    let modes: &[(PawnGenMode, &str)] = &[
        (PawnGenMode::Scalar, "scalar"),
        (PawnGenMode::ShiftCanStep, "shift"),
        (PawnGenMode::O1Lookup, "o1-lut"),
    ];

    let mut regression = false;

    // ── No-TT table ──────────────────────────────────────────────────────────
    println!("## No-TT (raw movegen cost)");
    println!();
    print!("| depth |");
    for (_, name) in modes {
        print!(" {name} nodes | {name} s |");
    }
    println!();
    print!("|-------|");
    for _ in modes {
        print!("----------:|------:|");
    }
    println!();

    for depth in 1u32..=4 {
        print!("| d{depth} |");
        for (mode, _) in modes {
            let mode = *mode;
            match timed(move || {
                let mut b = Board::new();
                perft_no_tt_mode(&mut b, depth, mode)
            }) {
                Some((n, s)) => {
                    print!(" {} {} | {:.3} |", n, check(n, depth), s);
                }
                None => {
                    print!(" TIMEOUT | >20s |");
                    regression = true;
                }
            }
        }
        println!();
    }
    // d5 no-TT: only O1 (others take minutes — not a regression gate)
    {
        let depth = 5u32;
        print!("| d{depth} |");
        for (mode, _) in modes {
            if *mode == PawnGenMode::O1Lookup {
                print!(" (skipped — ~3min no-TT) | — |");
            } else {
                print!(" (skipped — minutes) | — |");
            }
        }
        println!();
    }

    println!();

    // ── TT table — regression gate ───────────────────────────────────────────
    println!("## With TT (production path — O1 movegen, TT_BITS=22 by default)");
    println!("## Regression gate: >20s = major regression, exits 1");
    println!();
    println!("| depth | nodes | correct | seconds | status |");
    println!("|-------|------:|---------|--------:|--------|");

    for depth in 1u32..=5 {
        let mode = PawnGenMode::O1Lookup;
        match timed(move || {
            let mut b = Board::new();
            perft_fast_mode(&mut b, depth, mode)
        }) {
            Some((n, s)) => {
                let ok = check(n, depth);
                let status = if oracle(depth).map_or(false, |o| o != n) {
                    regression = true;
                    "WRONG NODE COUNT"
                } else {
                    "ok"
                };
                println!("| d{depth} | {n} | {ok} | {s:.3} | {status} |");
            }
            None => {
                println!("| d{depth} | — | — | >20s | TIMEOUT — REGRESSION |");
                regression = true;
            }
        }
    }

    if regression {
        eprintln!("\nREGRESSION DETECTED — see table above");
        std::process::exit(1);
    } else {
        println!("\nAll depths within 20s budget ✓");
    }
}
