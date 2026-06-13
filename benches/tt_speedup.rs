//! Isolates the Transposition Table's contribution to perft(4).
//! Both paths bulk-count at depth 1 and use the same pawn mode, so the ONLY
//! difference is the TT probe/store. Run: `cargo bench --bench tt_speedup`
//! (add `RUSTFLAGS="-C target-cpu=native"` to match production build flags).

use std::env;
use std::mem::{align_of, size_of};
use std::time::Instant;

use titanium::{perft_fast_mode, perft_no_tt_mode, Board, PawnGenMode};

fn main() {
    titanium::movegen::prewarm(); // build cold-start tables before timing
    // Depth from env: `$env:TT_DEPTH='5'`. Default 4. No-TT baseline only runs
    // at depth <= 4 (depth 5 no-TT is ~minutes — 28.8B nodes).
    let depth: u32 = env::var("TT_DEPTH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    let mode = PawnGenMode::ShiftCanStep; // production default

    println!("perft({depth}) — bulk-d1, mode = ShiftCanStep");
    println!("| config | seconds | speedup |");
    println!("|--------|--------:|--------:|");

    let mut node_check = None;

    if depth <= 4 {
        let mut b1 = Board::new();
        let t0 = Instant::now();
        let n_no_tt = perft_no_tt_mode(&mut b1, depth, mode);
        let no_tt = t0.elapsed().as_secs_f64();
        node_check = Some(n_no_tt);
        println!("| bulk-d1, TT OFF | {no_tt:.3} | 1.00x |");

        // TT path twice for noise.
        for _ in 0..2 {
            let mut b2 = Board::new();
            let t1 = Instant::now();
            let n_tt = perft_fast_mode(&mut b2, depth, mode);
            let tt = t1.elapsed().as_secs_f64();
            assert_eq!(n_tt, n_no_tt);
            println!("| bulk-d1, TT ON  | {tt:.3} | {:.2}x |", no_tt / tt);
        }
    } else {
        // Deep: TT-on only, 3 runs (fresh TT each) for noise.
        for _ in 0..3 {
            let mut b2 = Board::new();
            let t1 = Instant::now();
            let n_tt = perft_fast_mode(&mut b2, depth, mode);
            let tt = t1.elapsed().as_secs_f64();
            if let Some(prev) = node_check {
                assert_eq!(n_tt, prev);
            }
            node_check = Some(n_tt);
            println!("| bulk-d1, TT ON  | {tt:.3} | (nodes {n_tt}) |");
        }
    }

    // TT memory layout — the pasted analysis warned about awkward entry sizing.
    println!("\nTT layout (titanium::search::tt is private; sizes mirrored here):");
    println!(
        "  Entry {{ key:u64, depth:u8, nodes:u64 }} → {} bytes (align {})",
        size_of::<MirrorEntry>(),
        align_of::<MirrorEntry>()
    );
    println!(
        "  Cluster [Entry; 4] → {} bytes ({} cache lines @64B)",
        size_of::<MirrorCluster>(),
        (size_of::<MirrorCluster>() + 63) / 64
    );
}

// Mirrors src/search/tt.rs Entry/Cluster purely to print their layout.
#[derive(Clone, Copy)]
#[allow(dead_code)]
struct MirrorEntry {
    key: u64,
    depth: u8,
    nodes: u64,
}
#[derive(Clone, Copy)]
#[allow(dead_code)]
struct MirrorCluster {
    entries: [MirrorEntry; 4],
}
