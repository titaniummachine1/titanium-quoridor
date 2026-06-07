//! Titanium Engine CLI — perft / divide / bench entry points.

use std::env;
use std::time::Instant;

use titanium::{Board, generate_legal_moves, perft, perft_divide};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        print_usage();
        return;
    }

    match args[1].as_str() {
        "perft" => run_perft(&args),
        "divide" => run_divide(&args),
        "bench" => run_bench(&args),
        "moves" => run_moves(),
        _ => print_usage(),
    }
}

fn print_usage() {
    println!("Titanium Engine 0.1.0");
    println!("  titanium perft <depth>     — node count from startpos");
    println!("  titanium divide <depth>    — perft with move breakdown");
    println!("  titanium bench <depth> <n> — time perft iterations");
    println!("  titanium moves             — list legal moves at startpos");
}

fn run_perft(args: &[String]) {
    let depth: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1);
    let board = Board::new();
    let start = Instant::now();
    let nodes = perft(&board, depth);
    let elapsed = start.elapsed();
    println!("perft {} {}", depth, nodes);
    println!("time {:.3}s", elapsed.as_secs_f64());
}

fn run_divide(args: &[String]) {
    let depth: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1);
    let board = Board::new();
    let (total, lines) = perft_divide(&board, depth);
    for (mv, nodes) in &lines {
        println!("{} {}", mv, nodes);
    }
    println!();
    println!("Nodes searched: {}", total);
}

fn run_bench(args: &[String]) {
    let depth: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(2);
    let iterations: u32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(10);
    let board = Board::new();

    perft(&board, depth);

    let start = Instant::now();
    let mut nodes = 0u64;
    for _ in 0..iterations {
        nodes = perft(&board, depth);
    }
    let elapsed = start.elapsed();
    let total_nodes = nodes * iterations as u64;
    let nps = total_nodes as f64 / elapsed.as_secs_f64();

    println!(
        "bench depth={} iters={} nodes={} time={:.3}s nps={:.0}",
        depth,
        iterations,
        total_nodes,
        elapsed.as_secs_f64(),
        nps
    );
}

fn run_moves() {
    let board = Board::new();
    let moves = generate_legal_moves(&board);
    println!("{} legal moves at startpos", moves.len());
    for mv in moves {
        println!("{}", titanium::format_move(mv));
    }
}
