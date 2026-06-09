//! Titanium Engine CLI — perft / divide / bench / genmove entry points.

use std::env;
use std::time::Instant;

use titanium::{
    generate_legal_moves, genmove_algebraic, perft_divide, Board, Engine, GenmoveConfig,
    GenmoveEngine, MctsConfig, SearchConfig, DEFAULT_MAX_NODES, DEFAULT_TIME_MS,
    MCTS_DEFAULT_MAX_SIMULATIONS, MCTS_DEFAULT_UCT,
};

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
        "perft-race" => run_perft_race(&args),
        "perft-id" => run_perft_id(&args),
        "thread-bench" => run_thread_bench(&args),
        "moves" => run_moves(),
        "genmove" => run_genmove(&args),
        _ => print_usage(),
    }
}

fn print_usage() {
    println!("Titanium Engine 0.1.0");
    println!("  titanium perft [depth] [--threads N]  — node count (default depth 3, threads 1)");
    println!("  titanium divide <depth>                — perft with move breakdown");
    println!("  titanium bench <depth> <n> [--threads N]");
    println!("  titanium thread-bench [depth] [--threads N] — 1 vs N threads, same nodes");
    println!("  titanium perft-race <sec>              — max depth within time budget");
    println!("  titanium perft-id [depth]              — iterative deepening perft 0..depth");
    println!("  titanium moves                         — list legal moves at startpos");
    println!("  titanium genmove [moves...] [--engine mcts|minimax|greedy] [--cat]");
    println!("              [--time SEC] [--sims N] [--uct F] [--nodes N] [--log]");
    println!("              — default: Gorisanson-style MCTS in Rust");
}

const DEFAULT_PERFT_DEPTH: u32 = 3;
const DEFAULT_THREAD_BENCH_DEPTH: u32 = 4;

struct CliArgs {
    positional: Vec<String>,
    threads: usize,
}

fn parse_cli(args: &[String]) -> CliArgs {
    let mut threads = 1usize;
    let mut positional = Vec::new();
    let mut i = 0usize;
    while i < args.len() {
        if args[i] == "--threads" {
            threads = args
                .get(i + 1)
                .and_then(|s| s.parse().ok())
                .unwrap_or(1)
                .max(1);
            i += 2;
            continue;
        }
        positional.push(args[i].clone());
        i += 1;
    }
    CliArgs {
        positional,
        threads,
    }
}

fn default_parallel_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .max(2)
}

fn load_board(cli: &CliArgs, depth_index: usize) -> (Board, u32) {
    let depth: u32 = cli
        .positional
        .get(depth_index)
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_PERFT_DEPTH);
    let mut board = Board::new();
    for mv in cli.positional.iter().skip(depth_index + 1) {
        board.apply_algebraic(mv);
    }
    (board, depth)
}

fn make_engine(threads: usize) -> Engine {
    if threads <= 1 {
        Engine::new()
    } else {
        Engine::with_threads(threads)
    }
}

fn run_perft(args: &[String]) {
    let cli = parse_cli(args);
    let (board, depth) = load_board(&cli, 2);
    let mut engine = make_engine(cli.threads);
    let start = Instant::now();
    let nodes = engine.perft(&board, depth);
    let elapsed = start.elapsed();
    println!("perft {} {}", depth, nodes);
    println!("threads {}", cli.threads);
    println!("time {:.3}s", elapsed.as_secs_f64());
}

fn run_divide(args: &[String]) {
    let cli = parse_cli(args);
    let (board, depth) = load_board(&cli, 2);
    let (total, lines) = perft_divide(&board, depth);
    for (mv, nodes) in &lines {
        println!("{} {}", mv, nodes);
    }
    println!();
    println!("Nodes searched: {}", total);
}

fn run_bench(args: &[String]) {
    let cli = parse_cli(args);
    let depth: u32 = cli
        .positional
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_PERFT_DEPTH);
    let iterations: u32 = cli
        .positional
        .get(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    let board = Board::new();
    let mut engine = make_engine(cli.threads);

    engine.perft(&board, depth);

    let start = Instant::now();
    let mut nodes = 0u64;
    for _ in 0..iterations {
        nodes = engine.perft(&board, depth);
    }
    let elapsed = start.elapsed();
    let total_nodes = nodes * iterations as u64;
    let nps = total_nodes as f64 / elapsed.as_secs_f64();

    println!(
        "bench depth={} iters={} threads={} nodes={} time={:.3}s nps={:.0}",
        depth,
        iterations,
        cli.threads,
        total_nodes,
        elapsed.as_secs_f64(),
        nps
    );
}

fn run_perft_id(args: &[String]) {
    let cli = parse_cli(args);
    let max_depth: u32 = cli
        .positional
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_PERFT_DEPTH);
    let mut board = Board::new();
    let mut engine = make_engine(cli.threads);
    let start = Instant::now();
    let lines = engine.perft_iterative(&mut board, max_depth);
    let elapsed = start.elapsed();

    for (depth, nodes) in &lines {
        println!("perft {} {}", depth, nodes);
    }
    println!("threads {}", cli.threads);
    println!("perft-id total {:.3}s", elapsed.as_secs_f64());
}

fn run_thread_bench(args: &[String]) {
    let cli = parse_cli(args);
    let depth: u32 = cli
        .positional
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_THREAD_BENCH_DEPTH);
    let parallel = if cli.threads > 1 {
        cli.threads
    } else {
        default_parallel_threads()
    };
    let board = Board::new();

    let result = Engine::bench_threads(&board, depth, parallel);

    println!("thread-bench depth={} nodes={}", result.depth, result.nodes);
    println!("threads=1  time {:.3}s", result.threads_one_secs);
    println!(
        "threads={} time {:.3}s",
        result.threads_n, result.threads_n_secs
    );
    println!("speedup {:.2}x", result.speedup());
}

fn run_perft_race(args: &[String]) {
    let cli = parse_cli(args);
    let budget: f64 = cli
        .positional
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(3.0);
    let board = Board::new();
    let mut engine = make_engine(cli.threads);
    let mut best_depth = 0u32;
    let mut best_nodes = 0u64;
    let mut best_ms = 0.0f64;

    for depth in 1..=8 {
        let start = Instant::now();
        let nodes = engine.perft(&board, depth);
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        if ms > budget * 1000.0 {
            break;
        }
        best_depth = depth;
        best_nodes = nodes;
        best_ms = ms;
    }

    println!(
        "perft-race budget={:.1}s threads={} best_depth={} nodes={} time_ms={:.0}",
        budget, cli.threads, best_depth, best_nodes, best_ms
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

fn parse_genmove_config(args: &[String]) -> (GenmoveConfig, Vec<String>) {
    let log = std::env::var("TITANIUM_LOG").is_ok();
    let mut config = GenmoveConfig {
        engine: GenmoveEngine::Mcts,
        mcts: MctsConfig {
            time_ms: DEFAULT_TIME_MS,
            max_simulations: MCTS_DEFAULT_MAX_SIMULATIONS,
            uct: MCTS_DEFAULT_UCT,
            log,
            use_cat_guidance: false, // bridge is activated by the genmove handoff
            book_hint: None,
        },
        minimax: SearchConfig {
            time_ms: DEFAULT_TIME_MS,
            max_nodes: DEFAULT_MAX_NODES,
            log,
            book_hint: None,
        },
    };
    let mut moves = Vec::new();

    let mut i = 2usize;
    while i < args.len() {
        let arg = &args[i];
        if arg == "--engine" {
            if let Some(name) = args.get(i + 1) {
                config.engine = match name.as_str() {
                    "minimax" | "ab" => GenmoveEngine::Minimax,
                    "greedy" => GenmoveEngine::Greedy,
                    _ => GenmoveEngine::Mcts,
                };
                i += 2;
                continue;
            }
        } else if arg == "--time" {
            if let Some(sec) = args.get(i + 1).and_then(|s| s.parse::<f64>().ok()) {
                let ms = (sec * 1000.0) as u64;
                config.mcts.time_ms = ms;
                config.minimax.time_ms = ms;
                i += 2;
                continue;
            }
        } else if arg == "--sims" {
            if let Some(n) = args.get(i + 1).and_then(|s| s.parse().ok()) {
                config.mcts.max_simulations = n;
                i += 2;
                continue;
            }
        } else if arg == "--uct" {
            if let Some(u) = args.get(i + 1).and_then(|s| s.parse().ok()) {
                config.mcts.uct = u;
                i += 2;
                continue;
            }
        } else if arg == "--nodes" {
            if let Some(n) = args.get(i + 1).and_then(|s| s.parse().ok()) {
                config.minimax.max_nodes = n;
                i += 2;
                continue;
            }
        } else if arg == "--log" {
            config.mcts.log = true;
            config.minimax.log = true;
            i += 1;
            continue;
        } else if arg == "--cat" || arg == "--cat-guidance" || arg == "--bridge-mcts" {
            config.mcts.use_cat_guidance = true;
            i += 1;
            continue;
        } else if arg.starts_with("--") {
            i += 1;
            continue;
        } else {
            moves.push(arg.clone());
        }
        i += 1;
    }

    (config, moves)
}

fn run_genmove(args: &[String]) {
    let (config, moves) = parse_genmove_config(args);
    let mut board = Board::new();
    for mv in moves {
        board.apply_algebraic(&mv);
    }

    match genmove_algebraic(&mut board, config) {
        Some(algebraic) => println!("bestmove {}", algebraic),
        None => println!("bestmove (none)"),
    }
}
