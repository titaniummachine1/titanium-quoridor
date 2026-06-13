//! Titanium Engine CLI — perft / divide / bench / genmove entry points.

use std::env;
use std::time::Instant;

use titanium::{
    cat_snapshot_json, generate_legal_moves, genmove_algebraic, lmr_snapshot_json, perft_divide,
    run_session_stdio, Board, Engine, GenmoveConfig, GenmoveEngine, MctsConfig, SearchConfig,
    DEFAULT_MAX_NODES, DEFAULT_TIME_MS, MCTS_DEFAULT_MAX_SIMULATIONS, MCTS_DEFAULT_UCT,
};

#[cfg(not(target_arch = "wasm32"))]
fn maybe_pin_core() {
    use core_affinity::CoreId;

    let core = if let Ok(s) = env::var("TITANIUM_PIN_CORE") {
        s.parse::<usize>().ok().map(|id| CoreId { id })
    } else if env::var("TITANIUM_PIN_LAST").is_ok() {
        core_affinity::get_core_ids().and_then(|ids| ids.last().copied())
    } else {
        None
    };
    if let Some(c) = core {
        if core_affinity::set_for_current(c) {
            eprintln!("pinned: logical core {}", c.id);
        } else {
            eprintln!("warning: could not pin to core {}", c.id);
        }
    }
}

#[cfg(target_arch = "wasm32")]
fn maybe_pin_core() {}

fn main() {
    maybe_pin_core();
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        print_usage();
        return;
    }

    // Cold-start pawn tables (~1-2s, once per process). Long-lived server modes
    // kick the build off in the background AT LAUNCH so it overlaps the GUI
    // handshake (`isready`/first move blocks on it only if it isn't done yet);
    // one-shot timed commands build synchronously up front so the build is never
    // inside a measured region. Never rebuilds mid-session — that's the OnceLock.
    match args[1].as_str() {
        "uci" | "session" => {
            std::thread::spawn(|| titanium::movegen::prewarm());
        }
        _ => titanium::movegen::prewarm(),
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
        "ace-bench" => run_ace_bench(&args),
        "ace-perft" => run_ace_perft(&args),
        "cat" => run_cat(&args),
        "lmr" => run_lmr(&args),
        "uci" => titanium::run_uci_stdio(),
        "session" => match ace_engine_flag(&args) {
            Some(flag) => titanium::ace::run_ace_session_stdio(flag),
            None => run_session_stdio(),
        },
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
    println!("  titanium uci                           — UCI-style stdio protocol (testing infra)");
    println!("  titanium cat [moves...]                — CAT v3 heatmap JSON for current position");
    println!(
        "  titanium lmr [moves...] [--time SEC] [--depth N] — root LMR plan JSON (default depth 8)"
    );
    println!(
        "  titanium session [--engine ace-v8-ti-pmc] — long-lived REPL (TT persists between plies)"
    );
    println!("  titanium ace-perft [depth] [--iters N] — ACE vs Titanium movegen perft compare");
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
        engine: GenmoveEngine::Minimax,
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
            ..SearchConfig::default()
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

fn run_cat(args: &[String]) {
    let mut board = Board::new();
    for mv in args.iter().skip(2) {
        if mv.starts_with("--") {
            break;
        }
        board.apply_algebraic(mv);
    }
    println!("{}", cat_snapshot_json(&mut board));
}

fn looks_like_algebraic_move(arg: &str) -> bool {
    let b = arg.as_bytes();
    b.len() >= 2 && b[0].is_ascii_lowercase() && b[1].is_ascii_digit()
}

fn run_lmr(args: &[String]) {
    let mut board = Board::new();
    let mut time_ms = DEFAULT_TIME_MS;
    let mut id_depth = 8u32;
    let mut i = 2usize;
    while i < args.len() {
        let arg = &args[i];
        if arg == "--time" {
            if let Some(sec) = args.get(i + 1).and_then(|s| s.parse::<f64>().ok()) {
                time_ms = (sec * 1000.0).round() as u64;
                i += 2;
                continue;
            }
        } else if arg == "--depth" {
            if let Some(d) = args.get(i + 1).and_then(|s| s.parse::<u32>().ok()) {
                id_depth = d;
                i += 2;
                continue;
            }
        } else if arg.starts_with("--") {
            // Unknown flag — consume a numeric/value token so `8` is not parsed as a move.
            if let Some(next) = args.get(i + 1) {
                if !next.starts_with("--") && !looks_like_algebraic_move(next) {
                    i += 2;
                    continue;
                }
            }
            i += 1;
            continue;
        } else if looks_like_algebraic_move(arg) {
            board.apply_algebraic(arg);
        }
        i += 1;
    }
    println!("{}", lmr_snapshot_json(&mut board, time_ms, id_depth));
}

fn run_genmove(args: &[String]) {
    if is_ace_engine(args) {
        run_genmove_ace(args);
        return;
    }
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

// ── ACE v7 port (pure) ───────────────────────────────────────────────────────

fn ace_engine_flag(args: &[String]) -> Option<&str> {
    args.windows(2).find_map(|w| {
        if w[0] != "--engine" {
            return None;
        }
        match w[1].as_str() {
            "ace" | "ace-v8" | "ace-v10" | "ace-v11" | "ace-cat" | "ace-ti" | "ace-v8-ti"
            | "ace-v8-ti-pmc" | "ace-v10-ti" | "ace-v10-ti-pmc" | "ace-v11-ti"
            | "ace-v11-ti-pmc" | "ace-pmc" => Some(w[1].as_str()),
            _ => None,
        }
    })
}

fn ace_engine_mode(flag: &str) -> &'static str {
    match flag {
        "ace-cat" => "ace-cat",
        "ace-ti" | "ace-v8-ti" | "ace-v8-ti-pmc" | "ace-v10-ti" | "ace-v10-ti-pmc"
        | "ace-v11-ti" | "ace-v11-ti-pmc" => "ace-ti",
        _ => "ace",
    }
}

fn is_ace_engine(args: &[String]) -> bool {
    ace_engine_flag(args).is_some()
}

fn run_genmove_ace(args: &[String]) {
    let label = ace_engine_flag(args).unwrap_or("ace");
    let mode = ace_engine_mode(label);
    let mut params = titanium::ace::AceParams {
        cat: mode == "ace-cat",
        ti_movegen: mode == "ace-ti",
        eme: label.contains("pmc"),
        ..Default::default()
    };
    let mut moves = Vec::new();
    let mut i = 2usize;
    while i < args.len() {
        let arg = &args[i];
        if arg == "--time" {
            if let Some(sec) = args.get(i + 1).and_then(|s| s.parse::<f64>().ok()) {
                params.time_ms = (sec * 1000.0) as u64;
                i += 2;
                continue;
            }
        } else if arg == "--depth" {
            if let Some(d) = args.get(i + 1).and_then(|s| s.parse::<i32>().ok()) {
                params.max_depth = d;
                i += 2;
                continue;
            }
        } else if arg == "--full" {
            params.full = true;
            i += 1;
            continue;
        } else if arg == "--log" {
            params.log = true;
            i += 1;
            continue;
        } else if arg == "--eme" || arg == "--pseudo-mcts" {
            params.eme = true;
            i += 1;
            continue;
        } else if arg == "--engine" {
            i += 2;
            continue;
        } else if arg.starts_with("--") {
            // unknown flag with value (e.g. --sims N from the shared harness)
            if let Some(next) = args.get(i + 1) {
                if !next.starts_with("--") && !looks_like_algebraic_move(next) {
                    i += 2;
                    continue;
                }
            }
            i += 1;
            continue;
        } else if looks_like_algebraic_move(arg) {
            moves.push(arg.clone());
        }
        i += 1;
    }

    match titanium::ace::ace_genmove(&moves, params, label) {
        Some((algebraic, info)) => {
            if !params.log {
                let mut depth_json = String::new();
                for (i, e) in info.depth_log.iter().enumerate() {
                    if i > 0 {
                        depth_json.push(',');
                    }
                    let pv = e.pv.replace('\\', "\\\\").replace('"', "\\\"");
                    depth_json.push_str(&format!(
                        "{{\"depth\":{},\"score\":{},\"nodes\":{},\"elapsedMs\":{},\"marginalNodes\":{},\"pv\":\"{}\"}}",
                        e.depth, e.score, e.nodes, e.elapsed_ms, e.marginal_nodes, pv
                    ));
                }
                eprintln!(
                    "info json {{\"engine\":\"{}\",\"stoppedBy\":\"{}\",\"searchDepth\":{},\"nodes\":{},\"rootScore\":{},\"whiteDist\":{},\"blackDist\":{},\"elapsedMs\":{},\"depthLog\":[{}]}}",
                    label,
                    label,
                    info.depth,
                    info.nodes,
                    info.score,
                    info.white_dist,
                    info.black_dist,
                    info.ms,
                    depth_json
                );
            }
            println!("bestmove {}", algebraic);
        }
        None => println!("bestmove (none)"),
    }
}

/// Parity harness vs the JS reference — fixed depth, ACE numeric moves.
/// `--cat` switches to the hybrid wall filter.
fn run_ace_bench(args: &[String]) {
    let use_cat = args.iter().any(|a| a == "--cat");
    let depth: i32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(8);
    let mut g = titanium::ace::AceGame::new();
    for arg in args.iter().skip(3) {
        if let Ok(m) = arg.parse::<i16>() {
            g.make_move(m);
        }
    }
    println!("hash {} {}", g.hash_lo, g.hash_hi);
    let mut search = if use_cat {
        titanium::ace::AceSearch::with_cat(g)
    } else {
        titanium::ace::AceSearch::new(g)
    };
    let r = search.think(1_000_000_000, depth, true, false, "ace-bench");
    println!(
        "{{\"move\":{},\"score\":{},\"depth\":{},\"nodes\":{},\"ms\":{}}}",
        r.mv, r.score, r.depth, r.nodes, r.ms
    );
}

/// Compare perft: ACE v7 native movegen vs Titanium `perft_fast` (10s cap at depth 4).
fn run_ace_perft(args: &[String]) {
    use std::time::Duration;

    let depth: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(4);
    let mut timeout_secs = titanium::ace::default_timeout(depth).as_secs();
    let mut i = 3usize;
    while i < args.len() {
        if args[i] == "--timeout" {
            if let Some(sec) = args.get(i + 1).and_then(|s| s.parse::<u64>().ok()) {
                timeout_secs = sec;
                i += 2;
                continue;
            }
        }
        i += 1;
    }
    let timeout = Duration::from_secs(timeout_secs);

    fn print_line(r: &titanium::ace::TimedPerftResult) {
        if r.timed_out {
            println!(
                "  {:12} TIMEOUT after {:.1}s (no result)",
                r.label,
                r.elapsed_ms as f64 / 1000.0
            );
            return;
        }
        let nodes = r.nodes.unwrap_or(0);
        let secs = r.elapsed_ms as f64 / 1000.0;
        let nps = if secs > 0.0 { nodes as f64 / secs } else { 0.0 };
        println!(
            "  {:12} nodes={} time={:.3}s nps={:.0}",
            r.label, nodes, secs, nps
        );
    }

    println!(
        "ace-perft depth={} timeout={}s (oracle perft_fast + TT vs ACE v7 wall_legal)",
        depth, timeout_secs
    );

    let ti = titanium::ace::perft_titanium_timed(depth, timeout);
    print_line(&ti);

    let ace_ti = titanium::ace::perft_ace_ti_timed(depth, timeout);
    print_line(&ace_ti);

    let ace = titanium::ace::perft_ace_timed(depth, timeout);
    print_line(&ace);

    if let Some(exp) = titanium::ace::oracle_nodes(depth) {
        println!("  oracle depth{}={}", depth, exp);
        println!(
            "  perft_fast_ok={} ace_ti_ok={} ace_native_ok={}",
            ti.nodes == Some(exp),
            ace_ti.nodes == Some(exp),
            ace.nodes == Some(exp)
        );
        if let (Some(ti_n), Some(ati_n)) = (ti.nodes, ace_ti.nodes) {
            if ti_n == ati_n {
                let ratio = ace_ti.elapsed_ms as f64 / ti.elapsed_ms.max(1) as f64;
                println!("  ace_ti vs perft_fast: {:.2}x (1.0 = same speed)", ratio);
            }
        }
        if ace.timed_out {
            println!(
                "  ace-v7-native: TIMEOUT — ported wall_legal path unusable at depth {}",
                depth
            );
        } else if let (Some(an), Some(ati_n)) = (ace.nodes, ace_ti.nodes) {
            if an == ati_n {
                let ratio = ace.elapsed_ms as f64 / ace_ti.elapsed_ms.max(1) as f64;
                println!("  ace_ti vs ace-v7-native: {:.2}x faster", ratio);
            }
        }
    }

    if ace_ti.timed_out || (ace.nodes.is_some() && ace.nodes != ace_ti.nodes) {
        std::process::exit(1);
    }
}
