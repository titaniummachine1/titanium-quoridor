//! Titanium Engine CLI — perft / divide / bench / genmove entry points.

use std::env;
use std::time::Instant;

use titanium::{
    cat_snapshot_json, format_move, generate_legal_moves, genmove_algebraic, lmr_snapshot_json,
    perft_divide, run_search, run_session_stdio, Board, Engine, GameSearchSession, GenmoveConfig,
    GenmoveEngine, MctsConfig, SearchConfig, DEFAULT_MAX_NODES, DEFAULT_TIME_MS,
    MCTS_DEFAULT_MAX_SIMULATIONS, MCTS_DEFAULT_UCT,
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
        "rollout" => run_rollout(&args),
        "match" => run_match(&args),
        "uci" => titanium::run_uci_stdio(),
        "session" => match ace_engine_flag(&args) {
            Some(flag) if is_acev13(flag) => titanium::acev13::run_ace_session_stdio(flag),
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
        "  titanium session [--engine ace-v13-ti] — long-lived REPL (TT persists between plies)"
    );
    println!(
        "  titanium genmove --engine ace-v13 [moves...] — gen13 ACE port (O1 movegen; ace-v13-pure = faithful 1:1)"
    );
    println!("  titanium ace-perft [depth] [--iters N] — ACE vs Titanium movegen perft compare");
    println!(
        "  titanium rollout [moves...] [--sims K] [--plies P] [--cmp-depth D] [--seed S] [--time SEC]"
    );
    println!("              — EXPERIMENT: eval-guided rollout ranking vs deep αβ (see search::rollout)");
}

const DEFAULT_PERFT_DEPTH: u32 = 3;
const DEFAULT_THREAD_BENCH_DEPTH: u32 = 4;

struct CliArgs {
    positional: Vec<String>,
    threads: usize,
    no_tt: bool,
}

fn parse_cli(args: &[String]) -> CliArgs {
    let mut threads = 1usize;
    let mut no_tt = false;
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
        if args[i] == "--no-tt" {
            no_tt = true;
            i += 1;
            continue;
        }
        positional.push(args[i].clone());
        i += 1;
    }
    CliArgs {
        positional,
        threads,
        no_tt,
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
    let nodes = if cli.no_tt {
        let mut board_copy = board.clone();
        engine.perft_no_tt(&mut board_copy, depth)
    } else {
        engine.perft(&board, depth)
    };
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
                #[allow(deprecated)]
                let engine = match name.as_str() {
                    "minimax" | "ab" => GenmoveEngine::Minimax,
                    "greedy" => GenmoveEngine::Greedy,
                    _ => GenmoveEngine::Mcts,
                };
                config.engine = engine;
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

/// EXPERIMENTAL — measure how well eval-guided rollouts predict the deep
/// alpha-beta root-move ranking. Validates the "simulation-guided minimax"
/// hypothesis before any production wiring. See `search::rollout`.
///
/// Usage: `titanium rollout [moves...] [--sims K] [--plies P] [--cmp-depth D]
///         [--seed S] [--time SEC]`
fn run_rollout(args: &[String]) {
    let mut moves: Vec<String> = Vec::new();
    let mut sims = 64u32;
    let mut plies = 24u32;
    let mut cmp_depth = 8u32;
    let mut seed = 1u64;
    let mut time_ms = 5000u64;
    let mut i = 2usize;
    while i < args.len() {
        let arg = args[i].as_str();
        let next_u = args.get(i + 1).and_then(|s| s.parse::<u64>().ok());
        match arg {
            "--sims" if next_u.is_some() => {
                sims = next_u.unwrap() as u32;
                i += 2;
                continue;
            }
            "--plies" if next_u.is_some() => {
                plies = next_u.unwrap() as u32;
                i += 2;
                continue;
            }
            "--cmp-depth" if next_u.is_some() => {
                cmp_depth = next_u.unwrap() as u32;
                i += 2;
                continue;
            }
            "--seed" if next_u.is_some() => {
                seed = next_u.unwrap();
                i += 2;
                continue;
            }
            "--time" => {
                if let Some(sec) = args.get(i + 1).and_then(|s| s.parse::<f64>().ok()) {
                    time_ms = (sec * 1000.0).round() as u64;
                    i += 2;
                    continue;
                }
            }
            other if looks_like_algebraic_move(other) => moves.push(other.to_string()),
            _ => {}
        }
        i += 1;
    }

    // ── Rollout ranking ─────────────────────────────────────────────────────
    let mut board = Board::new();
    for mv in &moves {
        board.apply_algebraic(mv);
    }
    let t0 = Instant::now();
    let ranks = titanium::search::rollout::rollout_rank(&mut board, sims, plies, seed);
    let roll_ms = t0.elapsed().as_millis();

    // ── Ground-truth deep ranking ───────────────────────────────────────────
    let mut session = GameSearchSession::new();
    if !moves.is_empty() {
        let _ = session.set_position(&moves);
    }
    let config = SearchConfig {
        time_ms,
        max_nodes: DEFAULT_MAX_NODES,
        log: false,
        book_hint: None,
        max_id_depth: cmp_depth,
        cert_enabled: None,
    };
    let t1 = Instant::now();
    let report = run_search(&mut session, config);
    let deep_ms = t1.elapsed().as_millis();

    let Some(report) = report else {
        println!("position is terminal — nothing to rank");
        return;
    };
    let mut deep: Vec<_> = report.root_moves.clone();
    deep.sort_by(|a, b| b.score.cmp(&a.score));

    // Rank lookup tables keyed by algebraic notation.
    let deep_rank: std::collections::HashMap<String, usize> = deep
        .iter()
        .enumerate()
        .map(|(r, m)| (m.mv.clone(), r))
        .collect();
    let roll_algeb: Vec<String> = ranks.iter().map(|r| format_move(r.mv)).collect();

    // ── Agreement metrics ───────────────────────────────────────────────────
    let roll_top = roll_algeb.first().cloned().unwrap_or_default();
    let deep_top = deep.first().map(|m| m.mv.clone()).unwrap_or_default();
    let top1 = roll_top == deep_top;
    let top3 = deep
        .iter()
        .take(3)
        .any(|m| m.mv == roll_top);

    // Spearman over the moves common to both lists. Deep search root-filters to
    // a handful of candidates, so re-rank BOTH lists densely within that common
    // subset (0..n-1) — otherwise rollout ranks 0..130 vs deep ranks 0..18 make
    // the d² term meaningless.
    let mut common_moves: Vec<(usize, usize)> = Vec::new(); // (roll_pos, deep_pos)
    for (roll_r, alg) in roll_algeb.iter().enumerate() {
        if let Some(&deep_r) = deep_rank.get(alg) {
            common_moves.push((roll_r, deep_r));
        }
    }
    let common = common_moves.len();
    // Dense rank within the common subset for each ordering.
    let mut by_roll: Vec<usize> = (0..common).collect();
    by_roll.sort_by_key(|&k| common_moves[k].0);
    let mut by_deep: Vec<usize> = (0..common).collect();
    by_deep.sort_by_key(|&k| common_moves[k].1);
    let mut roll_dense = vec![0usize; common];
    let mut deep_dense = vec![0usize; common];
    for (rank, &k) in by_roll.iter().enumerate() {
        roll_dense[k] = rank;
    }
    for (rank, &k) in by_deep.iter().enumerate() {
        deep_dense[k] = rank;
    }
    let mut d2_sum = 0.0f64;
    for k in 0..common {
        let d = roll_dense[k] as f64 - deep_dense[k] as f64;
        d2_sum += d * d;
    }
    let spearman = if common > 1 {
        1.0 - (6.0 * d2_sum) / (common as f64 * ((common * common - 1) as f64))
    } else {
        f64::NAN
    };

    // ── Report ──────────────────────────────────────────────────────────────
    println!("=== rollout vs deep-search root ranking ===");
    println!(
        "sims/move={sims} plies={plies} cmp-depth={cmp_depth} seed={seed} \
         | rollout {roll_ms}ms, deep {deep_ms}ms ({} nodes)",
        report.nodes
    );
    println!(
        "top-1 match: {} | rollout #1 in deep top-3: {} | Spearman ρ = {:.3} (n={common})",
        if top1 { "YES" } else { "no" },
        if top3 { "YES" } else { "no" },
        spearman
    );
    println!();
    println!("  {:<5} {:<8} {:<8} {:<8} {:<8} {:<8}", "rank", "roll", "q", "prior", "deep@", "deepScore");
    for (r, rk) in ranks.iter().take(12).enumerate() {
        let alg = &roll_algeb[r];
        let dr = deep_rank
            .get(alg)
            .map(|x| x.to_string())
            .unwrap_or_else(|| "-".to_string());
        let ds = deep
            .iter()
            .find(|m| &m.mv == alg)
            .map(|m| m.score.to_string())
            .unwrap_or_else(|| "-".to_string());
        println!(
            "  {:<5} {:<8} {:<8.3} {:<8.3} {:<8} {:<8}",
            r, alg, rk.q, rk.prior, dr, ds
        );
    }
}

/// STRENGTH MATCH — Titanium+endgame-certificate vs plain Titanium, head to
/// head over `--games` games. Measures whether the v13 endgame proof oracle
/// makes the engine *win more*, not whether it searches the same nodes.
///
/// Each color-swapped PAIR shares one seeded random opening, so the two configs
/// face identical positions with both colors (fair + varied — a deterministic
/// engine plays the same game every time from a fixed start otherwise).
///
/// Usage: `titanium match [--games N] [--time SEC] [--seed S] [--open PLIES]
///         [--maxply N]`
fn run_match(args: &[String]) {
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
    use std::sync::Mutex;
    use titanium::search::alphabeta::CERT_PROOFS;

    let mut games = 100usize;
    let mut time_sec = 2.0f64;
    let mut seed = 1u64;
    let mut open_plies = 4u32;
    let mut max_ply = 200u32;
    let mut threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let mut engine_a = MatchEngine::TitaniumCert;
    let mut engine_b = MatchEngine::TitaniumPlain;
    let mut tt_bits: Option<usize> = None;
    let mut i = 2usize;
    while i < args.len() {
        match args[i].as_str() {
            "--games"  => { if let Some(v) = args.get(i+1).and_then(|s| s.parse().ok()) { games = v; i += 2; continue; } }
            "--time"   => { if let Some(v) = args.get(i+1).and_then(|s| s.parse().ok()) { time_sec = v; i += 2; continue; } }
            "--seed"   => { if let Some(v) = args.get(i+1).and_then(|s| s.parse().ok()) { seed = v; i += 2; continue; } }
            "--open"   => { if let Some(v) = args.get(i+1).and_then(|s| s.parse().ok()) { open_plies = v; i += 2; continue; } }
            "--maxply" => { if let Some(v) = args.get(i+1).and_then(|s| s.parse().ok()) { max_ply = v; i += 2; continue; } }
            "--threads"=> { if let Some(v) = args.get(i+1).and_then(|s| s.parse().ok()) { threads = v; i += 2; continue; } }
            "--tt-bits"=> { if let Some(v) = args.get(i+1).and_then(|s| s.parse().ok()) { tt_bits = Some(v); i += 2; continue; } }
            "--a"      => { if let Some(e) = args.get(i+1).and_then(|s| MatchEngine::parse(s)) { engine_a = e; i += 2; continue; } }
            "--b" | "--vs" => { if let Some(e) = args.get(i+1).and_then(|s| MatchEngine::parse(s)) { engine_b = e; i += 2; continue; } }
            _ => {}
        }
        i += 1;
    }
    let time_ms = (time_sec * 1000.0).round() as u64;

    // Halve threads: each "slot" runs TWO sequential games (a color-swapped pair).
    // So `threads` parallel slots = `threads * 2` games at once.
    let pair_threads = threads.max(1);
    rayon::ThreadPoolBuilder::new()
        .num_threads(pair_threads)
        .build_global()
        .ok();

    let a_w = AtomicU32::new(0);
    let b_w = AtomicU32::new(0);
    let draws   = AtomicU32::new(0);
    let cert_touched = AtomicU64::new(0);
    let games_done   = AtomicU32::new(0);
    let started = Instant::now();
    let print_mu: Mutex<()> = Mutex::new(());

    // Round pairs up to nearest multiple of thread count so every batch fills
    // all cores — no thread sits idle waiting for 1 straggler to finish.
    let raw_pairs = (games + 1) / 2;
    let pairs = raw_pairs.div_ceil(pair_threads) * pair_threads;
    let games = pairs * 2;

    let tt_note = tt_bits.map(|b| format!(", tt-bits={b}")).unwrap_or_default();
    eprintln!(
        "match: {games} games @ {time_sec}s/move, open={open_plies} plies, \
         maxply={max_ply}, threads={pair_threads}{tt_note}"
    );
    eprintln!("  A = {}   B = {}", engine_a.label(), engine_b.label());

    (0..pairs).into_par_iter().for_each(|pair| {
        let opening = match_random_opening(seed.wrapping_add(pair as u64), open_plies);

        for flip in 0..2u32 {
            let game_idx = pair * 2 + flip as usize;
            if game_idx >= games { break; }
            // Swap colors per game in the pair so the opening is played from both
            // sides — `a_is_one` true means engine A holds Player::One this game.
            let a_is_one = flip == 0;

            let proofs_before = CERT_PROOFS.load(Ordering::Relaxed);
            let outcome =
                play_one_game(&opening, a_is_one, time_ms, max_ply, engine_a, engine_b, tt_bits);
            if CERT_PROOFS.load(Ordering::Relaxed) > proofs_before {
                cert_touched.fetch_add(1, Ordering::Relaxed);
            }

            match outcome {
                Some(titanium::Player::One) => {
                    if a_is_one { a_w.fetch_add(1, Ordering::Relaxed); }
                    else        { b_w.fetch_add(1, Ordering::Relaxed); }
                }
                Some(titanium::Player::Two) => {
                    if a_is_one { b_w.fetch_add(1, Ordering::Relaxed); }
                    else        { a_w.fetch_add(1, Ordering::Relaxed); }
                }
                None => { draws.fetch_add(1, Ordering::Relaxed); }
            }

            let played = games_done.fetch_add(1, Ordering::Relaxed) + 1;
            let aw = a_w.load(Ordering::Relaxed);
            let bw = b_w.load(Ordering::Relaxed);
            let dr = draws.load(Ordering::Relaxed);
            let ct = cert_touched.load(Ordering::Relaxed);
            let score = aw as f64 + 0.5 * dr as f64;
            let _g = print_mu.lock().unwrap();
            eprintln!(
                "  [{played}/{games}] A {aw} - {bw} B  ({dr} draws)  \
                 A-score {score:.1}/{played}  cert-touched {ct} games  \
                 {:.0}s elapsed",
                started.elapsed().as_secs_f64()
            );
        }
    });

    let aw = a_w.load(Ordering::Relaxed);
    let bw = b_w.load(Ordering::Relaxed);
    let dr = draws.load(Ordering::Relaxed);
    let ct = cert_touched.load(Ordering::Relaxed);
    let n = games as f64;
    let score = aw as f64 + 0.5 * dr as f64;
    let p = score / n;
    let se = (p * (1.0 - p) / n).sqrt();
    let elo = if p > 0.0 && p < 1.0 {
        -400.0 * ((1.0 - p) / p).log10()
    } else {
        f64::INFINITY * (p - 0.5).signum()
    };
    println!("=== STRENGTH MATCH RESULT ===");
    println!("A = {},  B = {}{tt_note}", engine_a.label(), engine_b.label());
    println!("games {games} @ {time_sec}s/move");
    println!("A wins {aw}  |  B wins {bw}  |  draws {dr}");
    println!(
        "A score {score:.1}/{games} = {:.1}% (±{:.1}%)  →  ~{elo:+.0} Elo",
        p * 100.0, se * 196.0
    );
    println!("Titanium-cert fired in {ct}/{games} games");
}

/// Build a seeded random legal opening of `plies` moves from startpos.
fn match_random_opening(seed: u64, plies: u32) -> Vec<String> {
    use titanium::{generate_legal_moves, Board};
    let mut board = Board::new();
    let mut state = seed | 1;
    let mut next = || {
        // xorshift64*
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        state.wrapping_mul(0x2545_F491_4F6C_DD1D)
    };
    let mut out = Vec::new();
    for _ in 0..plies {
        if board.is_terminal().is_some() {
            break;
        }
        let moves = generate_legal_moves(&board);
        if moves.is_empty() {
            break;
        }
        let pick = (next() as usize) % moves.len();
        let mv = moves[pick];
        out.push(format_move(mv));
        board.apply_move(mv);
    }
    out
}

/// A selectable engine for either side of a match.
#[derive(Clone, Copy, PartialEq, Eq)]
enum MatchEngine {
    /// Titanium αβ + endgame certificate (distance eval).
    TitaniumCert,
    /// Titanium αβ, certificate disabled.
    TitaniumPlain,
    /// gen13 net search, O1 movegen only (the strong baseline).
    AceV13,
    /// gen13 net search + cheap hands-empty cert ONLY (no CAT). Isolates the
    /// certificate contribution from CAT pruning.
    AceV13Cert,
    /// gen13 net search + adaptive cache-tier TT ONLY. Isolates TT growth.
    AceV13AdaptiveTt,
    /// Production graft: gen13 net + cheap hands-empty cert + adaptive cache-tier
    /// TT (NO CAT — measured −25 Elo on the net engine).
    AceV13Grafted,
}

impl MatchEngine {
    fn parse(s: &str) -> Option<MatchEngine> {
        match s {
            "titanium" | "titanium-cert" => Some(MatchEngine::TitaniumCert),
            "titanium-plain" => Some(MatchEngine::TitaniumPlain),
            "ace-v13" => Some(MatchEngine::AceV13),
            "ace-v13-cert" => Some(MatchEngine::AceV13Cert),
            "ace-v13-att" | "ace-v13-adaptive-tt" => Some(MatchEngine::AceV13AdaptiveTt),
            "ace-v13-grafted" | "grafted" => Some(MatchEngine::AceV13Grafted),
            _ => None,
        }
    }
    fn label(self) -> &'static str {
        match self {
            MatchEngine::TitaniumCert => "Titanium+cert",
            MatchEngine::TitaniumPlain => "plain Titanium",
            MatchEngine::AceV13 => "ace-v13 (O1 movegen)",
            MatchEngine::AceV13Cert => "ace-v13 + cheap-cert (no CAT)",
            MatchEngine::AceV13AdaptiveTt => "ace-v13 + adaptive cache-tier TT",
            MatchEngine::AceV13Grafted => "ace-v13 grafted (cheap-cert + adaptive-TT)",
        }
    }
}

/// One side's warm engine state (TT/killers/history persist across the game).
enum Seat {
    Titanium {
        session: titanium::GameSearchSession,
        cert: bool,
    },
    Ace {
        search: Box<titanium::acev13::AceSearch>,
    },
}

impl Seat {
    fn new(engine: MatchEngine, opening: &[String], tt_bits: Option<usize>) -> Seat {
        use titanium::acev13::{algebraic_to_ace, AceGame, AceSearch};
        match engine {
            MatchEngine::TitaniumCert => Seat::Titanium {
                session: titanium::GameSearchSession::new(),
                cert: true,
            },
            MatchEngine::TitaniumPlain => Seat::Titanium {
                session: titanium::GameSearchSession::new(),
                cert: false,
            },
            MatchEngine::AceV13
            | MatchEngine::AceV13Cert
            | MatchEngine::AceV13AdaptiveTt
            | MatchEngine::AceV13Grafted => {
                let mut g = AceGame::new();
                for m in opening {
                    g.make_move(algebraic_to_ace(m));
                }
                let search = match engine {
                    MatchEngine::AceV13Grafted => AceSearch::grafted(g, tt_bits),
                    MatchEngine::AceV13Cert => AceSearch::with_ti_movegen_cheap_cert(g, tt_bits),
                    MatchEngine::AceV13AdaptiveTt => AceSearch::with_ti_movegen_adaptive_tt(g),
                    _ => {
                        // Plain ace-v13 — but still honor --tt-bits so a pure TT-size
                        // experiment (ace-v13 @ N bits vs ace-v13 default) is possible.
                        let mut s = AceSearch::with_ti_movegen(g);
                        if let Some(bits) = tt_bits {
                            s.resize_tt(bits);
                        }
                        s
                    }
                };
                Seat::Ace { search }
            }
        }
    }

    /// Pick a move for the current position (`moves` = full move list so far).
    fn pick(&mut self, moves: &[String], time_ms: u64) -> Option<String> {
        use titanium::{SearchConfig, DEFAULT_MAX_NODES};
        match self {
            Seat::Titanium { session, cert } => {
                session.set_position(moves).ok()?;
                let config = SearchConfig {
                    time_ms,
                    max_nodes: DEFAULT_MAX_NODES,
                    log: false,
                    book_hint: None,
                    cert_enabled: Some(*cert),
                    ..SearchConfig::default()
                };
                let report = run_search(session, config)?;
                Some(format_move(report.best_move))
            }
            Seat::Ace { search } => {
                let r = search.think(time_ms, 30, false, false, "match");
                if r.mv == titanium::acev13::ACE_NO_MOVE {
                    return None;
                }
                Some(titanium::acev13::ace_to_algebraic(r.mv))
            }
        }
    }

    /// Advance incremental state by one applied move (Ace keeps its TT warm; the
    /// Titanium seat re-syncs from the move list each `pick`, so it's a no-op).
    fn observe(&mut self, alg: &str) {
        if let Seat::Ace { search } = self {
            search.apply_move(titanium::acev13::algebraic_to_ace(alg));
        }
    }
}

/// Play one full game from `opening`. `a_is_one` decides which engine holds
/// Player::One. Returns the winner, or `None` for an adjudicated draw at the
/// ply cap (closer pawn wins; equal distance = draw).
fn play_one_game(
    opening: &[String],
    a_is_one: bool,
    time_ms: u64,
    max_ply: u32,
    engine_a: MatchEngine,
    engine_b: MatchEngine,
    tt_bits: Option<usize>,
) -> Option<titanium::Player> {
    use titanium::{Board, Player};

    let mut moves: Vec<String> = opening.to_vec();
    let mut board = Board::new();
    for m in &moves {
        board.apply_algebraic(m);
    }

    // `--tt-bits` configures the A-side candidate only; B always uses the engine
    // default, so a run like `--a ace-v13 --b ace-v13 --tt-bits 22` isolates TT size.
    let mut seat_a = Seat::new(engine_a, opening, tt_bits);
    let mut seat_b = Seat::new(engine_b, opening, None);

    for _ in 0..max_ply {
        if let Some(w) = board.is_terminal() {
            return Some(w);
        }
        let a_to_move = (board.side() == Player::One) == a_is_one;
        let alg = {
            let seat = if a_to_move { &mut seat_a } else { &mut seat_b };
            seat.pick(&moves, time_ms)?
        };
        board.apply_algebraic(&alg);
        seat_a.observe(&alg);
        seat_b.observe(&alg);
        moves.push(alg);
    }

    // Ply cap — adjudicate by shortest path.
    let mut bfs = titanium::BfsScratch::new();
    let d_one = bfs.shortest_distance(&board, Player::One).unwrap_or(255);
    let d_two = bfs.shortest_distance(&board, Player::Two).unwrap_or(255);
    match d_one.cmp(&d_two) {
        std::cmp::Ordering::Less => Some(Player::One),
        std::cmp::Ordering::Greater => Some(Player::Two),
        std::cmp::Ordering::Equal => None,
    }
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
            | "ace-v11-ti-pmc" | "ace-pmc" | "ace-v13" | "ace-v13-ti" | "ace-v13-ti-pmc"
            | "ace-v13-pure" => Some(w[1].as_str()),
            _ => None,
        }
    })
}

fn ace_engine_mode(flag: &str) -> &'static str {
    match flag {
        "ace-cat" => "ace-cat",
        // gen13: the headline `ace-v13` is the OPTIMIZED engine — it uses the
        // Titanium O1 movegen. `ace-v13-pure` is the faithful 1:1 (native ACE
        // `wall_legal` movegen) kept as the JS-matching reference.
        "ace-ti" | "ace-v8-ti" | "ace-v8-ti-pmc" | "ace-v10-ti" | "ace-v10-ti-pmc"
        | "ace-v11-ti" | "ace-v11-ti-pmc" | "ace-v13" | "ace-v13-ti" | "ace-v13-ti-pmc" => {
            "ace-ti"
        }
        _ => "ace",
    }
}

/// gen13 engine (`ACEV13.html` port in `crate::acev13`) vs the v11 `crate::ace`.
fn is_acev13(flag: &str) -> bool {
    flag.starts_with("ace-v13")
}

fn is_ace_engine(args: &[String]) -> bool {
    ace_engine_flag(args).is_some()
}

fn run_genmove_ace(args: &[String]) {
    let label = ace_engine_flag(args).unwrap_or("ace");
    let mode = ace_engine_mode(label);
    let cat = mode == "ace-cat";
    let ti_movegen = mode == "ace-ti";
    let eme0 = label.contains("pmc");
    let mut time_ms = 4000u64;
    let mut max_depth = 30i32;
    let mut full = false;
    let mut log = false;
    let mut eme = eme0;
    let mut moves = Vec::new();
    let mut i = 2usize;
    while i < args.len() {
        let arg = &args[i];
        if arg == "--time" {
            if let Some(sec) = args.get(i + 1).and_then(|s| s.parse::<f64>().ok()) {
                time_ms = (sec * 1000.0) as u64;
                i += 2;
                continue;
            }
        } else if arg == "--depth" {
            if let Some(d) = args.get(i + 1).and_then(|s| s.parse::<i32>().ok()) {
                max_depth = d;
                i += 2;
                continue;
            }
        } else if arg == "--full" {
            full = true;
            i += 1;
            continue;
        } else if arg == "--log" {
            log = true;
            i += 1;
            continue;
        } else if arg == "--eme" || arg == "--pseudo-mcts" {
            eme = true;
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

    // Both modules expose an identical `AceParams` / `ace_genmove` surface, so
    // the output handling is shared via a macro (field names match) and only
    // the module path differs between the v11 (`ace`) and gen13 (`acev13`) ports.
    macro_rules! emit_genmove {
        ($module:path) => {{
            use $module as ace_mod;
            let params = ace_mod::AceParams {
                cat,
                ti_movegen,
                eme,
                time_ms,
                max_depth,
                full,
                log,
                ..Default::default()
            };
            match ace_mod::ace_genmove(&moves, params, label) {
                Some((algebraic, info)) => {
                    if !log {
                        let mut depth_json = String::new();
                        for (j, e) in info.depth_log.iter().enumerate() {
                            if j > 0 {
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
                            label, label, info.depth, info.nodes, info.score,
                            info.white_dist, info.black_dist, info.ms, depth_json
                        );
                    }
                    println!("bestmove {}", algebraic);
                }
                None => println!("bestmove (none)"),
            }
        }};
    }

    if is_acev13(label) {
        emit_genmove!(titanium::acev13);
    } else {
        emit_genmove!(titanium::ace);
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
