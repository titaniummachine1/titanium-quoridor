//! UCI-style stdio protocol for Quoridor (Phase 1 of distributed testing plan).
//!
//! Supported commands:
//!   `uci`                                 → id + options + `uciok`
//!   `isready`                             → `readyok`
//!   `ucinewgame`                          → reset session (TT, killers, history)
//!   `position startpos [moves m1 m2 ...]` → set position (Quoridor algebraic moves)
//!   `go [movetime MS] [nodes N] [depth D]`→ search, print `bestmove MOVE`
//!   `quit`                                → exit
//!
//! Search progress is emitted on stdout as `info ...` lines.
//! Moves use the engine's native algebraic notation (pawn: `e2`; wall: `e3h`/`e3v`).

use std::io::{self, BufRead, Write};

use crate::search::alphabeta::{
    run_search, SearchConfig, DEFAULT_MAX_ID_DEPTH, DEFAULT_MAX_NODES, DEFAULT_TIME_MS,
};
use crate::search::session::GameSearchSession;
use crate::util::perft::format_move;

const ENGINE_NAME: &str = concat!("Titanium ", env!("CARGO_PKG_VERSION"));
const ENGINE_AUTHOR: &str = "terminator";

fn flush(stdout: &mut io::Stdout) {
    let _ = stdout.flush();
}

fn handle_uci(stdout: &mut io::Stdout) {
    let _ = writeln!(stdout, "id name {ENGINE_NAME}");
    let _ = writeln!(stdout, "id author {ENGINE_AUTHOR}");
    let _ = writeln!(stdout, "option name MoveTime type spin default {DEFAULT_TIME_MS} min 1 max 600000");
    let _ = writeln!(stdout, "uciok");
    flush(stdout);
}

fn handle_position(session: &mut GameSearchSession, parts: &[&str], stdout: &mut io::Stdout) {
    // Accept: `position startpos`, `position startpos moves e2 e8 ...`
    // Also tolerate `position moves ...` and bare `position e2 e8` for convenience.
    let mut moves: Vec<String> = Vec::new();
    let mut idx = 1;
    if parts.get(idx) == Some(&"startpos") {
        idx += 1;
    }
    if parts.get(idx) == Some(&"moves") {
        idx += 1;
    }
    moves.extend(parts[idx..].iter().map(|s| (*s).to_string()));

    match session.set_position(&moves) {
        Ok(_) => {}
        Err(msg) => {
            let _ = writeln!(stdout, "info string error {msg}");
            flush(stdout);
        }
    }
}

fn handle_go(session: &mut GameSearchSession, parts: &[&str], stdout: &mut io::Stdout) {
    if session.board.is_terminal().is_some() {
        let _ = writeln!(stdout, "info string terminal position");
        let _ = writeln!(stdout, "bestmove (none)");
        flush(stdout);
        return;
    }

    let mut time_ms: u64 = DEFAULT_TIME_MS;
    let mut max_nodes: u64 = DEFAULT_MAX_NODES;
    let mut max_depth: u32 = DEFAULT_MAX_ID_DEPTH;

    let mut i = 1;
    while i < parts.len() {
        match parts[i] {
            "movetime" => {
                if let Some(v) = parts.get(i + 1).and_then(|s| s.parse().ok()) {
                    time_ms = v;
                }
                i += 2;
            }
            "nodes" => {
                if let Some(v) = parts.get(i + 1).and_then(|s| s.parse().ok()) {
                    max_nodes = v;
                }
                i += 2;
            }
            "depth" => {
                if let Some(v) = parts.get(i + 1).and_then(|s| s.parse().ok()) {
                    max_depth = v;
                }
                i += 2;
            }
            "infinite" => {
                time_ms = 600_000;
                i += 1;
            }
            // wtime/btime/winc/binc accepted but unused (Quoridor batches use movetime)
            "wtime" | "btime" | "winc" | "binc" | "movestogo" => i += 2,
            _ => i += 1,
        }
    }

    let config = SearchConfig {
        time_ms: time_ms.max(1),
        max_nodes,
        log: true,
        book_hint: None,
        max_id_depth: max_depth,
    };

    match run_search(session, config) {
        Some(report) => {
            let _ = writeln!(stdout, "bestmove {}", format_move(report.best_move));
        }
        None => {
            let _ = writeln!(stdout, "bestmove (none)");
        }
    }
    flush(stdout);
}

/// Blocking UCI REPL — `titanium uci`.
pub fn run_uci_stdio() {
    let mut session = GameSearchSession::new();
    let stdin = io::stdin();
    let mut stdout = io::stdout();

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let parts: Vec<&str> = trimmed.split_whitespace().collect();
        match parts[0] {
            "uci" => handle_uci(&mut stdout),
            "isready" => {
                // Cold-start: build the pawn lookup tables now so they're ready
                // before any search/perft — `readyok` means truly ready.
                crate::movegen::prewarm();
                let _ = writeln!(stdout, "readyok");
                flush(&mut stdout);
            }
            "ucinewgame" => session.reset(),
            "position" => handle_position(&mut session, &parts, &mut stdout),
            "go" => handle_go(&mut session, &parts, &mut stdout),
            "stop" => { /* searches are synchronous; nothing to stop */ }
            "quit" => break,
            _ => {
                let _ = writeln!(stdout, "info string unknown command {}", parts[0]);
                flush(&mut stdout);
            }
        }
    }
}
