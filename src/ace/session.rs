//! Long-lived ACE session REPL — one process per web UI engine seat.
//!
//! Same wire protocol as `search::session_stdio` (`reset` / `position` /
//! `makemove` / `go TIME_SEC [MAX_NODES]` / `quit`), but holds a single
//! `AceSearch` for the whole game so the TT, killers, history, and
//! countermove tables stay warm between plies — each think starts from
//! the previous search's analysis instead of cold.

use std::io::{self, BufRead, Write};

use super::{algebraic_to_ace, ace_to_algebraic, AceGame, AceSearch};

fn reply_ready(stdout: &mut io::Stdout) {
    let _ = writeln!(stdout, "ready");
    let _ = stdout.flush();
}

fn reply_error(stdout: &mut io::Stdout, message: &str) {
    let _ = writeln!(stdout, "error {}", message);
    let _ = stdout.flush();
}

fn build_search(engine_flag: &str, g: AceGame) -> Box<AceSearch> {
    let mut search = match engine_flag {
        "ace-cat" => AceSearch::with_cat(g),
        "ace-ti" | "ace-v8-ti" | "ace-v8-ti-pmc" | "ace-v10-ti" | "ace-v10-ti-pmc"
        | "ace-v11-ti" | "ace-v11-ti-pmc" => {
            AceSearch::with_ti_movegen(g)
        }
        _ => AceSearch::new(g),
    };
    if engine_flag.contains("pmc") {
        search.enable_eme();
    }
    search
}

fn replay(moves: &[String]) -> Result<AceGame, String> {
    let mut g = AceGame::new();
    for text in moves {
        if g.winner() >= 0 {
            return Err(format!("move {text} past terminal position"));
        }
        g.make_move(algebraic_to_ace(text));
    }
    Ok(g)
}

/// Blocking REPL holding one warm `AceSearch` for the process lifetime.
pub fn run_ace_session_stdio(engine_flag: &str) {
    let mut search = build_search(engine_flag, AceGame::new());
    let mut applied: Vec<String> = Vec::new();
    let stdin = io::stdin();
    let mut stdout = io::stdout();

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                reply_error(&mut stdout, &e.to_string());
                break;
            }
        };
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let parts: Vec<&str> = trimmed.split_whitespace().collect();
        match parts[0] {
            "reset" => {
                search.set_position(AceGame::new());
                applied.clear();
                reply_ready(&mut stdout);
            }
            "position" => {
                let moves: Vec<String> = parts[1..].iter().map(|s| (*s).to_string()).collect();
                let extends = moves.len() >= applied.len()
                    && moves.iter().zip(applied.iter()).all(|(a, b)| a == b);
                if extends {
                    // common case: game advanced — push only the new plies,
                    // the search state stays fully warm.
                    let mut err = None;
                    for text in &moves[applied.len()..] {
                        if search.g.winner() >= 0 {
                            err = Some(format!("move {text} past terminal position"));
                            break;
                        }
                        search.apply_move(algebraic_to_ace(text));
                    }
                    if let Some(msg) = err {
                        reply_error(&mut stdout, &msg);
                        continue;
                    }
                } else {
                    // undo / divergence — rebuild the board, keep the TT.
                    match replay(&moves) {
                        Ok(g) => search.set_position(g),
                        Err(msg) => {
                            reply_error(&mut stdout, &msg);
                            continue;
                        }
                    }
                }
                applied = moves;
                let _ = writeln!(stdout, "ready {}", applied.len());
                let _ = stdout.flush();
            }
            "makemove" => {
                let Some(mv) = parts.get(1) else {
                    reply_error(&mut stdout, "makemove requires a move");
                    continue;
                };
                if search.g.winner() >= 0 {
                    reply_error(&mut stdout, "terminal position");
                    continue;
                }
                search.apply_move(algebraic_to_ace(mv));
                applied.push((*mv).to_string());
                reply_ready(&mut stdout);
            }
            "go" => {
                if search.g.winner() >= 0 {
                    reply_error(&mut stdout, "terminal position");
                    continue;
                }
                let time_sec: f64 = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(4.0);
                let time_ms = (time_sec * 1000.0).max(1.0) as u64;
                let result = search.think(time_ms, 30, false, true, engine_flag);
                if result.mv == 0 {
                    let _ = writeln!(stdout, "bestmove (none)");
                } else {
                    let _ = writeln!(stdout, "bestmove {}", ace_to_algebraic(result.mv));
                }
                let _ = stdout.flush();
            }
            "quit" => break,
            _ => reply_error(&mut stdout, "unknown command"),
        }
    }
}
