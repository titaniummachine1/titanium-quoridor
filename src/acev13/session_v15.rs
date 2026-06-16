//! **Titanium v15 engine session** — two-thread design: I/O on main thread,
//! search daemon thread.
//!
//! Titanium v15 is the production engine: grafts Titanium O1 movegen,
//! adaptive TT, win-certificate solver, and incremental HalfPW accumulator
//! onto the gen13 search core.  This session adds continuous-search support
//! on top of the standard game-server protocol.
//!
//! ## Protocol
//!
//! ### Standard commands (compatible with self_match.js and run_overnight.bat)
//!   reset / position [MOVES] / makemove MOVE / go TIME_SEC / quit
//!
//! ### Titanium v15 infinite-search extensions
//!   go infinite [PONDER_MOVE]   — start pondering; applies PONDER_MOVE first if given
//!   stop                        — stop pondering; replies "bestmove MOVE"
//!   ponderhit TIME_MS           — ponder move was correct; think for TIME_MS; replies "bestmove MOVE"
//!   movemiss MOVE TIME_MS       — opponent played MOVE (unexpected);
//!                                 migrate root and think for TIME_MS; replies "bestmove MOVE"
//!
//! For `go TIME_SEC` the daemon does a single blocking think and replies
//! "bestmove MOVE" — the standard path used by self_match.js.

use std::io::{self, BufRead, Write};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;

use super::{ace_to_algebraic, algebraic_to_ace, ACE_NO_MOVE, AceGame, AceSearch};

// ── Inter-thread messages ─────────────────────────────────────────────────────

enum Cmd {
    /// Replace the engine position (no I/O reply needed — I/O thread handles "ready").
    SetGame(AceGame),
    /// Timed search: think for `time_ms` and reply BestMove.
    GoTimed(u64),
    /// Start pondering on the current position (pre-apply `ponder_mv` if given).
    GoInfinite(i16),
    /// Stop pondering; reply with the last best move from the ponder search.
    StopAndGet,
    /// Ponder was correct: think for `time_ms` from the current (ponder) position.
    PonderHit(u64),
    /// Ponder was wrong: reset to `new_game` then think for `time_ms`.
    MoveMiss { new_game: AceGame, time_ms: u64 },
    Quit,
}

enum Reply {
    BestMove(i16),
    Error(String),
}

// ── Search daemon ─────────────────────────────────────────────────────────────

fn build_search(engine_flag: &str, g: AceGame) -> Box<AceSearch> {
    let mut search = match engine_flag {
        "ace-v13-pure"    => AceSearch::new(g),
        "ace-v13-ti-pure" => AceSearch::with_ti_movegen_pure(g),
        _                 => AceSearch::grafted(g, None),
    };
    if engine_flag.contains("pmc") {
        search.enable_eme();
    }
    search
}

fn search_daemon(engine_flag: String, rx: Receiver<Cmd>, tx: Sender<Reply>) {
    let mut search = build_search(&engine_flag, AceGame::new());
    let mut last_score: i32 = 0;
    let label = engine_flag.as_str();

    loop {
        let cmd = match rx.recv() {
            Ok(c) => c,
            Err(_) => return,
        };
        match cmd {
            Cmd::SetGame(g) => {
                search.set_position(g);
            }
            Cmd::GoTimed(time_ms) => {
                let r = search.think(time_ms, 30, false, true, label);
                last_score = r.score;
                let _ = tx.send(Reply::BestMove(r.mv));
            }
            Cmd::GoInfinite(ponder_mv) => {
                if ponder_mv != ACE_NO_MOVE {
                    search.apply_move(ponder_mv);
                }
                // Think in 100 ms chunks until interrupted.
                let mut last_mv = ACE_NO_MOVE;
                loop {
                    let r = search.think(100, 30, false, false, label);
                    if r.mv != ACE_NO_MOVE {
                        last_mv = r.mv;
                        last_score = r.score;
                    }
                    match rx.try_recv() {
                        Ok(Cmd::StopAndGet) => {
                            let _ = tx.send(Reply::BestMove(last_mv));
                            break;
                        }
                        Ok(Cmd::PonderHit(time_ms)) => {
                            // Already at the ponder position — just think for real.
                            let r2 = search.think(time_ms, 30, false, true, label);
                            last_score = r2.score;
                            let _ = tx.send(Reply::BestMove(r2.mv));
                            break;
                        }
                        Ok(Cmd::MoveMiss { new_game, time_ms }) => {
                            // Opponent played something unexpected.
                            search.set_position(new_game);
                            search.decay_history_by_surprise(last_score);
                            let r2 = search.think(time_ms, 30, false, true, label);
                            last_score = r2.score;
                            let _ = tx.send(Reply::BestMove(r2.mv));
                            break;
                        }
                        Ok(Cmd::Quit) | Err(mpsc::TryRecvError::Disconnected) => return,
                        Ok(Cmd::SetGame(g)) => {
                            // Position update mid-ponder — restart.
                            search.set_position(g);
                            last_mv = ACE_NO_MOVE;
                        }
                        Ok(_) | Err(mpsc::TryRecvError::Empty) => {}
                    }
                }
            }
            Cmd::StopAndGet => {
                // Not pondering — nothing to return.
                let _ = tx.send(Reply::BestMove(ACE_NO_MOVE));
            }
            Cmd::PonderHit(time_ms) => {
                let r = search.think(time_ms, 30, false, true, label);
                last_score = r.score;
                let _ = tx.send(Reply::BestMove(r.mv));
            }
            Cmd::MoveMiss { new_game, time_ms } => {
                search.set_position(new_game);
                let r = search.think(time_ms, 30, false, true, label);
                last_score = r.score;
                let _ = tx.send(Reply::BestMove(r.mv));
            }
            Cmd::Quit => return,
        }
    }
}

// ── I/O loop ──────────────────────────────────────────────────────────────────

pub fn run_v15_session_stdio(engine_flag: &str) {
    let (cmd_tx, cmd_rx) = mpsc::channel::<Cmd>();
    let (reply_tx, reply_rx) = mpsc::channel::<Reply>();

    let flag_owned = engine_flag.to_string();
    thread::spawn(move || search_daemon(flag_owned, cmd_rx, reply_tx));

    let stdin = io::stdin();
    let mut stdout = io::stdout();
    let mut applied: Vec<String> = Vec::new();
    // Track game state in I/O thread for position management.
    let mut current_g = AceGame::new();
    // Move the engine was asked to ponder on (ACE_NO_MOVE if none).
    let mut ponder_mv: i16 = ACE_NO_MOVE;

    macro_rules! ok {
        ($msg:expr) => {{
            let _ = writeln!(stdout, "{}", $msg);
            let _ = stdout.flush();
        }};
    }
    macro_rules! err {
        ($msg:expr) => {{
            let _ = writeln!(stdout, "error {}", $msg);
            let _ = stdout.flush();
        }};
    }

    fn replay_moves(moves: &[String]) -> Result<AceGame, String> {
        let mut g = AceGame::new();
        for text in moves {
            if g.winner() >= 0 {
                return Err(format!("move {text} past terminal position"));
            }
            g.make_move(algebraic_to_ace(text));
        }
        Ok(g)
    }

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(e) => { err!(e); break; }
        };
        let trimmed = line.trim();
        if trimmed.is_empty() { continue; }
        let parts: Vec<&str> = trimmed.splitn(4, ' ').collect();

        match parts[0] {
            "reset" => {
                current_g = AceGame::new();
                applied.clear();
                ponder_mv = ACE_NO_MOVE;
                let _ = cmd_tx.send(Cmd::SetGame(AceGame::new()));
                ok!("ready");
            }
            "position" => {
                let moves: Vec<String> = if parts.len() > 1 {
                    parts[1..].join(" ").split_whitespace().map(String::from).collect()
                } else {
                    Vec::new()
                };
                let extends = !applied.is_empty()
                    && moves.len() >= applied.len()
                    && moves.iter().zip(applied.iter()).all(|(a, b)| a == b);
                if extends {
                    let mut err = None;
                    for text in &moves[applied.len()..] {
                        if current_g.winner() >= 0 {
                            err = Some(format!("move {text} past terminal position"));
                            break;
                        }
                        current_g.make_move(algebraic_to_ace(text));
                    }
                    if let Some(msg) = err {
                        err!(msg);
                        continue;
                    }
                    // Incremental update: send only the new game state.
                    let _ = cmd_tx.send(Cmd::SetGame(current_g.clone()));
                } else {
                    match replay_moves(&moves) {
                        Ok(g) => {
                            current_g = g.clone();
                            let _ = cmd_tx.send(Cmd::SetGame(g));
                        }
                        Err(msg) => { err!(msg); continue; }
                    }
                }
                applied = moves;
                ponder_mv = ACE_NO_MOVE;
                let _ = writeln!(stdout, "ready {}", applied.len());
                let _ = stdout.flush();
            }
            "makemove" => {
                let Some(mv_str) = parts.get(1) else {
                    err!("makemove requires a move");
                    continue;
                };
                if current_g.winner() >= 0 {
                    err!("terminal position");
                    continue;
                }
                let mv = algebraic_to_ace(mv_str);
                current_g.make_move(mv);
                applied.push((*mv_str).to_string());
                ponder_mv = ACE_NO_MOVE;
                let _ = cmd_tx.send(Cmd::SetGame(current_g.clone()));
                ok!("ready");
            }
            "go" => {
                if current_g.winner() >= 0 {
                    err!("terminal position");
                    continue;
                }
                let arg1 = parts.get(1).copied().unwrap_or("4.0");
                if arg1 == "infinite" {
                    // go infinite [PONDER_MOVE]
                    let pm_str = parts.get(2).copied().unwrap_or("");
                    ponder_mv = if pm_str.is_empty() {
                        ACE_NO_MOVE
                    } else {
                        algebraic_to_ace(pm_str)
                    };
                    let _ = cmd_tx.send(Cmd::GoInfinite(ponder_mv));
                    // No reply expected — daemon starts pondering.
                } else {
                    let time_sec: f64 = arg1.parse().unwrap_or(4.0);
                    let time_ms = (time_sec * 1000.0).max(1.0) as u64;
                    let _ = cmd_tx.send(Cmd::GoTimed(time_ms));
                    match reply_rx.recv() {
                        Ok(Reply::BestMove(mv)) => {
                            if mv == ACE_NO_MOVE {
                                ok!("bestmove (none)");
                            } else {
                                ok!(format!("bestmove {}", ace_to_algebraic(mv)));
                            }
                        }
                        Ok(Reply::Error(msg)) => err!(msg),
                        Err(_) => break,
                    }
                }
            }
            "stop" => {
                let _ = cmd_tx.send(Cmd::StopAndGet);
                match reply_rx.recv() {
                    Ok(Reply::BestMove(mv)) => {
                        if mv == ACE_NO_MOVE {
                            ok!("bestmove (none)");
                        } else {
                            ok!(format!("bestmove {}", ace_to_algebraic(mv)));
                        }
                    }
                    Ok(Reply::Error(msg)) => err!(msg),
                    Err(_) => break,
                }
            }
            "ponderhit" => {
                // ponderhit TIME_MS  — ponder move was correct
                let time_ms: u64 = parts.get(1)
                    .and_then(|s| s.parse::<f64>().ok())
                    .map(|s| (s * 1000.0).max(1.0) as u64)
                    .unwrap_or(5000);
                // Update I/O position: the ponder move was played.
                if ponder_mv != ACE_NO_MOVE {
                    if current_g.winner() < 0 {
                        current_g.make_move(ponder_mv);
                        applied.push(ace_to_algebraic(ponder_mv));
                    }
                    ponder_mv = ACE_NO_MOVE;
                }
                let _ = cmd_tx.send(Cmd::PonderHit(time_ms));
                match reply_rx.recv() {
                    Ok(Reply::BestMove(mv)) => {
                        if mv == ACE_NO_MOVE {
                            ok!("bestmove (none)");
                        } else {
                            ok!(format!("bestmove {}", ace_to_algebraic(mv)));
                        }
                    }
                    Ok(Reply::Error(msg)) => err!(msg),
                    Err(_) => break,
                }
            }
            "movemiss" => {
                // movemiss MOVE TIME_MS  — opponent played MOVE, not the ponder move
                let Some(mv_str) = parts.get(1) else {
                    err!("movemiss requires MOVE TIME_MS");
                    continue;
                };
                let time_ms: u64 = parts.get(2)
                    .and_then(|s| s.parse::<f64>().ok())
                    .map(|s| (s * 1000.0).max(1.0) as u64)
                    .unwrap_or(5000);
                // Rewind to pre-ponder position, then apply actual move.
                // We do this by replaying applied (which doesn't include ponder_mv).
                let actual_mv = algebraic_to_ace(mv_str);
                if current_g.winner() < 0 {
                    current_g.make_move(actual_mv);
                    applied.push((*mv_str).to_string());
                }
                ponder_mv = ACE_NO_MOVE;
                let new_game = current_g.clone();
                let _ = cmd_tx.send(Cmd::MoveMiss { new_game, time_ms });
                match reply_rx.recv() {
                    Ok(Reply::BestMove(mv)) => {
                        if mv == ACE_NO_MOVE {
                            ok!("bestmove (none)");
                        } else {
                            ok!(format!("bestmove {}", ace_to_algebraic(mv)));
                        }
                    }
                    Ok(Reply::Error(msg)) => err!(msg),
                    Err(_) => break,
                }
            }
            "quit" => {
                let _ = cmd_tx.send(Cmd::Quit);
                break;
            }
            _ => err!("unknown command"),
        }
    }
}
