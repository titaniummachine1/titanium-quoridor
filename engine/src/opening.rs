//! Small tactical opening book and early-game guard.
//!
//! The book is intentionally human-readable: entries are algebraic prefixes with
//! algebraic replies, then converted to Zobrist-keyed entries at startup.  This
//! keeps the opening layer maintainable while making runtime lookup cheap.

use std::sync::OnceLock;

use crate::board::{Board, Move, Player, WallOrientation};
use crate::grid::goal_row;
use crate::moves::{generate_legal_moves_slice, MAX_LEGAL_MOVES};
use crate::path::BfsScratch;

const DISABLE_BOOK_ENV: &str = "TITANIUM_DISABLE_BOOK";
/// Phase 1 — theory window: book + guard only, no deep search.
pub const BOOK_MAX_PLY: u32 = 10;
// Guard covers the book window when no hash entry matches.
const OPENING_GUARD_FULL_MOVES: u32 = 6;

/// Half-move index (ply 1 = White's first turn at startpos).
pub fn ply_number(board: &Board) -> u32 {
    let half = match board.side() {
        Player::One => 1,
        Player::Two => 2,
    };
    (board.move_number - 1) * 2 + half
}

pub fn in_book_window(board: &Board) -> bool {
    ply_number(board) <= BOOK_MAX_PLY
}

/// Soft opening signal for search — never skips analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BookHint {
    pub mv: Move,
    /// Centimeters race lead for the player who plays `mv`.
    pub stm_bias: i16,
    pub priority: u8,
}

/// Plies covered by the opening guard's sprint hints (extends past book window).
const SPRINT_GUARD_MAX_MOVE: u32 = 10;

#[derive(Clone, Copy)]
struct BookLine {
    name: &'static str,
    prefix: &'static [&'static str],
    reply: &'static str,
    /// Higher wins when several lines share the same position key.
    priority: u8,
    /// Centimeters ahead for the player who plays `reply` (positive = good for STM).
    /// Zero means auto-compute from static race eval at book build time.
    stm_bias: i16,
}

#[derive(Clone)]
struct BookEntry {
    hash: u64,
    reply: Move,
    priority: u8,
    /// Eval in cm from reply-side perspective after `reply` is played.
    stm_bias: i16,
    name: &'static str,
}

const BOOK_LINES: &[BookLine] = &[
    // Orthodox center development into Standard/Shiller structure.  The key
    // transcript correction is ply 7: after e2 e8 e3 e7 e4 e6, do NOT play e5.
    // It donates a free forward jump to e4.  Instead anchor a vertical center
    // wall. e3v is the main Standard/Shiller weapon: it keeps W/B distances
    // balanced at 5/5 while splitting the board into committed lanes.
    BookLine {
        name: "center-start",
        prefix: &[],
        reply: "e2",
        priority: 100,
        stm_bias: 0,
    },
    BookLine {
        name: "center-mirror-1",
        prefix: &["e2"],
        reply: "e8",
        priority: 100,
        stm_bias: 0,
    },
    BookLine {
        name: "center-white-2",
        prefix: &["e2", "e8"],
        reply: "e3",
        priority: 100,
        stm_bias: 0,
    },
    BookLine {
        name: "center-black-2",
        prefix: &["e2", "e8", "e3"],
        reply: "e7",
        priority: 100,
        stm_bias: 0,
    },
    BookLine {
        name: "center-white-3",
        prefix: &["e2", "e8", "e3", "e7"],
        reply: "e4",
        priority: 100,
        stm_bias: 0,
    },
    BookLine {
        name: "center-black-3",
        prefix: &["e2", "e8", "e3", "e7", "e4"],
        reply: "e6",
        priority: 100,
        stm_bias: 0,
    },
    BookLine {
        name: "standard-shiller-center-vertical",
        prefix: &["e2", "e8", "e3", "e7", "e4", "e6"],
        reply: "e3v",
        priority: 150,
        stm_bias: 0,
    },
    // Shiller alternatives from the same center tabiya. They are deliberately
    // lower priority than e3v, but stay in the book as explicit analyzed
    // fallbacks and as documentation of the accepted vertical-wall family.
    BookLine {
        name: "shiller-d3v",
        prefix: &["e2", "e8", "e3", "e7", "e4", "e6"],
        reply: "d3v",
        priority: 135,
        stm_bias: 0,
    },
    BookLine {
        name: "shiller-c3v",
        prefix: &["e2", "e8", "e3", "e7", "e4", "e6"],
        reply: "c3v",
        priority: 130,
        stm_bias: 0,
    },
    // After e5 Black can jump to e4, leaving a tied race.  Playing d3h
    // immediately creates a 2-step forced detour for Black (d3h blocks both
    // e4→e3 and d4→d3), giving White the race lead before spending more walls.
    BookLine {
        name: "center-anti-jump",
        prefix: &["e2", "e8", "e3", "e7", "e4", "e6", "e5", "e4"],
        reply: "d3h",
        priority: 130,
        stm_bias: 0,
    },
    // If Black retreats north to e6 after d3h, White jumps over them to e7
    // (Black is at e6, White at e5 — jump lands at e7 = 2 steps from goal).
    BookLine {
        name: "center-anti-jump-retreat",
        prefix: &["e2", "e8", "e3", "e7", "e4", "e6", "e5", "e4", "d3h", "e6"],
        reply: "e7",
        priority: 130,
        stm_bias: 0,
    },
    // After d3h Black's optimal east detour is f4→f3→f2→f1 (4 steps).
    // White advances to keep tempo.
    BookLine {
        name: "center-anti-jump-detour-east",
        prefix: &["e2", "e8", "e3", "e7", "e4", "e6", "e5", "e4", "d3h", "f4"],
        reply: "e6",
        priority: 128,
        stm_bias: 0,
    },
    // Suboptimal west detour (c4 forces 5-step path after d3h); White still advances.
    BookLine {
        name: "center-anti-jump-detour-west",
        prefix: &["e2", "e8", "e3", "e7", "e4", "e6", "e5", "e4", "d3h", "c4"],
        reply: "e6",
        priority: 127,
        stm_bias: 0,
    },
    BookLine {
        name: "stonewall-tempo-reply",
        prefix: &["e2", "e8", "e3", "e7", "e4", "e6", "d1h"],
        reply: "d6h",
        priority: 125,
        stm_bias: 0,
    },
    BookLine {
        name: "stonewall-tempo-reply-mirror-anchor",
        prefix: &["e2", "e8", "e3", "e7", "e4", "e6", "f1h"],
        reply: "d6h",
        priority: 125,
        stm_bias: 0,
    },
    // Reed is treated as something to punish, not something Titanium should
    // mainline.  Edge counters narrow both path counts and force a tactical race.
    BookLine {
        name: "reed-counter-left",
        prefix: &["c3h"],
        reply: "a3h",
        priority: 110,
        stm_bias: 0,
    },
    BookLine {
        name: "reed-counter-right",
        prefix: &["c3h", "a3h", "f3h"],
        reply: "h3h",
        priority: 110,
        stm_bias: 0,
    },
    BookLine {
        name: "anti-black-reed-left",
        prefix: &["e2", "c7h"],
        reply: "a7h",
        priority: 105,
        stm_bias: 0,
    },
    BookLine {
        name: "anti-black-reed-right",
        prefix: &["e2", "c7h", "a7h", "f7h"],
        reply: "h7h",
        priority: 105,
        stm_bias: 0,
    },
    // Sidewall / Shatranj / Lee-style oddities: keep tempo first, then let
    // search handle the concrete box once corridors exist.
    BookLine {
        name: "anti-shatranj-d1v",
        prefix: &["d1v"],
        reply: "e8",
        priority: 100,
        stm_bias: 0,
    },
    BookLine {
        name: "anti-shatranj-e1v",
        prefix: &["e1v"],
        reply: "e8",
        priority: 100,
        stm_bias: 0,
    },
    BookLine {
        name: "anti-early-sidewall-d1v",
        prefix: &["e2", "d8v"],
        reply: "e3",
        priority: 100,
        stm_bias: 0,
    },
    BookLine {
        name: "anti-early-sidewall-e8v",
        prefix: &["e2", "e8v"],
        reply: "e3",
        priority: 100,
        stm_bias: 0,
    },
    // Ishtar often opens h2h — take the center anyway (mined from self-play).
    BookLine {
        name: "anti-h2h-black",
        prefix: &["h2h"],
        reply: "e8",
        priority: 115,
        stm_bias: 0,
    },
    BookLine {
        name: "anti-h2h-white",
        prefix: &["h2h", "e8"],
        reply: "e2",
        priority: 115,
        stm_bias: 0,
    },
    BookLine {
        name: "anti-h2h-center-line",
        prefix: &["h2h", "e8", "e2", "e7"],
        reply: "e3",
        priority: 115,
        stm_bias: 0,
    },
    // Ishtar Medium self-play (40 games, Jun 2026) — center through e6 is universal;
    // ply 7 splits evenly between h3h and a3h.  e5 stays higher priority (our tempo).
    BookLine {
        name: "mined-ply7-h3h",
        prefix: &["e2", "e8", "e3", "e7", "e4", "e6"],
        reply: "h3h",
        priority: 118,
        stm_bias: 0,
    },
    BookLine {
        name: "mined-ply7-a3h",
        prefix: &["e2", "e8", "e3", "e7", "e4", "e6"],
        reply: "a3h",
        priority: 118,
        stm_bias: 0,
    },
    BookLine {
        name: "mined-h3h-d6h",
        prefix: &["e2", "e8", "e3", "e7", "e4", "e6", "h3h"],
        reply: "d6h",
        priority: 112,
        stm_bias: 0,
    },
    BookLine {
        name: "mined-h3h-d6h-f3h",
        prefix: &["e2", "e8", "e3", "e7", "e4", "e6", "h3h", "d6h"],
        reply: "f3h",
        priority: 112,
        stm_bias: 0,
    },
    BookLine {
        name: "mined-a3h-e3v",
        prefix: &["e2", "e8", "e3", "e7", "e4", "e6", "a3h"],
        reply: "e3v",
        priority: 114,
        stm_bias: 0,
    },
    BookLine {
        name: "mined-a3h-e3v-d2h",
        prefix: &["e2", "e8", "e3", "e7", "e4", "e6", "a3h", "e3v"],
        reply: "d2h",
        priority: 114,
        stm_bias: 0,
    },
    BookLine {
        name: "mined-a3h-f6h",
        prefix: &["e2", "e8", "e3", "e7", "e4", "e6", "a3h"],
        reply: "f6h",
        priority: 110,
        stm_bias: 0,
    },
    // After e3h, Black plays e5h (ply 10) boxing White at e5.
    // White must advance to d5 to make progress — NOT place another wall.
    BookLine {
        name: "counter-e5h-d5",
        prefix: &["e2", "e8", "e3", "e7", "e4", "e6", "e5", "e4", "e3h", "e5h"],
        reply: "d5",
        priority: 125,
        stm_bias: 0,
    },
    // Mirror: after e3h + e5h mirrored (f5h), advance to f5.
    // After e3h, Black plays d5h (ply 10) blocking e5-e6 and d5-d6.
    // White still goes to d5 (horizontal move from e5 is fine; only vertical d5-d6 is blocked).
    BookLine {
        name: "counter-d5h-d5",
        prefix: &["e2", "e8", "e3", "e7", "e4", "e6", "e5", "e4", "e3h", "d5h"],
        reply: "d5",
        priority: 120,
        stm_bias: 0,
    },
];

static BOOK: OnceLock<Vec<BookEntry>> = OnceLock::new();

pub fn lookup(board: &mut Board) -> Option<Move> {
    book_hint(board).map(|hint| hint.mv)
}

/// Best book/guard reply for move ordering and aspiration bias. Search always runs.
pub fn book_hint(board: &mut Board) -> Option<BookHint> {
    if std::env::var(DISABLE_BOOK_ENV).is_ok_and(|value| value == "1") {
        return None;
    }
    lookup_book_hint(board).or_else(|| {
        // Guard fires within the book window AND in the early bridge phase.
        // This keeps "advance pawn" as a strong hint even at ply 11-20.
        if board.move_number <= SPRINT_GUARD_MAX_MOVE {
            opening_guard_hint(board)
        } else {
            None
        }
    })
}

fn lookup_book_hint(board: &mut Board) -> Option<BookHint> {
    let mut best: Option<&BookEntry> = None;
    for entry in book_entries() {
        if entry.hash != board.hash || !is_legal_reply(board, entry.reply) {
            continue;
        }
        if best.is_none_or(|current| book_entry_better(entry, current)) {
            best = Some(entry);
        }
    }
    best.map(|entry| BookHint {
        mv: entry.reply,
        stm_bias: entry.stm_bias,
        priority: entry.priority,
    })
}

fn opening_guard_hint(board: &mut Board) -> Option<BookHint> {
    opening_guard(board).map(|mv| BookHint {
        mv,
        stm_bias: 0,
        priority: 75,
    })
}

/// True when `candidate` is strictly better for the side to move than `current`.
fn book_entry_better(candidate: &BookEntry, current: &BookEntry) -> bool {
    if candidate.priority != current.priority {
        return candidate.priority > current.priority;
    }
    if candidate.stm_bias != current.stm_bias {
        return candidate.stm_bias > current.stm_bias;
    }
    // Deterministic tie-break — never pick at random.
    format_move_key(candidate.reply) > format_move_key(current.reply)
}

fn format_move_key(mv: Move) -> String {
    crate::perft::format_move(mv)
}

fn book_entries() -> &'static [BookEntry] {
    BOOK.get_or_init(build_book_entries)
}

fn build_book_entries() -> Vec<BookEntry> {
    let mut out = Vec::with_capacity(BOOK_LINES.len() * 2);
    for line in BOOK_LINES {
        push_book_variant(&mut out, line, false);
        push_book_variant(&mut out, line, true);
    }
    out
}

fn push_book_variant(out: &mut Vec<BookEntry>, line: &BookLine, mirrored: bool) {
    let prefix = materialize_prefix(line.prefix, mirrored);
    let reply_text = materialize_move(line.reply, mirrored);
    let mut board = Board::new();
    for mv in &prefix {
        board.apply_algebraic(mv);
    }
    let reply = parse_algebraic(&reply_text);
    if !is_legal_reply(&mut board, reply) {
        panic!(
            "illegal opening reply {} for {} ({})",
            reply_text,
            line.name,
            if mirrored { "mirrored" } else { "direct" }
        );
    }
    let stm_bias = if line.stm_bias != 0 {
        line.stm_bias
    } else {
        let undo = board.make_move(reply);
        let bias = race_bias_cm_for_mover(&mut board);
        board.unmake_move(undo);
        bias
    };
    out.push(BookEntry {
        hash: board.hash,
        reply,
        priority: line.priority,
        stm_bias,
        name: line.name,
    });
}

/// Centimeters race lead for the player who just moved (reply side).
fn race_bias_cm_for_mover(board: &Board) -> i16 {
    let mut scratch = BfsScratch::new();
    let mover = board.side().opposite();
    let opp = mover.opposite();
    let our = scratch.shortest_distance(board, mover).unwrap_or(255);
    let their = scratch.shortest_distance(board, opp).unwrap_or(255);
    let cm = (i32::from(their) - i32::from(our)) * 100;
    cm.clamp(i16::MIN as i32, i16::MAX as i32) as i16
}

fn materialize_prefix(prefix: &[&'static str], mirrored: bool) -> Vec<String> {
    prefix
        .iter()
        .map(|mv| materialize_move(mv, mirrored))
        .collect()
}

fn materialize_move(mv: &str, mirrored: bool) -> String {
    if mirrored {
        mirror_algebraic(mv)
    } else {
        mv.to_owned()
    }
}

fn mirror_algebraic(mv: &str) -> String {
    let bytes = mv.as_bytes();
    let col = bytes[0] - b'a';
    let row = bytes[1] as char;
    if bytes.len() == 2 {
        let mirrored_col = 8 - col;
        return format!("{}{}", (b'a' + mirrored_col) as char, row);
    }
    let mirrored_col = 7 - col;
    format!(
        "{}{}{}",
        (b'a' + mirrored_col) as char,
        row,
        bytes[2] as char
    )
}

fn parse_algebraic(text: &str) -> Move {
    let bytes = text.as_bytes();
    let col = bytes[0] - b'a';
    let row = bytes[1] - b'0' - 1;
    if bytes.len() == 2 {
        return Move::Pawn { row, col };
    }
    let orientation = match bytes[2] {
        b'h' => WallOrientation::Horizontal,
        b'v' => WallOrientation::Vertical,
        _ => panic!("bad opening wall suffix in {text}"),
    };
    Move::Wall {
        row,
        col,
        orientation,
    }
}

fn is_legal_reply(board: &mut Board, reply: Move) -> bool {
    let mut scratch = BfsScratch::new();
    let mut buf = [Move::Pawn { row: 0, col: 0 }; MAX_LEGAL_MOVES];
    let n = generate_legal_moves_slice(board, &mut buf, &mut scratch);
    buf[..n].contains(&reply)
}

fn opening_guard(board: &mut Board) -> Option<Move> {
    if board.move_number > OPENING_GUARD_FULL_MOVES {
        return None;
    }
    let mut scratch = BfsScratch::new();
    let mut buf = [Move::Pawn { row: 0, col: 0 }; MAX_LEGAL_MOVES];
    let n = generate_legal_moves_slice(board, &mut buf, &mut scratch);
    if n == 0 {
        return None;
    }
    advancing_pawn_move(board, &buf[..n], &mut scratch)
        .or_else(|| active_opening_wall(board, &buf[..n]))
}

fn advancing_pawn_move(
    board: &mut Board,
    moves: &[Move],
    scratch: &mut BfsScratch,
) -> Option<Move> {
    let stm = board.side();
    let goal = goal_row(stm);
    let (row, _) = board.pawn(stm);
    let current = row.abs_diff(goal);
    moves.iter().copied().find(|mv| match mv {
        Move::Pawn { row: next_row, .. } => {
            next_row.abs_diff(goal) < current
                && !allows_opponent_double_advance(board, *mv, scratch)
        }
        Move::Wall { .. } => false,
    })
}

fn allows_opponent_double_advance(
    board: &mut Board,
    candidate: Move,
    scratch: &mut BfsScratch,
) -> bool {
    let opp = board.side().opposite();
    let opp_before = scratch.shortest_distance(board, opp).unwrap_or(255);
    let undo_candidate = board.make_move(candidate);

    let mut legal = [Move::Pawn { row: 0, col: 0 }; MAX_LEGAL_MOVES];
    let n = generate_legal_moves_slice(board, &mut legal, scratch);
    let mut gives_jump = false;
    for &reply in &legal[..n] {
        let Move::Pawn { .. } = reply else {
            continue;
        };
        let undo_reply = board.make_move(reply);
        let opp_after = scratch.shortest_distance(board, opp).unwrap_or(255);
        board.unmake_move(undo_reply);
        if opp_before.saturating_sub(opp_after) >= 2 {
            gives_jump = true;
            break;
        }
    }

    board.unmake_move(undo_candidate);
    gives_jump
}

fn active_opening_wall(board: &mut Board, moves: &[Move]) -> Option<Move> {
    let stm = board.side();
    let opp = stm.opposite();
    let mut scratch = BfsScratch::new();
    let opp_before = scratch.shortest_distance(board, opp).unwrap_or(u8::MAX);
    let our_before = scratch.shortest_distance(board, stm).unwrap_or(u8::MAX);
    let mut best = None;
    let mut best_gain = 0i16;

    for &mv in moves {
        let Move::Wall {
            row,
            col: _,
            orientation: _,
        } = mv
        else {
            continue;
        };
        if is_passive_back_rank_wall(stm, row) {
            continue;
        }
        let undo = board.make_move(mv);
        let opp_after = scratch.shortest_distance(board, opp).unwrap_or(u8::MAX);
        let our_after = scratch.shortest_distance(board, stm).unwrap_or(u8::MAX);
        board.unmake_move(undo);

        let opp_gain = distance_gain(opp_before, opp_after);
        let self_loss = distance_gain(our_before, our_after);
        let score = opp_gain - self_loss;
        if score > best_gain {
            best_gain = score;
            best = Some(mv);
        }
    }
    best
}

fn distance_gain(before: u8, after: u8) -> i16 {
    if before == u8::MAX || after == u8::MAX {
        return 0;
    }
    i16::from(after) - i16::from(before)
}

fn is_passive_back_rank_wall(stm: Player, row: u8) -> bool {
    match stm {
        Player::One => row <= 1,
        Player::Two => row >= 6,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::perft::format_move;

    fn replay(moves: &[&str]) -> Board {
        let mut board = Board::new();
        for mv in moves {
            board.apply_algebraic(mv);
        }
        board
    }

    fn lookup_text(board: &mut Board) -> Option<String> {
        lookup(board).map(format_move)
    }

    fn is_standard_shiller(reply: Option<&str>) -> bool {
        matches!(reply, Some("e3v" | "d3v" | "c3v"))
    }

    fn is_center_anti_jump(reply: Option<&str>) -> bool {
        matches!(reply, Some("d3h" | "e3h"))
    }

    #[test]
    fn book_center_start() {
        let mut board = Board::new();
        assert_eq!(lookup_text(&mut board).as_deref(), Some("e2"));

        let mut board = replay(&["e2"]);
        assert_eq!(lookup_text(&mut board).as_deref(), Some("e8"));
    }

    #[test]
    fn book_reaches_active_center_reply() {
        let mut board = replay(&["e2", "e8", "e3", "e7", "e4", "e6"]);
        let reply = lookup_text(&mut board);
        assert!(
            is_standard_shiller(reply.as_deref()),
            "expected Standard/Shiller vertical wall, got {reply:?}"
        );
    }

    #[test]
    fn book_rejects_passive_stonewall_followup() {
        let mut board = replay(&["e2", "e8", "e3", "e7", "e4", "e6", "d1h"]);
        assert_eq!(lookup_text(&mut board).as_deref(), Some("d6h"));
    }

    #[test]
    fn book_blocks_jump_with_d3h() {
        // After e5 Black jumps over White to e4; d3h forces a 2-step detour.
        let mut board = replay(&["e2", "e8", "e3", "e7", "e4", "e6", "e5", "e4"]);
        let reply = lookup_text(&mut board);
        assert!(
            is_center_anti_jump(reply.as_deref()),
            "expected d3h or e3h, got {reply:?}"
        );
    }

    #[test]
    fn reed_counter_uses_edge_wall() {
        let mut board = replay(&["c3h"]);
        assert_eq!(lookup_text(&mut board).as_deref(), Some("a3h"));

        let mut board = replay(&["f3h"]);
        assert_eq!(lookup_text(&mut board).as_deref(), Some("h3h"));
    }

    #[test]
    fn every_book_reply_is_legal() {
        assert!(!book_entries().is_empty());
        for entry in book_entries() {
            // Construction validates every reply against legal move generation.
            // This confirms all generated direct/mirrored entries are materialized.
            assert_ne!(entry.hash, 0);
        }
    }

    #[test]
    fn guard_prefers_pawn_tempo_over_passive_back_rank_wall() {
        let mut board = replay(&["e2", "e8", "e3", "e7", "e4", "d7h"]);
        assert_eq!(lookup_text(&mut board).as_deref(), Some("e5"));
    }

    #[test]
    fn guard_rejects_sprint_that_donates_jump() {
        let mut board = replay(&["e2", "e8", "e3", "e7", "e4", "e6"]);
        assert_ne!(
            opening_guard(&mut board).map(format_move).as_deref(),
            Some("e5")
        );
    }

    #[test]
    fn book_picks_highest_priority_then_stm_bias() {
        let weak = BookEntry {
            hash: 1,
            reply: Move::Pawn { row: 0, col: 4 },
            priority: 120,
            stm_bias: 100,
            name: "weak",
        };
        let strong = BookEntry {
            hash: 1,
            reply: Move::Pawn { row: 1, col: 4 },
            priority: 130,
            stm_bias: 100,
            name: "strong",
        };
        let better_bias = BookEntry {
            hash: 1,
            reply: Move::Pawn { row: 2, col: 4 },
            priority: 120,
            stm_bias: 400,
            name: "better-bias",
        };
        assert!(book_entry_better(&strong, &weak));
        assert!(book_entry_better(&better_bias, &weak));
        assert!(!book_entry_better(&weak, &strong));
    }
}
