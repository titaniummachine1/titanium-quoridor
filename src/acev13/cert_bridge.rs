//! Bridge: run the proven v13 `certify_win` static win-certificate solver on a
//! Titanium `core::board::Board` position.
//!
//! The certificate solver ([`super::certify::certify`]) is a 1:1 port of
//! `certify_win.js` and operates on [`AceGame`]. Rather than re-port the
//! soundness-critical AND/OR search onto `Board`, we convert the position into a
//! throwaway `AceGame` and call the existing, oracle-tested solver. The ONLY new
//! trust-sensitive code is the converter below — and it is unit-tested against
//! `AceGame`'s own winner / distance / legal-move surface for parity.
//!
//! Coordinate mapping (same as `acev13::mod`):
//!   pawn cell      ace = (8 - row) * 9 + col          (Titanium row 0 = ACE row 8)
//!   wall ace slot  = (7 - wrow) * 8 + wcol            (Titanium wall bit = wrow*8+wcol)
//!   player index   identical (One→0, Two→1)
//!   turn hash term present iff side-to-move == 1 (ACE base hash is turn-0)

use crate::acev13::certify::{certify, CertifyOpts};
use crate::acev13::game::{AceGame, BORDER, DELTA, DIRBIT, ZOBRIST};
use crate::core::board::{Board, Player};
use crate::util::clock::{Duration, Instant};

/// Titanium pawn `(row, col)` → ACE cell index (0..80).
#[inline]
fn ace_cell(row: u8, col: u8) -> usize {
    (8 - row as usize) * 9 + col as usize
}

/// Build a throwaway `AceGame` mirroring `board`'s position, with a correct
/// incremental-zobrist base so the solver's memo keys are sound.
pub fn ace_from_board(board: &Board) -> AceGame {
    let z = &ZOBRIST;
    let mut g = AceGame::new();
    g.hw = [0u8; 64];
    g.vw = [0u8; 64];
    g.blocked = [0u8; 81];

    let p0 = ace_cell(board.pawns[0].0, board.pawns[0].1);
    let p1 = ace_cell(board.pawns[1].0, board.pawns[1].1);
    g.pawn = [p0, p1];
    g.wl = [
        board.walls_remaining[0] as i32,
        board.walls_remaining[1] as i32,
    ];
    g.turn = board.side_to_move as usize;

    let mut hl = z.pawn_lo[0][p0] ^ z.pawn_lo[1][p1];
    let mut hh = z.pawn_hi[0][p0] ^ z.pawn_hi[1][p1];

    for wrow in 0u8..8 {
        for wcol in 0u8..8 {
            let bit = (wrow as u64) * 8 + wcol as u64;
            let slot = (7 - wrow as usize) * 8 + wcol as usize;
            if (board.horizontal_walls >> bit) & 1 != 0 {
                g.hw[slot] = 1;
                g.set_wall_bits(0, slot, true);
                hl ^= z.hw_lo[slot];
                hh ^= z.hw_hi[slot];
            }
            if (board.vertical_walls >> bit) & 1 != 0 {
                g.vw[slot] = 1;
                g.set_wall_bits(1, slot, true);
                hl ^= z.vw_lo[slot];
                hh ^= z.vw_hi[slot];
            }
        }
    }
    if g.turn == 1 {
        hl ^= z.turn_lo;
        hh ^= z.turn_hi;
    }
    g.hash_lo = hl;
    g.hash_hi = hh;
    g
}

/// ACE player index (0/1) → Titanium [`Player`].
#[inline]
fn player_from_ace(idx: usize) -> Player {
    if idx == 0 {
        Player::One
    } else {
        Player::Two
    }
}

/// Run the v13 static win certificate on a Titanium board.
///
/// Returns `Some(side)` iff that side's win is PROVEN (sound — the certificate
/// only ever removes options from the maximizing side, so a proof transfers to
/// the real game). `None` = not proven within the budget/deadline.
///
/// * `budget` — certify node budget (JS default 200000; the search uses ~1200).
/// * `deadline_ms` — wall-clock cap; 0 = none.
/// * `side` — force a candidate side, or `None` to try the favored race winner
///   first then the other.
pub fn certify_board(
    board: &Board,
    budget: u64,
    deadline_ms: u64,
    side: Option<Player>,
) -> Option<Player> {
    let mut g = ace_from_board(board);
    let immediate = g.winner();
    if immediate >= 0 {
        let winner = player_from_ace(immediate as usize);
        return if side.is_none_or(|s| s == winner) {
            Some(winner)
        } else {
            None
        };
    }
    if g.wl[0] == 0 && g.wl[1] == 0 {
        let stm_wins = match hands_empty_race(&g) {
            RaceVerdict::Win => true,
            RaceVerdict::Loss => false,
            RaceVerdict::NeedsProof => match race_minimax(&mut g) {
                RaceProof::Win => true,
                RaceProof::Loss => false,
                RaceProof::Unknown => {
                    return None;
                }
            },
        };
        let winner_idx = if stm_wins { g.turn } else { 1 - g.turn };
        let winner = player_from_ace(winner_idx);
        return if side.is_none_or(|s| s == winner) {
            Some(winner)
        } else {
            None
        };
    }
    let deadline = if deadline_ms > 0 {
        Some(Instant::now() + Duration::from_millis(deadline_ms))
    } else {
        None
    };
    let report = certify(
        &mut g,
        &CertifyOpts {
            budget,
            deadline,
            mode_pruned: false,
            slack: 2,
            side: side.map(|p| p as usize),
            recommit: true,
        },
    );
    report.proven.map(player_from_ace)
}

// ── Forward race minimax (replaces the 13k-state retrograde oracle) ───────────
//
// Used ONLY for the NeedsProof edge case: paths overlap AND |adj| ≤ 1.
// Lazy forward search with a small memo; cycle/back-edge and depth-cap return
// [`RaceProof::Unknown`] so callers fall back to normal alpha-beta (not draw).

/// Exact zero-wall race proof from side-to-move perspective.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RaceProof {
    /// Side to move wins with certainty.
    Win,
    /// Side to move loses with certainty.
    Loss,
    /// No safe exact certificate (cycle, depth cap, or incomplete search).
    /// **Not a draw** — callers must decline endgame certification.
    Unknown,
}

pub const RACE_STATE_COUNT: usize = 81 * 81 * 2; // 13,122 — max index 13,121
const RACE_VISITING_WORDS: usize = RACE_STATE_COUNT.div_ceil(64); // 206 × 64 = 13,184 bits
const _: () = assert!(RACE_VISITING_WORDS == 206);
const _: () = assert!(RACE_VISITING_WORDS * 64 >= RACE_STATE_COUNT);
/// Resolver ply cap: 2× max observed legitimate depth on empty board (see tests).
pub const RACE_RESOLVER_DEPTH_CAP: u32 = 128;

#[inline]
pub fn race_state_index(p0: usize, p1: usize, turn: usize) -> usize {
    debug_assert!(p0 < 81 && p1 < 81 && turn < 2);
    (p0 * 81 + p1) * 2 + turn
}

#[inline]
fn visiting_test(words: &[u64; RACE_VISITING_WORDS], idx: usize) -> bool {
    words[idx / 64] & (1u64 << (idx % 64)) != 0
}

#[inline]
fn visiting_set(words: &mut [u64; RACE_VISITING_WORDS], idx: usize) {
    words[idx / 64] |= 1u64 << (idx % 64);
}

#[inline]
fn visiting_clear(words: &mut [u64; RACE_VISITING_WORDS], idx: usize) {
    words[idx / 64] &= !(1u64 << (idx % 64));
}

/// Three-valued minimax over child proofs (each child is from **opponent** stm).
pub fn aggregate_opponent_child_proofs(children: &[RaceProof]) -> RaceProof {
    if children.is_empty() {
        return RaceProof::Loss;
    }
    let mut has_stm_win = false;
    let mut all_opponent_win = true;
    for &child in children {
        match child {
            RaceProof::Loss => has_stm_win = true,
            RaceProof::Win => {}
            RaceProof::Unknown => all_opponent_win = false,
        }
    }
    if has_stm_win {
        RaceProof::Win
    } else if all_opponent_win {
        RaceProof::Loss
    } else {
        RaceProof::Unknown
    }
}

struct RaceResolver {
    visiting: [u64; RACE_VISITING_WORDS],
    memo: std::collections::HashMap<u32, RaceProof>,
    depth: u32,
    depth_cap: u32,
}

impl RaceResolver {
    fn new(depth_cap: u32) -> Self {
        Self {
            visiting: [0u64; RACE_VISITING_WORDS],
            memo: std::collections::HashMap::with_capacity(64),
            depth: 0,
            depth_cap,
        }
    }
}

/// Solve a hands-empty pawn race for the side to move.
///
/// Uses distance-decreasing pawn moves only (existing restricted subgame).
/// Returns [`RaceProof::Unknown`] on active-path cycle or depth cap — never draw.
pub fn race_minimax(g: &mut AceGame) -> RaceProof {
    race_minimax_with_cap(g, RACE_RESOLVER_DEPTH_CAP)
}

pub fn race_minimax_with_cap(g: &mut AceGame, depth_cap: u32) -> RaceProof {
    let mut resolver = RaceResolver::new(depth_cap);
    race_rec(g, &mut resolver)
}

/// Convenience for callers that need `Option<bool>` (None = decline certification).
#[inline]
pub fn race_minimax_stm_wins(g: &mut AceGame) -> Option<bool> {
    match race_minimax(g) {
        RaceProof::Win => Some(true),
        RaceProof::Loss => Some(false),
        RaceProof::Unknown => None,
    }
}

fn race_rec(g: &mut AceGame, resolver: &mut RaceResolver) -> RaceProof {
    let stm = g.turn;
    let pawn = g.pawn[stm];

    if (stm == 0 && pawn < 9) || (stm == 1 && pawn >= 72) {
        return RaceProof::Win;
    }

    if resolver.depth >= resolver.depth_cap {
        return RaceProof::Unknown;
    }

    let idx = race_state_index(g.pawn[0], g.pawn[1], g.turn);
    if visiting_test(&resolver.visiting, idx) {
        return RaceProof::Unknown;
    }

    let key = idx as u32;
    if let Some(&cached) = resolver.memo.get(&key) {
        return cached;
    }

    visiting_set(&mut resolver.visiting, idx);
    resolver.depth += 1;

    let mut d_goal = [255u8; 81];
    g.compute_dist(stm, &mut d_goal);
    let my_dist = d_goal[pawn];

    let mut buf = [0i16; 16];
    let cnt = g.gen_pawn_moves(&mut buf, 0);

    let mut child_proofs: [RaceProof; 16] = [RaceProof::Unknown; 16];
    let mut child_count = 0usize;

    for i in 0..cnt {
        let to = buf[i] as usize;
        if d_goal[to] < my_dist {
            if (stm == 0 && to < 9) || (stm == 1 && to >= 72) {
                visiting_clear(&mut resolver.visiting, idx);
                resolver.depth -= 1;
                return RaceProof::Win;
            }
            g.make_move(buf[i]);
            child_proofs[child_count] = race_rec(g, resolver);
            g.unmake_move();
            child_count += 1;
        }
    }

    resolver.depth -= 1;
    visiting_clear(&mut resolver.visiting, idx);

    let result = if child_count == 0 {
        RaceProof::Loss
    } else {
        aggregate_opponent_child_proofs(&child_proofs[..child_count])
    };

    if matches!(result, RaceProof::Win | RaceProof::Loss) {
        resolver.memo.insert(key, result);
    }
    result
}

// ── Path-aware hands-empty race classifier (jump-mechanics short-circuit) ─────

/// Walls-only BFS distance from `src` to every cell (255 = unreachable).
/// Mirror of `compute_dist` but seeded from an arbitrary cell.
fn bfs_from_cell(g: &AceGame, src: usize) -> [u8; 81] {
    let mut out = [255u8; 81];
    out[src] = 0;
    let mut queue = [0i16; 81];
    let mut head = 0usize;
    let mut tail = 0usize;
    queue[tail] = src as i16;
    tail += 1;
    while head < tail {
        let u = queue[head] as usize;
        head += 1;
        let du = out[u] + 1;
        let bm = g.blocked[u] | BORDER[u];
        for d in 0..4 {
            if bm & DIRBIT[d] != 0 {
                continue;
            }
            let v = (u as i16 + DELTA[d]) as usize;
            if out[v] > du {
                out[v] = du;
                queue[tail] = v as i16;
                tail += 1;
            }
        }
    }
    out
}

/// Do the two pawns' SHORTEST-PATH SETS share a cell? A cell `c` is on some
/// shortest path of player `p` iff `dist(pawn→c) + dist(c→goal) == D_p`. If the
/// sets are disjoint the pawns never contend for a square → no jump can occur →
/// the race is a pure parallel tempo race. `d_goal{0,1}` are `compute_dist`
/// fields (cell→goal); `D{0,1}` are the players' shortest distances.
fn paths_overlap(g: &AceGame, d_goal0: &[u8; 81], d_goal1: &[u8; 81]) -> bool {
    let s0 = bfs_from_cell(g, g.pawn[0]);
    let s1 = bfs_from_cell(g, g.pawn[1]);
    let big0 = d_goal0[g.pawn[0]] as u16;
    let big1 = d_goal1[g.pawn[1]] as u16;
    for c in 0..81 {
        let on0 = s0[c] != 255 && d_goal0[c] != 255 && s0[c] as u16 + d_goal0[c] as u16 == big0;
        if !on0 {
            continue;
        }
        let on1 = s1[c] != 255 && d_goal1[c] != 255 && s1[c] as u16 + d_goal1[c] as u16 == big1;
        if on1 {
            return true;
        }
    }
    false
}

/// Classifier + resolver: `Some(stm_wins)` when exact, `None` when certification
/// must be declined (resolver returned [`RaceProof::Unknown`]).
pub fn hands_empty_race_stm_wins(g: &mut AceGame) -> Option<bool> {
    match hands_empty_race(g) {
        RaceVerdict::Win => Some(true),
        RaceVerdict::Loss => Some(false),
        RaceVerdict::NeedsProof => race_minimax_stm_wins(g),
    }
}

/// Outcome of the wall-free (hands-empty) race classifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RaceVerdict {
    /// Side to move wins by force (deterministic — no proof needed).
    Win,
    /// Side to move loses by force (deterministic — no proof needed).
    Loss,
    /// Volatile (paths overlap and |tempo| ≤ 1): a jump can swing ±1 tempo and
    /// flip the result — the caller MUST run `race_minimax` to prove the outcome.
    NeedsProof,
}

/// Classify a hands-empty (no walls left) race for the side to move using
/// turn-adjusted tempo plus path-overlap, reserving the heavy proof for the
/// volatile band only.
///
/// Uses [`turn_adjusted_tempo_advantage`] (White perspective: `raw + 1` / `raw − 1`
/// for side to move). Wall-graph BFS distances ignore opposing pawn placement.
///
/// * `adj ≥ 2`: forced White win (stm wins iff White to move).
/// * `adj ≤ −2`: forced Black win (stm wins iff Black to move).
/// * `−1 ≤ adj ≤ 1` with overlapping shortest-path sets: jump may decide →
///   [`RaceVerdict::NeedsProof`] (caller runs [`race_minimax`]).
/// * `−1 ≤ adj ≤ 1` with separated paths: [`separated_pure_race_verdict`].
pub fn hands_empty_race(g: &AceGame) -> RaceVerdict {
    let adj = turn_adjusted_tempo_advantage(g);
    if adj >= 2 {
        return if g.turn == 0 {
            RaceVerdict::Win
        } else {
            RaceVerdict::Loss
        };
    }
    if adj <= -2 {
        return if g.turn == 0 {
            RaceVerdict::Loss
        } else {
            RaceVerdict::Win
        };
    }
    let mut d0 = [0u8; 81];
    let mut d1 = [0u8; 81];
    g.compute_dist(0, &mut d0);
    g.compute_dist(1, &mut d1);
    if d0[g.pawn[0]] == 255 || d1[g.pawn[1]] == 255 {
        return RaceVerdict::NeedsProof;
    }
    if paths_overlap(g, &d0, &d1) {
        return RaceVerdict::NeedsProof;
    }
    separated_pure_race_verdict(g)
}

// ── Turn-adjusted tempo (audit / proposed rule) ───────────────────────────────

/// Wall-graph distances for both players (255 = unreachable).
pub fn wall_graph_distances(g: &AceGame) -> (u8, u8) {
    let mut d0 = [0u8; 81];
    let mut d1 = [0u8; 81];
    g.compute_dist(0, &mut d0);
    g.compute_dist(1, &mut d1);
    (d0[g.pawn[0]], d1[g.pawn[1]])
}

/// White-perspective turn-adjusted tempo advantage for the proposed shortcut rule.
///
/// `raw = black_distance − white_distance` using wall-graph BFS (ignores pawns).
/// Adds one tempo when White is to move, subtracts one when Black is to move.
pub fn turn_adjusted_tempo_advantage(g: &AceGame) -> i32 {
    let (white_dist, black_dist) = wall_graph_distances(g);
    if white_dist == 255 || black_dist == 255 {
        return 0;
    }
    let raw = black_dist as i32 - white_dist as i32;
    if g.turn == 0 {
        raw + 1
    } else {
        raw - 1
    }
}

/// Pure parallel race once pawn interaction is impossible (paths separated).
/// Equal remaining distances are won by the side to move.
pub fn separated_pure_race_verdict(g: &AceGame) -> RaceVerdict {
    let (white_dist, black_dist) = wall_graph_distances(g);
    if white_dist == 255 || black_dist == 255 {
        return RaceVerdict::NeedsProof;
    }
    let stm_wins = if g.turn == 0 {
        white_dist <= black_dist
    } else {
        black_dist <= white_dist
    };
    if stm_wins {
        RaceVerdict::Win
    } else {
        RaceVerdict::Loss
    }
}

/// Proposed turn-adjusted classifier — mirrors [`hands_empty_race`] for audit diffs.
pub fn proposed_hands_empty_race(g: &AceGame) -> RaceVerdict {
    hands_empty_race(g)
}

/// Exact stm win/loss from retrograde oracle value (`+k`/`-k`/`0`).
pub fn exact_race_verdict(g: &AceGame, oracle_table: &[i16]) -> RaceVerdict {
    use crate::acev13::oracle::ORACLE_STATES;
    let id = (g.pawn[0] * 81 + g.pawn[1]) * 2 + g.turn;
    assert!(id < ORACLE_STATES);
    match oracle_table[id].cmp(&0) {
        std::cmp::Ordering::Greater => RaceVerdict::Win,
        std::cmp::Ordering::Less => RaceVerdict::Loss,
        std::cmp::Ordering::Equal => RaceVerdict::NeedsProof, // draw — treat as volatile
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::board::Board;

    /// The converter must reproduce AceGame's winner / pawn cells / distances
    /// for the same position reached by replaying moves on both sides.
    #[test]
    fn converter_matches_replayed_acegame() {
        use crate::acev13::algebraic_to_ace;
        let line = [
            "e2", "e8", "e3", "e7", "e4", "e6", "d3h", "d6h", "f3h", "f6h",
        ];
        let mut board = Board::new();
        let mut replayed = AceGame::new();
        for m in line {
            board.apply_algebraic(m);
            replayed.make_move(algebraic_to_ace(m));
        }
        let converted = ace_from_board(&board);
        assert_eq!(converted.pawn, replayed.pawn, "pawn cells");
        assert_eq!(converted.wl, replayed.wl, "walls left");
        assert_eq!(converted.turn, replayed.turn, "side to move");
        assert_eq!(converted.hw, replayed.hw, "h-walls");
        assert_eq!(converted.vw, replayed.vw, "v-walls");
        assert_eq!(converted.blocked, replayed.blocked, "blocked edges");
        assert_eq!(converted.hash_lo, replayed.hash_lo, "hash_lo");
        assert_eq!(converted.hash_hi, replayed.hash_hi, "hash_hi");
    }

    #[test]
    fn tempo_race_no_opp_walls_certifies_stm() {
        // One (stm) (3,1) dist 5, Two (5,7) dist 5 — DIFFERENT columns, so the
        // pawns never interact (no jump-tempo gift). Two has 0 walls ⇒ no
        // interdiction; One moves first and wins the independent race by a
        // tempo ⇒ a proven win for One. (Same column would let Two's jump over
        // One steal a tempo and flip the race — that position is NOT certifiable.)
        let mut board = Board::new();
        board.pawns = [(3, 1), (5, 7)];
        board.walls_remaining = [2, 0];
        board.side_to_move = Player::One;
        board.hash = crate::core::zobrist::hash_board(&board);
        assert_eq!(certify_board(&board, 5000, 0, None), Some(Player::One));
    }

    #[test]
    fn converter_matches_replay_on_random_positions() {
        // Soundness guard: the converter must reproduce a freshly-replayed
        // AceGame for ARBITRARY reachable positions, not just one line — a wrong
        // blocked[]/pawn/turn would feed the oracle/certificate a wrong board and
        // make it "prove" a win that isn't real. Fuzz many random legal games.
        use crate::acev13::algebraic_to_ace;
        use crate::movegen::generate_legal_moves;
        use crate::util::perft::format_move;

        let mut seed = 0x1234_5678_9abc_def0u64;
        let mut next = || {
            seed ^= seed >> 12;
            seed ^= seed << 25;
            seed ^= seed >> 27;
            seed.wrapping_mul(0x2545_F491_4F6C_DD1D)
        };

        for _ in 0..400 {
            let mut board = Board::new();
            let mut replayed = AceGame::new();
            let plies = 4 + (next() % 30) as usize;
            for _ in 0..plies {
                if board.is_terminal().is_some() {
                    break;
                }
                let moves = generate_legal_moves(&board);
                if moves.is_empty() {
                    break;
                }
                let mv = moves[(next() as usize) % moves.len()];
                let alg = format_move(mv);
                board.apply_move(mv);
                replayed.make_move(algebraic_to_ace(&alg));
            }
            let converted = ace_from_board(&board);
            assert_eq!(converted.pawn, replayed.pawn, "pawn cells");
            assert_eq!(converted.wl, replayed.wl, "walls left");
            assert_eq!(converted.turn, replayed.turn, "side to move");
            assert_eq!(converted.blocked, replayed.blocked, "blocked edges");
            assert_eq!(converted.hash_lo, replayed.hash_lo, "hash_lo");
            assert_eq!(converted.hash_hi, replayed.hash_hi, "hash_hi");
        }
    }

    fn race_board(p0: (u8, u8), p1: (u8, u8), stm: Player) -> Board {
        let mut b = Board::new();
        b.pawns = [p0, p1];
        b.walls_remaining = [0, 0];
        b.side_to_move = stm;
        b.hash = crate::core::zobrist::hash_board(&b);
        b
    }

    fn board_pos_from_ace(cell: usize) -> (u8, u8) {
        ((8 - cell / 9) as u8, (cell % 9) as u8)
    }

    #[test]
    fn hands_empty_classifier_matches_exact_when_decisive() {
        let mut decisive = 0usize;
        let mut volatile = 0usize;

        for p0 in 0..81 {
            // Skip already-terminal roots; those are handled by the normal
            // immediate winner path, not by race classification.
            if p0 < 9 {
                continue;
            }
            for p1 in 0..81 {
                if p1 >= 72 || p0 == p1 {
                    continue;
                }
                for stm in [Player::One, Player::Two] {
                    let board = race_board(board_pos_from_ace(p0), board_pos_from_ace(p1), stm);
                    let mut g = ace_from_board(&board);
                    let verdict = hands_empty_race(&g);
                    if verdict == RaceVerdict::NeedsProof {
                        volatile += 1;
                        continue;
                    }

                    decisive += 1;
                    let exact = match race_minimax(&mut g) {
                        RaceProof::Win => RaceVerdict::Win,
                        RaceProof::Loss => RaceVerdict::Loss,
                        RaceProof::Unknown => RaceVerdict::NeedsProof,
                    };
                    assert_eq!(
                        verdict, exact,
                        "p0={p0} p1={p1} stm={stm:?} classifier must match exact race solver"
                    );
                }
            }
        }

        assert!(decisive > 0, "test must cover deterministic certificates");
        assert!(volatile > 0, "test must cover proof-required close races");
    }

    #[test]
    fn parallel_race_equal_distance_stm_wins() {
        // Different columns ⇒ paths don't overlap; equal distance ⇒ stm wins.
        let g = ace_from_board(&race_board((3, 1), (5, 7), Player::One));
        assert_eq!(hands_empty_race(&g), RaceVerdict::Win);
    }

    #[test]
    fn certify_board_no_walls_uses_race_winner_and_side_filter() {
        let board = race_board((3, 1), (5, 7), Player::One);
        assert_eq!(certify_board(&board, 1, 0, None), Some(Player::One));
        assert_eq!(
            certify_board(&board, 1, 0, Some(Player::One)),
            Some(Player::One)
        );
        assert_eq!(certify_board(&board, 1, 0, Some(Player::Two)), None);
    }

    #[test]
    fn parallel_race_opponent_closer_stm_loses() {
        // Different columns (no overlap); One dist 7, Two dist 6 ⇒ One loses.
        let g = ace_from_board(&race_board((1, 1), (6, 7), Player::One));
        assert_eq!(hands_empty_race(&g), RaceVerdict::Loss);
    }

    #[test]
    fn same_column_close_race_needs_proof() {
        // Same column ⇒ paths overlap; equal distance (lead ≤1) ⇒ volatile.
        let g = ace_from_board(&race_board((3, 4), (5, 4), Player::One));
        assert_eq!(hands_empty_race(&g), RaceVerdict::NeedsProof);
    }

    #[test]
    fn two_tempo_lead_is_deterministic_even_when_overlapping() {
        // Same column (overlap) but One dist 2 vs Two dist 5 ⇒ lead ≥2 absorbs
        // any jump ⇒ deterministic win, no proof.
        let g = ace_from_board(&race_board((6, 4), (5, 4), Player::One));
        assert_eq!(hands_empty_race(&g), RaceVerdict::Win);
    }

    #[test]
    fn startpos_is_not_certifiable() {
        // No one can prove a win from the opening within a small budget.
        let board = Board::new();
        assert_eq!(certify_board(&board, 5000, 0, None), None);
    }

    #[test]
    fn reached_goal_certifies_immediately() {
        // Player One on the goal row is a trivially-proven (already-won) cert.
        let mut board = Board::new();
        board.pawns[0] = (8, 4);
        assert_eq!(certify_board(&board, 1000, 0, None), Some(Player::One));
    }

    // ── Turn-adjusted tempo audit (exhaustive zero-wall verifier) ─────────────

    #[derive(Default, Debug)]
    struct RaceAuditStats {
        states_checked: usize,
        exact_white_wins: usize,
        exact_black_wins: usize,
        exact_draws: usize,
        current_false_wins: usize,
        current_false_losses: usize,
        proposed_false_wins: usize,
        proposed_false_losses: usize,
        current_unnecessary_extensions: usize,
        proposed_avoids_extension: usize,
        current_extends_proposed_shortcuts: usize,
    }

    fn audit_zero_wall_state(
        p0: usize,
        p1: usize,
        stm: Player,
        oracle_table: &[i16],
        stats: &mut RaceAuditStats,
    ) -> Option<(RaceVerdict, RaceVerdict, RaceVerdict, i32)> {
        if p0 == p1 || p0 < 9 || p1 >= 72 {
            return None;
        }
        let board = race_board(board_pos_from_ace(p0), board_pos_from_ace(p1), stm);
        let g = ace_from_board(&board);
        let id = (p0 * 81 + p1) * 2 + (stm as usize);
        let exact_val = oracle_table[id];
        let exact = match exact_val.cmp(&0) {
            std::cmp::Ordering::Greater => RaceVerdict::Win,
            std::cmp::Ordering::Less => RaceVerdict::Loss,
            std::cmp::Ordering::Equal => RaceVerdict::NeedsProof,
        };
        let current = hands_empty_race(&g);
        let proposed = proposed_hands_empty_race(&g);
        let adj = turn_adjusted_tempo_advantage(&g);

        stats.states_checked += 1;
        if exact_val > 0 {
            if stm == Player::One {
                stats.exact_white_wins += 1;
            } else {
                stats.exact_black_wins += 1;
            }
        } else if exact_val < 0 {
            if stm == Player::One {
                stats.exact_black_wins += 1;
            } else {
                stats.exact_white_wins += 1;
            }
        } else {
            stats.exact_draws += 1;
        }

        if current == RaceVerdict::Win && exact != RaceVerdict::Win {
            stats.current_false_wins += 1;
        }
        if current == RaceVerdict::Loss && exact != RaceVerdict::Loss {
            stats.current_false_losses += 1;
        }
        if proposed == RaceVerdict::Win && exact != RaceVerdict::Win {
            stats.proposed_false_wins += 1;
        }
        if proposed == RaceVerdict::Loss && exact != RaceVerdict::Loss {
            stats.proposed_false_losses += 1;
        }
        if current == RaceVerdict::NeedsProof && proposed != RaceVerdict::NeedsProof {
            stats.proposed_avoids_extension += 1;
            if proposed == exact {
                stats.current_unnecessary_extensions += 1;
            }
        }
        if current == RaceVerdict::NeedsProof
            && (proposed == RaceVerdict::Win || proposed == RaceVerdict::Loss)
        {
            stats.current_extends_proposed_shortcuts += 1;
        }

        Some((exact, current, proposed, adj))
    }

    #[test]
    fn exhaustive_zero_wall_turn_adjusted_audit() {
        use crate::acev13::oracle::oracle_solve_board;

        let oracle_table = oracle_solve_board(&[0u8; 81]);
        let mut stats = RaceAuditStats::default();
        let mut counterexamples: Vec<String> = Vec::new();

        for p0 in 0..81 {
            for p1 in 0..81 {
                for stm in [Player::One, Player::Two] {
                    let Some((exact, current, proposed, adj)) =
                        audit_zero_wall_state(p0, p1, stm, &oracle_table, &mut stats)
                    else {
                        continue;
                    };
                    if proposed == RaceVerdict::Win && exact != RaceVerdict::Win {
                        counterexamples.push(format!(
                            "proposed false win p0={p0} p1={p1} stm={stm:?} adj={adj} exact={exact:?} current={current:?}"
                        ));
                    }
                    if proposed == RaceVerdict::Loss && exact != RaceVerdict::Loss {
                        counterexamples.push(format!(
                            "proposed false loss p0={p0} p1={p1} stm={stm:?} adj={adj} exact={exact:?} current={current:?}"
                        ));
                    }
                }
            }
        }

        eprintln!("zero-wall audit stats: {stats:#?}");
        if !counterexamples.is_empty() {
            for cx in counterexamples.iter().take(20) {
                eprintln!("COUNTEREXAMPLE: {cx}");
            }
            panic!(
                "proposed rule has {} false wins and {} false losses (showing up to 20)",
                stats.proposed_false_wins, stats.proposed_false_losses
            );
        }
        assert_eq!(stats.proposed_false_wins, 0);
        assert_eq!(stats.proposed_false_losses, 0);
        assert!(stats.states_checked > 10_000);
    }

    #[test]
    fn turn_adjusted_examples_from_spec() {
        // Discover concrete boards matching the spec's distance/adj examples.
        let mut plus_one_white_stm: Option<(usize, usize)> = None;
        let mut plus_one_black_stm: Option<(usize, usize)> = None;
        for p0 in 9..81 {
            for p1 in 0..72 {
                if p0 == p1 {
                    continue;
                }
                let g_w = ace_from_board(&race_board(
                    board_pos_from_ace(p0),
                    board_pos_from_ace(p1),
                    Player::One,
                ));
                let (wd, bd) = wall_graph_distances(&g_w);
                if bd as i32 - wd as i32 == 1 && turn_adjusted_tempo_advantage(&g_w) == 2 {
                    plus_one_white_stm = Some((p0, p1));
                }
                let g_b = ace_from_board(&race_board(
                    board_pos_from_ace(p0),
                    board_pos_from_ace(p1),
                    Player::Two,
                ));
                if bd as i32 - wd as i32 == 1 && turn_adjusted_tempo_advantage(&g_b) == 0 {
                    plus_one_black_stm = Some((p0, p1));
                }
            }
        }
        let (p0, p1) = plus_one_white_stm.expect("white +1 lead example");
        let g = ace_from_board(&race_board(
            board_pos_from_ace(p0),
            board_pos_from_ace(p1),
            Player::One,
        ));
        assert_eq!(turn_adjusted_tempo_advantage(&g), 2);
        assert_eq!(hands_empty_race(&g), RaceVerdict::Win);

        let (p0, p1) = plus_one_black_stm.expect("white +1 lead black-to-move example");
        let g = ace_from_board(&race_board(
            board_pos_from_ace(p0),
            board_pos_from_ace(p1),
            Player::Two,
        ));
        assert_eq!(turn_adjusted_tempo_advantage(&g), 0);
        assert_eq!(hands_empty_race(&g), RaceVerdict::NeedsProof);

        // Equal distance, White to move → adj=1 → resolve (same column overlaps).
        let g = ace_from_board(&race_board((3, 4), (5, 4), Player::One));
        assert_eq!(turn_adjusted_tempo_advantage(&g), 1);
        assert_eq!(hands_empty_race(&g), RaceVerdict::NeedsProof);

        // Equal distance, Black to move → adj=-1 → resolve.
        let g = ace_from_board(&race_board((3, 4), (5, 4), Player::Two));
        assert_eq!(turn_adjusted_tempo_advantage(&g), -1);
        assert_eq!(hands_empty_race(&g), RaceVerdict::NeedsProof);

        // Equal distance, separated paths, White to move → pure race win.
        let g = ace_from_board(&race_board((3, 1), (5, 7), Player::One));
        assert_eq!(turn_adjusted_tempo_advantage(&g), 1);
        assert_eq!(hands_empty_race(&g), RaceVerdict::Win);
    }

    #[test]
    fn separated_pure_race_matches_oracle_when_no_overlap() {
        use crate::acev13::oracle::oracle_solve_board;
        let oracle_table = oracle_solve_board(&[0u8; 81]);
        for p0 in 9..81 {
            for p1 in 0..72 {
                if p0 == p1 {
                    continue;
                }
                for stm in [Player::One, Player::Two] {
                    let board = race_board(board_pos_from_ace(p0), board_pos_from_ace(p1), stm);
                    let g = ace_from_board(&board);
                    let mut d0 = [0u8; 81];
                    let mut d1 = [0u8; 81];
                    g.compute_dist(0, &mut d0);
                    g.compute_dist(1, &mut d1);
                    if paths_overlap(&g, &d0, &d1) {
                        continue;
                    }
                    let id = (p0 * 81 + p1) * 2 + (stm as usize);
                    if oracle_table[id] == 0 {
                        continue;
                    }
                    let pure = separated_pure_race_verdict(&g);
                    let exact = if oracle_table[id] > 0 {
                        RaceVerdict::Win
                    } else {
                        RaceVerdict::Loss
                    };
                    assert_eq!(
                        pure, exact,
                        "separated pure race p0={p0} p1={p1} stm={stm:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn benchmark_extension_avoidance_vs_current() {
        use crate::acev13::oracle::oracle_solve_board;
        let oracle_table = oracle_solve_board(&[0u8; 81]);
        let mut stats = RaceAuditStats::default();
        for p0 in 9..81 {
            for p1 in 0..72 {
                if p0 == p1 {
                    continue;
                }
                for stm in [Player::One, Player::Two] {
                    let _ = audit_zero_wall_state(p0, p1, stm, &oracle_table, &mut stats);
                }
            }
        }
        eprintln!("zero-wall extension benchmark: {stats:#?}");
        assert!(stats.states_checked > 10_000);
    }

    // ── Cycle-safe race resolver tests ────────────────────────────────────────

    #[test]
    fn race_state_key_distinguishes_side_to_move() {
        let k0 = race_state_index(40, 50, 0);
        let k1 = race_state_index(40, 50, 1);
        assert_ne!(k0, k1);
        assert_eq!(k0, race_state_index(40, 50, 0));
    }

    #[test]
    fn race_state_index_fits_visiting_bitset() {
        assert_eq!(RACE_STATE_COUNT, 13_122);
        assert_eq!(RACE_VISITING_WORDS, 206);
        assert_eq!(race_state_index(80, 80, 1), 13_121);
        for p0 in 0..81 {
            for p1 in 0..81 {
                for turn in 0..2 {
                    let idx = race_state_index(p0, p1, turn);
                    assert!(
                        idx < RACE_VISITING_WORDS * 64,
                        "p0={p0} p1={p1} turn={turn} idx={idx}"
                    );
                }
            }
        }
        let mut visiting = [0u64; RACE_VISITING_WORDS];
        visiting_set(&mut visiting, 13_121);
        assert!(visiting_test(&visiting, 13_121));
        visiting_clear(&mut visiting, 13_121);
        assert!(!visiting_test(&visiting, 13_121));
    }

    #[test]
    fn aggregate_cycle_plus_winning_move() {
        assert_eq!(
            aggregate_opponent_child_proofs(&[RaceProof::Unknown, RaceProof::Loss]),
            RaceProof::Win
        );
    }

    #[test]
    fn aggregate_unknown_prevents_false_loss() {
        assert_eq!(
            aggregate_opponent_child_proofs(&[RaceProof::Win, RaceProof::Unknown]),
            RaceProof::Unknown
        );
    }

    #[test]
    fn aggregate_all_opponent_losses_is_stm_loss() {
        assert_eq!(
            aggregate_opponent_child_proofs(&[RaceProof::Win, RaceProof::Win]),
            RaceProof::Loss
        );
    }

    #[test]
    fn race_active_path_repetition_returns_unknown() {
        let mut visiting = [0u64; RACE_VISITING_WORDS];
        let idx = race_state_index(40, 50, 0);
        visiting_set(&mut visiting, idx);
        assert!(visiting_test(&visiting, idx));
    }

    #[test]
    fn race_depth_cap_returns_unknown() {
        let board = race_board((3, 4), (5, 4), Player::One);
        let mut g = ace_from_board(&board);
        assert_eq!(
            race_minimax_with_cap(&mut g, 0),
            RaceProof::Unknown,
            "depth cap must not certify"
        );
    }

    #[test]
    fn race_resolver_max_depth_on_empty_board() {
        let mut max_depth = 0u32;
        for p0 in 9..81 {
            for p1 in 0..72 {
                if p0 == p1 {
                    continue;
                }
                for stm in [Player::One, Player::Two] {
                    let board = race_board(board_pos_from_ace(p0), board_pos_from_ace(p1), stm);
                    let mut g = ace_from_board(&board);
                    if hands_empty_race(&g) != RaceVerdict::NeedsProof {
                        continue;
                    }
                    let mut resolver = RaceResolver::new(RACE_RESOLVER_DEPTH_CAP);
                    max_depth = max_depth.max(measure_race_depth(&mut g, &mut resolver));
                }
            }
        }
        eprintln!("race_minimax max observed depth (NeedsProof): {max_depth}");
        assert!(max_depth <= RACE_RESOLVER_DEPTH_CAP);
        assert!(max_depth > 0);
    }

    fn measure_race_depth(g: &mut AceGame, resolver: &mut RaceResolver) -> u32 {
        let saved_cap = resolver.depth_cap;
        resolver.depth_cap = u32::MAX / 2;
        let d = race_rec_depth_only(g, resolver, 0);
        resolver.depth_cap = saved_cap;
        d
    }

    fn race_rec_depth_only(g: &mut AceGame, resolver: &mut RaceResolver, cur: u32) -> u32 {
        let stm = g.turn;
        let pawn = g.pawn[stm];
        if (stm == 0 && pawn < 9) || (stm == 1 && pawn >= 72) {
            return cur;
        }
        let idx = race_state_index(g.pawn[0], g.pawn[1], g.turn);
        if visiting_test(&resolver.visiting, idx) {
            return cur;
        }
        visiting_set(&mut resolver.visiting, idx);
        let mut d_goal = [255u8; 81];
        g.compute_dist(stm, &mut d_goal);
        let my_dist = d_goal[pawn];
        let mut buf = [0i16; 16];
        let cnt = g.gen_pawn_moves(&mut buf, 0);
        let mut best = cur;
        for i in 0..cnt {
            let to = buf[i] as usize;
            if d_goal[to] < my_dist {
                g.make_move(buf[i]);
                best = best.max(race_rec_depth_only(g, resolver, cur + 1));
                g.unmake_move();
            }
        }
        visiting_clear(&mut resolver.visiting, idx);
        best
    }

    /// Pre-528d20b raw-tempo classifier (for regression counting only).
    fn old_raw_tempo_race(g: &AceGame) -> RaceVerdict {
        let mut d0 = [0u8; 81];
        let mut d1 = [0u8; 81];
        g.compute_dist(0, &mut d0);
        g.compute_dist(1, &mut d1);
        let big0 = d0[g.pawn[0]];
        let big1 = d1[g.pawn[1]];
        if big0 == 255 || big1 == 255 {
            return RaceVerdict::NeedsProof;
        }
        let (our, opp) = if g.turn == 0 {
            (big0 as i32, big1 as i32)
        } else {
            (big1 as i32, big0 as i32)
        };
        let tempo = opp - our;
        if tempo > 1 {
            return RaceVerdict::Win;
        }
        if tempo < -1 {
            return RaceVerdict::Loss;
        }
        if paths_overlap(g, &d0, &d1) {
            return RaceVerdict::NeedsProof;
        }
        if tempo < 0 {
            RaceVerdict::Loss
        } else {
            RaceVerdict::Win
        }
    }

    #[test]
    fn seventy_two_turn_adjusted_shortcuts_skip_resolver() {
        let mut shortcuts = 0usize;
        for p0 in 9..81 {
            for p1 in 0..72 {
                if p0 == p1 {
                    continue;
                }
                for stm in [Player::One, Player::Two] {
                    let g = ace_from_board(&race_board(
                        board_pos_from_ace(p0),
                        board_pos_from_ace(p1),
                        stm,
                    ));
                    if old_raw_tempo_race(&g) != RaceVerdict::NeedsProof {
                        continue;
                    }
                    if hands_empty_race(&g) == RaceVerdict::NeedsProof {
                        continue;
                    }
                    shortcuts += 1;
                }
            }
        }
        assert_eq!(shortcuts, 72, "528d20b shortcuts vs raw-tempo resolver");
    }

    #[derive(Default, Debug)]
    struct ExhaustiveResolverAudit {
        volatile: usize,
        exact_win: usize,
        exact_loss: usize,
        resolver_win: usize,
        resolver_loss: usize,
        resolver_unknown: usize,
        false_win: usize,
        false_loss: usize,
    }

    #[test]
    fn exhaustive_resolver_audit_no_false_certificates() {
        use crate::acev13::oracle::oracle_solve_board;
        let oracle_table = oracle_solve_board(&[0u8; 81]);
        let mut audit = ExhaustiveResolverAudit::default();

        for p0 in 9..81 {
            for p1 in 0..72 {
                if p0 == p1 {
                    continue;
                }
                for stm in [Player::One, Player::Two] {
                    let board = race_board(board_pos_from_ace(p0), board_pos_from_ace(p1), stm);
                    let mut g = ace_from_board(&board);
                    let id = (p0 * 81 + p1) * 2 + (stm as usize);
                    let exact_val = oracle_table[id];
                    let classifier = hands_empty_race(&g);

                    if classifier != RaceVerdict::NeedsProof {
                        if matches!(classifier, RaceVerdict::Win) && exact_val <= 0 {
                            audit.false_win += 1;
                        }
                        if matches!(classifier, RaceVerdict::Loss) && exact_val >= 0 {
                            audit.false_loss += 1;
                        }
                        continue;
                    }

                    audit.volatile += 1;
                    if exact_val > 0 {
                        audit.exact_win += 1;
                    } else if exact_val < 0 {
                        audit.exact_loss += 1;
                    }

                    match race_minimax(&mut g) {
                        RaceProof::Win => {
                            audit.resolver_win += 1;
                            if exact_val <= 0 {
                                audit.false_win += 1;
                            }
                        }
                        RaceProof::Loss => {
                            audit.resolver_loss += 1;
                            if exact_val >= 0 {
                                audit.false_loss += 1;
                            }
                        }
                        RaceProof::Unknown => audit.resolver_unknown += 1,
                    }
                }
            }
        }

        eprintln!("exhaustive resolver audit: {audit:#?}");
        assert_eq!(audit.false_win, 0);
        assert_eq!(audit.false_loss, 0);
        assert!(audit.volatile > 0);
    }
}
