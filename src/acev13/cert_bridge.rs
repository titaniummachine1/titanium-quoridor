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
            RaceVerdict::NeedsProof => race_minimax(&mut g) > 0,
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
// Used ONLY for the NeedsProof edge case: paths overlap AND delta < 2.
// The oracle builds all 13,122 (p0,p1,turn) states eagerly; here we only need
// the ~50-200 states reachable from the volatile position, so lazy forward
// minimax with a small HashMap is strictly cheaper.
//
// Soundness: in a wall-free pawn race, optimal play NEVER leaves the shortest-
// path set — deviating increases your distance while the opponent stays on
// schedule.  Jumps are included automatically: gen_pawn_moves emits them and
// they always satisfy `d_goal[to] < d_goal[pawn]` (land ≥1 step closer).

/// Solve a hands-empty pawn race for the side to move.
/// Returns `+1` (stm wins) or `-1` (stm loses). Never 0 — a race always ends.
pub fn race_minimax(g: &mut AceGame) -> i8 {
    let mut memo = std::collections::HashMap::with_capacity(64);
    race_rec(g, &mut memo)
}

fn race_rec(g: &mut AceGame, memo: &mut std::collections::HashMap<u32, i8>) -> i8 {
    let stm = g.turn;
    let pawn = g.pawn[stm];

    // Defensive guard: stm already at goal (we detect goal moves before recursing,
    // but be safe against the root being called on an already-won position).
    if (stm == 0 && pawn < 9) || (stm == 1 && pawn >= 72) {
        return 1;
    }

    // Key: 81 cells × 81 cells × 2 turns ≤ 13,122 states, fits in u32.
    let key = (g.pawn[0] * 162 + g.pawn[1] * 2 + g.turn) as u32;
    if let Some(&v) = memo.get(&key) {
        return v;
    }

    // Distance from every cell to stm's goal through the frozen wall graph.
    let mut d_goal = [255u8; 81];
    g.compute_dist(stm, &mut d_goal);
    let my_dist = d_goal[pawn];

    let mut buf = [0i16; 16];
    let cnt = g.gen_pawn_moves(&mut buf, 0);

    let mut result = -1i8; // pessimistic default: lose

    for i in 0..cnt {
        let to = buf[i] as usize;
        // Strict shortest-path filter: only moves that spend 1 tempo (strictly decrease distance).
        if d_goal[to] < my_dist {
            // Goal move → instant win.
            if (stm == 0 && to < 9) || (stm == 1 && to >= 72) {
                result = 1;
                break;
            }
            g.make_move(buf[i]);
            let child = race_rec(g, memo);
            g.unmake_move();
            if child < 0 {
                result = 1; // opponent loses from here → we win
                break;
            }
        }
    }

    memo.insert(key, result);
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

/// Classify a hands-empty (no walls left) race for the side to move using pure
/// tempo math plus path-overlap, reserving the heavy proof for the volatile case.
///
/// **Tempo** = `opp_dist − our_dist`: +1 tempo means stm is 1 move closer to their
/// goal. Delta=0 is broken by the stm's first-move advantage (they consume 1 tempo
/// by moving, so equal distance → stm wins in a pure race).
///
/// * `tempo > 1`: 2+ tempo lead absorbs any jump → stm wins, overlap or not.
/// * `tempo < -1`: 2+ tempo deficit, even a stm jump can't recover → stm loses.
/// * `-1 ≤ tempo ≤ 1`, no overlap: pure parallel race — stm wins if `tempo ≥ 0`.
/// * `-1 ≤ tempo ≤ 1`, paths overlap: jump can swing ±1 tempo → `NeedsProof`.
pub fn hands_empty_race(g: &AceGame) -> RaceVerdict {
    let mut d0 = [0u8; 81];
    let mut d1 = [0u8; 81];
    g.compute_dist(0, &mut d0);
    g.compute_dist(1, &mut d1);
    let big0 = d0[g.pawn[0]];
    let big1 = d1[g.pawn[1]];
    if big0 == 255 || big1 == 255 {
        return RaceVerdict::NeedsProof; // unreachable (shouldn't happen) → be safe
    }
    let (our, opp) = if g.turn == 0 {
        (big0 as i32, big1 as i32)
    } else {
        (big1 as i32, big0 as i32)
    };
    // tempo > 0 ⇒ stm is closer; tempo < 0 ⇒ stm is behind.
    let tempo = opp - our;

    if tempo > 1 {
        return RaceVerdict::Win; // ≥2 tempo lead absorbs any jump
    }
    if tempo < -1 {
        return RaceVerdict::Loss; // ≥2 tempo deficit, even a stm jump can't recover
    }
    // |tempo| ≤ 1: a jump can swing the result by ±1 tempo — overlap decides.
    if paths_overlap(g, &d0, &d1) {
        return RaceVerdict::NeedsProof;
    }
    // Pure parallel race (no jump possible): stm moves first.
    // tempo > 0: stm strictly closer → wins.
    // tempo == 0: equal distance but stm spends 1 tempo first → wins.
    // tempo < 0: opponent strictly closer → stm loses.
    if tempo < 0 {
        RaceVerdict::Loss
    } else {
        RaceVerdict::Win
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
                    let exact = if race_minimax(&mut g) > 0 {
                        RaceVerdict::Win
                    } else {
                        RaceVerdict::Loss
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
}
