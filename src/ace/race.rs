//! pathfix/RaceProof — wall-aware exact race endgame solver (both hands empty).
//!
//! Port of `solveRaceConfig` from quoridor_5.html (gen11_ghi). With no walls
//! left in either hand the wall graph is fixed, so the pawn race is a finite
//! two-player game over 81×81×2 states. A retrograde ply-round fixpoint
//! assigns every forced state its exact value:
//!   tbl[(p0*81+p1)*2+turn]:  +k = side to move wins in k plies (optimal),
//!                            -k = side to move loses in k plies (slowest),
//!                             0 = unresolved (proven draw / stalemate pocket).
//! Successors come from the engine's own `gen_pawn_moves` on the CURRENT
//! blocked[] board, so jump rules match the search exactly.

use crate::ace::game::AceGame;

/// 81 × 81 × 2 (p0 cell, p1 cell, side to move).
pub const RACE_STATES: usize = 13_122;

/// Race-proof score band: above every heuristic eval, below the true-mate band
/// (no TT mate ply-rescaling applies; race values are position-exact, so
/// storing them ply-unadjusted is sound). Win in k ⇒ `RACE_MATE - k`.
pub const RACE_MATE: i32 = 32_000;

/// Reusable solver scratch — successor graph + live worklist (~200 KB).
pub struct RaceScratch {
    succ: Box<[i16]>,  // 5 successor state ids per state; -1 = own goal row
    nsucc: Box<[u8]>,  // successor count per state
    live: Box<[i32]>,  // unresolved-state work list (compacted per round)
    buf: [i16; 16],    // gen_pawn_moves output (max 5 pawn moves)
}

impl RaceScratch {
    pub fn new() -> Self {
        Self {
            succ: vec![0i16; RACE_STATES * 5].into_boxed_slice(),
            nsucc: vec![0u8; RACE_STATES].into_boxed_slice(),
            live: vec![0i32; RACE_STATES].into_boxed_slice(),
            buf: [0; 16],
        }
    }
}

impl Default for RaceScratch {
    fn default() -> Self {
        Self::new()
    }
}

/// Solve the race for the game's current wall config into `tbl`.
/// `gen_pawn_moves` reads only pawn/turn/blocked — pawns and turn are
/// temporarily swept over all live states and restored afterwards.
pub fn solve_race_config(g: &mut AceGame, s: &mut RaceScratch, tbl: &mut [i16]) {
    debug_assert_eq!(tbl.len(), RACE_STATES);
    let (sp0, sp1, sturn) = (g.pawn[0], g.pawn[1], g.turn);
    tbl.fill(0);
    let mut n_live = 0usize;
    for p0 in 9..81usize {
        // p0 < 9: p0 already home -> not a live state
        g.pawn[0] = p0;
        for p1 in 0..72usize {
            // p1 >= 72: p1 already home -> not a live state
            if p1 == p0 {
                continue; // overlap: illegal
            }
            g.pawn[1] = p1;
            let base = (p0 * 81 + p1) * 2;
            g.turn = 0;
            let nm = g.gen_pawn_moves(&mut s.buf, 0);
            debug_assert!(nm <= 5);
            s.nsucc[base] = nm as u8;
            let off = base * 5;
            for j in 0..nm {
                let c = s.buf[j] as usize;
                s.succ[off + j] = if c < 9 {
                    -1
                } else {
                    ((c * 81 + p1) * 2 + 1) as i16
                };
            }
            s.live[n_live] = base as i32;
            n_live += 1;
            g.turn = 1;
            let nm = g.gen_pawn_moves(&mut s.buf, 0);
            debug_assert!(nm <= 5);
            s.nsucc[base + 1] = nm as u8;
            let off = (base + 1) * 5;
            for j in 0..nm {
                let c = s.buf[j] as usize;
                s.succ[off + j] = if c >= 72 {
                    -1
                } else {
                    ((p0 * 81 + c) * 2) as i16
                };
            }
            s.live[n_live] = (base + 1) as i32;
            n_live += 1;
        }
    }
    g.pawn[0] = sp0;
    g.pawn[1] = sp1;
    g.turn = sturn;

    // round k assigns exactly the value-k states: win-in-k iff cheapest assigned
    // loss successor is k-1 (immediate goal move = loss-in-0 for the new stm);
    // loss-in-k iff ALL successors are assigned wins and slowest is k-1. Values
    // assigned in the same round are visible but cannot satisfy the ==k filter,
    // so every state still gets its exact minimal (win) / maximal (loss) ply.
    let mut k: i32 = 1;
    while n_live > 0 && k < 1024 {
        let mut assigned = 0usize;
        let mut keep = 0usize;
        for i in 0..n_live {
            let id = s.live[i] as usize;
            let ns = s.nsucc[id] as usize;
            let mut min_loss: i32 = 32_767;
            let mut all_win = ns > 0;
            let mut max_win: i32 = 0;
            let off = id * 5;
            for j in 0..ns {
                let nid = s.succ[off + j];
                if nid < 0 {
                    min_loss = 0;
                    all_win = false;
                    continue;
                }
                let v = tbl[nid as usize] as i32;
                if v < 0 {
                    all_win = false;
                    if -v < min_loss {
                        min_loss = -v;
                    }
                } else if v > 0 {
                    if v > max_win {
                        max_win = v;
                    }
                } else {
                    all_win = false;
                }
            }
            if min_loss + 1 == k {
                tbl[id] = k as i16;
                assigned += 1;
                continue;
            }
            if all_win && max_win + 1 == k {
                tbl[id] = -k as i16;
                assigned += 1;
                continue;
            }
            s.live[keep] = id as i32;
            keep += 1;
        }
        n_live = keep;
        if assigned == 0 {
            break; // fixpoint: leftovers are genuine draws/pockets -> stay 0
        }
        k += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solved_empty_board() -> Vec<i16> {
        let mut g = AceGame::new();
        let mut s = RaceScratch::new();
        let mut tbl = vec![0i16; RACE_STATES];
        solve_race_config(&mut g, &mut s, &mut tbl);
        tbl
    }

    /// Every assigned value must satisfy the Bellman recurrence over the
    /// successor graph; the optimal-play definition admits exactly one such
    /// minimal fixpoint, so this validates soundness of every entry.
    #[test]
    fn race_table_is_bellman_consistent_on_sample_configs() {
        use crate::ace::algebraic_to_ace;
        let configs: [&[&str]; 3] = [
            &[],
            &["e2", "e8", "e3h", "e6h"],
            &["e2", "e8", "c3h", "f6v", "d7h", "b4v"],
        ];
        for moves in configs {
            let mut g = AceGame::new();
            for m in moves {
                g.make_move(algebraic_to_ace(m));
            }
            let mut s = RaceScratch::new();
            let mut tbl = vec![0i16; RACE_STATES];
            solve_race_config(&mut g, &mut s, &mut tbl);

            for id in 0..RACE_STATES {
                let v = tbl[id] as i32;
                if v == 0 {
                    continue;
                }
                let ns = s.nsucc[id] as usize;
                let off = id * 5;
                // recompute the state's value from successor values
                let mut min_loss = i32::MAX;
                let mut all_resolved_win = ns > 0;
                let mut max_win = 0i32;
                for j in 0..ns {
                    let nid = s.succ[off + j];
                    if nid < 0 {
                        min_loss = min_loss.min(0);
                        all_resolved_win = false;
                        continue;
                    }
                    let sv = tbl[nid as usize] as i32;
                    if sv < 0 {
                        all_resolved_win = false;
                        min_loss = min_loss.min(-sv);
                    } else if sv > 0 {
                        max_win = max_win.max(sv);
                    } else {
                        all_resolved_win = false;
                    }
                }
                if v > 0 {
                    assert_eq!(v, min_loss + 1, "win value mismatch at state {id}");
                } else {
                    assert!(all_resolved_win, "loss state {id} has a non-win successor");
                    assert_eq!(-v, max_win + 1, "loss value mismatch at state {id}");
                }
            }
        }
    }

    #[test]
    fn empty_board_head_on_race_is_movers_loss() {
        let tbl = solved_empty_board();
        // Symmetric same-file start (e1 vs e9): the side to move must
        // approach first and gets jumped over — mover LOSES in 16 plies.
        // (Real games never race head-on on one file; this pins the jump
        // dynamics the solver must capture.)
        let p0 = 76; // e1
        let p1 = 4; // e9
        assert_eq!(tbl[(p0 * 81 + p1) * 2], -16, "P0 to move loses head-on");
        assert_eq!(tbl[(p0 * 81 + p1) * 2 + 1], -16, "P1 to move loses head-on");
    }

    #[test]
    fn one_step_from_goal_wins_immediately() {
        let tbl = solved_empty_board();
        // p0 on row 1 (one step from its goal row 0), p1 mid-board (a cell
        // off its own goal row — goal-row cells are not live states).
        let p0 = 13; // row 1, col 4
        let p1 = 40; // row 4, col 4
        assert_eq!(tbl[(p0 * 81 + p1) * 2], 1);
    }
}
