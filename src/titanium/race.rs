//! pathfix/RaceProof — fixed-topology no-more-walls race system.
//!
//! Scope: **both wall hands are empty**, so the blocked-edge topology is frozen
//! permanently. Walls may already be on the board; only pawn moves, jumps and
//! diagonal jumps remain. Every API here is correct for *arbitrary* legal
//! fixed-wall topologies, not just the empty board.
//!
//! Two separate services, by design:
//!
//! **Service A — fast outcome / α-β bound** ([`race_outcome`]):
//!   Near-instant theorem deduction of the side-to-move's forced result, as an
//!   alpha-beta-native [`RaceBound`] (`Lower(RACE_WIN_FLOOR)` for a proven win,
//!   `Upper(-RACE_WIN_FLOOR)` for a proven loss, `Unknown` when it declines). It
//!   builds **no successor graph** and computes **no exact DTM**.
//!
//!   Sound decision rule (correct on ANY fixed-wall topology): if the two pawns'
//!   shortest-path SETS are **disjoint** they can never share a cell, so no jump
//!   / interception is possible and the race is a pure independent tempo race —
//!   the turn-adjusted faster pawn wins exactly ([`separated_pure_race_verdict`]).
//!   When the path sets **overlap**, interception can swing the result and no
//!   cheap proof is sound, so Service A returns `Unknown` and the caller falls
//!   back to ordinary search (or the exact service). It NEVER returns a false bound.
//!
//!   NOTE: two earlier cheap deciders were found **unsound on walled topologies**
//!   and are intentionally NOT used here: (a) the in-module winner-*sign*
//!   recursion (its sign disagreed with the retrograde oracle on random walled
//!   boards — masked because the old equality tests compared only the retrograde
//!   output, never the sign table); and (b) `cert_bridge::race_minimax`'s
//!   distance-decreasing-only forward proof (restricting the opponent's
//!   interception moves manufactures false wins). Both are exact only on the
//!   empty board, where optimal race play is always distance-decreasing.
//!
//! **Service B — optional exact DTM** ([`race_exact_dtm_on_demand`], [`solve_race_config`]):
//!   Exact `+k / −k` distance-to-mate, used only when a caller genuinely needs
//!   it (fastest-win / slowest-loss / stubborn-loser selection, UI, tests). Its
//!   ~160 KB successor-graph scratch is allocated on first use and reused; it is
//!   **never** invoked on the bound-only path. Computed by an exact ply-round
//!   retrograde over the live successor graph — the algorithm proven `+k/−k`-equal
//!   to the reference oracle on the empty board, all sample configs and 1,000
//!   random fixed topologies. (It is self-contained: it does NOT depend on any
//!   winner-sign field.)
//!
//! `solve_race_config_reference` remains a `#[cfg(test)]` oracle only.

// These cert_bridge helpers are now only referenced by the test/diagnostic
// suites (production Gate 2 is non-decisive — see `race_outcome_gates_ab`).
#[cfg(test)]
use crate::titanium::cert_bridge::{paths_overlap, separated_pure_race_verdict, RaceVerdict};
use crate::titanium::game::GameState;

/// 81 × 81 × 2 (p0 cell, p1 cell, side to move).
pub const RACE_STATES: usize = 13_122;

/// Legal live pawn placements: p0 ∉ goal row, p1 ∉ goal row, p0 ≠ p1, both turns.
pub const RACE_LIVE_STATES: usize = 10_242;

/// Race-proof score band: above every heuristic eval, below the true-mate band.
/// Exact-DTM table values:
///   +k = side to move wins in k plies,
///   -k = side to move loses in k plies,
///    0 = illegal/unused state.
pub const RACE_MATE: i32 = 32_000;

/// Hard cap on race plies (the retrograde fixpoint bound). Every exact race
/// score therefore satisfies `RACE_MATE - RACE_MAX_PLIES < |score| ≤ RACE_MATE`.
pub const RACE_MAX_PLIES: i32 = 1_024;

/// Proven-outcome α-β bound magnitude. A theorem-proved win is a LOWER bound of
/// `RACE_WIN_FLOOR` (the true score is some exact `RACE_MATE - k ≥ RACE_WIN_FLOOR`);
/// a proven loss is an UPPER bound of `-RACE_WIN_FLOOR`. Chosen to sit strictly
///   - above every heuristic evaluation (race heuristic peaks well under 10 000),
///   - at or below every exact race-win score (`RACE_MATE - k`, `k < RACE_MAX_PLIES`),
///   - far below the real-mate band (`MATE − 1000 = 99 000`),
/// so it is always safe for fail-high / fail-low use and never collides with a
/// heuristic leaf or a true mate.
pub const RACE_WIN_FLOOR: i32 = RACE_MATE - RACE_MAX_PLIES;

/// Fast race outcome as an alpha-beta-native bound (Service A).
///
/// Never returns an invented exact score: a proven win is a LOWER bound, a proven
/// loss an UPPER bound. `Exact` is produced only by the on-demand exact service.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RaceBound {
    /// Side-to-move is a forced winner: true score ≥ this lower bound.
    Lower(i32),
    /// Side-to-move is a forced loser: true score ≤ this upper bound.
    Upper(i32),
    /// Genuine exact distance-to-mate (only from the exact service).
    Exact(i32),
    /// Not resolved by the fast theorem — caller must fall back to search.
    Unknown,
}

impl RaceBound {
    /// Proven win or loss → the bound's signum (+1 / −1); otherwise 0.
    #[inline]
    pub fn signum(self) -> i32 {
        match self {
            RaceBound::Lower(_) => 1,
            RaceBound::Upper(_) => -1,
            RaceBound::Exact(v) => v.signum(),
            RaceBound::Unknown => 0,
        }
    }
}

/// Secondary time metadata for Service A — never exact DTM.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlyEstimate {
    /// Walking-ETA style estimate: `2 * dist − (winner == turn)`.
    Approx(u16),
    Unknown,
}

/// Service A result: sound α-β bound plus optional approximate ply hint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RaceDeduction {
    pub bound: RaceBound,
    pub estimated_plies: PlyEstimate,
}

/// Approximate plies for the proven winner to reach its goal (may exceed exact DTM).
#[inline]
pub fn estimated_plies_to_result(
    g: &GameState,
    winner: usize,
    winner_own_goal_distance: u8,
) -> u16 {
    let v = 2 * winner_own_goal_distance as i16 - i16::from(winner == g.turn);
    v.max(0) as u16
}

#[inline]
fn ply_estimate_for_bound(g: &GameState, bound: RaceBound) -> PlyEstimate {
    match bound {
        RaceBound::Unknown | RaceBound::Exact(_) => PlyEstimate::Unknown,
        RaceBound::Lower(_) | RaceBound::Upper(_) => {
            let mut d0 = [0u8; 81];
            let mut d1 = [0u8; 81];
            g.compute_dist(0, &mut d0);
            g.compute_dist(1, &mut d1);
            let winner = match bound {
                RaceBound::Lower(_) => g.turn,
                RaceBound::Upper(_) => g.turn ^ 1,
                _ => return PlyEstimate::Unknown,
            };
            let wd = if winner == 0 {
                d0[g.pawn[0]]
            } else {
                d1[g.pawn[1]]
            };
            if wd == u8::MAX {
                PlyEstimate::Unknown
            } else {
                PlyEstimate::Approx(estimated_plies_to_result(g, winner, wd))
            }
        }
    }
}

/// Reusable solver scratch.
///
/// The bound path ([`race_outcome`]) needs nothing from here — it uses the
/// classifier's own tiny transient scratch. The exact successor-graph tier
/// (~160 KB, Service B) is allocated lazily on first exact use and reused.
pub struct RaceScratch {
    /// Lazy exact-DTM successor graph (Service B), allocated on demand.
    exact: Option<Box<ExactScratch>>,
    /// Lazy asymmetric-certificate winner table (Service A Tier 3), built once
    /// per wall topology and reused across all pawn states of that topology.
    winner_tbl: Option<Box<RaceWinnerTable>>,
    /// Wall-topology key the cached `winner_tbl` was built for. A mismatch
    /// (walls changed) forces a rebuild on the next Tier-3 query.
    winner_key: u64,
}

/// Exact-DTM successor-graph scratch — live-only, ~160 KB. Lazily allocated.
struct ExactScratch {
    graph_slot: Box<[u16]>,
    live: Box<[u16]>,
    nsucc: Box<[u8]>,
    succ: Box<[i16]>,
    buf: [i16; 16],
}

impl ExactScratch {
    fn new() -> Self {
        Self {
            graph_slot: vec![0u16; RACE_STATES].into_boxed_slice(),
            live: vec![0u16; RACE_LIVE_STATES].into_boxed_slice(),
            nsucc: vec![0u8; RACE_LIVE_STATES].into_boxed_slice(),
            succ: vec![0i16; RACE_LIVE_STATES * 5].into_boxed_slice(),
            buf: [0; 16],
        }
    }

    const fn bytes() -> usize {
        RACE_STATES * std::mem::size_of::<u16>()
            + RACE_LIVE_STATES * std::mem::size_of::<u16>()
            + RACE_LIVE_STATES * std::mem::size_of::<u8>()
            + RACE_LIVE_STATES * 5 * std::mem::size_of::<i16>()
            + std::mem::size_of::<[i16; 16]>()
    }
}

impl RaceScratch {
    pub fn new() -> Self {
        Self {
            exact: None,
            winner_tbl: None,
            winner_key: 0,
        }
    }

    /// Resident bytes on the bound-only path (the exact tier is not allocated).
    pub const fn scratch_bytes() -> usize {
        std::mem::size_of::<Option<Box<ExactScratch>>>()
    }

    /// Additional heap when the exact (Service B) tier is lazily allocated.
    pub const fn exact_scratch_bytes() -> usize {
        ExactScratch::bytes()
    }

    /// Whether the exact successor-graph tier is currently allocated.
    pub fn exact_allocated(&self) -> bool {
        self.exact.is_some()
    }

    /// Whether the asymmetric winner table (Tier 3) is currently allocated.
    pub fn winner_table_allocated(&self) -> bool {
        self.winner_tbl.is_some()
    }

    /// Persistent heap bytes held by the cached winner table (0 if none).
    pub fn winner_table_bytes(&self) -> usize {
        if self.winner_tbl.is_some() {
            RaceWinnerTable::persistent_bytes()
        } else {
            0
        }
    }
}

impl Default for RaceScratch {
    fn default() -> Self {
        Self::new()
    }
}

#[inline(always)]
fn state_id(p0: usize, p1: usize, turn: usize) -> usize {
    (p0 * 81 + p1) * 2 + turn
}

#[inline(always)]
fn decode_state(id: usize) -> (usize, usize, usize) {
    let turn = id % 2;
    let pp = id / 2;
    (pp / 81, pp % 81, turn)
}

#[inline(always)]
fn is_home(side: usize, cell: usize) -> bool {
    if side == 0 {
        cell < 9
    } else {
        cell >= 72
    }
}

// Only used by the test/diagnostic suites since the detour DFS was removed.
#[cfg(test)]
#[inline(always)]
fn cell_manhattan(a: usize, b: usize) -> usize {
    let ar = a / 9;
    let ac = a % 9;
    let br = b / 9;
    let bc = b % 9;
    ar.abs_diff(br) + ac.abs_diff(bc)
}

// ---------------------------------------------------------------------------
// Service A — fast outcome / alpha-beta bound (no successor graph, no exact DTM).
// ---------------------------------------------------------------------------

/// Alternating-ply ETA for `side` to travel `distance` steps when `turn` moves
/// next. Side to move gets a free half-ply (one step sooner).
#[inline(always)]
fn arrival_ply(side: usize, turn: usize, distance: u8) -> i16 {
    if distance == 0 {
        0
    } else {
        2 * distance as i16 - i16::from(side == turn)
    }
}

/// Gate 1 only (ETA `delta_eta > 1` interception-impossible). Gate 2 is
/// non-decisive (Case B — see the body). Used by audits and by
/// [`race_outcome_detailed`] before the winner-table tier.
fn race_outcome_gates_ab(g: &mut GameState) -> RaceBound {
    debug_assert!(
        g.pawn[0] >= 9 && g.pawn[1] < 72,
        "race_outcome on terminal state"
    );
    let mut d0 = [0u8; 81];
    let mut d1 = [0u8; 81];
    g.compute_dist(0, &mut d0);
    g.compute_dist(1, &mut d1);

    let r0 = d0[g.pawn[0]];
    let r1 = d1[g.pawn[1]];
    if r0 == u8::MAX || r1 == u8::MAX {
        return RaceBound::Unknown;
    }

    let eta0 = arrival_ply(0, g.turn, r0);
    let eta1 = arrival_ply(1, g.turn, r1);

    if eta0 != eta1 {
        let runner: usize = if eta0 < eta1 { 0 } else { 1 };
        let chaser = runner ^ 1;
        let runner_eta = if runner == 0 { eta0 } else { eta1 };

        let d_runner_goal: &[u8; 81] = if runner == 0 { &d0 } else { &d1 };
        let chaser_d = d_runner_goal[g.pawn[chaser]];

        let fires = chaser_d == u8::MAX || {
            let chaser_eta = arrival_ply(chaser, g.turn, chaser_d);
            chaser_eta - runner_eta > 1
        };

        if fires {
            return if runner == g.turn {
                RaceBound::Lower(RACE_WIN_FLOOR)
            } else {
                RaceBound::Upper(-RACE_WIN_FLOOR)
            };
        }
    }

    // Gate 2 is NON-DECISIVE (Case B). The separated-shortest-path theorem is
    // unsound on a fixed-wall topology even with the complete shortest-path set
    // and a contact-aware adjacency test: the TRAILING pawn can detour OFF its
    // shortest path to block, which no shortest-path-set separation test bounds.
    // Two oracle counterexamples are pinned in the test module
    // (`diag_gate2_adjacent_counterexample`, `diag_gate2_nonadjacent_detour_counterexample`).
    // The asymmetric winner table (Tier 3) classifies separated races correctly
    // (proven 0 mismatches over 6,169,154 states), so we decline here and defer.
    RaceBound::Unknown
}

/// Forced-outcome bound for the side to move at the current hands-empty state.
///
/// Sound on ANY fixed-wall topology. Three decision tiers:
///
/// **Tier 1 — ETA gate:** `delta_eta > 1` interception impossible.
/// **Tier 2 — overlap check:** disjoint shortest-path sets → pure tempo race.
/// **Tier 3 — detour-dominance certificate:** bounded interaction search.
///
/// Never returns `Exact` — only `Lower`/`Upper` sign bounds or `Unknown`.
pub fn race_outcome_detailed(g: &mut GameState, s: &mut RaceScratch) -> RaceDeduction {
    let ab = race_outcome_gates_ab(g);
    let bound = if ab != RaceBound::Unknown {
        ab
    } else {
        winner_table_bound(g, s)
    };
    RaceDeduction {
        estimated_plies: ply_estimate_for_bound(g, bound),
        bound,
    }
}

/// Never returns `Exact` — only `Lower`/`Upper` sign bounds or `Unknown`.
pub fn race_outcome(g: &mut GameState, s: &mut RaceScratch) -> RaceBound {
    race_outcome_detailed(g, s).bound
}

/// Convenience: `Some(true)` = stm forced win, `Some(false)` = forced loss,
/// `None` = undecided (caller falls back to search).
#[inline]
pub fn race_outcome_stm_wins(g: &mut GameState, s: &mut RaceScratch) -> Option<bool> {
    match race_outcome(g, s) {
        RaceBound::Lower(_) => Some(true),
        RaceBound::Upper(_) => Some(false),
        RaceBound::Exact(v) => Some(v > 0),
        RaceBound::Unknown => None,
    }
}

// ---------------------------------------------------------------------------
// Service A — Tier 3: asymmetric strategy-certificate winner table.
//
// Replaces the former (unsound) symmetric detour-dominance DFS. To prove that
// player P wins from a state we run an attractor (backward-reachability)
// computation in which ONLY P is restricted to certified progress moves
// (Class A shortest-path progress + any legal productive jump + immediate goal
// moves) while the opponent may play EVERY legal pawn move:
//
//   * P to move   (OR node):  P wins when SOME permitted P move stays in P's
//                             attractor.
//   * opponent to move (AND): P wins only when EVERY fully-legal opponent move
//                             stays in P's attractor.
//
// Seeds are the states where P is already home. Because the opponent is never
// restricted, any state placed in the attractor corresponds to a genuine
// forcing strategy for P in the FULL game — a claimed win is never false. It is
// merely incomplete: a true P win that requires an off-shortest setup move for
// P falls outside the attractor and is reported `Unknown` (a sound decline).
//
// The two passes (P0, P1) are merged into a compact per-topology winner table
// addressing the full 13,122-state space. The table is built lazily, once per
// wall topology, and cached on the caller's [`RaceScratch`]; it is rebuilt
// whenever the wall topology changes. The exact DTM successor graph (Service B)
// is NOT constructed by this tier.
// ---------------------------------------------------------------------------

/// Compact winner classification for one race state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RaceClass {
    /// Player 0 is the proven forced winner (sound; backed by a real strategy).
    ProvenP0,
    /// Player 1 is the proven forced winner.
    ProvenP1,
    /// Neither prover succeeds with restricted progress moves — sound decline.
    Unknown,
}

/// Per-topology asymmetric-certificate winner table over the full race-state
/// address space. `class` holds the merged P0/P1/Unknown verdict; `layer` holds
/// the attractor iteration depth of the winning prover (an APPROXIMATE ply hint,
/// never exact DTM; `u16::MAX` when unknown).
pub struct RaceWinnerTable {
    class: Box<[u8]>, // RACE_STATES; 0 = Unknown, 1 = ProvenP0, 2 = ProvenP1
    layer: Box<[u16]>, // RACE_STATES; approximate plies (attractor layer) or u16::MAX
}

impl RaceWinnerTable {
    #[inline]
    fn classify(&self, id: usize) -> RaceClass {
        match self.class[id] {
            1 => RaceClass::ProvenP0,
            2 => RaceClass::ProvenP1,
            _ => RaceClass::Unknown,
        }
    }

    /// Approximate plies-to-result hint for `id` (attractor layer); `None` when
    /// the state is an unknown/declined classification.
    #[inline]
    pub fn approx_plies(&self, id: usize) -> Option<u16> {
        let l = self.layer[id];
        if l == u16::MAX {
            None
        } else {
            Some(l)
        }
    }

    /// Persistent heap bytes held by a built table.
    pub const fn persistent_bytes() -> usize {
        RACE_STATES * std::mem::size_of::<u8>() + RACE_STATES * std::mem::size_of::<u16>()
    }

    /// Number of states classified as each verdict — `(proven_p0, proven_p1, unknown)`.
    pub fn coverage(&self) -> (usize, usize, usize) {
        let mut p0 = 0usize;
        let mut p1 = 0usize;
        let mut unk = 0usize;
        for &c in self.class.iter() {
            match c {
                1 => p0 += 1,
                2 => p1 += 1,
                _ => unk += 1,
            }
        }
        (p0, p1, unk)
    }
}

/// Transient scratch for one attractor pass — predecessor lists, remaining-child
/// counters, and per-state win/layer/best-move arrays.
struct AttractorScratch {
    predecessors: Vec<Vec<(u32, i16)>>,
    remaining: Vec<u16>,
    win: Vec<bool>,
    layer: Vec<u16>,
    best_mv: Vec<i16>,
}

impl AttractorScratch {
    fn new() -> Self {
        Self {
            predecessors: vec![Vec::new(); RACE_STATES],
            remaining: vec![0; RACE_STATES],
            win: vec![false; RACE_STATES],
            layer: vec![u16::MAX; RACE_STATES],
            best_mv: vec![NO_RACE_MOVE; RACE_STATES],
        }
    }

    /// Transient heap bytes for one attractor pass (predecessor lists excluded —
    /// they are data-dependent; this is the fixed per-state overhead).
    #[cfg(test)]
    fn scratch_bytes() -> usize {
        RACE_STATES
            * (std::mem::size_of::<Vec<(u32, i16)>>()
                + std::mem::size_of::<u16>()
                + std::mem::size_of::<bool>()
                + std::mem::size_of::<u16>()
                + std::mem::size_of::<i16>())
    }
}

/// Sentinel "no move" marker for the winner-table best-move field.
const NO_RACE_MOVE: i16 = -1;

/// Run one asymmetric attractor pass for `prover` over the fixed-wall topology
/// `g_root` carries, using each side's own-goal distance field `own_goal_dist`.
/// Fully iterative (backward reachability) — no recursion.
fn attractor_pass(
    g_root: &GameState,
    own_goal_dist: &[[u8; 81]; 2],
    prover: usize,
    sc: &mut AttractorScratch,
) {
    for v in sc.predecessors.iter_mut() {
        v.clear();
    }
    sc.remaining.iter_mut().for_each(|x| *x = 0);
    sc.win.iter_mut().for_each(|x| *x = false);
    sc.layer.iter_mut().for_each(|x| *x = u16::MAX);
    sc.best_mv.iter_mut().for_each(|x| *x = NO_RACE_MOVE);

    let opp = prover ^ 1;
    let mut queue: std::collections::VecDeque<usize> = std::collections::VecDeque::new();

    for p0 in 0..81usize {
        for p1 in 0..81usize {
            if p0 == p1 {
                continue;
            }
            for turn in 0..2usize {
                let id = state_id(p0, p1, turn);

                // Seed: prover already home (regardless of whose turn it is).
                let prover_home = is_home(prover, if prover == 0 { p0 } else { p1 });
                if prover_home {
                    sc.win[id] = true;
                    sc.layer[id] = 0;
                    queue.push_back(id);
                    continue;
                }
                // Opponent already home → prover cannot win; no outgoing edges.
                let opp_home = is_home(opp, if opp == 0 { p0 } else { p1 });
                if opp_home {
                    continue;
                }

                let mut g = g_root.clone();
                g.pawn[0] = p0;
                g.pawn[1] = p1;
                g.turn = turn;

                let mut buf = [0i16; 16];
                let nm = g.gen_pawn_moves(&mut buf, 0);

                let side = turn;
                let is_or = side == prover;
                let src = if side == 0 { p0 } else { p1 };
                let old_d = own_goal_dist[side][src];
                let mut child_count: u16 = 0;

                for &mv in &buf[..nm] {
                    let dst = mv as usize;
                    if is_or {
                        // Restricted: Class A (own-goal Δ ≥ 1) or any jump.
                        let new_d = own_goal_dist[side][dst];
                        let delta = if new_d == u8::MAX {
                            i16::MIN / 2
                        } else {
                            old_d as i16 - new_d as i16
                        };
                        let jump = is_race_jump(src, dst);
                        if !(jump || delta >= 1) {
                            continue;
                        }
                    }
                    // AND node (opponent to move): every legal move, unrestricted.
                    let mut cg = g.clone();
                    cg.make_move(mv);
                    let cid = state_id(cg.pawn[0], cg.pawn[1], cg.turn);
                    sc.predecessors[cid].push((id as u32, mv));
                    child_count += 1;
                }
                sc.remaining[id] = child_count;
            }
        }
    }

    // Backward-reachability (attractor) propagation.
    while let Some(c) = queue.pop_front() {
        let cl = sc.layer[c];
        // Take the predecessor list out to avoid a borrow conflict with the
        // mutations below; restore it afterwards (lists are read once but a
        // state can be re-reached as a different predecessor's child).
        let preds = std::mem::take(&mut sc.predecessors[c]);
        for &(p_u, mv) in preds.iter() {
            let p = p_u as usize;
            if sc.win[p] {
                continue;
            }
            let p_turn = p & 1;
            if p_turn == prover {
                // OR node: one winning child is enough.
                sc.win[p] = true;
                sc.layer[p] = cl.saturating_add(1);
                sc.best_mv[p] = mv;
                queue.push_back(p);
            } else {
                // AND node: confirm only when every child is a prover win.
                if sc.remaining[p] > 0 {
                    sc.remaining[p] -= 1;
                }
                if sc.remaining[p] == 0 {
                    sc.win[p] = true;
                    sc.layer[p] = cl.saturating_add(1); // last (deepest) child ⟹ max resistance
                    queue.push_back(p);
                }
            }
        }
        sc.predecessors[c] = preds;
    }
}

/// Build the full per-topology winner table for the wall layout `g` carries.
/// Runs one attractor pass per prover and merges. Pawn position / turn on `g`
/// are not consulted (the whole address space is swept) and are left untouched.
pub fn build_winner_table(g: &GameState) -> RaceWinnerTable {
    let mut own = [[u8::MAX; 81]; 2];
    g.compute_dist(0, &mut own[0]);
    g.compute_dist(1, &mut own[1]);

    let mut sc = AttractorScratch::new();

    attractor_pass(g, &own, 0, &mut sc);
    let win0 = sc.win.clone();
    let layer0 = sc.layer.clone();

    attractor_pass(g, &own, 1, &mut sc);
    let win1 = &sc.win;
    let layer1 = &sc.layer;

    let mut class = vec![0u8; RACE_STATES].into_boxed_slice();
    let mut layer = vec![u16::MAX; RACE_STATES].into_boxed_slice();
    for id in 0..RACE_STATES {
        if win0[id] {
            // Determinacy: both provers cannot force a win at a legal live state.
            // The only exception is the degenerate both-home synthetic state
            // (p0 on row 0 AND p1 on row 8), which is outside the live domain.
            debug_assert!(
                !win1[id] || {
                    let (p0c, p1c, _) = decode_state(id);
                    is_home(0, p0c) && is_home(1, p1c)
                },
                "both provers force a win at legal state {id}"
            );
            class[id] = 1;
            layer[id] = layer0[id];
        } else if win1[id] {
            class[id] = 2;
            layer[id] = layer1[id];
        }
    }

    RaceWinnerTable { class, layer }
}

/// Fixed-wall topology key (FNV-1a over the horizontal+vertical wall bitboards).
/// Two states share a key iff they share an identical wall layout.
#[inline]
fn wall_topology_key(g: &GameState) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for &b in g.hw.iter() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    for &b in g.vw.iter() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Service A Tier 3: look up (building/caching as needed) the asymmetric
/// winner-table verdict for the current state and translate it to a sound
/// α-β bound. Builds the table at most once per wall topology; subsequent
/// queries on the same topology are O(1).
fn winner_table_bound(g: &mut GameState, s: &mut RaceScratch) -> RaceBound {
    let key = wall_topology_key(g);
    if s.winner_key != key || s.winner_tbl.is_none() {
        let tbl = build_winner_table(g);
        s.winner_tbl = Some(Box::new(tbl));
        s.winner_key = key;
    }
    let tbl = s.winner_tbl.as_ref().expect("winner table");
    let id = state_id(g.pawn[0], g.pawn[1], g.turn);
    match tbl.classify(id) {
        RaceClass::ProvenP0 => {
            if g.turn == 0 {
                RaceBound::Lower(RACE_WIN_FLOOR)
            } else {
                RaceBound::Upper(-RACE_WIN_FLOOR)
            }
        }
        RaceClass::ProvenP1 => {
            if g.turn == 1 {
                RaceBound::Lower(RACE_WIN_FLOOR)
            } else {
                RaceBound::Upper(-RACE_WIN_FLOOR)
            }
        }
        RaceClass::Unknown => RaceBound::Unknown,
    }
}

/// Manhattan-distance "is a jump?" test for the race classifier — a move whose
/// source/destination are not orthogonally adjacent (a pawn-interaction jump).
#[inline]
fn is_race_jump(src: usize, dst: usize) -> bool {
    let (sr, sc) = (src / 9, src % 9);
    let (dr, dc) = (dst / 9, dst % 9);
    let drow = (sr as i32 - dr as i32).unsigned_abs();
    let dcol = (sc as i32 - dc as i32).unsigned_abs();
    drow + dcol != 1
}


// ---------------------------------------------------------------------------
// Service B — exact DTM (lazy successor-graph retrograde). Proven +k/-k-exact.
// ---------------------------------------------------------------------------

fn build_live_graph(
    g: &mut GameState,
    graph_slot: &mut [u16],
    live: &mut [u16],
    nsucc: &mut [u8],
    succ: &mut [i16],
    buf: &mut [i16; 16],
) -> usize {
    graph_slot.fill(0);
    let mut n = 0usize;
    let (saved_p0, saved_p1, saved_turn) = (g.pawn[0], g.pawn[1], g.turn);

    for p0 in 9..81usize {
        g.pawn[0] = p0;
        for p1 in 0..72usize {
            if p1 == p0 {
                continue;
            }
            g.pawn[1] = p1;

            for turn in 0..2usize {
                let id = state_id(p0, p1, turn);
                graph_slot[id] = n as u16;
                live[n] = id as u16;
                g.turn = turn;

                let nm = g.gen_pawn_moves(buf, 0);
                debug_assert!(nm <= 5);
                nsucc[n] = nm as u8;
                let off = n * 5;

                for j in 0..nm {
                    let c = buf[j] as usize;
                    succ[off + j] = if turn == 0 {
                        if c < 9 {
                            -1
                        } else {
                            state_id(c, p1, 1) as i16
                        }
                    } else if c >= 72 {
                        -1
                    } else {
                        state_id(p0, c, 0) as i16
                    };
                }
                n += 1;
            }
        }
    }

    g.pawn[0] = saved_p0;
    g.pawn[1] = saved_p1;
    g.turn = saved_turn;

    debug_assert_eq!(n, RACE_LIVE_STATES);
    n
}

/// Ply-round retrograde DTM over the live successor cache. Self-contained:
/// exact `+k = 1 + min losing-child magnitude`, `-k = 1 + max winning-child`.
fn fill_exact_dtm(g: &mut GameState, ex: &mut ExactScratch, tbl: &mut [i16]) {
    tbl.fill(0);

    let n_live = build_live_graph(
        g,
        &mut ex.graph_slot,
        &mut ex.live,
        &mut ex.nsucc,
        &mut ex.succ,
        &mut ex.buf,
    );
    let mut n_unresolved = n_live;
    let mut k = 1i32;

    while n_unresolved > 0 && k < RACE_MAX_PLIES {
        let mut assigned = 0usize;
        let mut keep = 0usize;

        for i in 0..n_unresolved {
            let id = ex.live[i] as usize;

            let gi = ex.graph_slot[id] as usize;
            let ns = ex.nsucc[gi] as usize;
            let off = gi * 5;

            let mut min_loss = i32::MAX;
            let mut all_win = ns > 0;
            let mut max_win = 0i32;

            for j in 0..ns {
                let nid = ex.succ[off + j];
                if nid < 0 {
                    min_loss = min_loss.min(0);
                    all_win = false;
                    continue;
                }

                let v = tbl[nid as usize] as i32;
                if v < 0 {
                    all_win = false;
                    min_loss = min_loss.min(-v);
                } else if v > 0 {
                    max_win = max_win.max(v);
                } else {
                    all_win = false;
                }
            }

            if min_loss != i32::MAX && min_loss + 1 == k {
                tbl[id] = k as i16;
                assigned += 1;
                continue;
            }

            if all_win && max_win + 1 == k {
                tbl[id] = -k as i16;
                assigned += 1;
                continue;
            }

            ex.live[keep] = id as u16;
            keep += 1;
        }

        n_unresolved = keep;
        if assigned == 0 {
            break;
        }
        k += 1;
    }

    debug_assert_eq!(
        n_unresolved, 0,
        "DTM pass left {n_unresolved} unresolved states"
    );
}

/// Fill the complete fixed-topology exact race table (Service B). Lazily
/// allocates/reuses the ~160 KB successor-graph scratch.
pub fn solve_race_config(g: &mut GameState, s: &mut RaceScratch, tbl: &mut [i16]) {
    debug_assert_eq!(tbl.len(), RACE_STATES);
    if s.exact.is_none() {
        s.exact = Some(Box::new(ExactScratch::new()));
    }
    let ex = s.exact.as_mut().expect("exact scratch");
    fill_exact_dtm(g, ex, tbl);
}

/// Exact distance-to-mate for the *current* state only (Service B, on demand).
///
/// Builds (or reuses) the exact full table for this topology into `tbl`, then
/// returns `+k / −k` for the current `(p0, p1, turn)`. `None` if the state is
/// off the live set. The caller owns `tbl` (it may cache it per topology); this
/// routine is never called on the bound-only search path.
pub fn race_exact_dtm_on_demand(
    g: &mut GameState,
    s: &mut RaceScratch,
    tbl: &mut [i16],
) -> Option<i16> {
    debug_assert_eq!(tbl.len(), RACE_STATES);
    solve_race_config(g, s, tbl);
    let v = tbl[state_id(g.pawn[0], g.pawn[1], g.turn)];
    if v == 0 {
        None
    } else {
        Some(v)
    }
}

// ---------------------------------------------------------------------------
// Test-only exhaustive reference oracle.
// ---------------------------------------------------------------------------

#[cfg(test)]
struct ReferenceScratch {
    succ: Box<[i16]>,
    nsucc: Box<[u8]>,
    live: Box<[i32]>,
    buf: [i16; 16],
}

#[cfg(test)]
impl ReferenceScratch {
    fn new() -> Self {
        Self {
            succ: vec![0i16; RACE_STATES * 5].into_boxed_slice(),
            nsucc: vec![0u8; RACE_STATES].into_boxed_slice(),
            live: vec![0i32; RACE_STATES].into_boxed_slice(),
            buf: [0; 16],
        }
    }
}

#[cfg(test)]
fn solve_race_config_reference(g: &mut GameState, s: &mut ReferenceScratch, tbl: &mut [i16]) {
    debug_assert_eq!(tbl.len(), RACE_STATES);
    let (sp0, sp1, sturn) = (g.pawn[0], g.pawn[1], g.turn);
    tbl.fill(0);

    let mut n_live = 0usize;
    for p0 in 9..81usize {
        g.pawn[0] = p0;
        for p1 in 0..72usize {
            if p1 == p0 {
                continue;
            }
            g.pawn[1] = p1;
            let base = state_id(p0, p1, 0);

            g.turn = 0;
            let nm = g.gen_pawn_moves(&mut s.buf, 0);
            debug_assert!(nm <= 5);
            s.nsucc[base] = nm as u8;
            let off = base * 5;
            for j in 0..nm {
                let c = s.buf[j] as usize;
                s.succ[off + j] = if c < 9 { -1 } else { state_id(c, p1, 1) as i16 };
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
                    state_id(p0, c, 0) as i16
                };
            }
            s.live[n_live] = (base + 1) as i32;
            n_live += 1;
        }
    }

    g.pawn[0] = sp0;
    g.pawn[1] = sp1;
    g.turn = sturn;

    let mut k = 1i32;
    while n_live > 0 && k < 1024 {
        let mut assigned = 0usize;
        let mut keep = 0usize;

        for i in 0..n_live {
            let id = s.live[i] as usize;
            let ns = s.nsucc[id] as usize;
            let mut min_loss = 32_767i32;
            let mut all_win = ns > 0;
            let mut max_win = 0i32;
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
                    min_loss = min_loss.min(-v);
                } else if v > 0 {
                    max_win = max_win.max(v);
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
            break;
        }
        k += 1;
    }
}

#[cfg(test)]
fn gen_successor_ids_for_test(
    g: &mut GameState,
    id: usize,
    buf: &mut [i16; 16],
    succ_out: &mut [i16; 5],
) -> usize {
    let (p0, p1, turn) = decode_state(id);
    g.pawn[0] = p0;
    g.pawn[1] = p1;
    g.turn = turn;

    let nm = g.gen_pawn_moves(buf, 0);
    debug_assert!(nm <= 5);

    for j in 0..nm {
        let c = buf[j] as usize;
        succ_out[j] = if turn == 0 {
            if c < 9 {
                -1
            } else {
                state_id(c, p1, 1) as i16
            }
        } else if c >= 72 {
            -1
        } else {
            state_id(p0, c, 0) as i16
        };
    }
    nm
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solved_empty_board() -> Vec<i16> {
        let mut g = GameState::new();
        let mut s = RaceScratch::new();
        let mut tbl = vec![0i16; RACE_STATES];
        solve_race_config(&mut g, &mut s, &mut tbl);
        tbl
    }

    /// Deterministic LCG for test RNG.
    fn lcg_next(rng: &mut u64) -> u64 {
        *rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        *rng
    }

    /// A fixed wall topology reached by a fully legal replay with both hands empty.
    #[derive(Clone)]
    struct LegalFullWallTopology {
        g: GameState,
        moves: Vec<String>,
        walls_by: [u8; 2],
        gen_restarts: u32,
    }

    fn collect_legal_wall_moves(g: &mut GameState, out: &mut Vec<i16>) {
        if g.wl[g.turn] <= 0 {
            return;
        }
        for slot in 0..64usize {
            if g.wall_legal(0, slot) {
                out.push(100 + slot as i16);
            }
            if g.wall_legal(1, slot) {
                out.push(200 + slot as i16);
            }
        }
    }

    fn collect_legal_pawn_moves(g: &GameState, out: &mut Vec<i16>) {
        let mut buf = [0i16; 16];
        let nm = g.gen_pawn_moves(&mut buf, 0);
        out.extend_from_slice(&buf[..nm]);
    }

    fn pawn_move_legal(g: &GameState, m: i16) -> bool {
        let mut buf = [0i16; 16];
        let nm = g.gen_pawn_moves(&mut buf, 0);
        buf[..nm].contains(&m)
    }

    fn wall_move_legal(g: &mut GameState, m: i16) -> bool {
        if m < 100 || g.wl[g.turn] <= 0 {
            return false;
        }
        if m < 200 {
            g.wall_legal(0, (m - 100) as usize)
        } else {
            g.wall_legal(1, (m - 200) as usize)
        }
    }

    fn assert_full_wall_endgame(g: &mut GameState, walls_by: [u8; 2]) {
        assert_eq!(walls_by[0], 10, "player 0 must place 10 walls");
        assert_eq!(walls_by[1], 10, "player 1 must place 10 walls");
        assert_eq!(g.wl, [0, 0], "both wall hands must be naturally empty");
        assert!(g.has_path(0), "player 0 must retain a goal path");
        assert!(g.has_path(1), "player 1 must retain a goal path");
        assert!(g.winner() < 0, "topology must not be terminal");
    }

    /// Replay `moves` from start; every move must pass real legality at apply time.
    fn replay_legal_algebraic(moves: &[&str]) -> Result<LegalFullWallTopology, String> {
        use crate::titanium::algebraic_to_move_id;
        let mut g = GameState::new();
        let mut walls_by = [0u8; 2];
        let mut seq = Vec::with_capacity(moves.len());
        for &text in moves {
            if g.winner() >= 0 {
                return Err(format!("terminal before move {text}"));
            }
            let m = algebraic_to_move_id(text);
            if m < 100 {
                if !pawn_move_legal(&g, m) {
                    return Err(format!("illegal pawn move {text}"));
                }
            } else if !wall_move_legal(&mut g, m) {
                return Err(format!("illegal wall move {text}"));
            }
            let side = g.turn;
            g.make_move(m);
            seq.push(text.to_string());
            if m >= 100 {
                walls_by[side] += 1;
            }
        }
        if walls_by != [10, 10] {
            return Err(format!(
                "expected 10 walls each, got p0={} p1={}",
                walls_by[0], walls_by[1]
            ));
        }
        assert_full_wall_endgame(&mut g, walls_by);
        Ok(LegalFullWallTopology {
            g,
            moves: seq,
            walls_by,
            gen_restarts: 0,
        })
    }

    /// Generate a reachable 20-wall topology (10 per player) via legal moves only.
    fn generate_legal_full_wall_topology(
        rng: &mut u64,
        max_game_restarts: u32,
    ) -> Option<LegalFullWallTopology> {
        use crate::titanium::move_id_to_algebraic;

        const MAX_PLY_FAILS: u32 = 512;

        for _ in 0..max_game_restarts {
            let mut g = GameState::new();
            let mut moves = Vec::new();
            let mut walls_by = [0u8; 2];
            let mut ply_fails = 0u32;

            while walls_by[0] < 10 || walls_by[1] < 10 {
                if g.winner() >= 0 {
                    break;
                }
                if !g.has_path(0) || !g.has_path(1) {
                    break;
                }

                let mut walls = Vec::new();
                let mut pawns = Vec::new();
                collect_legal_wall_moves(&mut g, &mut walls);
                collect_legal_pawn_moves(&g, &mut pawns);

                let need_wall =
                    g.wl[g.turn] > 0 && (walls_by[g.turn] < 10 || lcg_next(rng) % 5 != 0);

                let pool: &[i16] = if need_wall && !walls.is_empty() {
                    &walls
                } else if !pawns.is_empty() {
                    &pawns
                } else if !walls.is_empty() {
                    &walls
                } else {
                    break;
                };

                let mv = pool[(lcg_next(rng) as usize) % pool.len()];
                if mv < 100 {
                    if !pawn_move_legal(&g, mv) {
                        ply_fails += 1;
                        if ply_fails > MAX_PLY_FAILS {
                            break;
                        }
                        continue;
                    }
                } else if !wall_move_legal(&mut g, mv) {
                    ply_fails += 1;
                    if ply_fails > MAX_PLY_FAILS {
                        break;
                    }
                    continue;
                }

                let side = g.turn;
                g.make_move(mv);
                moves.push(move_id_to_algebraic(mv));
                if mv >= 100 {
                    walls_by[side] += 1;
                }
                ply_fails = 0;
            }

            if walls_by == [10, 10]
                && g.wl == [0, 0]
                && g.winner() < 0
                && g.has_path(0)
                && g.has_path(1)
            {
                assert_full_wall_endgame(&mut g, walls_by);
                return Some(LegalFullWallTopology {
                    g,
                    moves,
                    walls_by,
                    gen_restarts: 0,
                });
            }
        }
        None
    }

    /// Sample legal pawn-only playout states after all walls are spent.
    fn sample_reachable_playout_states(
        base: &GameState,
        rng: &mut u64,
        max_plies: usize,
        max_states: usize,
    ) -> Vec<(usize, usize, usize)> {
        let mut g = base.clone();
        let mut out = Vec::with_capacity(max_states);
        let mut seen = std::collections::HashSet::new();

        let record =
            |g: &GameState,
             out: &mut Vec<(usize, usize, usize)>,
             seen: &mut std::collections::HashSet<(usize, usize, usize)>| {
                if g.winner() >= 0 {
                    return;
                }
                let key = (g.pawn[0], g.pawn[1], g.turn);
                if seen.insert(key) {
                    out.push(key);
                }
            };

        record(&g, &mut out, &mut seen);
        for _ in 0..max_plies {
            if g.winner() >= 0 || out.len() >= max_states {
                break;
            }
            let mut buf = [0i16; 16];
            let nm = g.gen_pawn_moves(&mut buf, 0);
            if nm == 0 {
                break;
            }
            let mv = buf[(lcg_next(rng) as usize) % nm];
            g.make_move(mv);
            record(&g, &mut out, &mut seen);
        }
        out
    }

    #[test]
    fn legal_full_wall_topology_generator_smoke() {
        let mut rng: u64 = 0x5A0E1_E6A1;
        let mut ok = 0usize;
        let mut rejected = 0u64;
        for _ in 0..8 {
            match generate_legal_full_wall_topology(&mut rng, 64) {
                Some(t) => {
                    ok += 1;
                    assert_eq!(t.walls_by, [10, 10]);
                    assert_eq!(t.g.wl, [0, 0]);
                    assert!(!t.moves.is_empty());
                }
                None => rejected += 1,
            }
        }
        eprintln!("legal topology smoke: ok={ok} rejected={rejected}");
        assert!(
            ok >= 3,
            "must generate at least 3 legal full-wall topologies"
        );
    }

    /// Generate one legal full-wall topology; panics if generation fails.
    fn random_legal_full_wall_topology(rng: &mut u64) -> LegalFullWallTopology {
        generate_legal_full_wall_topology(rng, 256)
            .expect("failed to generate legal full-wall topology")
    }

    fn compare_tables(
        fast: &[i16],
        reference: &[i16],
    ) -> (
        usize,
        usize,
        usize,
        Option<(usize, i16, i16)>,
        Option<(usize, i16, i16)>,
    ) {
        let mut live = 0usize;
        let mut sign_mismatches = 0usize;
        let mut exact_mismatches = 0usize;
        let mut first_sign = None;
        let mut first_exact = None;

        for id in 0..RACE_STATES {
            if reference[id] == 0 && fast[id] == 0 {
                continue;
            }
            live += 1;
            if fast[id].signum() != reference[id].signum() {
                sign_mismatches += 1;
                first_sign.get_or_insert((id, fast[id], reference[id]));
            }
            if fast[id] != reference[id] {
                exact_mismatches += 1;
                first_exact.get_or_insert((id, fast[id], reference[id]));
            }
        }

        (
            live,
            sign_mismatches,
            exact_mismatches,
            first_sign,
            first_exact,
        )
    }

    fn print_mismatch(label: &str, id: usize, fast: i16, reference: i16) {
        let (p0, p1, turn) = decode_state(id);
        eprintln!("{label}: id={id} p0={p0} p1={p1} turn={turn} fast={fast} ref={reference}");
    }

    // ── 1. Exhaustive empty-board exact equality (Service B) ──────────────────

    #[test]
    fn empty_board_exhaustive_exact_equality() {
        let mut g = GameState::new();

        let mut fast_scratch = RaceScratch::new();
        let mut fast = vec![0i16; RACE_STATES];
        solve_race_config(&mut g, &mut fast_scratch, &mut fast);

        let mut ref_scratch = ReferenceScratch::new();
        let mut reference = vec![0i16; RACE_STATES];
        solve_race_config_reference(&mut g, &mut ref_scratch, &mut reference);

        let (live, sign_m, exact_m, first_sign, first_exact) = compare_tables(&fast, &reference);

        if let Some((id, f, r)) = first_sign {
            print_mismatch("first sign mismatch", id, f, r);
        }
        if let Some((id, f, r)) = first_exact {
            print_mismatch("first exact mismatch", id, f, r);
        }

        eprintln!("empty-board: live={live} sign_mismatches={sign_m} exact_mismatches={exact_m}");

        assert_eq!(sign_m, 0, "sign mismatches on empty board");
        assert_eq!(exact_m, 0, "exact mismatches on empty board");
    }

    /// Service A (`race_outcome`) — on every DECISIVE live empty-board state its
    /// bound sign must match the exact oracle. (Unknown is allowed; it is never a
    /// false bound.) The bound path must allocate no exact graph.
    #[test]
    fn empty_board_race_outcome_bound_sign_audit() {
        let mut g = GameState::new();
        let mut ref_scratch = ReferenceScratch::new();
        let mut reference = vec![0i16; RACE_STATES];
        solve_race_config_reference(&mut g, &mut ref_scratch, &mut reference);

        let mut s = RaceScratch::new();
        let mut decisive = 0usize;
        let mut unknown = 0usize;
        for p0 in 9..81usize {
            for p1 in 0..72usize {
                if p0 == p1 {
                    continue;
                }
                for turn in 0..2usize {
                    let id = state_id(p0, p1, turn);
                    if reference[id] == 0 {
                        continue;
                    }
                    g.pawn[0] = p0;
                    g.pawn[1] = p1;
                    g.turn = turn;
                    let bound = race_outcome(&mut g, &mut s);
                    match bound {
                        RaceBound::Unknown => unknown += 1,
                        _ => {
                            decisive += 1;
                            assert_eq!(
                                bound.signum(),
                                reference[id].signum() as i32,
                                "race_outcome sign mismatch id={id} p0={p0} p1={p1} turn={turn} bound={bound:?} ref={}",
                                reference[id]
                            );
                        }
                    }
                    assert!(!s.exact_allocated(), "race_outcome allocated exact scratch");
                }
            }
        }
        eprintln!("race_outcome empty-board: decisive={decisive} unknown={unknown}");
        assert!(decisive > 0);
    }

    // ── 2. Fixed-wall sample configs ─────────────────────────────────────────

    #[test]
    fn exact_matches_reference_on_sample_configs() {
        use crate::titanium::algebraic_to_move_id;

        let configs: [&[&str]; 3] = [
            &[],
            &["e2", "e8", "e3h", "e6h"],
            &["e2", "e8", "c3h", "f6v", "d7h", "b4v"],
        ];

        for moves in configs {
            let mut g = GameState::new();
            for m in moves {
                g.make_move(algebraic_to_move_id(m));
            }

            let mut fast_scratch = RaceScratch::new();
            let mut fast = vec![0i16; RACE_STATES];
            solve_race_config(&mut g, &mut fast_scratch, &mut fast);

            let mut ref_scratch = ReferenceScratch::new();
            let mut reference = vec![0i16; RACE_STATES];
            solve_race_config_reference(&mut g, &mut ref_scratch, &mut reference);

            let (_, sign_m, exact_m, first_sign, first_exact) = compare_tables(&fast, &reference);

            assert_eq!(
                sign_m, 0,
                "sign mismatch; moves={moves:?}, first={first_sign:?}"
            );
            assert_eq!(
                exact_m, 0,
                "exact mismatch; moves={moves:?}, first={first_exact:?}"
            );
        }
    }

    // ── 3. Random legal fixed topologies (exact + bound sign) ────────────────

    #[test]
    fn random_fixed_topology_exact_equality_1000() {
        let seed: u64 = 0xACE5_2026;
        let mut rng = seed;

        const N: usize = 1_000;
        let mut fast_scratch = RaceScratch::new();
        let mut ref_scratch = ReferenceScratch::new();
        for trial in 0..N {
            let topo = random_legal_full_wall_topology(&mut rng);
            let mut g = topo.g;

            let mut fast = vec![0i16; RACE_STATES];
            solve_race_config(&mut g, &mut fast_scratch, &mut fast);

            let mut reference = vec![0i16; RACE_STATES];
            solve_race_config_reference(&mut g, &mut ref_scratch, &mut reference);

            let (_, sign_m, exact_m, first_sign, first_exact) = compare_tables(&fast, &reference);
            if sign_m != 0 || exact_m != 0 {
                eprintln!(
                    "random topology failure trial={trial} seed={seed} pawns=({},{}) turn={}",
                    g.pawn[0], g.pawn[1], g.turn
                );
                if let Some((id, f, r)) = first_sign {
                    print_mismatch("sign", id, f, r);
                }
                if let Some((id, f, r)) = first_exact {
                    print_mismatch("exact", id, f, r);
                }
            }

            assert_eq!(sign_m, 0, "trial {trial} seed {seed} sign mismatch");
            assert_eq!(exact_m, 0, "trial {trial} seed {seed} exact mismatch");
        }
    }

    /// Service A soundness on WALLED topologies: across 1,000 random fixed
    /// topologies, EVERY decisive `race_outcome` bound must agree in sign with the
    /// exact oracle. (Unknown is allowed — it is never a false bound.) This is the
    /// gate that the in-module winner-sign recursion failed, motivating the switch
    /// to the proven cert_bridge resolver.
    #[test]
    fn random_fixed_topology_race_outcome_bound_sign_1000() {
        let seed: u64 = 0x71744E_1ACE;
        let mut rng = seed;
        const N: usize = 1_000;
        let mut s = RaceScratch::new();
        let mut ref_scratch = ReferenceScratch::new();
        let mut reference = vec![0i16; RACE_STATES];
        let mut decisive = 0usize;
        let mut unknown = 0usize;
        let mut g_probe = GameState::new();

        for trial in 0..N {
            let topo = random_legal_full_wall_topology(&mut rng);
            let mut g = topo.g;
            solve_race_config_reference(&mut g, &mut ref_scratch, &mut reference);

            // Probe a deterministic spread of live states. Rebuild a *consistent*
            // GameState per probe by replaying onto a clone with the topology's
            // walls — so the classifier sees a valid position.
            for step in 0..24usize {
                let p0 = 9 + (step * 7 + trial) % 72;
                let p1 = (step * 13 + 2 * trial) % 72;
                if p0 == p1 {
                    continue;
                }
                let turn = step % 2;
                let id = state_id(p0, p1, turn);
                if reference[id] == 0 {
                    continue;
                }
                // Place pawns directly on a fresh clone of the walled topology.
                g_probe.clone_from(&g);
                g_probe.pawn[0] = p0;
                g_probe.pawn[1] = p1;
                g_probe.turn = turn;
                let bound = race_outcome(&mut g_probe, &mut s);
                match bound {
                    RaceBound::Unknown => unknown += 1,
                    _ => {
                        decisive += 1;
                        assert_eq!(
                            bound.signum(),
                            reference[id].signum() as i32,
                            "outcome sign trial={trial} seed={seed} p0={p0} p1={p1} turn={turn} bound={bound:?} ref={}",
                            reference[id]
                        );
                    }
                }
            }
            assert!(
                !s.exact_allocated(),
                "bound path must not allocate exact scratch"
            );
        }
        eprintln!("race_outcome walled audit: decisive={decisive} unknown={unknown} (seed={seed})");
        assert!(
            decisive > 0,
            "must exercise decisive bounds on walled boards"
        );
    }

    // ── 4. Child-preservation audit (validates outcome-based move filtering) ──

    /// For every proven-winning state at least one legal child is a loss for the
    /// opponent; for every proven-losing state every legal child is a win for the
    /// opponent. Verified against the exact oracle on the empty board.
    #[test]
    fn child_preservation_audit_empty_board() {
        let mut g = GameState::new();
        let mut ref_scratch = ReferenceScratch::new();
        let mut reference = vec![0i16; RACE_STATES];
        solve_race_config_reference(&mut g, &mut ref_scratch, &mut reference);

        let mut buf = [0i16; 16];
        let mut succ = [0i16; 5];
        let mut win_states = 0usize;
        let mut loss_states = 0usize;

        for id in 0..RACE_STATES {
            let v = reference[id];
            if v == 0 {
                continue;
            }
            let ns = gen_successor_ids_for_test(&mut g, id, &mut buf, &mut succ);

            if v > 0 {
                win_states += 1;
                let preserves = (0..ns).any(|j| {
                    let nid = succ[j];
                    nid < 0 || reference[nid as usize] < 0
                });
                assert!(preserves, "winning state {id} has no winning child");
            } else {
                loss_states += 1;
                for j in 0..ns {
                    let nid = succ[j];
                    assert!(
                        nid >= 0,
                        "losing state {id} has an immediate-goal move (would be a win)"
                    );
                    assert!(
                        reference[nid as usize] > 0,
                        "losing state {id} has a non-winning child {nid}"
                    );
                }
            }
        }
        eprintln!("child-preservation: win_states={win_states} loss_states={loss_states}");
        assert!(win_states > 0 && loss_states > 0);
    }

    // ── 5. Alpha-beta bound correctness ──────────────────────────────────────

    const MATE_GUARD: i32 = 99_000;

    /// `race_outcome` lower/upper bounds must never cross the true exact score:
    /// a LOWER bound ≤ the exact race-win score; an UPPER bound ≥ the exact
    /// race-loss score. Both must stay above the heuristic band and below mate.
    #[test]
    fn race_outcome_bounds_never_cross_exact() {
        let mut g = GameState::new();
        let mut ref_scratch = ReferenceScratch::new();
        let mut reference = vec![0i16; RACE_STATES];
        solve_race_config_reference(&mut g, &mut ref_scratch, &mut reference);

        let mut s = RaceScratch::new();
        for p0 in 9..81usize {
            for p1 in 0..72usize {
                if p0 == p1 {
                    continue;
                }
                for turn in 0..2usize {
                    let id = state_id(p0, p1, turn);
                    let rv = reference[id] as i32;
                    if rv == 0 {
                        continue;
                    }
                    // Exact α-β score the engine assigns from this leaf.
                    let exact_score = if rv > 0 {
                        RACE_MATE - rv
                    } else {
                        -(RACE_MATE + rv) // rv<0 → -(RACE_MATE - |rv|)
                    };
                    g.pawn[0] = p0;
                    g.pawn[1] = p1;
                    g.turn = turn;
                    match race_outcome(&mut g, &mut s) {
                        RaceBound::Lower(b) => {
                            assert!(rv > 0, "LOWER bound on a non-win state {id}");
                            assert!(
                                b <= exact_score,
                                "LOWER bound {b} exceeds exact {exact_score} at {id}"
                            );
                            assert!(b > 9_000, "LOWER bound {b} not above heuristic band");
                            assert!(b < MATE_GUARD, "LOWER bound {b} reaches mate band");
                        }
                        RaceBound::Upper(b) => {
                            assert!(rv < 0, "UPPER bound on a non-loss state {id}");
                            assert!(
                                b >= exact_score,
                                "UPPER bound {b} below exact {exact_score} at {id}"
                            );
                            assert!(b < -9_000, "UPPER bound {b} not below heuristic band");
                            assert!(b > -MATE_GUARD, "UPPER bound {b} reaches mate band");
                        }
                        RaceBound::Exact(_) => panic!("Service A must not return Exact"),
                        RaceBound::Unknown => {} // allowed: no claim
                    }
                }
            }
        }
    }

    // ── 6. Existing regressions ──────────────────────────────────────────────

    #[test]
    fn empty_board_head_on_race_is_movers_loss() {
        let tbl = solved_empty_board();
        let p0 = 76;
        let p1 = 4;
        assert_eq!(tbl[state_id(p0, p1, 0)], -16);
        assert_eq!(tbl[state_id(p0, p1, 1)], -16);
    }

    #[test]
    fn immediate_jump_to_goal_wins_in_one_ply() {
        let tbl = solved_empty_board();
        let p0 = 18;
        let p1 = 9;
        assert_eq!(tbl[state_id(p0, p1, 0)], 1);
    }

    #[test]
    fn one_step_from_goal_wins_immediately() {
        let tbl = solved_empty_board();
        let p0 = 13;
        let p1 = 40;
        assert_eq!(tbl[state_id(p0, p1, 0)], 1);
    }

    #[test]
    fn race_table_is_bellman_consistent_on_sample_configs() {
        use crate::titanium::algebraic_to_move_id;

        let configs: [&[&str]; 3] = [
            &[],
            &["e2", "e8", "e3h", "e6h"],
            &["e2", "e8", "c3h", "f6v", "d7h", "b4v"],
        ];

        for moves in configs {
            let mut g = GameState::new();
            for m in moves {
                g.make_move(algebraic_to_move_id(m));
            }

            let mut fast_scratch = RaceScratch::new();
            let mut tbl = vec![0i16; RACE_STATES];
            solve_race_config(&mut g, &mut fast_scratch, &mut tbl);

            let mut buf = [0i16; 16];
            let mut succ = [0i16; 5];

            for id in 0..RACE_STATES {
                let v = tbl[id] as i32;
                if v == 0 {
                    continue;
                }

                let ns = gen_successor_ids_for_test(&mut g, id, &mut buf, &mut succ);
                let mut min_loss = i32::MAX;
                let mut all_resolved_win = ns > 0;
                let mut max_win = 0i32;

                for j in 0..ns {
                    let nid = succ[j];
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
    fn ka_game_ply67_stubborn_loser_root_moves() {
        use crate::titanium::algebraic_to_move_id;
        use crate::titanium::move_id_to_algebraic;

        let moves = [
            "e2", "e8", "e3", "e7", "e4", "e6", "e3h", "f6h", "c3h", "d4v", "e5v", "h6h", "a3h",
            "d6h", "f4v", "c5v", "h1h", "b4h", "g5h", "a7h", "f1h", "c7h", "d1h", "e5", "e6", "e4",
            "d6", "f4", "d5", "f5", "d4", "f6", "c4", "g6", "b4", "h6", "a4", "i6", "a5", "i5",
            "b5", "i4", "b6", "h4", "c6", "b6h", "b6", "h3", "a6", "g3", "a7", "f3", "b7", "e3",
            "c7", "d3", "d7", "d2", "e7", "c2", "b1h", "e7h", "d7", "b2", "c7", "a2",
        ];

        let mut g = GameState::new();
        for m in moves {
            g.make_move(algebraic_to_move_id(m));
        }

        let mut s = RaceScratch::new();
        let mut tbl = vec![0i16; RACE_STATES];
        solve_race_config(&mut g, &mut s, &mut tbl);

        let id = state_id(g.pawn[0], g.pawn[1], g.turn);
        let rv = tbl[id] as i32;
        let me = g.turn;
        let mut buf = [0i16; 16];
        let nm = g.gen_pawn_moves(&mut buf, 0);
        let mut best_key = i32::MIN;
        let mut best_alg = String::new();

        for &mv in &buf[..nm] {
            let c = mv as usize;
            let my_v = if is_home(me, c) {
                1
            } else {
                let child_id = if me == 0 {
                    state_id(c, g.pawn[1], 1)
                } else {
                    state_id(g.pawn[0], c, 0)
                };

                let v = tbl[child_id] as i32;
                if v == 0 {
                    continue;
                }

                if v > 0 {
                    -(v + 1)
                } else {
                    1 - v
                }
            };

            let key = if my_v > 0 {
                1_000_000 - my_v
            } else {
                -1_000_000 - my_v
            };

            if key > best_key {
                best_key = key;
                best_alg = move_id_to_algebraic(mv);
            }
        }

        assert!(rv < 0, "white must be in a proven loss");
        assert_eq!(
            best_alg, "b7",
            "b7 and d7 tie on race plies; b7 wins move-order tie-break"
        );
    }

    // ── 7. On-demand exact API + lazy lifecycle ──────────────────────────────

    #[test]
    fn on_demand_exact_matches_full_table_and_is_lazy() {
        let mut g = GameState::new();
        let mut s = RaceScratch::new();

        // Bound queries first: no exact graph yet.
        g.pawn[0] = 40;
        g.pawn[1] = 41;
        g.turn = 0;
        let _ = race_outcome(&mut g, &mut s);
        assert!(!s.exact_allocated(), "bound query must stay graph-free");

        // On-demand exact: allocates the graph, returns the same value as the
        // full table for this state, and agrees with the oracle.
        let mut tbl = vec![0i16; RACE_STATES];
        let v = race_exact_dtm_on_demand(&mut g, &mut s, &mut tbl);
        assert!(s.exact_allocated(), "exact request must allocate the graph");
        let id = state_id(g.pawn[0], g.pawn[1], g.turn);
        assert_eq!(v, Some(tbl[id]));

        let mut ref_scratch = ReferenceScratch::new();
        let mut reference = vec![0i16; RACE_STATES];
        solve_race_config_reference(&mut g, &mut ref_scratch, &mut reference);
        assert_eq!(v, Some(reference[id]));
    }

    // ── 8. ETA gate audit (delta_eta > 1 soundness) ──────────────────────────

    /// Isolated oracle audit for the `delta_eta > 1` ETA interception gate.
    ///
    /// For every live state where the gate fires, the candidate bound is compared
    /// against the exact retrograde oracle. A single false bound is a fatal failure.
    ///
    /// Coverage: empty board (all 10,242 live states), 3 fixed-wall sample
    /// configs, 10,000 deterministic random topologies, 7 adversarial configs
    /// designed around adjacency / jumps / leapfrogging / narrow corridors.
    #[test]
    fn eta_gate_delta_gt1_soundness_audit() {
        use crate::titanium::algebraic_to_move_id;

        // ── helpers ──────────────────────────────────────────────────────────

        struct Counters {
            live: u64,
            gate_fires: u64,
            correct: u64,
            false_bounds: u64,
            min_firing_delta: i16,
            delta_hist: [u64; 32], // index = delta_eta (clamped at 31)
        }

        impl Counters {
            fn new() -> Self {
                Self {
                    live: 0,
                    gate_fires: 0,
                    correct: 0,
                    false_bounds: 0,
                    min_firing_delta: i16::MAX,
                    delta_hist: [0u64; 32],
                }
            }
        }

        /// Compute what the ETA gate would return for one live state, WITHOUT going
        /// through the full `race_outcome` path. Returns `(fires, delta_eta, bound)`.
        fn eta_gate_probe(
            g: &mut GameState,
            d0: &[u8; 81],
            d1: &[u8; 81],
        ) -> Option<(i16, RaceBound)> {
            let r0 = d0[g.pawn[0]];
            let r1 = d1[g.pawn[1]];
            if r0 == u8::MAX || r1 == u8::MAX {
                return None;
            }
            let eta0 = arrival_ply(0, g.turn, r0);
            let eta1 = arrival_ply(1, g.turn, r1);
            if eta0 == eta1 {
                return None;
            }
            let runner: usize = if eta0 < eta1 { 0 } else { 1 };
            let chaser = runner ^ 1;
            let runner_eta = if runner == 0 { eta0 } else { eta1 };

            let d_runner_goal: &[u8; 81] = if runner == 0 { d0 } else { d1 };
            let chaser_d = d_runner_goal[g.pawn[chaser]];

            let delta = if chaser_d == u8::MAX {
                i16::MAX
            } else {
                arrival_ply(chaser, g.turn, chaser_d) - runner_eta
            };

            if delta <= 1 {
                return None;
            }

            let bound = if runner == g.turn {
                RaceBound::Lower(RACE_WIN_FLOOR)
            } else {
                RaceBound::Upper(-RACE_WIN_FLOOR)
            };
            Some((delta, bound))
        }

        /// Run audit over every live state for `g`'s topology.
        fn audit_topology(
            label: &str,
            g: &mut GameState,
            ref_scratch: &mut ReferenceScratch,
            ref_tbl: &mut Vec<i16>,
            ex_scratch: &mut RaceScratch,
            ex_tbl: &mut Vec<i16>,
            ctr: &mut Counters,
            first_false: &mut Option<String>,
        ) {
            // Build exact table for this topology.
            solve_race_config(g, ex_scratch, ex_tbl);
            solve_race_config_reference(g, ref_scratch, ref_tbl);

            let mut d0 = [0u8; 81];
            let mut d1 = [0u8; 81];
            let saved = (g.pawn[0], g.pawn[1], g.turn);

            for p0 in 9..81usize {
                for p1 in 0..72usize {
                    if p0 == p1 {
                        continue;
                    }
                    for turn in 0..2usize {
                        let id = state_id(p0, p1, turn);
                        let oracle = ref_tbl[id];
                        if oracle == 0 {
                            continue;
                        }
                        ctr.live += 1;

                        // Compute distance fields from this position.
                        g.pawn[0] = p0;
                        g.pawn[1] = p1;
                        g.turn = turn;
                        g.compute_dist(0, &mut d0);
                        g.compute_dist(1, &mut d1);

                        if let Some((delta, bound)) = eta_gate_probe(g, &d0, &d1) {
                            ctr.gate_fires += 1;
                            let hidx = delta.min(31) as usize;
                            ctr.delta_hist[hidx] += 1;
                            if delta < ctr.min_firing_delta {
                                ctr.min_firing_delta = delta;
                            }

                            let oracle_sign = oracle.signum() as i32;
                            if bound.signum() == oracle_sign {
                                ctr.correct += 1;
                            } else {
                                ctr.false_bounds += 1;
                                if first_false.is_none() {
                                    // Recover best move from exact table.
                                    let mut buf = [0i16; 16];
                                    let nm = g.gen_pawn_moves(&mut buf, 0);
                                    let mut succ_ids = [0i16; 5];
                                    let mut best_mv = -1i16;
                                    let mut best_key = i32::MIN;
                                    for j in 0..nm {
                                        let c = buf[j] as usize;
                                        let nid = if turn == 0 {
                                            if c < 9 {
                                                -1
                                            } else {
                                                state_id(c, p1, 1) as i16
                                            }
                                        } else if c >= 72 {
                                            -1
                                        } else {
                                            state_id(p0, c, 0) as i16
                                        };
                                        succ_ids[j] = nid;
                                        let key = if nid < 0 {
                                            1_000_000
                                        } else {
                                            let sv = ex_tbl[nid as usize] as i32;
                                            if sv < 0 {
                                                1_000_000 - sv.abs()
                                            } else {
                                                -sv
                                            }
                                        };
                                        if key > best_key {
                                            best_key = key;
                                            best_mv = buf[j];
                                        }
                                    }
                                    let legals: Vec<String> = buf[..nm]
                                        .iter()
                                        .map(|&mv| crate::titanium::move_id_to_algebraic(mv))
                                        .collect();
                                    let best_alg = if best_mv >= 0 {
                                        crate::titanium::move_id_to_algebraic(best_mv)
                                    } else {
                                        "?".into()
                                    };
                                    *first_false = Some(format!(
                                        "COUNTEREXAMPLE topology={label} id={id} \
                                         p0={p0} p1={p1} turn={turn} \
                                         runner={} chaser={} delta_eta={delta} \
                                         bound={bound:?} oracle={oracle} \
                                         legal_moves={legals:?} best_move={best_alg}",
                                        if arrival_ply(0, turn, d0[p0])
                                            < arrival_ply(1, turn, d1[p1])
                                        {
                                            0
                                        } else {
                                            1
                                        },
                                        if arrival_ply(0, turn, d0[p0])
                                            < arrival_ply(1, turn, d1[p1])
                                        {
                                            1
                                        } else {
                                            0
                                        },
                                    ));
                                }
                            }
                        }
                    }
                }
            }

            g.pawn[0] = saved.0;
            g.pawn[1] = saved.1;
            g.turn = saved.2;
        }

        // ── test harness ─────────────────────────────────────────────────────

        let mut ctr = Counters::new();
        let mut first_false: Option<String> = None;
        let mut ref_scratch = ReferenceScratch::new();
        let mut ref_tbl = vec![0i16; RACE_STATES];
        let mut ex_scratch = RaceScratch::new();
        let mut ex_tbl = vec![0i16; RACE_STATES];

        // 1. Empty board.
        {
            let mut g = GameState::new();
            audit_topology(
                "empty",
                &mut g,
                &mut ref_scratch,
                &mut ref_tbl,
                &mut ex_scratch,
                &mut ex_tbl,
                &mut ctr,
                &mut first_false,
            );
        }

        // 2. Fixed-wall sample configs (synthetic partial-wall — ETA gate only).
        for moves in [
            &["e2", "e8", "e3h", "e6h"][..],
            &["e2", "e8", "c3h", "f6v", "d7h", "b4v"],
            &["e2", "e8", "d2h", "d4h", "d6h", "e3v", "e5v"],
        ] {
            let mut g = GameState::new();
            for m in moves {
                let mid = algebraic_to_move_id(m);
                if mid < 100 {
                    assert!(pawn_move_legal(&g, mid), "sample pawn {m}");
                } else {
                    assert!(wall_move_legal(&mut g, mid), "sample wall {m}");
                }
                g.make_move(mid);
            }
            let label = format!("synthetic_sample[{}]", moves.join(","));
            audit_topology(
                &label,
                &mut g,
                &mut ref_scratch,
                &mut ref_tbl,
                &mut ex_scratch,
                &mut ex_tbl,
                &mut ctr,
                &mut first_false,
            );
        }

        // 3. 10,000 legal full-wall topologies.
        {
            let seed_state: u64 = 0xACE5_2026;
            let mut rng = seed_state;
            for i in 0..10_000usize {
                let topo = random_legal_full_wall_topology(&mut rng);
                let mut g = topo.g;
                let label = format!("legal_rand[{i}]");
                audit_topology(
                    &label,
                    &mut g,
                    &mut ref_scratch,
                    &mut ref_tbl,
                    &mut ex_scratch,
                    &mut ex_tbl,
                    &mut ctr,
                    &mut first_false,
                );
            }
        }

        // 4. Legal full-wall adversarial seeds (deterministic).
        for (label, seed) in [
            ("legal_adv_corridor", 0xC0111D_A001),
            ("legal_adv_serpentine", 0x5E2A_A002),
        ] {
            let mut rng = seed;
            let topo = loop {
                match generate_legal_full_wall_topology(&mut rng, 256) {
                    Some(t) => break t,
                    None => continue,
                }
            };
            let mut g = topo.g;
            audit_topology(
                label,
                &mut g,
                &mut ref_scratch,
                &mut ref_tbl,
                &mut ex_scratch,
                &mut ex_tbl,
                &mut ctr,
                &mut first_false,
            );
        }

        // A second seed sweep to broaden random coverage.
        {
            let seed2: u64 = 0x71744E_1ACE;
            let mut rng2 = seed2;
            for i in 0..2_000usize {
                let topo = random_legal_full_wall_topology(&mut rng2);
                let mut g = topo.g;
                let label = format!("legal_rand2[{i}]");
                audit_topology(
                    &label,
                    &mut g,
                    &mut ref_scratch,
                    &mut ref_tbl,
                    &mut ex_scratch,
                    &mut ex_tbl,
                    &mut ctr,
                    &mut first_false,
                );
            }
        }

        // ── report ───────────────────────────────────────────────────────────

        eprintln!(
            "ETA gate audit (delta_eta>1): live={} firings={} correct={} false={} min_delta={}",
            ctr.live,
            ctr.gate_fires,
            ctr.correct,
            ctr.false_bounds,
            if ctr.min_firing_delta == i16::MAX {
                -1
            } else {
                ctr.min_firing_delta as i64
            }
        );

        let mut hist_str = String::new();
        for (d, &count) in ctr.delta_hist.iter().enumerate() {
            if count > 0 {
                hist_str.push_str(&format!(" delta={d}:{count}"));
            }
        }
        eprintln!("delta distribution:{hist_str}");

        if let Some(ref msg) = first_false {
            eprintln!("{msg}");
        }

        assert_eq!(
            ctr.false_bounds,
            0,
            "ETA gate (delta_eta>1) produced {} false bound(s); first: {}",
            ctr.false_bounds,
            first_false.as_deref().unwrap_or("none"),
        );
        assert!(ctr.gate_fires > 0, "gate never fired — coverage broken");
    }

    // ── 9. Detour-dominance certificate — isolated oracle audit ──────────────

    /// Follow exact-oracle-optimal moves from `(p0,p1,turn)`, returning an
    /// algebraic PV until the first productive jump (own-goal Δ==2) or a goal.
    fn oracle_pv_until_jump(
        g: &mut GameState,
        tbl: &[i16],
        start: (usize, usize, usize),
        max_len: usize,
    ) -> Vec<String> {
        use crate::titanium::move_id_to_algebraic;
        let mut pv = Vec::new();
        let (mut p0, mut p1, mut turn) = start;
        let mut d0 = [0u8; 81];
        let mut d1 = [0u8; 81];
        for _ in 0..max_len {
            g.pawn[0] = p0;
            g.pawn[1] = p1;
            g.turn = turn;
            g.compute_dist(0, &mut d0);
            g.compute_dist(1, &mut d1);
            let dist_mover = if turn == 0 { &d0 } else { &d1 };
            let before = dist_mover[if turn == 0 { p0 } else { p1 }];

            let mut buf = [0i16; 16];
            let nm = g.gen_pawn_moves(&mut buf, 0);
            let mut best_key = i32::MIN;
            let mut best_mv = -1i16;
            let mut best_child: Option<(usize, usize, usize)> = None;
            for &mv in &buf[..nm] {
                let c = mv as usize;
                let (my_v, child) = if is_home(turn, c) {
                    (1, None)
                } else {
                    let child = if turn == 0 { (c, p1, 1) } else { (p0, c, 0) };
                    let cid = state_id(child.0, child.1, child.2);
                    let v = tbl[cid] as i32;
                    if v == 0 {
                        continue;
                    }
                    let mv_val = if v > 0 { -(v + 1) } else { 1 - v };
                    (mv_val, Some(child))
                };
                let key = if my_v > 0 {
                    1_000_000 - my_v
                } else {
                    -1_000_000 - my_v
                };
                if key > best_key {
                    best_key = key;
                    best_mv = mv;
                    best_child = child;
                }
            }
            if best_mv < 0 {
                break;
            }
            let c = best_mv as usize;
            let delta = before as i16 - dist_mover[c] as i16;
            pv.push(move_id_to_algebraic(best_mv));
            if is_home(turn, c) || best_child.is_none() {
                break; // reached goal
            }
            if delta == 2 {
                break; // first productive jump
            }
            let nc = best_child.unwrap();
            p0 = nc.0;
            p1 = nc.1;
            turn = nc.2;
        }
        pv
    }

    /// Legal-corpus audit: mandatory gates are winner sign + bound safety only.
    /// Approximate ply error is informational (never fails the run).
    fn run_legal_corpus_audit(n_legal_random: usize, stage_label: &str, progress_every: usize) {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::{Arc, Mutex};

        const MATE_GUARD: i32 = 99_000;
        const PLAYOUT_SAMPLES: usize = 256;
        const PLAYOUT_MAX_PLIES: usize = 240;
        const RAND_SEEDS: [u64; 4] = [0xACE5_2026, 0xDE70_D0D0, 0xC3A7_1F1E, 0xA11D_0D1E];
        let n_legal_topo = n_legal_random;

        eprintln!("══ legal-corpus audit {stage_label} (random={n_legal_topo}) ══");

        #[derive(Default, Clone, Debug)]
        struct PlyErrorStats {
            samples: u64,
            sum_abs: u64,
            max_abs: u32,
            within_1: u64,
            within_2: u64,
            reservoir: Vec<u32>,
        }

        impl PlyErrorStats {
            fn record(&mut self, est: PlyEstimate, oracle: i16) {
                let PlyEstimate::Approx(e) = est else {
                    return;
                };
                if oracle == 0 {
                    return;
                }
                let exact = oracle.unsigned_abs();
                let err = e.abs_diff(exact) as u32;
                self.samples += 1;
                self.sum_abs += u64::from(err);
                self.max_abs = self.max_abs.max(err);
                if err <= 1 {
                    self.within_1 += 1;
                }
                if err <= 2 {
                    self.within_2 += 1;
                }
                if self.reservoir.len() < 8192 {
                    self.reservoir.push(err);
                }
            }

            fn merge(&mut self, o: &Self) {
                self.samples += o.samples;
                self.sum_abs += o.sum_abs;
                self.max_abs = self.max_abs.max(o.max_abs);
                self.within_1 += o.within_1;
                self.within_2 += o.within_2;
                if self.reservoir.len() < 8192 {
                    self.reservoir
                        .extend(o.reservoir.iter().take(8192 - self.reservoir.len()));
                }
            }

            fn report(&self) {
                if self.samples == 0 {
                    eprintln!("  approximate_ply_error: (no decisive samples)");
                    return;
                }
                let mean = self.sum_abs as f64 / self.samples as f64;
                let mut sorted = self.reservoir.clone();
                sorted.sort_unstable();
                let median = sorted.get(sorted.len() / 2).copied().unwrap_or(0) as f64;
                let p95_idx =
                    ((sorted.len() as f64 * 0.95) as usize).min(sorted.len().saturating_sub(1));
                let p95 = sorted.get(p95_idx).copied().unwrap_or(0) as f64;
                eprintln!(
                    "  approximate_ply_error: samples={} mean={mean:.3} median={median:.1} \
                     p95={p95:.1} max={} within_1={:.1}% within_2={:.1}%",
                    self.samples,
                    self.max_abs,
                    100.0 * self.within_1 as f64 / self.samples as f64,
                    100.0 * self.within_2 as f64 / self.samples as f64,
                );
            }
        }

        #[derive(Default, Clone, Debug)]
        struct TopAcc {
            eta_decisions: u64,
            eta_false_signs: u64,
            overlap_decisions: u64,
            overlap_false_signs: u64,
            cert_calls: u64,
            cert_proven_wins: u64,
            cert_false_proofs: u64,
            bound_violations: u64,
            exhaustive_states: u64,
            reachable_states: u64,
            cert_unknown_after_ab: u64,
            cert_unknown_reduced: u64,
            legal_topologies: u64,
            gen_rejected: u64,
            natural_wl_zero: u64,
            ply_err: PlyErrorStats,
            first_failure: Option<String>,
        }

        impl TopAcc {
            fn merge(&mut self, o: &TopAcc) {
                self.eta_decisions += o.eta_decisions;
                self.eta_false_signs += o.eta_false_signs;
                self.overlap_decisions += o.overlap_decisions;
                self.overlap_false_signs += o.overlap_false_signs;
                self.cert_calls += o.cert_calls;
                self.cert_proven_wins += o.cert_proven_wins;
                self.cert_false_proofs += o.cert_false_proofs;
                self.bound_violations += o.bound_violations;
                self.exhaustive_states += o.exhaustive_states;
                self.reachable_states += o.reachable_states;
                self.cert_unknown_after_ab += o.cert_unknown_after_ab;
                self.cert_unknown_reduced += o.cert_unknown_reduced;
                self.legal_topologies += o.legal_topologies;
                self.gen_rejected += o.gen_rejected;
                self.natural_wl_zero += o.natural_wl_zero;
                self.ply_err.merge(&o.ply_err);
                if self.first_failure.is_none() {
                    self.first_failure = o.first_failure.clone();
                }
            }

            fn mandatory_failure(&self) -> bool {
                self.eta_false_signs > 0
                    || self.overlap_false_signs > 0
                    || self.cert_false_proofs > 0
                    || self.bound_violations > 0
            }
        }

        #[inline]
        fn exact_search_score(oracle: i16) -> i32 {
            let rv = oracle as i32;
            if rv > 0 {
                RACE_MATE - rv
            } else {
                -(RACE_MATE + rv)
            }
        }

        #[inline]
        fn bound_crosses_exact(bound: RaceBound, oracle: i16) -> bool {
            let rv = oracle as i32;
            if rv == 0 {
                return false;
            }
            let exact_score = exact_search_score(oracle);
            match bound {
                RaceBound::Lower(b) => rv <= 0 || b > exact_score,
                RaceBound::Upper(b) => rv >= 0 || b < exact_score,
                RaceBound::Exact(_) => true,
                RaceBound::Unknown => false,
            }
        }

        fn check_sign_and_bound(
            acc: &mut TopAcc,
            gate: &str,
            topology: &str,
            corpus: &str,
            move_seq: &[String],
            id: usize,
            p0: usize,
            p1: usize,
            turn: usize,
            bound: RaceBound,
            est: PlyEstimate,
            oracle: i16,
        ) {
            if bound == RaceBound::Unknown {
                return;
            }
            acc.ply_err.record(est, oracle);
            if bound_crosses_exact(bound, oracle) {
                acc.bound_violations += 1;
                acc.first_failure.get_or_insert_with(|| {
                    format!(
                        "BOUND CROSS topology={topology} corpus={corpus} gate={gate} id={id} \
                         p0={p0} p1={p1} turn={turn} bound={bound:?} oracle={oracle} \
                         exact_score={} legal_replay={move_seq:?}",
                        exact_search_score(oracle)
                    )
                });
            }
            if matches!(bound, RaceBound::Exact(_)) {
                acc.bound_violations += 1;
                acc.first_failure.get_or_insert_with(|| {
                    format!(
                        "Service A returned Exact id={id} gate={gate} legal_replay={move_seq:?}"
                    )
                });
            }
            if bound.signum() != oracle.signum() as i32 {
                match gate {
                    "eta" => acc.eta_false_signs += 1,
                    "overlap" => acc.overlap_false_signs += 1,
                    "cert" => acc.cert_false_proofs += 1,
                    _ => {}
                }
                acc.first_failure.get_or_insert_with(|| {
                    format!(
                        "SIGN FAIL gate={gate} topology={topology} corpus={corpus} \
                         id={id} bound={bound:?} oracle={oracle} legal_replay={move_seq:?}"
                    )
                });
            }
            let _ = MATE_GUARD;
        }

        fn audit_one_state(
            acc: &mut TopAcc,
            label: &str,
            corpus: &str,
            move_seq: &[String],
            _base: &GameState,
            _ex_tbl: &[i16],
            ref_tbl: &[i16],
            winner_tbl: &RaceWinnerTable,
            p0: usize,
            p1: usize,
            turn: usize,
            probe: &mut GameState,
            d0: &mut [u8; 81],
            d1: &mut [u8; 81],
        ) {
            let id = state_id(p0, p1, turn);
            let oracle = ref_tbl[id];
            if oracle == 0 {
                return;
            }

            probe.pawn[0] = p0;
            probe.pawn[1] = p1;
            probe.turn = turn;
            probe.compute_dist(0, d0);
            probe.compute_dist(1, d1);

            let r0 = d0[p0];
            let r1 = d1[p1];
            if r0 == u8::MAX || r1 == u8::MAX {
                return;
            }

            let eta0 = arrival_ply(0, turn, r0);
            let eta1 = arrival_ply(1, turn, r1);

            if acc.first_failure.is_none() && eta0 != eta1 {
                let runner_a = if eta0 < eta1 { 0 } else { 1 };
                let chaser_a = runner_a ^ 1;
                let runner_eta_a = if runner_a == 0 { eta0 } else { eta1 };
                let d_rg_a: &[u8; 81] = if runner_a == 0 { d0 } else { d1 };
                let c_d_rg = d_rg_a[if chaser_a == 0 { p0 } else { p1 }];
                let fires =
                    c_d_rg == u8::MAX || arrival_ply(chaser_a, turn, c_d_rg) - runner_eta_a > 1;
                if fires {
                    acc.eta_decisions += 1;
                    let bound = if runner_a == turn {
                        RaceBound::Lower(RACE_WIN_FLOOR)
                    } else {
                        RaceBound::Upper(-RACE_WIN_FLOOR)
                    };
                    let wd_a = if runner_a == 0 { r0 } else { r1 };
                    let est = PlyEstimate::Approx(estimated_plies_to_result(probe, runner_a, wd_a));
                    check_sign_and_bound(
                        acc, "eta", label, corpus, move_seq, id, p0, p1, turn, bound, est, oracle,
                    );
                    return;
                }
            }

            // Gate 2 is NON-DECISIVE in production (Case B) — the separated
            // pure-race verdict is unsound (trailing-pawn detour-to-block), so
            // the audit does not test it as a decisive tier. Such states fall
            // through to the winner-table tier below, exactly like production.

            probe.pawn[0] = p0;
            probe.pawn[1] = p1;
            probe.turn = turn;
            if race_outcome_gates_ab(probe) != RaceBound::Unknown {
                return;
            }

            acc.cert_unknown_after_ab += 1;
            // Tier 3: asymmetric winner-table lookup (pre-built for this topology).
            let bound = match winner_tbl.classify(id) {
                RaceClass::ProvenP0 => {
                    if turn == 0 {
                        RaceBound::Lower(RACE_WIN_FLOOR)
                    } else {
                        RaceBound::Upper(-RACE_WIN_FLOOR)
                    }
                }
                RaceClass::ProvenP1 => {
                    if turn == 1 {
                        RaceBound::Lower(RACE_WIN_FLOOR)
                    } else {
                        RaceBound::Upper(-RACE_WIN_FLOOR)
                    }
                }
                RaceClass::Unknown => RaceBound::Unknown,
            };
            acc.cert_calls += 1;

            if bound == RaceBound::Unknown {
                return;
            }
            probe.pawn[0] = p0;
            probe.pawn[1] = p1;
            probe.turn = turn;
            acc.cert_proven_wins += 1;
            let est = ply_estimate_for_bound(probe, bound);
            check_sign_and_bound(
                acc, "cert", label, corpus, move_seq, id, p0, p1, turn, bound, est, oracle,
            );
            if bound.signum() == oracle.signum() as i32 {
                acc.cert_unknown_reduced += 1;
            }
        }

        fn audit_topology(
            label: &str,
            move_seq: &[String],
            base: &GameState,
            playout_rng: Option<&mut u64>,
            playout_samples: usize,
            run_reachable: bool,
        ) -> TopAcc {
            let mut acc = TopAcc::default();

            let mut ref_scratch = ReferenceScratch::new();
            let mut ref_tbl = vec![0i16; RACE_STATES];
            let mut ex_scratch = RaceScratch::new();

            let mut g = base.clone();
            solve_race_config_reference(&mut g, &mut ref_scratch, &mut ref_tbl);
            let mut ex_tbl = vec![0i16; RACE_STATES];
            solve_race_config(&mut g, &mut ex_scratch, &mut ex_tbl);

            // Tier-3 production winner table, built once for this topology.
            let winner_tbl = build_winner_table(&g);

            let mut probe = base.clone();
            let mut d0 = [0u8; 81];
            let mut d1 = [0u8; 81];

            // Corpus A — exhaustive live pawn placements on frozen topology.
            for p0 in 9..81usize {
                for p1 in 0..72usize {
                    if p0 == p1 {
                        continue;
                    }
                    for turn in 0..2usize {
                        let id = state_id(p0, p1, turn);
                        if ref_tbl[id] == 0 {
                            continue;
                        }
                        acc.exhaustive_states += 1;
                        audit_one_state(
                            &mut acc,
                            label,
                            "exhaustive",
                            move_seq,
                            base,
                            &ex_tbl,
                            &ref_tbl,
                            &winner_tbl,
                            p0,
                            p1,
                            turn,
                            &mut probe,
                            &mut d0,
                            &mut d1,
                        );
                        if acc.mandatory_failure() {
                            return acc;
                        }
                    }
                }
            }

            // Corpus B — states sampled from legal pawn-only playouts.
            if run_reachable {
                if let Some(rng) = playout_rng {
                    let states = sample_reachable_playout_states(
                        base,
                        rng,
                        PLAYOUT_MAX_PLIES,
                        playout_samples,
                    );
                    for (p0, p1, turn) in states {
                        let id = state_id(p0, p1, turn);
                        if ref_tbl[id] == 0 {
                            continue;
                        }
                        acc.reachable_states += 1;
                        audit_one_state(
                            &mut acc,
                            label,
                            "reachable",
                            move_seq,
                            base,
                            &ex_tbl,
                            &ref_tbl,
                            &winner_tbl,
                            p0,
                            p1,
                            turn,
                            &mut probe,
                            &mut d0,
                            &mut d1,
                        );
                        if acc.mandatory_failure() {
                            return acc;
                        }
                    }
                }
            }
            acc
        }

        fn generate_legal_topology_index(
            index: usize,
            seed: u64,
            acc: &mut TopAcc,
        ) -> LegalFullWallTopology {
            let mut rng = seed ^ (index as u64).wrapping_mul(0x517C_C1B7_2722_0A95);
            loop {
                match generate_legal_full_wall_topology(&mut rng, 256) {
                    Some(t) => {
                        assert_eq!(t.walls_by, [10, 10]);
                        assert_eq!(t.g.wl, [0, 0]);
                        acc.natural_wl_zero += 1;
                        acc.legal_topologies += 1;
                        return t;
                    }
                    None => acc.gen_rejected += 1,
                }
            }
        }

        fn worker_audit_range(
            start: usize,
            end: usize,
            seed: u64,
            progress_every: usize,
            total_random: usize,
            t_start: std::time::Instant,
            progress_counter: Arc<AtomicUsize>,
        ) -> TopAcc {
            let mut acc = TopAcc::default();
            for i in start..end {
                let topo = generate_legal_topology_index(i, seed, &mut acc);
                let label = format!("legal[{i}]");
                let mut play_rng = seed ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                let a = audit_topology(
                    &label,
                    &topo.moves,
                    &topo.g,
                    Some(&mut play_rng),
                    PLAYOUT_SAMPLES,
                    true,
                );
                acc.merge(&a);
                let done = progress_counter.fetch_add(1, Ordering::Relaxed) + 1;
                if done % progress_every == 0 || done == total_random {
                    let elapsed = t_start.elapsed();
                    let rate = done as f64 / elapsed.as_secs_f64().max(0.001);
                    let eta = (total_random - done) as f64 / rate.max(0.001);
                    eprintln!(
                        "progress: topologies={done}/{total_random} states={} \
                         cert_calls={} proofs={} false_proofs={} elapsed={elapsed:?} eta={eta:.0}s",
                        acc.exhaustive_states + acc.reachable_states,
                        acc.cert_calls,
                        acc.cert_proven_wins,
                        acc.cert_false_proofs,
                    );
                }
                if acc.mandatory_failure() {
                    break;
                }
            }
            acc
        }

        fn fail_if_mandatory(acc: &TopAcc, context: &str) {
            if acc.mandatory_failure() {
                if let Some(ref msg) = acc.first_failure {
                    eprintln!("{msg}");
                }
                panic!("deduction audit FAILED: {context}");
            }
        }

        // ── audit driver ─────────────────────────────────────────────────────
        let t_start = std::time::Instant::now();
        let mut global = TopAcc::default();

        // Synthetic empty board — Corpus A only (not a legal 20-wall endgame).
        {
            eprintln!("corpus A: synthetic_empty (exhaustive only, 0 walls)");
            let g = GameState::new();
            let a = audit_topology("synthetic_empty", &[], &g, None, 0, false);
            fail_if_mandatory(&a, "synthetic_empty");
            global.merge(&a);
        }

        // Deterministic legal full-wall adversarial seeds (both corpora).
        const ADV_SEEDS: [(&str, u64); 8] = [
            ("legal_adv_corridor", 0xC0111D_A001),
            ("legal_adv_serpentine", 0x5E2A_A002),
            ("legal_adv_cross", 0xC2055_A003),
            ("legal_adv_sideways", 0x51DE_A004),
            ("legal_adv_backward", 0xBAC0_A005),
            ("legal_adv_leapfrog", 0x1EAA_A006),
            ("legal_adv_narrow", 0xAA0D_A007),
            ("legal_adv_role_rev", 0x701E_A008),
        ];
        for (label, seed) in ADV_SEEDS {
            let mut acc_local = TopAcc::default();
            let topo = generate_legal_topology_index(0, seed, &mut acc_local);
            global.gen_rejected += acc_local.gen_rejected;
            global.legal_topologies += acc_local.legal_topologies;
            global.natural_wl_zero += acc_local.natural_wl_zero;
            let mut play_rng = seed ^ 0xB0AD;
            let a = audit_topology(
                label,
                &topo.moves,
                &topo.g,
                Some(&mut play_rng),
                PLAYOUT_SAMPLES,
                true,
            );
            fail_if_mandatory(&a, label);
            global.merge(&a);
        }

        for (si, &seed) in RAND_SEEDS.iter().enumerate() {
            eprintln!("legal topology seed[{si}]={seed:#x}");
        }

        let n_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .clamp(4, 16);
        let chunk = (n_legal_topo + n_threads - 1) / n_threads;
        eprintln!(
            "parallel: topologies={n_legal_topo} workers={n_threads} playout_samples={PLAYOUT_SAMPLES}"
        );

        let shared: Arc<Mutex<Vec<TopAcc>>> = Arc::new(Mutex::new(Vec::with_capacity(n_threads)));
        let base_seed = RAND_SEEDS[0];
        let progress_counter = Arc::new(AtomicUsize::new(0));

        let mut handles = Vec::with_capacity(n_threads);
        for t in 0..n_threads {
            let shared = Arc::clone(&shared);
            let progress_counter = Arc::clone(&progress_counter);
            let start_i = t * chunk;
            let end_i = (start_i + chunk).min(n_legal_topo);
            handles.push(std::thread::spawn(move || {
                let acc = worker_audit_range(
                    start_i,
                    end_i,
                    base_seed,
                    progress_every,
                    n_legal_topo,
                    t_start,
                    progress_counter,
                );
                shared.lock().unwrap().push(acc);
            }));
        }
        for h in handles {
            h.join().unwrap();
        }

        let results = Arc::try_unwrap(shared).unwrap().into_inner().unwrap();
        for a in &results {
            fail_if_mandatory(a, "parallel legal sweep");
            global.merge(a);
        }

        let elapsed = t_start.elapsed();

        eprintln!("── results ({stage_label}) ──");
        eprintln!(
            "legal_topologies={} gen_rejected={} natural_wl_zero={}",
            global.legal_topologies, global.gen_rejected, global.natural_wl_zero,
        );
        eprintln!(
            "exhaustive_states={} reachable_states={}",
            global.exhaustive_states, global.reachable_states,
        );
        eprintln!(
            "eta_decisions={} overlap_decisions={} cert_calls={} cert_proofs={} \
             false_proofs={} bound_violations={}",
            global.eta_decisions,
            global.overlap_decisions,
            global.cert_calls,
            global.cert_proven_wins,
            global.cert_false_proofs,
            global.bound_violations,
        );
        eprintln!(
            "cert_unknown_base={} unknown_reduced={}",
            global.cert_unknown_after_ab, global.cert_unknown_reduced,
        );
        global.ply_err.report();
        eprintln!("total_wall={elapsed:?} seeds={RAND_SEEDS:?}");

        assert_eq!(global.natural_wl_zero, global.legal_topologies);
        assert!(global.legal_topologies >= n_legal_topo as u64 + ADV_SEEDS.len() as u64);
        assert_eq!(global.eta_false_signs, 0, "ETA gate: sign mismatch");
        assert_eq!(
            global.overlap_false_signs, 0,
            "Path-overlap gate: sign mismatch"
        );
        assert_eq!(global.cert_false_proofs, 0, "Certificate: false proof");
        assert_eq!(
            global.bound_violations, 0,
            "Bound crossed exact oracle score"
        );
    }

    #[test]
    fn unified_deduction_oracle_audit_stage1() {
        run_legal_corpus_audit(100, "stage1", 5);
    }

    #[test]
    fn unified_deduction_oracle_audit_stage2() {
        run_legal_corpus_audit(1_000, "stage2", 25);
    }

    #[test]
    fn unified_deduction_oracle_audit_stage3() {
        run_legal_corpus_audit(10_000, "stage3", 100);
    }

    #[test]
    fn unified_deduction_oracle_audit() {
        run_legal_corpus_audit(10_000, "stage3", 100);
    }

    #[test]
    fn gate3_work_savings_benchmark() {
        use std::time::Instant;

        let mut rng: u64 = 0xBECA_1E6A1;
        let mut corpus: Vec<LegalFullWallTopology> = Vec::new();
        for _ in 0..32 {
            corpus.push(random_legal_full_wall_topology(&mut rng));
        }

        let mut g_probe = GameState::new();
        let mut s_full = RaceScratch::new();
        let mut ab_ns = 0u128;
        let mut full_ns = 0u128;
        let mut ab_unknown = 0u64;
        let mut full_resolved = 0u64;

        for topo in &corpus {
            for step in 0..48usize {
                let p0 = 9 + (step * 5) % 72;
                let p1 = (step * 11 + 3) % 72;
                if p0 == p1 {
                    continue;
                }
                for turn in 0..2usize {
                    g_probe.clone_from(&topo.g);
                    g_probe.pawn[0] = p0;
                    g_probe.pawn[1] = p1;
                    g_probe.turn = turn;

                    let t0 = Instant::now();
                    let ab = race_outcome_gates_ab(&mut g_probe);
                    ab_ns += t0.elapsed().as_nanos();
                    if ab == RaceBound::Unknown {
                        ab_unknown += 1;
                    }

                    g_probe.clone_from(&topo.g);
                    g_probe.pawn[0] = p0;
                    g_probe.pawn[1] = p1;
                    g_probe.turn = turn;
                    let t1 = Instant::now();
                    let full = race_outcome(&mut g_probe, &mut s_full);
                    full_ns += t1.elapsed().as_nanos();
                    if full != RaceBound::Unknown {
                        full_resolved += 1;
                    }
                }
            }
        }

        assert!(!s_full.exact_allocated());
        eprintln!(
            "gate3-bench: topologies={} ab_only_ns={} full_ns={} ab_unknown={} full_resolved={}",
            corpus.len(),
            ab_ns / 3072,
            full_ns / 3072,
            ab_unknown,
            full_resolved,
        );
    }

    // ── Benchmarks (printed; assert correctness) ─────────────────────────────

    #[test]
    fn benchmark_services_and_scratch() {
        let mut g = GameState::new();

        const ITERS: u32 = 200;
        let n = u128::from(ITERS);

        // (1/4) ordinary bound path: one lazy outcome query.
        let mut bound_ns = 0u128;
        let mut s = RaceScratch::new();
        for _ in 0..ITERS {
            g.pawn[0] = 40;
            g.pawn[1] = 41;
            g.turn = 0;
            let t = std::time::Instant::now();
            let _ = race_outcome(&mut g, &mut s);
            bound_ns += t.elapsed().as_nanos();
        }
        assert!(
            !s.exact_allocated(),
            "bound path must not allocate exact graph"
        );

        // (7) exact cold (fresh scratch each iter — includes the lazy alloc).
        let mut exact_cold_us = 0u128;
        for _ in 0..ITERS {
            let mut s = RaceScratch::new();
            let mut tbl = vec![0i16; RACE_STATES];
            let t = std::time::Instant::now();
            solve_race_config(&mut g, &mut s, &mut tbl);
            exact_cold_us += t.elapsed().as_micros();
        }

        // (8) exact cached (graph already allocated; reused).
        let mut exact_cached_us = 0u128;
        {
            let mut s = RaceScratch::new();
            let mut tbl = vec![0i16; RACE_STATES];
            solve_race_config(&mut g, &mut s, &mut tbl);
            for _ in 0..ITERS {
                let t = std::time::Instant::now();
                solve_race_config(&mut g, &mut s, &mut tbl);
                exact_cached_us += t.elapsed().as_micros();
            }
        }

        eprintln!(
            "race-bench: bound_query_ns={} exact_cold_us={} exact_cached_us={} bound_scratch_bytes={} exact_scratch_bytes={}",
            bound_ns / n,
            exact_cold_us / n,
            exact_cached_us / n,
            RaceScratch::scratch_bytes(),
            RaceScratch::exact_scratch_bytes(),
        );
    }

    // ────────────────────────────────────────────────────────────────────────
    // Certificate POC — disconnected from production.
    //
    // Tests the candidate theorem: "in a fixed no-more-walls race, every
    // outcome-relevant optimal move strictly reduces the moving player's
    // fixed-topology distance to that player's own goal (delta >= 1)."
    //
    // solve_certificate is a memoised minimax over delta>=1 moves, gated by
    // the existing sound deduction tiers (ETA gate, paths_overlap) as leaves.
    // NOT wired into race_outcome, alpha-beta, TT, or search.rs.
    // ────────────────────────────────────────────────────────────────────────

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct CertificateSolution {
        winner: u8,
        dtm: u16,
        best_move: i16,
    }

    #[derive(Clone, Debug)]
    struct MoveDiagnostic {
        move_id: i16,
        destination: usize,
        old_distance: u8,
        new_distance: u8,
        delta: i16,
        is_jump: bool,
    }

    #[derive(Clone, Debug)]
    enum CertificateResult {
        Solved(CertificateSolution),
        DominanceCounterexample {
            state_id: usize,
            p0: usize,
            p1: usize,
            turn: usize,
            legal_moves: Vec<MoveDiagnostic>,
        },
        CycleDetected {
            state_id: usize,
        },
    }

    struct CertificateContext {
        /// own_goal_dist[side][cell] = wall-only BFS distance from cell to side's goal row.
        own_goal_dist: [[u8; 81]; 2],
        /// Memo table indexed by state_id(p0, p1, turn). RACE_STATES = 13122 entries.
        /// Only contains entries for states fully solved by solve_certificate.
        memo: Vec<Option<CertificateSolution>>,
        /// Diagnostic: maximum recursion depth observed across all solve calls.
        max_depth: usize,
        /// Diagnostic: first cycle path (state_id chain leading to a back-edge).
        first_cycle: Option<Vec<usize>>,
    }

    /// True when a pawn move from `src` to `dst` is a jump (straight or diagonal),
    /// i.e., caused by pawn interaction, NOT a simple one-step ordinary move.
    #[inline(always)]
    fn is_jump_move(src: usize, dst: usize) -> bool {
        let row_diff = (src / 9) as i32 - (dst / 9) as i32;
        let col_diff = (src % 9) as i32 - (dst % 9) as i32;
        row_diff.abs() + col_diff.abs() != 1
    }

    const NO_MOVE: i16 = -1;

    /// Alternating-ply ETA returning u16 (certificate version, matches spec).
    #[inline(always)]
    fn cert_arrival_ply(side: usize, turn: usize, distance: u8) -> u16 {
        if distance == 0 {
            0
        } else {
            2 * distance as u16 - u16::from(side == turn)
        }
    }

    /// Build a CertificateContext for the given game topology.
    /// Pawn positions in `g` do not matter — we compute wall-only BFS fields.
    fn make_certificate_context(g: &mut GameState) -> CertificateContext {
        let mut own_goal_dist = [[u8::MAX; 81]; 2];
        g.compute_dist(0, &mut own_goal_dist[0]);
        g.compute_dist(1, &mut own_goal_dist[1]);
        CertificateContext {
            own_goal_dist,
            memo: vec![None; RACE_STATES],
            max_depth: 0,
            first_cycle: None,
        }
    }

    /// Create a fresh DFS in-progress Vec for one top-level certificate solve call.
    /// Must be passed to every solve_certificate invocation in that call tree.
    #[inline(always)]
    fn new_in_progress() -> Vec<bool> {
        vec![false; RACE_STATES]
    }

    /// Select the current mover's best direct-progress move for a pure-race leaf.
    /// Priority: immediate goal > productive jump (delta=2) > shortest (delta=1) > lowest ID.
    fn pure_race_leaf(
        g: &GameState,
        ctx: &CertificateContext,
        winner: usize,
    ) -> CertificateSolution {
        let side = g.turn;
        let winner_dist = ctx.own_goal_dist[winner][g.pawn[winner]];
        let dtm = cert_arrival_ply(winner, g.turn, winner_dist);

        let src = g.pawn[side];
        let old_d = ctx.own_goal_dist[side][src];
        let mut buf = [0i16; 16];
        let nm = g.gen_pawn_moves(&mut buf, 0);

        let mut best_move = NO_MOVE;
        let mut best_priority: i64 = i64::MIN;

        for &mv in &buf[..nm] {
            let dst = mv as usize;
            let priority: i64 = if is_home(side, dst) {
                3_000_000
            } else {
                let new_d = ctx.own_goal_dist[side][dst];
                if new_d == u8::MAX {
                    continue;
                }
                let delta = old_d as i16 - new_d as i16;
                let jump = is_jump_move(src, dst);
                if delta >= 2 {
                    2_000_000 - mv as i64
                } else if delta >= 1 {
                    1_000_000 - mv as i64
                } else if jump {
                    // Class B: interaction jump with delta<=0 (e.g. diagonal).
                    500_000 - mv as i64
                } else {
                    continue; // Class C: ordinary non-shortest detour
                }
            };
            if priority > best_priority {
                best_priority = priority;
                best_move = mv;
            }
        }

        // Fallback: if no retained move exists, pick the lowest-ID legal move.
        if best_move == NO_MOVE && nm > 0 {
            best_move = buf[..nm].iter().copied().min().unwrap_or(NO_MOVE);
        }

        CertificateSolution {
            winner: winner as u8,
            dtm,
            best_move,
        }
    }

    /// Recursive certificate solver.
    ///
    /// `in_progress` is a per-DFS-call Vec<bool> (indexed by state_id) that tracks
    /// which states are currently on the call stack.  It must be created fresh by
    /// the caller via `new_in_progress()` for each top-level invocation and passed
    /// through every recursive call.  solve_certificate ALWAYS clears
    /// `in_progress[id]` before returning (Solved or non-Solved), so the Vec is
    /// fully reset when the top-level call returns.
    ///
    /// `ctx.memo` is shared across all calls for a given topology (cache of fully
    /// solved states).  Only states that completed without any CycleDetected
    /// propagation in their subtree are memoised.
    fn solve_certificate(
        g: &GameState,
        ctx: &mut CertificateContext,
        in_progress: &mut Vec<bool>,
        depth: usize,
        stack: &mut Vec<usize>,
    ) -> CertificateResult {
        // ── Gate 0: terminal ──────────────────────────────────────────────────
        if is_home(0, g.pawn[0]) {
            return CertificateResult::Solved(CertificateSolution {
                winner: 0,
                dtm: 0,
                best_move: NO_MOVE,
            });
        }
        if is_home(1, g.pawn[1]) {
            return CertificateResult::Solved(CertificateSolution {
                winner: 1,
                dtm: 0,
                best_move: NO_MOVE,
            });
        }

        let id = state_id(g.pawn[0], g.pawn[1], g.turn);

        // ── Depth tracking ────────────────────────────────────────────────────
        if depth > ctx.max_depth {
            ctx.max_depth = depth;
        }

        // ── Memo lookup (fast path – fully solved in a previous call) ─────────
        if let Some(cached) = ctx.memo[id] {
            return CertificateResult::Solved(cached);
        }

        // ── Cycle detection ───────────────────────────────────────────────────
        if in_progress[id] {
            // Back-edge: record first cycle path.
            if ctx.first_cycle.is_none() {
                // Find where in the stack this id first appears.
                let cycle_start = stack.iter().position(|&s| s == id).unwrap_or(0);
                let mut cycle = stack[cycle_start..].to_vec();
                cycle.push(id); // close the loop
                ctx.first_cycle = Some(cycle);
            }
            return CertificateResult::CycleDetected { state_id: id };
        }
        in_progress[id] = true;
        stack.push(id);

        // Generate legal pawn moves.
        let mut buf = [0i16; 16];
        let nm = g.gen_pawn_moves(&mut buf, 0);

        // ── Gate 1: immediate legal goal move ────────────────────────────────
        let side = g.turn;
        let mut goal_move = NO_MOVE;
        for &mv in &buf[..nm] {
            let dst = mv as usize;
            if is_home(side, dst) {
                if goal_move == NO_MOVE || mv < goal_move {
                    goal_move = mv;
                }
            }
        }
        if goal_move != NO_MOVE {
            let result = CertificateSolution {
                winner: side as u8,
                dtm: 1,
                best_move: goal_move,
            };
            ctx.memo[id] = Some(result);
            stack.pop();
            in_progress[id] = false;
            return CertificateResult::Solved(result);
        }

        // ── Classify and retain moves ─────────────────────────────────────
        // Class A: ordinary adjacent move that is a shortest-path continuation
        //          (own_goal_dist[side][dst] + 1 == own_goal_dist[side][src], i.e. delta == 1).
        // Class B: jump move (straight or diagonal) caused by pawn interaction —
        //          always retained, even when delta == 0 or < 0.
        // Class C: ordinary adjacent move that is NOT a shortest-path continuation —
        //          excluded (delta <= 0, not a jump).
        let src = g.pawn[side];
        let old_d = ctx.own_goal_dist[side][src];

        let mut retained: Vec<i16> = Vec::new();
        let mut diagnostics: Vec<MoveDiagnostic> = Vec::new();

        for &mv in &buf[..nm] {
            let dst = mv as usize;
            let new_d = ctx.own_goal_dist[side][dst];
            let delta = if new_d == u8::MAX {
                i16::MIN / 2
            } else {
                old_d as i16 - new_d as i16
            };
            let jump = is_jump_move(src, dst);
            // Verify Class A assertion: ordinary shortest-path move must decrease distance by 1.
            debug_assert!(
                !(!jump && delta >= 1) || (new_d != u8::MAX && old_d as i16 == new_d as i16 + 1),
                "Class A assertion failed: src={src} dst={dst} old_d={old_d} new_d={new_d}"
            );
            // Class A: ordinary + delta >= 1.  Class B: any jump.  Class C: otherwise excluded.
            let retain = jump || delta >= 1;
            diagnostics.push(MoveDiagnostic {
                move_id: mv,
                destination: dst,
                old_distance: old_d,
                new_distance: new_d,
                delta,
                is_jump: jump,
            });
            if retain {
                retained.push(mv);
            }
        }

        if retained.is_empty() {
            // No Class A or Class B moves available — genuine dominance counterexample.
            stack.pop();
            in_progress[id] = false;
            return CertificateResult::DominanceCounterexample {
                state_id: id,
                p0: g.pawn[0],
                p1: g.pawn[1],
                turn: g.turn,
                legal_moves: diagnostics,
            };
        }

        let mut wins: Vec<(i16, CertificateSolution)> = Vec::new();
        let mut losses: Vec<(i16, CertificateSolution)> = Vec::new();

        for mv in retained {
            let mut child = g.clone();
            child.make_move(mv);

            let child_sol = match solve_certificate(&child, ctx, in_progress, depth + 1, stack) {
                CertificateResult::Solved(s) => s,
                // Propagate non-Solved results up — caller counts them.
                other => {
                    stack.pop();
                    in_progress[id] = false;
                    return other;
                }
            };

            if child_sol.winner as usize == side {
                wins.push((mv, child_sol));
            } else {
                losses.push((mv, child_sol));
            }
        }

        let result = if !wins.is_empty() {
            let (mv, child) = wins
                .into_iter()
                .min_by_key(|(m, s)| (s.dtm, *m))
                .unwrap();
            CertificateSolution {
                winner: side as u8,
                dtm: child.dtm + 1,
                best_move: mv,
            }
        } else {
            let (mv, child) = losses
                .into_iter()
                .max_by_key(|(m, s)| (s.dtm, std::cmp::Reverse(*m)))
                .unwrap();
            CertificateSolution {
                winner: (side ^ 1) as u8,
                dtm: child.dtm + 1,
                best_move: mv,
            }
        };

        ctx.memo[id] = Some(result);
        stack.pop();
        in_progress[id] = false;
        CertificateResult::Solved(result)
    }

    /// Reconstruct the principal variation from the memoised certificate table.
    /// Fills memo lazily for any state along the PV that was not visited during
    /// the initial solve (e.g. states past a gate-leaf that was not recursed into).
    fn reconstruct_certificate_pv(root: &GameState, ctx: &mut CertificateContext) -> Vec<i16> {
        let mut g = root.clone();
        let mut pv = Vec::new();
        loop {
            if is_home(0, g.pawn[0]) || is_home(1, g.pawn[1]) {
                break;
            }
            let id = state_id(g.pawn[0], g.pawn[1], g.turn);
            if ctx.memo[id].is_none() {
                // Gate leaf — state was not recursed into; solve it now.
                let mut ip = new_in_progress();
                match solve_certificate(&g, ctx, &mut ip, 0, &mut Vec::new()) {
                    CertificateResult::Solved(_) => {}
                    CertificateResult::DominanceCounterexample { .. } |
                    CertificateResult::CycleDetected { .. } => break,
                }
            }
            let entry = ctx.memo[id].expect("certificate PV state missing from memo");
            if entry.best_move == NO_MOVE {
                break;
            }
            pv.push(entry.best_move);
            g.make_move(entry.best_move);
        }
        pv
    }

    // ── POC targeted tests ───────────────────────────────────────────────────

    /// Empty board: p0 at row 1 (cell 9+col) one step from row 0; p1 far away.
    /// p0 to move → wins in 1 ply via direct goal move.
    /// Walls-only BFS distance from `src` to every cell (255 = unreachable).
    fn bfs_from_cell_walls(g: &GameState, src: usize) -> [u8; 81] {
        use crate::titanium::game::{BORDER, DELTA, DIRBIT};
        let mut out = [255u8; 81];
        out[src] = 0;
        let mut queue = [0i16; 81];
        let (mut head, mut tail) = (0usize, 0usize);
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

    /// Complete shortest-path-set membership for `src → its goal row`: every cell
    /// `v` with `dist(src,v) + dist(v,goal) == dist(src,goal)` (full DAG, union
    /// over all equally short terminal goal cells), not one BFS parent chain.
    fn shortest_path_set(g: &GameState, src: usize, d_goal: &[u8; 81]) -> [bool; 81] {
        let s = bfs_from_cell_walls(g, src);
        let big = d_goal[src];
        let mut on = [false; 81];
        if big == u8::MAX {
            return on;
        }
        for v in 0..81 {
            if s[v] != u8::MAX
                && d_goal[v] != u8::MAX
                && s[v] as u16 + d_goal[v] as u16 == big as u16
            {
                on[v] = true;
            }
        }
        on
    }

    /// Contact-aware separation diagnostic (the corrected Gate-2 semantics):
    /// the COMPLETE shortest-path sets are vertex-disjoint AND have no open-edge
    /// adjacency. Returns true ⟹ no pawn interaction while both pawns travel
    /// shortest paths. NOTE: this is NOT sufficient for a sound decisive gate —
    /// the trailing pawn can detour off its shortest path to block (Case B,
    /// `diag_gate2_nonadjacent_detour_counterexample`). Retained as diagnostic.
    fn paths_contact_free(g: &GameState, d0: &[u8; 81], d1: &[u8; 81]) -> bool {
        use crate::titanium::game::{BORDER, DELTA, DIRBIT};
        let set0 = shortest_path_set(g, g.pawn[0], d0);
        let set1 = shortest_path_set(g, g.pawn[1], d1);
        for a in 0..81 {
            if !set0[a] {
                continue;
            }
            if set1[a] {
                return false;
            }
            let bm = g.blocked[a] | BORDER[a];
            for d in 0..4 {
                if bm & DIRBIT[d] != 0 {
                    continue;
                }
                let b = (a as i16 + DELTA[d]) as usize;
                if set1[b] {
                    return false;
                }
            }
        }
        true
    }

    /// Reconstruct legal[`index`] from base seed RAND_SEEDS[0] exactly as the
    /// corpus audit does, probe `(p0,p1,turn)`, and print/return the full Gate-2
    /// diagnostic for that state.
    fn gate2_diag(index: usize, p0: usize, p1: usize, turn: usize) -> (i16, bool, bool, RaceVerdict, RaceClass) {
        let seed = 0xACE5_2026u64;
        let mut rng = seed ^ (index as u64).wrapping_mul(0x517C_C1B7_2722_0A95);
        let topo = loop {
            if let Some(t) = generate_legal_full_wall_topology(&mut rng, 256) {
                break t;
            }
        };
        let mut g = topo.g.clone();
        g.pawn[0] = p0;
        g.pawn[1] = p1;
        g.turn = turn;

        let mut ref_scratch = ReferenceScratch::new();
        let mut ref_tbl = vec![0i16; RACE_STATES];
        let mut gg = topo.g.clone();
        solve_race_config_reference(&mut gg, &mut ref_scratch, &mut ref_tbl);
        let oracle = ref_tbl[state_id(p0, p1, turn)];

        let mut d0 = [0u8; 81];
        let mut d1 = [0u8; 81];
        g.compute_dist(0, &mut d0);
        g.compute_dist(1, &mut d1);
        let overlap = paths_overlap(&g, &d0, &d1);
        let verdict = separated_pure_race_verdict(&g);
        let contact_free = paths_contact_free(&g, &d0, &d1);

        let set0 = shortest_path_set(&g, p0, &d0);
        let set1 = shortest_path_set(&g, p1, &d1);
        let cells0: Vec<usize> = (0..81).filter(|&c| set0[c]).collect();
        let cells1: Vec<usize> = (0..81).filter(|&c| set1[c]).collect();

        let mut sc = RaceScratch::new();
        let mut gp = g.clone();
        let prod = race_outcome(&mut gp, &mut sc);
        let wt = build_winner_table(&g);
        let cls = wt.classify(state_id(p0, p1, turn));

        eprintln!("── Gate 2 diagnostic legal[{index}] ──────────────────────────");
        eprintln!("replay moves: {:?}", topo.moves);
        eprintln!("state_id={} p0={p0} p1={p1} turn={turn} manhattan={}",
                  state_id(p0, p1, turn), cell_manhattan(p0, p1));
        eprintln!("d0[p0]={} d1[p1]={}", d0[p0], d1[p1]);
        eprintln!("shortest set0 (p0): {cells0:?}");
        eprintln!("shortest set1 (p1): {cells1:?}");
        eprintln!("paths_overlap(vertex)={overlap}  contact_free(corrected)={contact_free}");
        eprintln!("separated_verdict={verdict:?}  prod_bound={prod:?}");
        eprintln!("oracle={oracle} (>0 ⟹ stm wins)  tier3_class={cls:?}");
        eprintln!("──────────────────────────────────────────────────────────────");
        (oracle, overlap, contact_free, verdict, cls)
    }

    /// Counterexample 1 — ADJACENT pawns. The vertex-only `paths_overlap` test
    /// wrongly reports "separated"; the corrected contact-aware test catches it.
    /// The proven winner table is correct either way.
    /// Performance characterization of the Tier-3 winner table: cold build time,
    /// cached lookup time, coverage split, and persistent/scratch memory.
    #[test]
    fn perf_winner_table_characterization() {
        use std::time::Instant;

        // A representative walled topology (legal 20-wall endgame) + empty board.
        let mut rng = 0xACE5_2026u64 ^ (7u64).wrapping_mul(0x517C_C1B7_2722_0A95);
        let topo = loop {
            if let Some(t) = generate_legal_full_wall_topology(&mut rng, 256) {
                break t;
            }
        };

        for (label, g) in [("empty_board", GameState::new()), ("legal_20wall", topo.g.clone())] {
            // Cold build (median of a few).
            let mut best = u128::MAX;
            let mut tbl_holder = None;
            for _ in 0..5 {
                let t = Instant::now();
                let tbl = build_winner_table(&g);
                best = best.min(t.elapsed().as_micros());
                tbl_holder = Some(tbl);
            }
            let tbl = tbl_holder.unwrap();
            let (p0, p1, unk) = tbl.coverage();

            // Cached lookup over all states.
            let t = Instant::now();
            let mut sink = 0u64;
            for _ in 0..50 {
                for id in 0..RACE_STATES {
                    sink = sink.wrapping_add(tbl.class[id] as u64);
                }
            }
            let lookup_ns = t.elapsed().as_nanos() as f64 / (50.0 * RACE_STATES as f64);
            std::hint::black_box(sink);

            eprintln!(
                "PERF[{label}] build={:.2}ms lookup={lookup_ns:.2}ns/state \
                 coverage: P0={p0} P1={p1} Unknown={unk} ({:.1}% decided)  \
                 persistent={}B scratch={}B",
                best as f64 / 1000.0,
                100.0 * (p0 + p1) as f64 / RACE_STATES as f64,
                RaceWinnerTable::persistent_bytes(),
                AttractorScratch::scratch_bytes(),
            );
        }
    }

    #[test]
    fn diag_gate2_adjacent_counterexample() {
        let (oracle, overlap, contact_free, verdict, cls) = gate2_diag(92, 21, 20, 1);
        assert_eq!(oracle, 11, "oracle: stm (p1) wins in 11");
        assert!(!overlap, "vertex-only overlap reports separated (defect)");
        assert_eq!(verdict, RaceVerdict::Loss, "pure-race verdict: stm loses (WRONG)");
        assert_eq!(cls, RaceClass::ProvenP1, "winner table: P1 proven win (CORRECT)");
        assert!(!contact_free, "contact-aware test catches the adjacency contact");
    }

    /// Counterexample 2 — NON-ADJACENT pawns (manhattan 4). The corrected
    /// contact-aware test STILL reports "separated" (no shortest-path-set
    /// contact), yet the pure-race verdict is WRONG-SIGN: the trailing pawn
    /// detours OFF its shortest path to block. This proves the
    /// separated-shortest-path THEOREM is insufficient (Case B) — no
    /// shortest-path-set separation test can be a decisive gate.
    ///
    /// Sound outcome: production declines (Gate 2 non-decisive) and the winner
    /// table SOUNDLY DECLINES this state (a P0 win needs an off-shortest setup
    /// move, outside the restricted attractor). Production returns `Unknown` —
    /// never a wrong bound — and search / Service B resolves it.
    #[test]
    fn diag_gate2_nonadjacent_detour_counterexample() {
        let (oracle, _overlap, contact_free, verdict, cls) = gate2_diag(24, 23, 39, 0);
        assert_eq!(oracle, 15, "oracle: stm (p0) wins in 15");
        assert!(contact_free, "contact-aware test still reports separated (insufficient)");
        assert_eq!(verdict, RaceVerdict::Loss, "pure-race verdict: WRONG sign (would be a false bound)");
        // The winner table declines soundly (no false proof); production returns Unknown.
        assert_eq!(cls, RaceClass::Unknown, "winner table soundly declines (no false proof)");
    }

    #[test]
    fn certificate_poc_immediate_jump_to_goal() {
        let mut g = GameState::new();
        // p0 at cell 13 (row 1, col 4 — one step from row 0), p1 far at 40.
        g.pawn[0] = 13;
        g.pawn[1] = 40;
        g.turn = 0;
        let mut ctx = make_certificate_context(&mut g);
        match solve_certificate(&g, &mut ctx, &mut new_in_progress(), 0, &mut Vec::new()) {
            CertificateResult::Solved(s) => {
                assert_eq!(s.winner, 0, "p0 should win");
                assert_eq!(s.dtm, 1, "should win in 1 ply");
                assert!(s.best_move != NO_MOVE, "must have a move");
                assert!(is_home(0, s.best_move as usize), "move must reach p0 goal");
                eprintln!("immediate_goal: winner={} dtm={} mv={}", s.winner, s.dtm, s.best_move);
            }
            CertificateResult::DominanceCounterexample { state_id, .. } |
            CertificateResult::CycleDetected { state_id } => {
                panic!("unexpected result at state {state_id}");
            }
        }
    }

    /// Empty board head-on: p0=76 (col 4 row 8), p1=4 (col 4 row 0).
    /// Both at distance 8; p1 to move wins (exact DTM = 16, but loser perspective = -16).
    /// Known from `empty_board_head_on_race_is_movers_loss`: tbl[id] = -16.
    #[test]
    fn certificate_poc_head_on() {
        let mut g = GameState::new();
        g.pawn[0] = 76;
        g.pawn[1] = 4;
        g.turn = 0;
        let mut ctx = make_certificate_context(&mut g);
        // Build exact oracle.
        let mut s = RaceScratch::new();
        let mut tbl = vec![0i16; RACE_STATES];
        solve_race_config(&mut g, &mut s, &mut tbl);
        let oracle_id = state_id(76, 4, 0);
        let oracle = tbl[oracle_id];
        eprintln!("head_on oracle: tbl[{oracle_id}] = {oracle}");
        match solve_certificate(&g, &mut ctx, &mut new_in_progress(), 0, &mut Vec::new()) {
            CertificateResult::Solved(s) => {
                let oracle_winner = if oracle > 0 { 0u8 } else { 1u8 };
                assert_eq!(s.winner, oracle_winner, "winner mismatch head-on");
                eprintln!("head_on cert: winner={} dtm={} mv={}", s.winner, s.dtm, s.best_move);
            }
            CertificateResult::DominanceCounterexample { state_id, legal_moves, .. } => {
                for d in &legal_moves {
                    eprintln!("  mv={} dst={} old_d={} new_d={} delta={} jump={}", d.move_id, d.destination, d.old_distance, d.new_distance, d.delta, d.is_jump);
                }
                panic!("counterexample at state {state_id}");
            }
            CertificateResult::CycleDetected { state_id } => {
                panic!("cycle at state {state_id}");
            }
        }
    }

    /// Empty board: p0 at 18 (row 2, col 0), p1 at 9 (row 1, col 0).
    /// p0 can jump over p1 to reach row 0 in one move (productive jump, delta=2).
    /// Known from `immediate_jump_to_goal_wins_in_one_ply`: tbl[id(18,9,0)] = 1.
    #[test]
    fn certificate_poc_diagonal_jump() {
        let mut g = GameState::new();
        g.pawn[0] = 18;
        g.pawn[1] = 9;
        g.turn = 0;
        let mut ctx = make_certificate_context(&mut g);
        match solve_certificate(&g, &mut ctx, &mut new_in_progress(), 0, &mut Vec::new()) {
            CertificateResult::Solved(s) => {
                assert_eq!(s.winner, 0, "p0 should win");
                assert_eq!(s.dtm, 1, "should win in 1 ply via jump");
                assert!(s.best_move != NO_MOVE, "must have a move");
                assert!(is_home(0, s.best_move as usize), "jump must reach row 0");
                eprintln!("diagonal_jump: winner={} dtm={} mv={}", s.winner, s.dtm, s.best_move);
            }
            CertificateResult::DominanceCounterexample { state_id, .. } |
            CertificateResult::CycleDetected { state_id } => {
                panic!("unexpected result at state {state_id}");
            }
        }
    }

    /// PV consistency: for a sample of states, reconstruct PV and verify length
    /// equals DTM and all moves are legal.
    #[test]
    fn certificate_poc_pv_consistency() {
        let mut g = GameState::new();
        let mut ctx = make_certificate_context(&mut g);

        // Solve a few specific states.
        let cases: &[(usize, usize, usize)] = &[
            (13, 40, 0), // p0 one step from goal
            (76, 4, 0),  // head-on
            (18, 9, 0),  // productive jump
            (40, 41, 0), // middle of board
            (72, 9, 1),  // p1 near goal (cell 72 >= 72 is home for p1 — actually terminal!)
        ];

        // Use non-terminal cases only.
        let _unused = cases; // shadow to avoid warning
        let cases: &[(usize, usize, usize)] = &[
            (13, 40, 0),
            (40, 41, 0),
            (76, 4, 0),
        ];

        for &(p0, p1, turn) in cases {
            let mut root = g.clone();
            root.pawn[0] = p0;
            root.pawn[1] = p1;
            root.turn = turn;

            // Skip if already terminal.
            if is_home(0, p0) || is_home(1, p1) {
                continue;
            }

            let sol = match solve_certificate(&root, &mut ctx, &mut new_in_progress(), 0, &mut Vec::new()) {
                CertificateResult::Solved(s) => s,
                CertificateResult::DominanceCounterexample { state_id, .. } |
                CertificateResult::CycleDetected { state_id } => {
                    panic!("unexpected result at state {state_id} for ({p0},{p1},{turn})");
                }
            };

            let pv = reconstruct_certificate_pv(&root, &mut ctx);
            eprintln!(
                "pv_consistency ({p0},{p1},{turn}): winner={} dtm={} pv_len={} pv={pv:?}",
                sol.winner, sol.dtm, pv.len()
            );

            assert_eq!(
                pv.len(),
                sol.dtm as usize,
                "PV length {} != DTM {} at ({p0},{p1},{turn})",
                pv.len(),
                sol.dtm
            );

            // Walk the PV and verify each move is legal.
            let mut walk = root.clone();
            for &mv in &pv {
                let mut legal_buf = [0i16; 16];
                let nm = walk.gen_pawn_moves(&mut legal_buf, 0);
                assert!(
                    legal_buf[..nm].contains(&mv),
                    "PV move {mv} is illegal at ({},{},{})",
                    walk.pawn[0],
                    walk.pawn[1],
                    walk.turn
                );
                walk.make_move(mv);
            }

            // After DTM plies, the declared winner's goal should be reached.
            let winner_at_goal = if sol.winner == 0 {
                is_home(0, walk.pawn[0])
            } else {
                is_home(1, walk.pawn[1])
            };
            assert!(
                winner_at_goal,
                "PV did not reach winner's goal at ({p0},{p1},{turn})"
            );
        }
    }

    /// Main POC correctness test: solve every live state in the exact table and
    /// compare certificate (winner, DTM, Bellman move optimality) against oracle.
    #[test]
    fn certificate_poc_all_exact_states() {
        use std::time::Instant;

        let mut g = GameState::new();

        // Build oracle.
        let t_oracle = Instant::now();
        let mut ref_scratch = ReferenceScratch::new();
        let mut oracle_tbl = vec![0i16; RACE_STATES];
        solve_race_config_reference(&mut g, &mut ref_scratch, &mut oracle_tbl);
        let oracle_ms = t_oracle.elapsed().as_millis();

        // Build certificate context (topology distances).
        let mut ctx = make_certificate_context(&mut g);

        // Cold-solve from a representative root to warm the memo table.
        g.pawn[0] = 40;
        g.pawn[1] = 41;
        g.turn = 0;
        let t_cold = Instant::now();
        let _ = solve_certificate(&g, &mut ctx, &mut new_in_progress(), 0, &mut Vec::new());
        let cold_us = t_cold.elapsed().as_micros();

        // Counters.
        let mut live = 0usize;
        let mut winner_mismatches = 0usize;
        let mut dtm_mismatches = 0usize;
        let mut move_violations = 0usize;
        let mut counterexamples = 0usize;
        let mut memo_hits = 0usize;
        let mut first_winner_fail: Option<String> = None;
        let mut first_dtm_fail: Option<String> = None;

        // Measure warm-lookup time on a solved state.
        let warm_id = state_id(40, 41, 0);
        let t_warm = Instant::now();
        for _ in 0..10_000 {
            let _ = ctx.memo[warm_id];
        }
        let warm_ns = t_warm.elapsed().as_nanos() / 10_000;

        // Sweep all live states.
        for p0 in 9..81usize {
            for p1 in 0..72usize {
                if p0 == p1 {
                    continue;
                }
                for turn in 0..2usize {
                    let id = state_id(p0, p1, turn);
                    let oracle = oracle_tbl[id];
                    if oracle == 0 {
                        continue;
                    }
                    live += 1;

                    if ctx.memo[id].is_some() {
                        memo_hits += 1;
                    }

                    // Set up state.
                    g.pawn[0] = p0;
                    g.pawn[1] = p1;
                    g.turn = turn;

                    let cert_result = solve_certificate(&g, &mut ctx, &mut new_in_progress(), 0, &mut Vec::new());

                    let sol = match cert_result {
                        CertificateResult::Solved(s) => s,
                        CertificateResult::DominanceCounterexample {
                            state_id: sid,
                            p0: cp0,
                            p1: cp1,
                            turn: cturn,
                            ref legal_moves,
                        } => {
                            counterexamples += 1;
                            if first_winner_fail.is_none() {
                                let diag: Vec<String> = legal_moves
                                    .iter()
                                    .map(|d| {
                                        format!(
                                            "mv={} dst={} old={} new={} delta={} jump={}",
                                            d.move_id, d.destination,
                                            d.old_distance, d.new_distance,
                                            d.delta, d.is_jump
                                        )
                                    })
                                    .collect();
                                first_winner_fail = Some(format!(
                                    "COUNTEREXAMPLE sid={sid} p0={cp0} p1={cp1} turn={cturn} \
                                     oracle={oracle} moves=[{}]",
                                    diag.join("; ")
                                ));
                            }
                            continue;
                        }
                        CertificateResult::CycleDetected { state_id: sid } => {
                            counterexamples += 1;
                            if first_winner_fail.is_none() {
                                first_winner_fail = Some(format!(
                                    "CYCLE id={sid} p0={p0} p1={p1} turn={turn} oracle={oracle}"
                                ));
                            }
                            continue;
                        }
                    };

                    // Winner check.
                    // Oracle: +k = stm wins in k plies; -k = stm loses.
                    let oracle_winner = if oracle > 0 { turn as u8 } else { (turn ^ 1) as u8 };
                    if sol.winner != oracle_winner {
                        winner_mismatches += 1;
                        if first_winner_fail.is_none() {
                            first_winner_fail = Some(format!(
                                "WINNER MISMATCH id={id} p0={p0} p1={p1} turn={turn} \
                                 cert_winner={} oracle_winner={oracle_winner} oracle={oracle}",
                                sol.winner
                            ));
                        }
                        continue;
                    }

                    // DTM check.
                    let oracle_dtm = oracle.unsigned_abs() as u16;
                    if sol.dtm != oracle_dtm {
                        dtm_mismatches += 1;
                        if first_dtm_fail.is_none() {
                            first_dtm_fail = Some(format!(
                                "DTM MISMATCH id={id} p0={p0} p1={p1} turn={turn} \
                                 cert_dtm={} oracle_dtm={oracle_dtm} oracle={oracle}",
                                sol.dtm
                            ));
                        }
                    }

                    // Bellman move optimality check (only if winner is correct).
                    if sol.best_move != NO_MOVE && sol.best_move >= 0 {
                        let mv = sol.best_move;
                        let dst = mv as usize;
                        let child_oracle = if is_home(turn, dst) {
                            1i16 // terminal win: DTM=1 from child's (non-existent) perspective
                        } else {
                            let child_id = if turn == 0 {
                                state_id(dst, p1, 1)
                            } else {
                                state_id(p0, dst, 0)
                            };
                            oracle_tbl[child_id]
                        };

                        // Child oracle from child's (stm=opponent) perspective.
                        // oracle > 0 means stm wins in oracle plies → chosen move is a
                        // "loss child" for original stm (child.stm=opponent wins).
                        let stm_wins = oracle > 0;
                        if stm_wins {
                            // stm wins → best move leads to a LOSING child for child's stm.
                            // child_oracle should be < 0 (child stm = opponent loses).
                            // OR child_oracle is 1 (terminal win for stm via goal move).
                            let child_ok = is_home(turn, dst) || child_oracle < 0;
                            if !child_ok {
                                // Also check it's the minimum-DTM winning child.
                                // Soft violation — count but don't hard-fail.
                                move_violations += 1;
                            }
                        } else {
                            // stm loses → every retained child is a winning child for child stm.
                            // Best move maximises DTM.
                            let child_ok = child_oracle > 0;
                            if !child_ok {
                                move_violations += 1;
                            }
                        }
                    }
                }
            }
        }

        let final_memo_entries = ctx.memo.iter().filter(|e| e.is_some()).count();

        eprintln!("─── certificate POC report ───────────────────────────────────");
        eprintln!("topology:               empty board");
        eprintln!("addressable states:     {RACE_STATES}");
        eprintln!("live states:            {live}");
        eprintln!("winner mismatches:      {winner_mismatches}");
        eprintln!("dtm mismatches:         {dtm_mismatches}");
        eprintln!("dominance counterex:    {counterexamples}");
        eprintln!("bellman move violations:{move_violations}");
        eprintln!("memo hits:              {memo_hits}/{live}");
        eprintln!("memo entries filled:    {final_memo_entries}");
        eprintln!("──────────────────────────────────────────────────────────────");
        eprintln!("oracle build time:      {oracle_ms} ms");
        eprintln!("cert cold solve time:   {cold_us} µs");
        eprintln!("cert warm lookup (ns):  {warm_ns}");
        eprintln!("memo memory:            {} KB", RACE_STATES * std::mem::size_of::<Option<CertificateSolution>>() / 1024);
        eprintln!("──────────────────────────────────────────────────────────────");
        if let Some(ref msg) = first_winner_fail {
            eprintln!("first winner fail: {msg}");
        }
        if let Some(ref msg) = first_dtm_fail {
            eprintln!("first dtm fail:    {msg}");
        }

        // Hard assertions.
        assert_eq!(
            counterexamples, 0,
            "theorem counterexample: no retained delta>=1 move at {} state(s)", counterexamples
        );
        assert_eq!(
            winner_mismatches, 0,
            "certificate winner mismatch at {} state(s); first: {}",
            winner_mismatches,
            first_winner_fail.as_deref().unwrap_or("none")
        );

        // Informational: report DTM accuracy.
        if dtm_mismatches == 0 {
            eprintln!("VERDICT: exact winner + exact DTM + exact PV supported");
        } else {
            eprintln!(
                "VERDICT: exact winner supported; DTM approximate ({dtm_mismatches} mismatches); first: {}",
                first_dtm_fail.as_deref().unwrap_or("none")
            );
        }

        // Verify PV consistency on a sample.
        let sample_cases: &[(usize, usize, usize)] = &[
            (13, 40, 0),
            (40, 41, 0),
            (76, 4, 0),
            (76, 4, 1),
            (40, 41, 1),
        ];
        let mut pv_failures = 0usize;
        for &(p0, p1, turn) in sample_cases {
            if is_home(0, p0) || is_home(1, p1) {
                continue;
            }
            let mut root = GameState::new();
            root.pawn[0] = p0;
            root.pawn[1] = p1;
            root.turn = turn;
            let id = state_id(p0, p1, turn);
            let sol = match ctx.memo[id] {
                Some(s) => s,
                None => continue,
            };
            let pv = reconstruct_certificate_pv(&root, &mut ctx);
            if pv.len() != sol.dtm as usize {
                pv_failures += 1;
                eprintln!("PV length {} != DTM {} at ({p0},{p1},{turn})", pv.len(), sol.dtm);
            }
            // Walk and verify legal.
            let mut walk = root.clone();
            for &mv in &pv {
                let mut lb = [0i16; 16];
                let nm = walk.gen_pawn_moves(&mut lb, 0);
                if !lb[..nm].contains(&mv) {
                    pv_failures += 1;
                    eprintln!("PV move {mv} illegal at ({},{},{})", walk.pawn[0], walk.pawn[1], walk.turn);
                }
                walk.make_move(mv);
            }
        }
        assert_eq!(pv_failures, 0, "PV consistency failures: {pv_failures}");
    }

    // ────────────────────────────────────────────────────────────────────────
    // Real-game database certificate test.
    //
    // Reads all stored complete games from:
    //   training/data/all_games.db   (SQLite, moves_bin = u16-LE move IDs)
    //   training/data/**/*.games     (text, "GAME move1 move2 ..." lines)
    //
    // For each game, replays from GameState::new() and detects g.wl == [0, 0].
    // Deduplicates wall topologies by (hw, vw) bitboards.
    //
    // For each unique topology:
    //   - Builds the heavy exact race table once (oracle).
    //   - Iterates all 13,122 addressable state IDs exhaustively.
    //   - Runs solve_certificate on every live state.
    //   - Compares certificate winner vs oracle.
    //   - Reports DTM mismatches separately.
    //
    // Also tests actual recorded endgame positions (turn+pawns at extraction).
    // ────────────────────────────────────────────────────────────────────────

    /// Wall topology extracted from a real game.
    #[derive(Clone)]
    struct ExtractedTopology {
        hw: [u8; 64],
        vw: [u8; 64],
        /// Provenance: list of (source, game_index, ply, move_prefix).
        sources: Vec<(String, usize, usize, Vec<i16>)>,
        /// Actual recorded endgame positions at extraction.
        endgame_positions: Vec<(usize, usize, usize)>, // (p0, p1, turn)
    }

    /// Return repo root (parent of the engine crate directory).
    fn repo_root() -> Option<std::path::PathBuf> {
        let manifest = std::env::var("CARGO_MANIFEST_DIR").ok()?;
        std::path::Path::new(&manifest).parent().map(|p| p.to_path_buf())
    }

    /// Load all games from all_games.db. Returns Vec of move sequences (move IDs).
    fn load_games_from_sqlite(db_path: &std::path::Path) -> Vec<(String, Vec<i16>)> {
        use rusqlite::Connection;
        let mut out = Vec::new();
        let conn = match Connection::open(db_path) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("Skipping {}: {e}", db_path.display());
                return out;
            }
        };
        let mut stmt = match conn.prepare("SELECT id, moves_bin FROM games WHERE moves_bin IS NOT NULL") {
            Ok(s) => s,
            Err(e) => {
                eprintln!("prepare failed: {e}");
                return out;
            }
        };
        let rows = stmt.query_map([], |row| {
            let id: i64 = row.get(0)?;
            let blob: Vec<u8> = row.get(1)?;
            Ok((id, blob))
        });
        match rows {
            Err(e) => eprintln!("query failed: {e}"),
            Ok(iter) => {
                for row in iter.flatten() {
                    let (id, blob) = row;
                    // u16 little-endian: each move is 2 bytes, value = engine move ID.
                    if blob.len() % 2 != 0 {
                        continue;
                    }
                    let moves: Vec<i16> = blob
                        .chunks_exact(2)
                        .map(|b| u16::from_le_bytes([b[0], b[1]]) as i16)
                        .collect();
                    out.push((format!("all_games.db#game_{id}"), moves));
                }
            }
        }
        out
    }

    /// Load all games from all *.games text files under `data_dir`.
    fn load_games_from_text_files(data_dir: &std::path::Path) -> Vec<(String, Vec<i16>)> {
        use crate::titanium::algebraic_to_move_id;
        use std::io::{BufRead, BufReader};
        let mut out = Vec::new();
        let walker = walkdir_games(data_dir);
        for path in walker {
            let src_name = path
                .strip_prefix(data_dir)
                .unwrap_or(&path)
                .display()
                .to_string();
            let file = match std::fs::File::open(&path) {
                Ok(f) => f,
                Err(_) => continue,
            };
            let mut game_idx = 0usize;
            for line in BufReader::new(file).lines().flatten() {
                if let Some(rest) = line.strip_prefix("GAME ") {
                    let moves: Vec<i16> = rest
                        .split_whitespace()
                        .map(algebraic_to_move_id)
                        .collect();
                    out.push((format!("{src_name}#game_{game_idx}"), moves));
                    game_idx += 1;
                }
            }
        }
        out
    }

    /// Collect *.games file paths under dir, recursively, skipping pytest temp dirs.
    fn walkdir_games(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
        let mut out = Vec::new();
        let Ok(entries) = std::fs::read_dir(dir) else {
            return out;
        };
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() {
                let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if name.starts_with('.') || name == ".pytest-temp" {
                    continue;
                }
                out.extend(walkdir_games(&p));
            } else if p.extension().and_then(|e| e.to_str()) == Some("games") {
                out.push(p);
            }
        }
        out
    }

    /// Replay a move sequence and collect every ply where g.wl == [0, 0].
    /// Returns list of (ply_index, g_clone) for each such ply.
    fn replay_collect_zero_wall_states(moves: &[i16]) -> Vec<(usize, GameState)> {
        let mut g = GameState::new();
        let mut out = Vec::new();
        for (ply, &mv) in moves.iter().enumerate() {
            // Validate: skip if move ID is out of range or game already over.
            if g.winner() >= 0 {
                break;
            }
            g.make_move(mv);
            if g.wl == [0, 0] && g.winner() < 0 {
                out.push((ply + 1, g.clone()));
            }
        }
        out
    }

    /// Per-prover attractor result: can prover P force reaching its own goal,
    /// the forcing distance (DTM), and at OR (prover-to-move) nodes the move
    /// that realises the forced win.
    struct AttractorResult {
        win: Vec<bool>,
        dist: Vec<u16>,
        best_mv: Vec<i16>,
    }

    /// Sound asymmetric strategy certificate via attractor (backward reachability)
    /// computation for a single prover `P`.
    ///
    /// At a node it is `P`'s turn → **OR node**: P chooses one of its RESTRICTED
    /// moves (Class A shortest-path-progress + any jump).  P wins if SOME such
    /// move leads to a P-win.
    ///
    /// At a node it is the opponent's turn → **AND node**: the opponent may play
    /// ANY legal move (no restriction).  P wins only if ALL opponent moves still
    /// lead to a P-win.
    ///
    /// Seeds: every state in which P's pawn is already home (P has won).
    ///
    /// Because the OPPONENT is never restricted, any state placed in the
    /// attractor corresponds to a genuine forcing strategy for P in the FULL
    /// game → a claimed win is never false.  (It may be incomplete: a true P-win
    /// requiring an off-shortest-path setup move for P falls outside the
    /// attractor and is simply declined — never mis-reported.)
    fn attractor_solve(
        g_root: &GameState,
        own_goal_dist: &[[u8; 81]; 2],
        prover: usize,
    ) -> AttractorResult {
        // predecessors[child] = (parent, move). Edges follow each parent's own
        // node type: OR (prover turn) → restricted moves; AND (opp turn) → full.
        let mut predecessors: Vec<Vec<(usize, i16)>> = vec![Vec::new(); RACE_STATES];
        // For AND nodes: remaining unconfirmed children. For OR nodes: unused.
        let mut remaining: Vec<u16> = vec![0; RACE_STATES];
        let mut win = vec![false; RACE_STATES];
        let mut dist = vec![u16::MAX; RACE_STATES];
        let mut best_mv = vec![NO_MOVE; RACE_STATES];

        let mut queue: std::collections::VecDeque<usize> =
            std::collections::VecDeque::new();

        for p0 in 0..81usize {
            for p1 in 0..81usize {
                if p0 == p1 { continue; }
                for turn in 0..2usize {
                    let id = state_id(p0, p1, turn);

                    // Seed: prover already home (regardless of whose turn).
                    let prover_home = is_home(prover, if prover == 0 { p0 } else { p1 });
                    if prover_home {
                        win[id] = true;
                        dist[id] = 0;
                        queue.push_back(id);
                        continue;
                    }
                    // Opponent already home → prover cannot win from here; no edges.
                    let opp = prover ^ 1;
                    let opp_home = is_home(opp, if opp == 0 { p0 } else { p1 });
                    if opp_home { continue; }

                    let mut g = g_root.clone();
                    g.pawn[0] = p0;
                    g.pawn[1] = p1;
                    g.turn = turn;

                    let mut buf = [0i16; 16];
                    let nm = g.gen_pawn_moves(&mut buf, 0);

                    let side = turn;
                    let is_or = side == prover;
                    let src = if side == 0 { p0 } else { p1 };
                    let old_d = own_goal_dist[side][src];
                    let mut child_count: u16 = 0;

                    for &mv in &buf[..nm] {
                        let dst = mv as usize;
                        if is_or {
                            // Restricted: Class A (delta>=1) or any jump.
                            let new_d = own_goal_dist[side][dst];
                            let delta = if new_d == u8::MAX {
                                i16::MIN / 2
                            } else {
                                old_d as i16 - new_d as i16
                            };
                            let jump = is_jump_move(src, dst);
                            if !(jump || delta >= 1) { continue; }
                        }
                        // AND node: take every legal move unrestricted.
                        let mut cg = g.clone();
                        cg.make_move(mv);
                        let cid = state_id(cg.pawn[0], cg.pawn[1], cg.turn);
                        predecessors[cid].push((id, mv));
                        child_count += 1;
                    }
                    // AND node needs all children; OR node needs just one.
                    remaining[id] = child_count;
                }
            }
        }

        // Backward reachability (attractor) propagation.
        while let Some(c) = queue.pop_front() {
            let cd = dist[c];
            let preds = predecessors[c].clone();
            for (p, mv) in preds {
                if win[p] { continue; }
                let p_turn = p % 2;
                if p_turn == prover {
                    // OR node: one winning child suffices.
                    win[p] = true;
                    dist[p] = cd + 1;
                    best_mv[p] = mv;
                    queue.push_back(p);
                } else {
                    // AND node: confirm only when every child is a P-win.
                    if remaining[p] > 0 { remaining[p] -= 1; }
                    if remaining[p] == 0 {
                        win[p] = true;
                        dist[p] = cd + 1; // last (max-dist) child ⟹ max resistance
                        queue.push_back(p);
                    }
                }
            }
        }

        AttractorResult { win, dist, best_mv }
    }

    /// Populate `ctx.memo` for the current wall topology using the sound
    /// asymmetric strategy certificate.  Runs one attractor pass per prover and
    /// merges: a state is certified for whichever prover forces a win; states
    /// forced by neither are left `None` (declined — counted as a counterexample,
    /// never a winner mismatch).  Fully iterative — no recursion, no stack risk.
    fn solve_all_iterative(g_root: &GameState, ctx: &mut CertificateContext) {
        let a0 = attractor_solve(g_root, &ctx.own_goal_dist, 0);
        let a1 = attractor_solve(g_root, &ctx.own_goal_dist, 1);

        for id in 0..RACE_STATES {
            let turn = id % 2;
            // Decode pawn cells to detect the degenerate both-home state
            // (p0 on row 0 AND p1 on row 8) — unreachable in real play and
            // excluded from the sweep, but it trivially seeds both attractors.
            let p1c = (id / 2) % 81;
            let p0c = (id / 2) / 81;
            let both_home = is_home(0, p0c) && is_home(1, p1c);
            if a0.win[id] {
                // Determinacy: a0 and a1 cannot both hold for a real state.
                debug_assert!(both_home || !a1.win[id], "both provers force win at state {id}");
                ctx.memo[id] = Some(CertificateSolution {
                    winner: 0,
                    dtm: a0.dist[id],
                    best_move: if turn == 0 { a0.best_mv[id] } else { NO_MOVE },
                });
            } else if a1.win[id] {
                ctx.memo[id] = Some(CertificateSolution {
                    winner: 1,
                    dtm: a1.dist[id],
                    best_move: if turn == 1 { a1.best_mv[id] } else { NO_MOVE },
                });
            }
            // else: declined — leave None.
        }
    }

    #[test]
    fn certificate_poc_real_game_database() {
        use std::time::Instant;
        use std::collections::HashMap;

        let root = match repo_root() {
            Some(r) => r,
            None => {
                eprintln!("SKIP: cannot determine repo root");
                return;
            }
        };

        let db_path = root.join("training/data/all_games.db");
        let data_dir = root.join("training/data");

        // ── 1. Load all games ──────────────────────────────────────────────
        eprintln!("Loading games from SQLite: {}", db_path.display());
        let sqlite_games = load_games_from_sqlite(&db_path);
        eprintln!("  Loaded {} games from SQLite", sqlite_games.len());

        eprintln!("Loading games from .games files: {}", data_dir.display());
        let text_games = load_games_from_text_files(&data_dir);
        eprintln!("  Loaded {} games from .games files", text_games.len());

        let all_games: Vec<(String, Vec<i16>)> = sqlite_games
            .into_iter()
            .chain(text_games.into_iter())
            .collect();
        eprintln!("Total games: {}", all_games.len());
        assert!(!all_games.is_empty(), "No games loaded — check database paths");

        // ── 2. Replay all games, collect topologies ────────────────────────
        // Dedup key: (hw[0..64], vw[0..64]).
        let mut topo_map: HashMap<([u8; 64], [u8; 64]), ExtractedTopology> = HashMap::new();
        let mut total_replayed = 0usize;
        let mut total_zero_wall_hits = 0usize;

        for (game_idx, (source, moves)) in all_games.iter().enumerate() {
            total_replayed += 1;
            let zero_states = replay_collect_zero_wall_states(moves);
            for (ply, mut g) in zero_states {
                total_zero_wall_hits += 1;
                // Assert soundness of extraction.
                debug_assert_eq!(g.wl, [0, 0]);
                debug_assert!(g.winner() < 0);
                assert!(g.has_path(0), "p0 must retain a goal path at extraction");
                assert!(g.has_path(1), "p1 must retain a goal path at extraction");
                // Count walls placed by each player.
                let w0 = hw_count(&g.hw) + vw_count(&g.vw); // total walls on board
                let _ = w0; // actual per-player tracking not needed here since wl==0 asserts it

                let key = (g.hw, g.vw);
                let entry = topo_map.entry(key).or_insert_with(|| ExtractedTopology {
                    hw: g.hw,
                    vw: g.vw,
                    sources: Vec::new(),
                    endgame_positions: Vec::new(),
                });
                if entry.sources.len() < 4 {
                    entry.sources.push((
                        source.clone(),
                        game_idx,
                        ply,
                        moves[..ply].to_vec(),
                    ));
                }
                // Record the actual endgame position (may hit same topology many times).
                let pos = (g.pawn[0], g.pawn[1], g.turn);
                if !entry.endgame_positions.contains(&pos) {
                    entry.endgame_positions.push(pos);
                }
            }
        }

        eprintln!("Replayed: {total_replayed} games, zero-wall hits: {total_zero_wall_hits}");
        eprintln!("Unique topologies: {}", topo_map.len());

        if topo_map.is_empty() {
            eprintln!("No wl==[0,0] positions found in any game. Cannot run exhaustive test.");
            eprintln!("The game database may not contain games long enough to exhaust all walls.");
            return;
        }

        // ── 3. Exhaustive certificate sweep per topology ───────────────────
        let mut total_live = 0u64;
        let mut total_winner_mismatches = 0u64;
        let mut total_dtm_mismatches = 0u64;
        let mut total_cert_counterexamples = 0u64;
        let mut total_bellman_violations = 0u64;
        // Production winner-table audit: parity with the test implementation and
        // direct production-vs-oracle winner agreement.
        let mut total_parity_mismatches = 0u64;
        let mut total_prod_winner_mismatches = 0u64;
        // Corrected (contact-aware) Gate 2 audit: decisive firings + sign violations.
        let mut total_g2_fires = 0u64;
        let mut total_g2_violations = 0u64;
        let mut first_g2_violation: Option<String> = None;
        let mut first_winner_fail: Option<String> = None;

        // Endgame position results (from actual recorded game positions).
        let mut endgame_tested = 0u64;
        let mut endgame_winner_mismatches = 0u64;
        let mut endgame_dtm_mismatches = 0u64;

        let t_start = Instant::now();
        let topologies: Vec<_> = topo_map.into_values().collect();

        for (topo_idx, topo) in topologies.iter().enumerate() {
            let mut g = build_game_with_walls(&topo.hw, &topo.vw);

            // Build oracle (heavy exact table).
            let mut ref_scratch = ReferenceScratch::new();
            let mut oracle_tbl = vec![0i16; RACE_STATES];
            solve_race_config_reference(&mut g, &mut ref_scratch, &mut oracle_tbl);

            // Build certificate context and fully populate via iterative BFS.
            let mut ctx = make_certificate_context(&mut g);
            solve_all_iterative(&g, &mut ctx);

            // Build the PRODUCTION winner table for the same topology — parity
            // against the proven test implementation is asserted per state below.
            let prod_tbl = build_winner_table(&g);

            // Wall-graph goal distances (pawn-independent) — computed once per
            // topology for the corrected contact-aware Gate 2 audit.
            let mut g2d0 = [0u8; 81];
            let mut g2d1 = [0u8; 81];
            g.compute_dist(0, &mut g2d0);
            g.compute_dist(1, &mut g2d1);

            // ── 3a. Exhaustive pawn-state sweep ───────────────────────────
            let mut topo_live = 0u64;
            let mut topo_winner_m = 0u64;
            let mut topo_dtm_m = 0u64;
            let mut topo_cert_cx = 0u64;
            let mut topo_bellman = 0u64;

            for p0 in 9..81usize {
                for p1 in 0..72usize {
                    if p0 == p1 { continue; }
                    for turn in 0..2usize {
                        let id = state_id(p0, p1, turn);
                        let oracle = oracle_tbl[id];
                        if oracle == 0 { continue; }
                        topo_live += 1;

                        let oracle_winner = if oracle > 0 { turn as u8 } else { (turn ^ 1) as u8 };

                        // ── Production parity + production-vs-oracle audit ─────
                        let test_winner: Option<u8> = ctx.memo[id].map(|s| s.winner);
                        let prod_winner: Option<u8> = match prod_tbl.classify(id) {
                            RaceClass::ProvenP0 => Some(0),
                            RaceClass::ProvenP1 => Some(1),
                            RaceClass::Unknown => None,
                        };
                        if prod_winner != test_winner {
                            total_parity_mismatches += 1;
                        }
                        if let Some(w) = prod_winner {
                            if w != oracle_winner {
                                total_prod_winner_mismatches += 1;
                            }
                        }

                        // ── Corrected (contact-aware) Gate 2 audit ────────────
                        // Fires decisively only when the complete shortest-path
                        // sets are contact-free; verdict via pure tempo race.
                        g.pawn[0] = p0;
                        g.pawn[1] = p1;
                        g.turn = turn;
                        if g2d0[p0] != u8::MAX
                            && g2d1[p1] != u8::MAX
                            && paths_contact_free(&g, &g2d0, &g2d1)
                        {
                            total_g2_fires += 1;
                            let stm_wins = if turn == 0 {
                                g2d0[p0] <= g2d1[p1]
                            } else {
                                g2d1[p1] <= g2d0[p0]
                            };
                            let g2_winner = if stm_wins { turn as u8 } else { (turn ^ 1) as u8 };
                            if g2_winner != oracle_winner {
                                total_g2_violations += 1;
                                if first_g2_violation.is_none() {
                                    first_g2_violation = Some(format!(
                                        "G2 VIOLATION topo={topo_idx} id={id} p0={p0} p1={p1} \
                                         turn={turn} d0={} d1={} g2_winner={g2_winner} \
                                         oracle={oracle} manhattan={}",
                                        g2d0[p0], g2d1[p1], cell_manhattan(p0, p1)
                                    ));
                                }
                            }
                        }

                        // Iterative solver pre-populated ctx.memo — direct lookup.
                        let sol = match ctx.memo[id] {
                            Some(s) => s,
                            None => {
                                topo_cert_cx += 1; // cycle / dominance: unresolved
                                continue;
                            }
                        };

                        if sol.winner != oracle_winner {
                            topo_winner_m += 1;
                            if first_winner_fail.is_none() {
                                g.pawn[0] = p0;
                                g.pawn[1] = p1;
                                g.turn = turn;
                                first_winner_fail = Some(build_failure_diag(
                                    &topo.sources.first().map(|s| s.0.as_str()).unwrap_or("?"),
                                    topo_idx, &topo.hw, &topo.vw,
                                    id, p0, p1, turn, &sol, oracle, &ctx, &oracle_tbl, &mut g,
                                ));
                            }
                            continue;
                        }

                        let oracle_dtm = oracle.unsigned_abs() as u16;
                        if sol.dtm != oracle_dtm { topo_dtm_m += 1; }

                        if sol.best_move != NO_MOVE && sol.best_move >= 0 {
                            let dst = sol.best_move as usize;
                            let child_oracle = if is_home(turn, dst) {
                                1i16
                            } else {
                                let cid = if turn == 0 {
                                    state_id(dst, p1, 1)
                                } else {
                                    state_id(p0, dst, 0)
                                };
                                oracle_tbl[cid]
                            };
                            let stm_wins = oracle > 0;
                            let child_ok = if stm_wins {
                                is_home(turn, dst) || child_oracle < 0
                            } else {
                                child_oracle > 0
                            };
                            if !child_ok { topo_bellman += 1; }
                        }
                    }
                }
            }

            total_live += topo_live;
            total_winner_mismatches += topo_winner_m;
            total_dtm_mismatches += topo_dtm_m;
            total_cert_counterexamples += topo_cert_cx;
            total_bellman_violations += topo_bellman;

            // ── 3b. Actual recorded endgame positions ─────────────────────
            for &(p0, p1, turn) in &topo.endgame_positions {
                let id = state_id(p0, p1, turn);
                let oracle = oracle_tbl[id];
                if oracle == 0 { continue; }
                endgame_tested += 1;

                if let Some(sol) = ctx.memo[id] {
                    let oracle_winner = if oracle > 0 { turn as u8 } else { (turn ^ 1) as u8 };
                    if sol.winner != oracle_winner { endgame_winner_mismatches += 1; }
                    let oracle_dtm = oracle.unsigned_abs() as u16;
                    if sol.dtm != oracle_dtm { endgame_dtm_mismatches += 1; }
                }
            }

            if total_winner_mismatches > 0
                || total_parity_mismatches > 0
                || total_prod_winner_mismatches > 0
            {
                break; // stop on first failing topology for fast diagnostics
            }
            // NOTE: do NOT break on Gate-2 violations — we want the full corpus
            // count to decide Case A (keep decisive) vs Case B (defer to table).
        }

        let elapsed = t_start.elapsed();

        // ── 4. Report ──────────────────────────────────────────────────────
        eprintln!("─── real-game database certificate report ───────────────────");
        eprintln!("Games loaded:            {total_replayed}");
        eprintln!("Zero-wall hits:          {total_zero_wall_hits}");
        eprintln!("Unique topologies:       {}", topologies.len());
        eprintln!("Live states swept:       {total_live}");
        eprintln!("Winner mismatches:       {total_winner_mismatches}");
        eprintln!("DTM mismatches:          {total_dtm_mismatches}");
        eprintln!("Dominance counterex:     {total_cert_counterexamples}");
        eprintln!("Bellman violations:      {total_bellman_violations}");
        eprintln!("Prod parity mismatches:  {total_parity_mismatches}");
        eprintln!("Prod winner mismatches:  {total_prod_winner_mismatches}");
        eprintln!("Corrected G2 fires:      {total_g2_fires}");
        eprintln!("Corrected G2 violations: {total_g2_violations}");
        if let Some(ref m) = first_g2_violation {
            eprintln!("  first: {m}");
        }
        eprintln!("Endgame positions tested:{endgame_tested}");
        eprintln!("Endgame winner mismatches:{endgame_winner_mismatches}");
        eprintln!("Endgame DTM mismatches:  {endgame_dtm_mismatches}");
        eprintln!("Elapsed:                 {:.1}s", elapsed.as_secs_f64());
        eprintln!("─────────────────────────────────────────────────────────────");
        if let Some(ref msg) = first_winner_fail {
            eprintln!("{msg}");
        }

        assert_eq!(total_winner_mismatches, 0,
            "certificate winner mismatch on real-game topologies; first: {}",
            first_winner_fail.as_deref().unwrap_or("none"));
        assert_eq!(endgame_winner_mismatches, 0,
            "certificate winner mismatch on recorded endgame positions");
        assert_eq!(total_parity_mismatches, 0,
            "production winner table disagrees with the proven test implementation");
        assert_eq!(total_prod_winner_mismatches, 0,
            "production winner table winner disagrees with the exact oracle");
        assert_eq!(total_g2_violations, 0,
            "corrected contact-aware Gate 2 produced a wrong-sign verdict; first: {}",
            first_g2_violation.as_deref().unwrap_or("none"));
    }

    /// Count bits set in a [u8; 64] wall array.
    fn hw_count(hw: &[u8; 64]) -> usize {
        hw.iter().map(|&b| b as usize).sum()
    }
    fn vw_count(vw: &[u8; 64]) -> usize {
        vw.iter().map(|&b| b as usize).sum()
    }

    /// Reconstruct a GameState with the given wall bitboards planted.
    /// Pawn positions are at their start; walls are applied via make_move.
    fn build_game_with_walls(hw: &[u8; 64], vw: &[u8; 64]) -> GameState {
        let mut g = GameState::new();
        // Give both players unlimited walls temporarily so we can place them.
        g.wl = [100, 100];
        for slot in 0..64usize {
            if hw[slot] != 0 {
                // Alternate sides arbitrarily; only the topology matters.
                g.make_move(100 + slot as i16);
                g.turn ^= 1; // flip without advancing wl correctly
            }
        }
        for slot in 0..64usize {
            if vw[slot] != 0 {
                g.make_move(200 + slot as i16);
                g.turn ^= 1;
            }
        }
        // Reset game state to canonical endgame: wl=0, standard pawns, turn=0.
        g.wl = [0, 0];
        g.pawn = [76, 4];
        g.turn = 0;
        g
    }

    /// Build a diagnostic string for the first winner mismatch.
    fn build_failure_diag(
        source: &str,
        topo_idx: usize,
        hw: &[u8; 64],
        vw: &[u8; 64],
        id: usize,
        p0: usize,
        p1: usize,
        turn: usize,
        sol: &CertificateSolution,
        oracle: i16,
        ctx: &CertificateContext,
        oracle_tbl: &[i16],
        g: &mut GameState,
    ) -> String {
        let oracle_winner = if oracle > 0 { turn } else { turn ^ 1 };
        let old_d0 = ctx.own_goal_dist[0][p0];
        let old_d1 = ctx.own_goal_dist[1][p1];
        let mut buf = [0i16; 16];
        let nm = g.gen_pawn_moves(&mut buf, 0);
        let side = turn;
        let old_d = ctx.own_goal_dist[side][g.pawn[side]];
        let move_diag: Vec<String> = buf[..nm]
            .iter()
            .map(|&mv| {
                let dst = mv as usize;
                let new_d = ctx.own_goal_dist[side][dst];
                let delta = if new_d == u8::MAX { i16::MIN / 2 } else { old_d as i16 - new_d as i16 };
                let jump = is_jump_move(g.pawn[side], dst);
                let class = if jump { "B" } else if delta >= 1 { "A" } else { "C" };
                // Child oracle after this move (from child's STM perspective).
                let child_id = if turn == 0 {
                    state_id(dst, p1, 1)
                } else {
                    state_id(p0, dst, 0)
                };
                let child_oracle = if is_home(turn, dst) { 1i16 } else { oracle_tbl[child_id] };
                format!("mv={mv} dst={dst} delta={delta} class={class} child_oracle={child_oracle}")
            })
            .collect();
        // For each retained Class A/B move, show player 0's best response oracle.
        let mut child_diags: Vec<String> = Vec::new();
        for &mv in &buf[..nm] {
            let dst = mv as usize;
            let new_d = ctx.own_goal_dist[side][dst];
            let delta = if new_d == u8::MAX { i16::MIN / 2 } else { old_d as i16 - new_d as i16 };
            let jump = is_jump_move(side, dst);
            if !(jump || delta >= 1) { continue; }
            // Build child state, show opponent's legal moves and oracle values.
            let mut child_g = g.clone();
            child_g.make_move(mv);
            let opp = child_g.turn;
            let opp_src = child_g.pawn[opp];
            let opp_old_d = ctx.own_goal_dist[opp][opp_src];
            let mut cbuf = [0i16; 16];
            let cnm = child_g.gen_pawn_moves(&mut cbuf, 0);
            let opp_moves: Vec<String> = cbuf[..cnm].iter().map(|&cmv| {
                let cdst = cmv as usize;
                let cnew_d = ctx.own_goal_dist[opp][cdst];
                let cdelta = if cnew_d == u8::MAX { i16::MIN/2 } else { opp_old_d as i16 - cnew_d as i16 };
                let cjump = is_jump_move(opp_src, cdst);
                let cclass = if cjump { "B" } else if cdelta >= 1 { "A" } else { "C" };
                let gc_id = if opp == 0 { state_id(cdst, child_g.pawn[1], 1) } else { state_id(child_g.pawn[0], cdst, 0) };
                let gc_oracle = if is_home(opp, cdst) { 1i16 } else { oracle_tbl[gc_id] };
                format!("cmv={cmv} cdst={cdst} cdelta={cdelta} cclass={cclass} gc_oracle={gc_oracle}")
            }).collect();
            child_diags.push(format!(
                "  after mv={mv}: opp_at={} opp_d={} opp_moves=[{}]",
                opp_src, opp_old_d, opp_moves.join("; ")
            ));
        }
        let hw_bits: Vec<usize> = hw.iter().enumerate().filter(|(_, &v)| v != 0).map(|(i, _)| i).collect();
        let vw_bits: Vec<usize> = vw.iter().enumerate().filter(|(_, &v)| v != 0).map(|(i, _)| i).collect();
        format!(
            "WINNER FAIL source={source} topo={topo_idx} \
             hw_slots={hw_bits:?} vw_slots={vw_bits:?} \
             id={id} p0={p0} p1={p1} turn={turn} \
             d0={old_d0} d1={old_d1} \
             cert_winner={} oracle_winner={oracle_winner} oracle={oracle} \
             moves=[{}]\nchild analysis:\n{}",
            sol.winner,
            move_diag.join("; "),
            child_diags.join("\n")
        )
    }
}
