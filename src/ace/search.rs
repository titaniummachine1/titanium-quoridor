//! ACE v7 search — 1:1 port of the JS `Search` object.
//!
//! Iterative-deepening αβ with aspiration windows, typed TT, killers/history/
//! countermoves, null move, graduated LMR / EME, frontier LMP, reverse futility,
//! lazy wall legality, repetition detection, wall-stamp dist caching,
//! easy-move early stop, HalfPW net eval. Mirrors the JS node-for-node.

use crate::ace::ace_move_to_board;
use crate::util::clock::{Duration, Instant};

use crate::ace::game::{AceGame, ZOBRIST};
use crate::ace::net::{net, Net, NET_BKT, NET_H, NET_MIRC, NET_MIRS};
use crate::cat::prune::{gap_play_zone_mask, get_shortest_path, wall_should_search};
use crate::cat::CorridorAttention;
use crate::core::board::{Board, Move as BoardMove, Player, Undo, WallOrientation};
use crate::movegen::{generate_legal_moves_slice, MAX_LEGAL_MOVES};
use crate::path::BfsScratch;
pub const MATE: i32 = 100_000;
pub const MAX_PLY: usize = 64;
const INF: i32 = 2 * MATE;

/// Graduated LMR starts after this move index (JS acev10: `i >= 4`).
const ACE_LMR_AFTER_MOVE: usize = 4;
/// Both LMR and EME require at least this remaining depth.
const ACE_LMR_MIN_DEPTH: i32 = 3;

/// Late-move reduction plies — same formula as JS graduated LMR.
fn ace_graduated_lmr_reduction(move_index: usize, depth: i32) -> i32 {
    let mut red = 1;
    if move_index >= 12 {
        red += 1;
    }
    if depth >= 6 && move_index >= 24 {
        red += 1;
    }
    red
}

/// EME extends only the first ordered wall moves after the TT/best move.
/// Index 0 (TT move) already gets full depth; extending more siblings
/// compounds multiplicatively down the tree and explodes the node count.
const ACE_EME_TOP_MOVES: usize = 2;

/// Early Move Extension — +1 ply for the top ordered walls; +2 only for
/// the very first non-TT wall when there is real depth left to spend.
fn ace_graduated_eme_extension(move_index: usize, depth: i32) -> i32 {
    if move_index == 1 && depth >= 8 {
        2
    } else {
        1
    }
}

const TT_BITS: usize = 20;
const TT_SIZE: usize = 1 << TT_BITS;
const TT_MASK: u32 = (TT_SIZE - 1) as u32;

/// Time-abort marker — propagates like the JS `throw "time"`.
pub struct TimeUp;

/// Titanium `Board` kept in sync with the ACE game — fast movegen + optional CAT.
pub struct TiBridge {
    pub board: Board,
    pub bfs: BfsScratch,
    undo_stack: Vec<Undo>,
}

impl TiBridge {
    fn from_game(g: &AceGame) -> Box<Self> {
        let mut board = Board::new();
        for i in 0..g.hist_len {
            let _ = board.make_move(ace_move_to_board(g.hist_m[i]));
        }
        Box::new(Self {
            board,
            bfs: BfsScratch::new(),
            undo_stack: Vec::with_capacity(256),
        })
    }

    fn push(&mut self, m: i16) {
        let undo = self.board.make_move(ace_move_to_board(m));
        self.undo_stack.push(undo);
    }

    fn pop(&mut self) {
        if let Some(undo) = self.undo_stack.pop() {
            self.board.unmake_move(undo);
        }
    }

    /// Full legal moves via Titanium `movegen` → ACE encoding.
    fn gen_legal_ace(&mut self, out: &mut [i16; 160]) -> usize {
        let mut ti_buf = [BoardMove::Pawn { row: 0, col: 0 }; MAX_LEGAL_MOVES];
        let n = generate_legal_moves_slice(&mut self.board, &mut ti_buf, &mut self.bfs);
        for i in 0..n {
            out[i] = board_move_to_ace(ti_buf[i]);
        }
        n
    }
}

/// Titanium board move → ACE numeric encoding.
pub fn board_move_to_ace(mv: BoardMove) -> i16 {
    match mv {
        BoardMove::Pawn { row, col } => ((8 - row as i16) * 9 + col as i16) as i16,
        BoardMove::Wall {
            row,
            col,
            orientation,
        } => {
            let slot = (7 - row as i16) * 8 + col as i16;
            match orientation {
                WallOrientation::Horizontal => 100 + slot,
                WallOrientation::Vertical => 200 + slot,
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct AceDepthLogEntry {
    pub depth: i32,
    pub score: i32,
    pub nodes: u64,
    pub elapsed_ms: u64,
    pub marginal_nodes: u64,
    pub pv: String,
}

pub struct ThinkResult {
    pub mv: i16,
    pub score: i32,
    pub depth: i32,
    pub nodes: u64,
    pub ms: u64,
    pub white_dist: u8,
    pub black_dist: u8,
    pub depth_log: Vec<AceDepthLogEntry>,
}

fn emit_ace_progress(
    engine_label: &str,
    depth_log: &[AceDepthLogEntry],
    search_depth: i32,
    nodes: u64,
    root_score: i32,
    white_dist: u8,
    black_dist: u8,
    elapsed_ms: u64,
) {
    let mut depth_json = String::new();
    for (i, e) in depth_log.iter().enumerate() {
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
        engine_label,
        engine_label,
        search_depth,
        nodes,
        root_score,
        white_dist,
        black_dist,
        elapsed_ms,
        depth_json
    );
    let _ = std::io::Write::flush(&mut std::io::stderr());
}

pub struct AceSearch {
    pub g: AceGame,
    tt_key_hi: Vec<u32>,
    tt_key_lo: Vec<u32>,
    tt_meta: Vec<i32>, // move | flag<<10 | depth<<12, 0 = empty
    tt_score: Vec<i32>,
    history_tbl: [i32; 512],
    cm: [i16; 512], // countermove table
    killers: [[i16; 2]; MAX_PLY],
    path_lo: [u32; MAX_PLY],
    path_hi: [u32; MAX_PLY],
    d0: [[u8; 81]; MAX_PLY],
    d1: [[u8; 81]; MAX_PLY],
    dist0_idx: usize, // active ply slot in d0 (JS: this.dist0 array ref)
    dist1_idx: usize,
    cached_stamp: i32,
    // HalfPW accumulator cache
    np_acc0: [f64; NET_H],
    np_acc1: [f64; NET_H],
    np_hw: [u8; 64],
    np_vw: [u8; 64],
    np_b0: i32,
    np_b1v: i32,
    net: &'static Net,
    /// Mirrored Titanium board (movegen and/or CAT).
    bridge: Option<Box<TiBridge>>,
    /// Use Titanium `generate_legal_moves_slice` instead of ACE `wall_legal`.
    ti_movegen: bool,
    /// CAT-filter walls at inner nodes (requires `bridge`).
    cat_walls: bool,
    /// Early Move Extensions on the first ordered wall moves (mirror of graduated LMR).
    eme: bool,
    pub nodes: u64,
    deadline: Instant,
    root_best: i16,
    root_score: i32,
    /// Live `info json` during `think(..., log=true)` — cleared when search ends.
    stream_log: bool,
    stream_label: String,
    stream_t0: Instant,
    stream_root_score: i32,
    stream_search_depth: i32,
    stream_depth_log: Vec<AceDepthLogEntry>,
    stream_last_emit_nodes: u64,
    stream_last_emit_ms: u64,
    stream_last_best: i16,
}

/// Periodic progress cadence: every 64K nodes AND ≥ 100ms apart — stdout/stderr
/// writes are expensive; spamming them steals think time from the search.
const STREAM_EMIT_NODE_MASK: u64 = 65535;
const STREAM_EMIT_MIN_INTERVAL_MS: u64 = 100;

impl AceSearch {
    pub fn new(g: AceGame) -> Box<Self> {
        Box::new(Self {
            g,
            tt_key_hi: vec![0; TT_SIZE],
            tt_key_lo: vec![0; TT_SIZE],
            tt_meta: vec![0; TT_SIZE],
            tt_score: vec![0; TT_SIZE],
            history_tbl: [0; 512],
            cm: [0; 512],
            killers: [[0; 2]; MAX_PLY],
            path_lo: [0; MAX_PLY],
            path_hi: [0; MAX_PLY],
            d0: [[0; 81]; MAX_PLY],
            d1: [[0; 81]; MAX_PLY],
            dist0_idx: 0,
            dist1_idx: 0,
            cached_stamp: -1,
            np_acc0: [0.0; NET_H],
            np_acc1: [0.0; NET_H],
            np_hw: [0; 64],
            np_vw: [0; 64],
            np_b0: -1,
            np_b1v: -1,
            net: net(),
            bridge: None,
            ti_movegen: false,
            cat_walls: false,
            eme: false,
            nodes: 0,
            deadline: Instant::now(),
            root_best: 0,
            root_score: 0,
            stream_log: false,
            stream_label: String::new(),
            stream_t0: Instant::now(),
            stream_root_score: 0,
            stream_search_depth: 0,
            stream_depth_log: Vec::new(),
            stream_last_emit_nodes: 0,
            stream_last_emit_ms: 0,
            stream_last_best: 0,
        })
    }

    /// Enable Early Move Extensions — same gates/tuning as graduated LMR, early indices.
    pub fn enable_eme(&mut self) {
        self.eme = true;
    }

    /// Titanium movegen on a mirrored board — same legal set, much faster than `wall_legal`.
    pub fn with_ti_movegen(g: AceGame) -> Box<Self> {
        let mut search = Self::new(g);
        search.bridge = Some(TiBridge::from_game(&search.g));
        search.ti_movegen = true;
        search
    }

    /// CAT hybrid: walls at inner nodes must pass `wall_should_search`.
    pub fn with_cat(g: AceGame) -> Box<Self> {
        let mut search = Self::new(g);
        search.bridge = Some(TiBridge::from_game(&search.g));
        search.cat_walls = true;
        search
    }

    /// Fast Titanium movegen + CAT wall filter.
    pub fn with_ti_movegen_and_cat(g: AceGame) -> Box<Self> {
        let mut search = Self::with_ti_movegen(g);
        search.cat_walls = true;
        search
    }

    /// Advance the live game one ply, keeping TT/killers/history warm.
    /// Long-lived session path — the next `think` reuses prior analysis.
    pub fn apply_move(&mut self, m: i16) {
        self.g.make_move(m);
        self.position_changed();
    }

    /// Replace the position outright (undo, new game) without clearing the
    /// TT — entries are hash-keyed, stale ones simply never match.
    pub fn set_position(&mut self, g: AceGame) {
        self.g = g;
        self.position_changed();
    }

    fn position_changed(&mut self) {
        if self.bridge.is_some() {
            self.bridge = Some(TiBridge::from_game(&self.g));
        }
        self.cached_stamp = -1;
        self.np_b0 = -1; // force full accumulator rebuild (v10: no stamp gate)
        self.np_b1v = -1;
    }

    fn sync_stream_meta(
        &mut self,
        depth_log: &[AceDepthLogEntry],
        search_depth: i32,
        root_score: i32,
    ) {
        self.stream_depth_log.clear();
        self.stream_depth_log.extend_from_slice(depth_log);
        self.stream_search_depth = search_depth;
        self.stream_root_score = root_score;
    }

    /// Periodic + forced progress for website SSE (matches JS cumulative `search.nodes`).
    /// Periodic emits are throttled by node count AND wall time; forced emits
    /// (depth complete, root best-move change, deadline) always go out.
    fn emit_stream_progress(&mut self, force: bool) {
        if !self.stream_log {
            return;
        }
        let elapsed_ms = self.stream_t0.elapsed().as_millis() as u64;
        if !force {
            if self.nodes == 0 || self.nodes == self.stream_last_emit_nodes {
                return;
            }
            if (self.nodes & STREAM_EMIT_NODE_MASK) != 0 {
                return;
            }
            if elapsed_ms.saturating_sub(self.stream_last_emit_ms) < STREAM_EMIT_MIN_INTERVAL_MS {
                return;
            }
        }
        self.stream_last_emit_ms = elapsed_ms;
        self.stream_last_emit_nodes = self.nodes;
        self.refresh_dist(0);
        let white_dist = self.d0[self.dist0_idx][self.g.pawn[0]];
        let black_dist = self.d1[self.dist1_idx][self.g.pawn[1]];
        let elapsed_ms = self.stream_t0.elapsed().as_millis() as u64;
        emit_ace_progress(
            &self.stream_label,
            &self.stream_depth_log,
            self.stream_search_depth,
            self.nodes,
            self.stream_root_score,
            white_dist,
            black_dist,
            elapsed_ms,
        );
    }

    #[inline(always)]
    fn check_time(&mut self) -> Result<(), TimeUp> {
        if (self.nodes & 1023) == 0 {
            if Instant::now() > self.deadline {
                self.emit_stream_progress(true);
                return Err(TimeUp);
            }
            self.emit_stream_progress(false);
        }
        Ok(())
    }

    fn ace_time_fraction(last_score: i32) -> f64 {
        if last_score < -80 {
            0.92
        } else {
            0.85
        }
    }

    fn ace_over_time_budget(t0: Instant, time_ms: u64, last_score: i32) -> bool {
        let budget = time_ms as f64 * Self::ace_time_fraction(last_score);
        t0.elapsed().as_millis() as f64 > budget
    }

    fn refresh_dist(&mut self, ply: usize) {
        let stamp = self.g.wall_stamp;
        if self.cached_stamp == stamp {
            return; // refs already valid for these walls
        }
        if self.cached_stamp == stamp - 1 && self.g.hist_len > 0 {
            // exactly one wall added since the cached config: slots hold its dists.
            // recompute a player's field only if the wall cuts a shortest-path edge
            // (|dist diff| === 1); equal-dist edges lie on no shortest path.
            let m = self.g.hist_m[self.g.hist_len - 1];
            if m >= 100 {
                let slot = (m % 100) as usize;
                let a = (slot >> 3) * 9 + (slot & 7);
                let (b2, c2, e2) = if m < 200 {
                    (a + 9, a + 1, a + 10) // hw: two vertical edges
                } else {
                    (a + 1, a + 9, a + 10) // vw: two horizontal edges
                };
                let d0 = &self.d0[self.dist0_idx];
                if d0[a] != d0[b2] || d0[c2] != d0[e2] {
                    self.dist0_idx = ply; // redirect first: never write an ancestor's array
                    self.g.compute_dist(0, &mut self.d0[ply]);
                }
                let d1 = &self.d1[self.dist1_idx];
                if d1[a] != d1[b2] || d1[c2] != d1[e2] {
                    self.dist1_idx = ply;
                    self.g.compute_dist(1, &mut self.d1[ply]);
                }
                self.cached_stamp = stamp;
                return;
            }
        }
        self.dist0_idx = ply; // own arrays: ancestors stay intact
        self.dist1_idx = ply;
        self.g.compute_dist(0, &mut self.d0[ply]);
        self.g.compute_dist(1, &mut self.d1[ply]);
        self.cached_stamp = stamp;
    }

    fn evaluate(&mut self) -> i32 {
        let me = self.g.turn;
        let opp = 1 - me;
        let d_me_u = if me == 0 {
            self.d0[self.dist0_idx][self.g.pawn[0]]
        } else {
            self.d1[self.dist1_idx][self.g.pawn[1]]
        };
        let d_opp_u = if opp == 0 {
            self.d0[self.dist0_idx][self.g.pawn[0]]
        } else {
            self.d1[self.dist1_idx][self.g.pawn[1]]
        };
        let w_me_i = self.g.wl[me];
        let w_opp_i = self.g.wl[opp];
        let d_me_i = d_me_u as i32;
        let d_opp_i = d_opp_u as i32;
        if w_me_i == 0 && w_opp_i == 0 {
            // pure race
            if d_me_i <= d_opp_i {
                return 3000 + (d_opp_i - d_me_i) * 50 - d_me_i;
            }
            return -3000 - (d_me_i - d_opp_i) * 50 + d_opp_i;
        }

        let d_me = d_me_i as f64;
        let d_opp = d_opp_i as f64;
        let w_me = w_me_i as f64;
        let w_opp = w_opp_i as f64;
        let nw = self.net;
        let ws = &nw.ws;

        let pd = d_opp - d_me;
        let wd = w_me - w_opp;
        let mut out = ws[0]
            + ws[1] * pd
            + ws[2] * wd
            + ws[3] * d_me
            + ws[4] * d_opp
            + ws[9] * pd * (w_me + w_opp) / 20.0
            + ws[10] * wd * (d_me + d_opp) / 16.0;
        if w_opp_i == 0 {
            out += ws[6];
            if d_me <= d_opp {
                out += ws[5];
            }
        } else if w_me_i == 0 {
            out += ws[8];
            if d_opp <= d_me - 1.0 {
                out += ws[7];
            }
        }
        if d_opp <= 4.0 {
            out += ws[11] * if w_me < 3.0 { w_me } else { 3.0 };
        }
        if d_me <= 4.0 {
            out += ws[12] * if w_opp < 3.0 { w_opp } else { 3.0 };
        }

        let b0 = NET_BKT[self.g.pawn[0]] as i32;
        let b1 = NET_BKT[NET_MIRC[self.g.pawn[1]]] as i32;
        if b0 != self.np_b0 || b1 != self.np_b1v {
            // bucket cross: rebuild BOTH perspectives (ACE v10 audit blocker 5:
            // rebuilding only the crossed side dropped pending wall diffs for
            // the other accumulator)
            self.np_acc0.fill(0.0);
            self.np_acc1.fill(0.0);
            for s in 0..64 {
                if self.g.hw[s] != 0 {
                    let o = (b0 as usize * 128 + s) * NET_H;
                    for j in 0..NET_H {
                        self.np_acc0[j] += nw.w1c[o + j];
                    }
                    let o = (b1 as usize * 128 + NET_MIRS[s]) * NET_H;
                    for j in 0..NET_H {
                        self.np_acc1[j] += nw.w1c[o + j];
                    }
                }
                if self.g.vw[s] != 0 {
                    let o = (b0 as usize * 128 + 64 + s) * NET_H;
                    for j in 0..NET_H {
                        self.np_acc0[j] += nw.w1c[o + j];
                    }
                    let o = (b1 as usize * 128 + 64 + NET_MIRS[s]) * NET_H;
                    for j in 0..NET_H {
                        self.np_acc1[j] += nw.w1c[o + j];
                    }
                }
                self.np_hw[s] = self.g.hw[s];
                self.np_vw[s] = self.g.vw[s];
            }
            self.np_b0 = b0;
            self.np_b1v = b1;
        } else {
            // NO stamp gate (ACE v10 audit blocker 4: wall_stamp is a count,
            // aliases across sibling wall configs): always diff the wall snapshot
            for s in 0..64 {
                if self.g.hw[s] != self.np_hw[s] {
                    let sg = if self.g.hw[s] != 0 { 1.0 } else { -1.0 };
                    let o0 = (b0 as usize * 128 + s) * NET_H;
                    let o1 = (b1 as usize * 128 + NET_MIRS[s]) * NET_H;
                    for j in 0..NET_H {
                        self.np_acc0[j] += sg * nw.w1c[o0 + j];
                        self.np_acc1[j] += sg * nw.w1c[o1 + j];
                    }
                    self.np_hw[s] = self.g.hw[s];
                }
                if self.g.vw[s] != self.np_vw[s] {
                    let sg = if self.g.vw[s] != 0 { 1.0 } else { -1.0 };
                    let o0 = (b0 as usize * 128 + 64 + s) * NET_H;
                    let o1 = (b1 as usize * 128 + 64 + NET_MIRS[s]) * NET_H;
                    for j in 0..NET_H {
                        self.np_acc0[j] += sg * nw.w1c[o0 + j];
                        self.np_acc1[j] += sg * nw.w1c[o1 + j];
                    }
                    self.np_vw[s] = self.g.vw[s];
                }
            }
        }

        let mut hid = [0.0f64; NET_H];
        if me == 0 {
            for j in 0..NET_H {
                hid[j] = nw.b1[j] + self.np_acc0[j];
            }
            let o0 = self.g.pawn[0] * NET_H;
            for j in 0..NET_H {
                hid[j] += nw.po[o0 + j];
            }
            let o1 = self.g.pawn[1] * NET_H;
            for j in 0..NET_H {
                hid[j] += nw.px[o1 + j];
            }
        } else {
            for j in 0..NET_H {
                hid[j] = nw.b1[j] + self.np_acc1[j];
            }
            let o0 = NET_MIRC[self.g.pawn[1]] * NET_H;
            for j in 0..NET_H {
                hid[j] += nw.po[o0 + j];
            }
            let o1 = NET_MIRC[self.g.pawn[0]] * NET_H;
            for j in 0..NET_H {
                hid[j] += nw.px[o1 + j];
            }
        }
        for j in 0..NET_H {
            let a2 = hid[j].clamp(0.0, 1.0);
            out += nw.w2[j] * a2 * 200.0;
        }
        out as i32
    }

    fn gen_moves(&mut self, ply: usize, depth: i32, tt_move: i16, out: &mut [i16; 160]) -> usize {
        let check_legal = ply == 0;
        // MoveGen+ : Titanium legal movegen at EVERY node (perft-parity search).
        // Fully legal walls — no lazy seal checks needed downstream, and inner
        // nodes can never search (or suggest via TT) a Titanium-illegal move.
        // The CAT hybrid keeps its own filtered path at inner nodes.
        if self.ti_movegen && (check_legal || !self.cat_walls) {
            return self
                .bridge
                .as_mut()
                .expect("ti movegen needs bridge")
                .gen_legal_ace(out);
        }
        let mut n = self.g.gen_pawn_moves(out, 0);
        if self.g.wl[self.g.turn] <= 0 {
            return n;
        }
        if self.cat_walls && !check_legal {
            return self.gen_walls_cat_filtered(depth, tt_move, out, n);
        }
        for slot in 0..64 {
            if check_legal {
                if self.g.wall_legal(0, slot) {
                    out[n] = 100 + slot as i16;
                    n += 1;
                }
                if self.g.wall_legal(1, slot) {
                    out[n] = 200 + slot as i16;
                    n += 1;
                }
            } else {
                // lazy: geometry only; path-seal checked when the move is searched
                if self.g.wall_fits(0, slot) {
                    out[n] = 100 + slot as i16;
                    n += 1;
                }
                if self.g.wall_fits(1, slot) {
                    out[n] = 200 + slot as i16;
                    n += 1;
                }
            }
        }
        n
    }

    /// Hybrid wall generation: lazy geometry + CAT relevance filter.
    ///
    /// CAT (multi-route corridor heat) only above the leaf layer — depth-1 nodes
    /// dominate the tree and only need witness-path tactics, not breadth
    /// (mirrors `search::alphabeta`). The TT move always survives the filter.
    fn gen_walls_cat_filtered(
        &mut self,
        depth: i32,
        tt_move: i16,
        out: &mut [i16; 160],
        mut n: usize,
    ) -> usize {
        let me = self.g.turn;
        let our_dist = if me == 0 {
            self.d0[self.dist0_idx][self.g.pawn[0]]
        } else {
            self.d1[self.dist1_idx][self.g.pawn[1]]
        };
        let opp_dist = if me == 0 {
            self.d1[self.dist1_idx][self.g.pawn[1]]
        } else {
            self.d0[self.dist0_idx][self.g.pawn[0]]
        };
        let opp_player = if me == 0 { Player::Two } else { Player::One };

        let bridge = self.bridge.as_mut().expect("cat bridge");
        let cat = if depth >= 2 {
            bridge.bfs.build_corridor_attention(&bridge.board)
        } else {
            CorridorAttention::default()
        };
        let mut opp_path = [0u8; 81];
        let opp_path_len =
            get_shortest_path(&bridge.board, opp_player, &mut bridge.bfs, &mut opp_path);
        let reachable = bridge.bfs.both_reachable_mask(&bridge.board);
        let gap_zone = gap_play_zone_mask(reachable);

        for slot in 0..64 {
            for (wall_type, base) in [(0usize, 100i16), (1usize, 200i16)] {
                if !self.g.wall_fits(wall_type, slot) {
                    continue;
                }
                let m = base + slot as i16;
                let keep = m == tt_move
                    || wall_should_search(
                        ace_move_to_board(m),
                        &cat,
                        reachable,
                        gap_zone,
                        &mut bridge.board,
                        our_dist,
                        opp_dist,
                        &opp_path,
                        opp_path_len,
                        &mut bridge.bfs,
                    );
                if keep {
                    out[n] = m;
                    n += 1;
                }
            }
        }
        n
    }

    fn order_moves(&self, ply: usize, moves: &mut [i16], tt_move: i16, cm_move: i16) {
        let dist_me = if self.g.turn == 0 {
            &self.d0[self.dist0_idx]
        } else {
            &self.d1[self.dist1_idx]
        };
        let k = &self.killers[ply];
        let n = moves.len();
        let mut sc = [0i32; 160];
        for i in 0..n {
            let m = moves[i];
            sc[i] = if m == tt_move {
                2_000_000_000
            } else if m < 100 {
                1_000_000 - dist_me[m as usize] as i32 * 1000
            } else if m == k[0] {
                900_000
            } else if m == cm_move {
                870_000
            } else if m == k[1] {
                850_000
            } else {
                self.history_tbl[m as usize]
            };
        }
        // stable insertion sort, descending — must match JS tie order exactly
        for a in 1..n {
            let mv = moves[a];
            let ms = sc[a];
            let mut b = a as isize - 1;
            while b >= 0 && sc[b as usize] < ms {
                moves[(b + 1) as usize] = moves[b as usize];
                sc[(b + 1) as usize] = sc[b as usize];
                b -= 1;
            }
            moves[(b + 1) as usize] = mv;
            sc[(b + 1) as usize] = ms;
        }
    }

    /// True when the current board hash already appeared in real game history
    /// (since the last wall — same rule as the in-search repetition cutoff).
    fn repeats_game_history(&self) -> bool {
        let lwp = self.g.last_wall_ply as isize;
        let mut gi = self.g.hist_len as isize * 2 - 4;
        while gi >= lwp * 2 {
            if self.g.hashes_u[gi as usize] == self.g.hash_lo
                && self.g.hashes_u[gi as usize + 1] == self.g.hash_hi
            {
                return true;
            }
            gi -= 2;
        }
        false
    }

    fn move_repeats_game_history(&mut self, m: i16) -> bool {
        self.g.make_move(m);
        let rep = self.repeats_game_history();
        self.g.unmake_move();
        rep
    }

    fn ab(
        &mut self,
        depth: i32,
        mut alpha: i32,
        beta: i32,
        ply: usize,
        allow_null: bool,
        prev_move: i16,
    ) -> Result<i32, TimeUp> {
        self.nodes += 1;
        self.check_time()?;
        let prev = 1 - self.g.turn;
        if (prev == 0 && self.g.pawn[0] < 9) || (prev == 1 && self.g.pawn[1] >= 72) {
            return Ok(-(MATE - ply as i32));
        }
        if ply >= MAX_PLY - 1 {
            return Ok(0);
        }
        self.path_lo[ply] = self.g.hash_lo;
        self.path_hi[ply] = self.g.hash_hi;
        if ply > 0 {
            // repetition: search line, then game history back to last wall
            for ri in (0..ply).rev() {
                if self.path_lo[ri] == self.g.hash_lo && self.path_hi[ri] == self.g.hash_hi {
                    return Ok(0);
                }
            }
            let lwp = self.g.last_wall_ply as isize;
            let mut gi = self.g.hist_len as isize * 2 - 4;
            while gi >= lwp * 2 {
                if self.g.hashes_u[gi as usize] == self.g.hash_lo
                    && self.g.hashes_u[gi as usize + 1] == self.g.hash_hi
                {
                    return Ok(0);
                }
                gi -= 2;
            }
        }

        self.refresh_dist(ply);
        let nd0 = self.dist0_idx; // restored on every unmake
        let nd1 = self.dist1_idx;
        let nst = self.cached_stamp;
        if depth <= 0 {
            return Ok(self.evaluate());
        }

        // TT probe (typed, always-replace)
        let idx = (self.g.hash_lo & TT_MASK) as usize;
        let mut tt_move: i16 = 0;
        let meta = self.tt_meta[idx];
        if meta != 0
            && self.tt_key_hi[idx] == self.g.hash_hi
            && self.tt_key_lo[idx] == self.g.hash_lo
        {
            tt_move = (meta & 1023) as i16;
            let tdepth = meta >> 12;
            let tflag = (meta >> 10) & 3;
            if tdepth >= depth && ply > 0 {
                let mut es = self.tt_score[idx]; // mate scores stored node-relative
                if es > MATE - 2 * MAX_PLY as i32 {
                    es -= ply as i32;
                } else if es < -(MATE - 2 * MAX_PLY as i32) {
                    es += ply as i32;
                }
                if tflag == 0 {
                    return Ok(es);
                }
                if tflag == 1 && es >= beta {
                    return Ok(es);
                }
                if tflag == 2 && es <= alpha {
                    return Ok(es);
                }
            }
        }

        // reverse futility: hopeless to fall below beta at shallow depth
        if depth <= 4 && beta > -2000 && beta < 2000 {
            let sev = self.evaluate();
            if sev - 90 * depth >= beta {
                return Ok(sev);
            }
        }

        // null move
        if allow_null && depth >= 3 && ply > 0 {
            let ev = self.evaluate();
            if ev >= beta {
                let z = &ZOBRIST;
                self.g.turn ^= 1;
                self.g.hash_lo ^= z.turn_lo;
                self.g.hash_hi ^= z.turn_hi;
                if let Some(bridge) = self.bridge.as_mut() {
                    // keep the mirrored board's side in sync (wall accounting)
                    bridge.board.side_to_move = bridge.board.side_to_move.opposite();
                }
                let res = self.ab(depth - 3, -beta, -beta + 1, ply + 1, false, 0);
                let z = &ZOBRIST;
                self.g.turn ^= 1;
                self.g.hash_lo ^= z.turn_lo;
                self.g.hash_hi ^= z.turn_hi;
                if let Some(bridge) = self.bridge.as_mut() {
                    bridge.board.side_to_move = bridge.board.side_to_move.opposite();
                }
                self.dist0_idx = nd0;
                self.dist1_idx = nd1;
                self.cached_stamp = nst;
                let ns = -res?;
                if ns >= beta && ns < MATE - 200 {
                    return Ok(beta);
                }
            }
        }

        let mut moves = [0i16; 160];
        let n = self.gen_moves(ply, depth, tt_move, &mut moves);
        if n == 0 {
            return Ok(self.evaluate());
        }
        let cm_move = if prev_move > 0 {
            self.cm[prev_move as usize]
        } else {
            0
        };
        self.order_moves(ply, &mut moves[..n], tt_move, cm_move);

        let mut best = i32::MIN; // JS -Infinity
        let mut best_move: i16 = 0;
        let mut flag = 2;

        for i in 0..n {
            let m = moves[i];
            // frontier LMP
            if depth <= 2
                && ply > 0
                && i >= 10
                && m >= 100
                && m != tt_move
                && self.history_tbl[m as usize] <= 0
                && best > -MATE + 200
            {
                continue;
            }
            // Seal check only needed for ACE's lazy pseudo-legal walls; with
            // MoveGen+ (Titanium legal gen at every node) all walls are legal.
            let lazy_walls = !(self.ti_movegen && !self.cat_walls);
            if m >= 100 && ply > 0 && lazy_walls {
                let wt = if m < 200 { 0 } else { 1 };
                let slot = (m % 100) as usize;
                if self.g.wall_needs_path_check(wt, slot) {
                    self.g.set_wall_bits(wt, slot, true);
                    let paths_ok = self.g.has_path(0) && self.g.has_path(1);
                    self.g.set_wall_bits(wt, slot, false);
                    if !paths_ok {
                        continue; // sealing wall: pseudo-legal only
                    }
                }
            }
            self.g.make_move(m);
            if let Some(bridge) = self.bridge.as_mut() {
                bridge.push(m);
            }
            let new_depth = depth - 1;
            let result = if self.eme
                && i > 0
                && i <= ACE_EME_TOP_MOVES
                && depth >= ACE_LMR_MIN_DEPTH
                && m >= 100
                && m != tt_move
            {
                // EME — extend only the top ordered walls (see ACE_EME_TOP_MOVES)
                let ext = ace_graduated_eme_extension(i, depth);
                let ed = new_depth + ext;
                self.ab(ed, -beta, -alpha, ply + 1, true, m).map(|s| -s)
            } else if i >= ACE_LMR_AFTER_MOVE
                && depth >= ACE_LMR_MIN_DEPTH
                && m >= 100
                && m != tt_move
            {
                // graduated LMR
                let red = ace_graduated_lmr_reduction(i, depth);
                let rd = (new_depth - red).max(0);
                match self.ab(rd, -alpha - 1, -alpha, ply + 1, true, m) {
                    Ok(s) => {
                        let mut score = -s;
                        if score > alpha {
                            match self.ab(new_depth, -beta, -alpha, ply + 1, true, m) {
                                Ok(s2) => score = -s2,
                                Err(e) => {
                                    self.unwind_move(nd0, nd1, nst);
                                    return Err(e);
                                }
                            }
                        }
                        Ok(score)
                    }
                    Err(e) => Err(e),
                }
            } else if i > 0 {
                match self.ab(new_depth, -alpha - 1, -alpha, ply + 1, true, m) {
                    Ok(s) => {
                        let mut score = -s;
                        if score > alpha && score < beta {
                            match self.ab(new_depth, -beta, -alpha, ply + 1, true, m) {
                                Ok(s2) => score = -s2,
                                Err(e) => {
                                    self.unwind_move(nd0, nd1, nst);
                                    return Err(e);
                                }
                            }
                        }
                        Ok(score)
                    }
                    Err(e) => Err(e),
                }
            } else {
                self.ab(new_depth, -beta, -alpha, ply + 1, true, m)
                    .map(|s| -s)
            };
            self.g.unmake_move();
            if let Some(bridge) = self.bridge.as_mut() {
                bridge.pop();
            }
            self.dist0_idx = nd0;
            self.dist1_idx = nd1;
            self.cached_stamp = nst;
            let score = result?;

            let prefer_non_repeat = ply == 0
                && score == best
                && best_move != 0
                && self.move_repeats_game_history(best_move)
                && !self.move_repeats_game_history(m);

            if score > best || prefer_non_repeat {
                best = score;
                best_move = m;
                if score > alpha || prefer_non_repeat {
                    alpha = score;
                    flag = 0;
                    if ply == 0 {
                        self.root_best = m;
                        self.root_score = score;
                        // New best move at root → push an info-card update now
                        // (forced; bypasses the periodic throttle).
                        if self.stream_last_best != m {
                            self.stream_last_best = m;
                            self.stream_root_score = score;
                            self.emit_stream_progress(true);
                        }
                    }
                    if alpha >= beta {
                        flag = 1;
                        if m >= 100 {
                            if self.killers[ply][0] != m {
                                self.killers[ply][1] = self.killers[ply][0];
                                self.killers[ply][0] = m;
                            }
                            self.history_tbl[m as usize] += depth * depth;
                            if self.history_tbl[m as usize] > 100_000_000 {
                                for h in self.history_tbl.iter_mut() {
                                    *h >>= 1;
                                }
                            }
                        }
                        if prev_move > 0 {
                            self.cm[prev_move as usize] = m;
                        }
                        break;
                    }
                }
            }
        }

        if best == i32::MIN {
            return Ok(self.evaluate()); // all pseudo-legal moves were sealing walls
        }
        let mut ts = best; // store mate scores node-relative
        if ts > MATE - 2 * MAX_PLY as i32 {
            ts += ply as i32;
        } else if ts < -(MATE - 2 * MAX_PLY as i32) {
            ts -= ply as i32;
        }
        self.tt_key_hi[idx] = self.g.hash_hi;
        self.tt_key_lo[idx] = self.g.hash_lo;
        self.tt_meta[idx] = best_move as i32 | (flag << 10) | (depth << 12);
        self.tt_score[idx] = ts;
        Ok(best)
    }

    /// Restore after a time abort mid-move (JS `finally` semantics).
    fn unwind_move(&mut self, nd0: usize, nd1: usize, nst: i32) {
        self.g.unmake_move();
        if let Some(bridge) = self.bridge.as_mut() {
            bridge.pop();
        }
        self.dist0_idx = nd0;
        self.dist1_idx = nd1;
        self.cached_stamp = nst;
    }

    /// Entry: iterative deepening within `time_ms`. `full` disables the easy-move stop.
    pub fn think(
        &mut self,
        time_ms: u64,
        max_depth: i32,
        full: bool,
        log: bool,
        engine_label: &str,
    ) -> ThinkResult {
        let t0 = Instant::now();
        self.deadline = t0 + Duration::from_millis(time_ms);
        self.nodes = 0;
        self.root_best = 0;
        self.root_score = 0;
        self.stream_log = log;
        self.stream_label = engine_label.to_string();
        self.stream_t0 = t0;
        self.stream_root_score = 0;
        self.stream_search_depth = 0;
        self.stream_depth_log.clear();
        self.stream_last_emit_nodes = 0;
        self.stream_last_emit_ms = 0;
        self.stream_last_best = 0;
        // Re-sync the mirrored Titanium board from the authoritative ACE game.
        // Kills any drift left over from a previous search (e.g. an unbalanced
        // push/pop on time-abort) before it can poison this move's root list.
        if self.bridge.is_some() {
            self.bridge = Some(TiBridge::from_game(&self.g));
        }
        let mut last_best: i16 = 0;
        let mut last_score = 0;
        let mut last_depth = 0;
        let mut stable = 0;
        let mut depth_log: Vec<AceDepthLogEntry> = Vec::new();
        let max_depth = if max_depth > 0 { max_depth } else { 30 };

        for d in 1..=max_depth {
            if d > 1 && Self::ace_over_time_budget(t0, time_ms, last_score) {
                break;
            }
            if Instant::now() >= self.deadline {
                break;
            }
            self.stream_root_score = last_score;
            self.stream_search_depth = d;
            let nodes_at_depth = self.nodes;
            let result = if d >= 4 && last_score > -2000 && last_score < 2000 {
                // aspiration
                let mut lo = last_score - 75;
                let mut hi = last_score + 75;
                loop {
                    match self.ab(d, lo, hi, 0, true, 0) {
                        Ok(sc) => {
                            if sc <= lo {
                                lo = -INF;
                            } else if sc >= hi {
                                hi = INF;
                            } else {
                                break Ok(sc);
                            }
                        }
                        Err(e) => break Err(e),
                    }
                }
            } else {
                self.ab(d, -INF, INF, 0, true, 0)
            };
            match result {
                Ok(sc) => {
                    stable = if self.root_best == last_best {
                        stable + 1
                    } else {
                        0
                    };
                    last_best = self.root_best;
                    last_score = sc;
                    last_depth = d;
                    let elapsed_ms = t0.elapsed().as_millis() as u64;
                    let pv = if last_best != 0 {
                        super::ace_to_algebraic(last_best)
                    } else {
                        String::new()
                    };
                    depth_log.push(AceDepthLogEntry {
                        depth: d,
                        score: last_score,
                        nodes: self.nodes,
                        elapsed_ms,
                        marginal_nodes: self.nodes.saturating_sub(nodes_at_depth),
                        pv,
                    });
                    if log {
                        self.sync_stream_meta(&depth_log, d, last_score);
                        self.emit_stream_progress(true);
                    }
                    if sc > MATE - 200 || sc < -(MATE - 200) {
                        break; // forced result
                    }
                    // v8 easy-move stop (acev8_engine.js)
                    if !full
                        && d >= 9
                        && stable >= 3
                        && last_score > -120
                        && t0.elapsed().as_millis() as u64 > time_ms * 3 / 10
                    {
                        break;
                    }
                }
                Err(TimeUp) => break, // state already restored by unwinding unmakes
            }
            if Self::ace_over_time_budget(t0, time_ms, last_score) {
                break;
            }
        }

        // Bridge desync detector: whenever control is back at the root the
        // mirrored board's undo stack MUST be empty. If not, a make/unmake
        // path leaked a frame (this is how "illegal move" crashes happen) —
        // log it loudly and rebuild from the authoritative game.
        if let Some(bridge) = self.bridge.as_ref() {
            if !bridge.undo_stack.is_empty() {
                eprintln!(
                    "info string ace bridge DESYNC: {} unpopped frames after search — rebuilding",
                    bridge.undo_stack.len()
                );
                self.bridge = Some(TiBridge::from_game(&self.g));
            }
        }

        // Root legality guard: never emit a move the true position rejects.
        // Regenerates the legal root list from clean state; if the searched
        // best move is not in it, substitute the best legal alternative.
        self.refresh_dist(0);
        let mut legal = [0i16; 160];
        let nlegal = self.gen_moves(0, 1, last_best, &mut legal);
        if last_best == 0 || !legal[..nlegal].contains(&last_best) {
            if last_best != 0 {
                eprintln!(
                    "info string ace root guard: searched best {} is illegal in true position — substituting",
                    super::ace_to_algebraic(last_best)
                );
            }
            if nlegal > 0 {
                self.order_moves(0, &mut legal[..nlegal], 0, 0);
                last_best = legal[0];
            } else {
                last_best = 0;
            }
        }

        self.refresh_dist(0);
        let white_dist = self.d0[self.dist0_idx][self.g.pawn[0]];
        let black_dist = self.d1[self.dist1_idx][self.g.pawn[1]];
        let ms = t0.elapsed().as_millis() as u64;

        if log {
            self.sync_stream_meta(&depth_log, last_depth, last_score);
            self.emit_stream_progress(true);
        }

        ThinkResult {
            mv: last_best,
            score: last_score,
            depth: last_depth,
            nodes: self.nodes,
            ms,
            white_dist,
            black_dist,
            depth_log,
        }
    }
}
