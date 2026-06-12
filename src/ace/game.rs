//! ACE v7 game state — 1:1 port of the JS `Quoridor` object.
//!
//! Coordinates are ACE-native: cell = r*9+c with r=0 the TOP row.
//! Player 0 starts at 76 (bottom) and races to row 0; player 1 starts at 4
//! and races to row 8. Moves: 0..80 pawn target, 100+slot hw, 200+slot vw.

use crate::core::board::WallOrientation;

pub const DELTA: [i16; 4] = [-9, 9, -1, 1];
pub const DIRBIT: [u8; 4] = [1, 2, 4, 8];

// ── Zobrist (exact JS xorshift sequence so hashes match the reference) ───────

pub struct Zobrist {
    pub pawn_lo: [[u32; 81]; 2],
    pub pawn_hi: [[u32; 81]; 2],
    pub hw_lo: [u32; 64],
    pub hw_hi: [u32; 64],
    pub vw_lo: [u32; 64],
    pub vw_hi: [u32; 64],
    pub turn_lo: u32,
    pub turn_hi: u32,
}

const fn zrand(seed: u32) -> u32 {
    let mut s = seed;
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    s
}

const fn build_zobrist() -> Zobrist {
    let mut z = Zobrist {
        pawn_lo: [[0; 81]; 2],
        pawn_hi: [[0; 81]; 2],
        hw_lo: [0; 64],
        hw_hi: [0; 64],
        vw_lo: [0; 64],
        vw_hi: [0; 64],
        turn_lo: 0,
        turn_hi: 0,
    };
    let mut seed: u32 = 0x9e3779b9;
    let mut zi = 0;
    while zi < 2 {
        let mut zj = 0;
        while zj < 81 {
            seed = zrand(seed);
            z.pawn_lo[zi][zj] = seed;
            seed = zrand(seed);
            z.pawn_hi[zi][zj] = seed;
            zj += 1;
        }
        zi += 1;
    }
    let mut zs = 0;
    while zs < 64 {
        seed = zrand(seed);
        z.hw_lo[zs] = seed;
        seed = zrand(seed);
        z.hw_hi[zs] = seed;
        seed = zrand(seed);
        z.vw_lo[zs] = seed;
        seed = zrand(seed);
        z.vw_hi[zs] = seed;
        zs += 1;
    }
    seed = zrand(seed);
    z.turn_lo = seed;
    seed = zrand(seed);
    z.turn_hi = seed;
    z
}

pub static ZOBRIST: Zobrist = build_zobrist();

const fn build_border() -> [u8; 81] {
    let mut border = [0u8; 81];
    let mut bc = 0;
    while bc < 81 {
        let br = bc / 9;
        let bcl = bc % 9;
        border[bc] = (if br == 0 { 1 } else { 0 })
            | (if br == 8 { 2 } else { 0 })
            | (if bcl == 0 { 4 } else { 0 })
            | (if bcl == 8 { 8 } else { 0 });
        bc += 1;
    }
    border
}

pub static BORDER: [u8; 81] = build_border();

// ── Game state ────────────────────────────────────────────────────────────────

pub struct AceGame {
    pub pawn: [usize; 2],
    pub wl: [i32; 2],
    pub turn: usize,
    pub hw: [u8; 64],
    pub vw: [u8; 64],
    /// Wall-blocked direction bits per cell: N=1 S=2 W=4 E=8 (bounds via BORDER).
    pub blocked: [u8; 81],
    pub hash_lo: u32,
    pub hash_hi: u32,
    pub hist_m: [i16; 1024],
    pub hist_from: [i16; 1024],
    pub hist_lw: [i16; 1024],
    pub hashes_u: [u32; 2048],
    pub hist_len: usize,
    /// Repetition can only reach back to the last wall placement.
    pub last_wall_ply: usize,
    /// Bumped on every wall make/unmake; dist fields depend only on walls.
    pub wall_stamp: i32,
    seen: [u8; 81],
}

impl Default for AceGame {
    fn default() -> Self {
        Self::new()
    }
}

impl AceGame {
    pub fn new() -> Self {
        let z = &ZOBRIST;
        Self {
            pawn: [76, 4],
            wl: [10, 10],
            turn: 0,
            hw: [0; 64],
            vw: [0; 64],
            blocked: [0; 81],
            hash_lo: z.pawn_lo[0][76] ^ z.pawn_lo[1][4],
            hash_hi: z.pawn_hi[0][76] ^ z.pawn_hi[1][4],
            hist_m: [0; 1024],
            hist_from: [0; 1024],
            hist_lw: [0; 1024],
            hashes_u: [0; 2048],
            hist_len: 0,
            last_wall_ply: 0,
            wall_stamp: 0,
            seen: [0; 81],
        }
    }

    #[inline(always)]
    pub fn can_step(&self, cell: usize, dir: usize) -> bool {
        ((self.blocked[cell] | BORDER[cell]) & DIRBIT[dir]) == 0
    }

    pub fn winner(&self) -> i32 {
        if self.pawn[0] < 9 {
            return 0;
        }
        if self.pawn[1] >= 72 {
            return 1;
        }
        -1
    }

    // ── Wall mechanics ──────────────────────────────────────────────────────

    pub fn set_wall_bits(&mut self, wall_type: usize, slot: usize, on: bool) {
        let r = slot / 8;
        let c = slot % 8;
        if wall_type == 0 {
            let a = r * 9 + c;
            let b = a + 1;
            let cc = a + 9;
            let dd = b + 9;
            if on {
                self.blocked[a] |= 2;
                self.blocked[b] |= 2;
                self.blocked[cc] |= 1;
                self.blocked[dd] |= 1;
            } else {
                self.blocked[a] &= !2;
                self.blocked[b] &= !2;
                self.blocked[cc] &= !1;
                self.blocked[dd] &= !1;
            }
        } else {
            let a = r * 9 + c;
            let b = a + 9;
            let cc = a + 1;
            let dd = b + 1;
            if on {
                self.blocked[a] |= 8;
                self.blocked[b] |= 8;
                self.blocked[cc] |= 4;
                self.blocked[dd] |= 4;
            } else {
                self.blocked[a] &= !8;
                self.blocked[b] &= !8;
                self.blocked[cc] &= !4;
                self.blocked[dd] &= !4;
            }
        }
    }

    pub fn wall_fits(&self, wall_type: usize, slot: usize) -> bool {
        let r = slot / 8;
        let c = slot % 8;
        if self.hw[slot] != 0 || self.vw[slot] != 0 {
            return false;
        }
        if wall_type == 0 {
            if c > 0 && self.hw[slot - 1] != 0 {
                return false;
            }
            if c < 7 && self.hw[slot + 1] != 0 {
                return false;
            }
        } else {
            if r > 0 && self.vw[slot - 8] != 0 {
                return false;
            }
            if r < 7 && self.vw[slot + 8] != 0 {
                return false;
            }
        }
        true
    }

    /// Conservative "cannot possibly seal" precheck (over-counts anchors, so safe to skip BFS).
    pub fn wall_needs_path_check(&self, wall_type: usize, slot: usize) -> bool {
        let r = (slot / 8) as i32;
        let c = (slot % 8) as i32;
        let mut anchors = 0;
        if wall_type == 0 {
            if c == 0 {
                anchors += 1;
            }
            if c == 7 {
                anchors += 1;
            }
        } else {
            if r == 0 {
                anchors += 1;
            }
            if r == 7 {
                anchors += 1;
            }
        }
        let mut dr = -2;
        while dr <= 2 && anchors < 2 {
            let rr = r + dr;
            if rr < 0 || rr > 7 {
                dr += 1;
                continue;
            }
            let mut dc = -2;
            while dc <= 2 {
                let ccc = c + dc;
                if ccc < 0 || ccc > 7 {
                    dc += 1;
                    continue;
                }
                let ss = (rr * 8 + ccc) as usize;
                if self.hw[ss] != 0 || self.vw[ss] != 0 {
                    anchors += 1;
                    if anchors >= 2 {
                        break;
                    }
                }
                dc += 1;
            }
            dr += 1;
        }
        anchors >= 2
    }

    pub fn has_path(&mut self, player: usize) -> bool {
        let goal = if player == 0 { 0 } else { 8 };
        let start = self.pawn[player];
        if start / 9 == goal {
            return true;
        }
        self.seen.fill(0);
        let mut queue = [0i16; 81];
        let mut head = 0;
        let mut tail = 0;
        queue[tail] = start as i16;
        tail += 1;
        self.seen[start] = 1;
        while head < tail {
            let u = queue[head] as usize;
            head += 1;
            let bm = self.blocked[u] | BORDER[u];
            for d in 0..4 {
                if bm & DIRBIT[d] != 0 {
                    continue;
                }
                let v = (u as i16 + DELTA[d]) as usize;
                if self.seen[v] != 0 {
                    continue;
                }
                if v / 9 == goal {
                    return true;
                }
                self.seen[v] = 1;
                queue[tail] = v as i16;
                tail += 1;
            }
        }
        false
    }

    // gen11: `wallCanBlockTopology` and the Titanium path oracle are GONE —
    // the JS engine deleted the topology fast path (unreachable right-edge
    // condition let trapping walls skip the path check); `wall_legal` now runs
    // the anchor-count precheck + own BFS, exactly like the JS.

    pub fn wall_legal(&mut self, wall_type: usize, slot: usize) -> bool {
        if self.wl[self.turn] <= 0 {
            return false;
        }
        if !self.wall_fits(wall_type, slot) {
            return false;
        }
        // gen11: the `wallCanBlockTopology` gate is GONE from the JS engine —
        // its right-edge condition was unreachable (the same off-by-one fixed
        // in Titanium `can_wall_block_topology`), letting trapping walls skip
        // the path check. Only the sound anchor-count precheck remains.
        if !self.wall_needs_path_check(wall_type, slot) {
            return true;
        }
        self.set_wall_bits(wall_type, slot, true);
        let ok = self.has_path(0) && self.has_path(1);
        self.set_wall_bits(wall_type, slot, false);
        ok
    }

    // ── Pawn moves ──────────────────────────────────────────────────────────

    pub fn gen_pawn_moves(&self, out: &mut [i16], mut n: usize) -> usize {
        let me = self.turn;
        let s = self.pawn[me];
        let o = self.pawn[1 - me];
        for d in 0..4 {
            if !self.can_step(s, d) {
                continue;
            }
            let t = (s as i16 + DELTA[d]) as usize;
            if t != o {
                out[n] = t as i16;
                n += 1;
                continue;
            }
            if self.can_step(o, d) {
                out[n] = o as i16 + DELTA[d];
                n += 1;
                continue;
            }
            let p1 = if d < 2 { 2 } else { 0 };
            let p2 = if d < 2 { 3 } else { 1 };
            if self.can_step(o, p1) {
                let w1 = (o as i16 + DELTA[p1]) as usize;
                if w1 != s {
                    out[n] = w1 as i16;
                    n += 1;
                }
            }
            if self.can_step(o, p2) {
                let w2 = (o as i16 + DELTA[p2]) as usize;
                if w2 != s {
                    out[n] = w2 as i16;
                    n += 1;
                }
            }
        }
        n
    }

    // ── Make / unmake (allocation-free) ─────────────────────────────────────

    pub fn make_move(&mut self, m: i16) {
        let z = &ZOBRIST;
        let hl = self.hist_len;
        self.hist_m[hl] = m;
        self.hist_lw[hl] = self.last_wall_ply as i16;
        if m < 100 {
            let p = self.turn;
            let to = m as usize;
            self.hist_from[hl] = self.pawn[p] as i16;
            self.hash_lo ^= z.pawn_lo[p][self.pawn[p]] ^ z.pawn_lo[p][to];
            self.hash_hi ^= z.pawn_hi[p][self.pawn[p]] ^ z.pawn_hi[p][to];
            self.pawn[p] = to;
        } else if m < 200 {
            let s0 = (m - 100) as usize;
            self.hw[s0] = 1;
            self.set_wall_bits(0, s0, true);
            self.wl[self.turn] -= 1;
            self.wall_stamp += 1;
            self.hash_lo ^= z.hw_lo[s0];
            self.hash_hi ^= z.hw_hi[s0];
            self.last_wall_ply = hl + 1;
        } else {
            let s1 = (m - 200) as usize;
            self.vw[s1] = 1;
            self.set_wall_bits(1, s1, true);
            self.wl[self.turn] -= 1;
            self.wall_stamp += 1;
            self.hash_lo ^= z.vw_lo[s1];
            self.hash_hi ^= z.vw_hi[s1];
            self.last_wall_ply = hl + 1;
        }
        self.turn ^= 1;
        self.hash_lo ^= z.turn_lo;
        self.hash_hi ^= z.turn_hi;
        self.hashes_u[hl * 2] = self.hash_lo;
        self.hashes_u[hl * 2 + 1] = self.hash_hi;
        self.hist_len = hl + 1;
    }

    pub fn unmake_move(&mut self) {
        let z = &ZOBRIST;
        self.hist_len -= 1;
        let hl = self.hist_len;
        let m = self.hist_m[hl];
        self.last_wall_ply = self.hist_lw[hl] as usize;
        self.turn ^= 1;
        self.hash_lo ^= z.turn_lo;
        self.hash_hi ^= z.turn_hi;
        if m < 100 {
            let p = self.turn;
            let from = self.hist_from[hl] as usize;
            let to = m as usize;
            self.hash_lo ^= z.pawn_lo[p][from] ^ z.pawn_lo[p][to];
            self.hash_hi ^= z.pawn_hi[p][from] ^ z.pawn_hi[p][to];
            self.pawn[p] = from;
        } else if m < 200 {
            let s0 = (m - 100) as usize;
            self.hw[s0] = 0;
            self.set_wall_bits(0, s0, false);
            self.wl[self.turn] += 1;
            self.wall_stamp -= 1;
            self.hash_lo ^= z.hw_lo[s0];
            self.hash_hi ^= z.hw_hi[s0];
        } else {
            let s1 = (m - 200) as usize;
            self.vw[s1] = 0;
            self.set_wall_bits(1, s1, false);
            self.wl[self.turn] += 1;
            self.wall_stamp -= 1;
            self.hash_lo ^= z.vw_lo[s1];
            self.hash_hi ^= z.vw_hi[s1];
        }
    }

    // ── Distance fields ─────────────────────────────────────────────────────

    pub fn compute_dist(&self, player: usize, dist: &mut [u8; 81]) {
        dist.fill(255);
        let goal = if player == 0 { 0 } else { 8 };
        let mut queue = [0i16; 81];
        let mut head = 0;
        let mut tail = 0;
        for c in 0..9 {
            let cell = goal * 9 + c;
            dist[cell] = 0;
            queue[tail] = cell as i16;
            tail += 1;
        }
        while head < tail {
            let u = queue[head] as usize;
            head += 1;
            let du = dist[u] + 1;
            let bm = self.blocked[u] | BORDER[u];
            for d in 0..4 {
                if bm & DIRBIT[d] != 0 {
                    continue;
                }
                let v = (u as i16 + DELTA[d]) as usize;
                if dist[v] > du {
                    dist[v] = du;
                    queue[tail] = v as i16;
                    tail += 1;
                }
            }
        }
    }
}
