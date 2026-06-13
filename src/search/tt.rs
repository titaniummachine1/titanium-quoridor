//! Transposition table — perft node cache now, αβ search later.
//!
//! Stockfish-style **clustered buckets** (4 slots per index) to cut collisions.

const TT_CLUSTER: usize = 4;
/// Default index bits → `2^bits` clusters of [`TT_CLUSTER`] slots (~384 MB at 22).
/// Overridable at construction via the `TT_BITS` env var (10..=27).
///
/// Size is a depth-dependent tradeoff, measured (native, perft, `tt_speedup` bench):
///   | bits | RAM | perft(4) TT-on | perft(5) TT-on |
///   | 18 | 24 MB | 0.30 s | 20.7 s |
///   | 20 | 96 MB | 0.33 s | 16.1 s |
///   | 22 | 384 MB | 0.41 s | **12.6 s** |
///   | 23 | 768 MB | — | 11.0 s (≈ 22) |
///   | 24 | 1.5 GB | 0.76 s | 13.4 s |
/// **22 is the chosen default — the optimal speed/size tradeoff:** it owns the
/// deep perft (d5 ~1.6× over 18) for 384 MB, while 24 doubles to 1.5 GB *and*
/// regresses (the table's own cache pressure outweighs the marginal hit-rate
/// gain). **Why not 23?** measured d5 (best of 3, this machine): 22 = 11.04 s,
/// 23 = 11.00 s, 24 = 11.76 s — 23 is a noise-level tie with 22 but costs **2×
/// the RAM (768 MB vs 384)** for nothing, so 22 is the better point. The only
/// cost of 22 is shallow perft(4) (0.41 s vs 0.30 s at 18 — the tiny working set
/// can't amortise the scattered-probe page-fault/TLB cost); a memory-constrained
/// caller can drop back with `TT_BITS=18`.
///
/// Flag guidance: `TT_BITS=23` (768 MB) is a good choice for deeper-than-d5
/// searches (ties 22 at d5, more headroom beyond), and `24` (1.5 GB) for even
/// deeper still. Default stays 22.
const DEFAULT_TT_BITS: usize = 22;

// ── adaptive sizing (this `adaptive-tt` branch) ──────────────────────────────
// Instead of pre-allocating the static default, start small — sized to the
// d3/d4 working set (measured: d3 fits by ~14 bits, d4's knee is ~18) — and grow
// ONE bit at a time when the table passes a load threshold, REHASHING the live
// entries into the bigger table so nothing already computed is thrown away. RAM
// then tracks the actual depth reached instead of a fixed 384 MB. A/B vs `main`:
//   adaptive (default here):   `cargo run --bin titanium -- perft 5`
//   static, like main:         `TT_BITS=22 cargo run --bin titanium -- perft 5`
/// Adaptive start (env `TT_START_BITS`): holds the d3 working set in ~1.5 MB.
const DEFAULT_START_BITS: usize = 14;
/// Adaptive ceiling (env `TT_MAX_BITS`): growth stops here. RAM = `2^bits × 96 B`
/// per cluster: 24 = 1.6 GB, **25 = 3.2 GB (default cap)**, 26 = 6.4 GB (≈ the
/// "~8 GB" budget), 27 = 12.9 GB. Raise with `TT_MAX_BITS=26` for ~8 GB headroom.
const DEFAULT_MAX_BITS: usize = 25;

// NOTE: a 16-byte packed layout (`key` + `depth<<56 | nodes`, cluster = one
// 64-byte cache line) was tried and measured at BOTH perft(4) and perft(5) — no
// speedup at either (d4 ~0.32s, d5 ~22s, indistinguishable from 24-byte). Even
// at depth 5, where the TT thrashes, halving the cluster cache footprint did
// nothing: the engine is compute-bound on TT-miss nodes, not TT-memory-bound.
// Kept the clear 24-byte struct. See `benches/tt_speedup.rs`.
#[derive(Clone, Copy, Default)]
struct Entry {
    key: u64,
    depth: u8,
    nodes: u64,
}

#[derive(Clone, Copy)]
struct Cluster {
    entries: [Entry; TT_CLUSTER],
}

impl Default for Cluster {
    fn default() -> Self {
        Self {
            entries: [Entry::default(); TT_CLUSTER],
        }
    }
}

pub struct TranspositionTable {
    clusters: Vec<Cluster>,
    mask: usize,
    bits: usize,
    /// Non-empty slots currently held (the grow trigger).
    filled: usize,
    /// Growth ceiling; `bits == max_bits` (or `!adaptive`) disables growth.
    max_bits: usize,
    /// Adaptive grow-on-load is on iff `TT_BITS` was NOT pinned.
    adaptive: bool,
}

impl Default for TranspositionTable {
    fn default() -> Self {
        Self::new()
    }
}

fn env_bits(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&b| (10..=27).contains(&b))
}

impl TranspositionTable {
    pub fn new() -> Self {
        // `TT_BITS` pins a static size (A/B baseline, matches `main`); otherwise
        // start small and grow adaptively.
        if let Some(bits) = env_bits("TT_BITS") {
            return Self::with_size(bits, bits, false);
        }
        let start = env_bits("TT_START_BITS").unwrap_or(DEFAULT_START_BITS);
        let max = env_bits("TT_MAX_BITS")
            .unwrap_or(DEFAULT_MAX_BITS)
            .max(start);
        Self::with_size(start, max, true)
    }

    fn with_size(bits: usize, max_bits: usize, adaptive: bool) -> Self {
        let size = 1usize << bits;
        Self {
            clusters: vec![Cluster::default(); size],
            mask: size - 1,
            bits,
            filled: 0,
            max_bits,
            adaptive,
        }
    }

    pub fn clear(&mut self) {
        self.clusters.fill(Cluster::default());
        self.filled = 0;
    }

    /// Total slots = `TT_CLUSTER` per index. Grow at load factor ≥ 1/2.
    #[inline]
    fn should_grow(&self) -> bool {
        self.adaptive && self.bits < self.max_bits && self.filled * 2 >= self.clusters.len() * TT_CLUSTER
    }

    #[inline]
    pub fn probe(&self, key: u64, depth: u8) -> Option<u64> {
        let cluster = &self.clusters[(key as usize) & self.mask];
        for entry in &cluster.entries {
            if entry.key == key && entry.depth == depth {
                return Some(entry.nodes);
            }
        }
        None
    }

    #[inline]
    pub fn store(&mut self, key: u64, depth: u8, nodes: u64) {
        if Self::insert_into(&mut self.clusters, self.mask, key, depth, nodes) {
            self.filled += 1;
            if self.should_grow() {
                self.grow();
            }
        }
    }

    /// Insert into a (clusters, mask) table. Returns `true` iff a previously
    /// EMPTY slot was consumed (a new distinct entry), so the caller can track
    /// occupancy. Matching-key updates and shallowest-evictions return `false`.
    #[inline]
    fn insert_into(
        clusters: &mut [Cluster],
        mask: usize,
        key: u64,
        depth: u8,
        nodes: u64,
    ) -> bool {
        let cluster = &mut clusters[(key as usize) & mask];
        let mut replace = 0usize;
        let mut shallowest = u8::MAX;
        for (i, entry) in cluster.entries.iter().enumerate() {
            if entry.key == key {
                if entry.depth <= depth {
                    cluster.entries[i] = Entry { key, depth, nodes };
                }
                return false;
            }
            if entry.key == 0 {
                cluster.entries[i] = Entry { key, depth, nodes };
                return true;
            }
            if entry.depth < shallowest {
                shallowest = entry.depth;
                replace = i;
            }
        }
        cluster.entries[replace] = Entry { key, depth, nodes };
        false
    }

    /// Step up one bit, REHASHING the live entries into the doubled table so
    /// nothing already computed is lost. O(old slots); cheap vs the search that
    /// fills it (and amortised — each entry is rehashed ≤ once per doubling).
    fn grow(&mut self) {
        let new_bits = self.bits + 1;
        let new_size = 1usize << new_bits;
        let new_mask = new_size - 1;
        let mut next = vec![Cluster::default(); new_size];
        for cluster in &self.clusters {
            for e in &cluster.entries {
                if e.key != 0 {
                    Self::insert_into(&mut next, new_mask, e.key, e.depth, e.nodes);
                }
            }
        }
        self.clusters = next;
        self.mask = new_mask;
        self.bits = new_bits;
    }
}
