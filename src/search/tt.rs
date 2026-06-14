//! Transposition table — perft node cache now, αβ search later.
//!
//! Stockfish-style **clustered buckets** (4 slots per index) to cut collisions.

const TT_CLUSTER: usize = 4;
/// Default index bits → `2^bits` clusters of [`TT_CLUSTER`] slots.
/// Overridable at construction via the `TT_BITS` env var (8..=27).
///
/// Size is a depth-dependent tradeoff, measured (native, perft, `tt_speedup` bench):
///   | bits | RAM    | perft(3) | perft(4) | perft(5) |
///   |   8  |  24 KB |  111 ms  |  1.12 s  |    —     |
///   |  11  | 192 KB |  103 ms  |  0.83 s  |    —     |  ← d3 sweet spot (L2/core)
///   |  14  | 1.5 MB |  104 ms  |  0.53 s  |    —     |
///   |  16  |   6 MB |  111 ms  |  0.44 s  |    —     |
///   |  18  |  24 MB |  119 ms  |  0.37 s  | **20.7 s** |  ← d4 sweet spot
///   |  20  |  96 MB |  110 ms  |  0.42 s  |  16.1 s  |
///   |  22  | 384 MB |  119 ms  |  0.63 s  | **12.6 s** |  ← d5 sweet spot (default)
///   |  24  | 1.5 GB |    —     |  0.76 s  |  13.4 s  |
///
/// Working set scales ~7 bits per depth (Quoridor branches ~100× per ply):
///   d3 knee ≈ 11, d4 knee ≈ 18, d5 knee ≈ 22.
/// Larger tables regress at shallow depths (page-fault / TLB pressure outweighs
/// the marginal hit-rate gain once the working set is already resident).
///
/// **Adaptive mode (default)**: starts at `DEFAULT_START_BITS` (11, L2-sized),
/// grows one bit at a time when the load factor hits 50%, rehashing live entries
/// so nothing computed is thrown away. RAM tracks actual depth reached instead of
/// pre-allocating 384 MB. Pin a static size with `TT_BITS=N` (disables growth) to
/// match a specific depth or memory budget; use `TT_START_BITS` / `TT_MAX_BITS`
/// to tune the adaptive range without disabling it.
///
/// Pin a static size for benchmarking with `TT_BITS=22` (d5 optimal) or
/// `TT_BITS=18` (d4 optimal, 24 MB). Adaptive is the default.
/// Adaptive start — holds the d3 working set in ~192 KB (L2 per core).
const DEFAULT_START_BITS: usize = 11;
/// Adaptive ceiling — growth stops here. `2^25 × 96 B ≈ 3.2 GB`.
const DEFAULT_MAX_BITS: usize = 25;

// NOTE: a 16-byte packed layout was tried and measured at BOTH perft(4) and
// perft(5) — no speedup at either. Even at depth 5, halving cluster footprint
// did nothing: the engine is compute-bound on TT-miss nodes, not
// TT-memory-bound. Kept the clear 24-byte struct. See `benches/tt_speedup.rs`.
//
// COLLISION SAFETY: the 64-bit `key` alone can't prove two boards are identical.
// `verify` is a SECOND, independent 32-bit hash (`Board::tt_verify`); a false
// hit needs BOTH `key` (64) and `verify` (32) to collide (~2^-96/pair —
// negligible even at game-solve scale). FREE: { key:8, nodes:8, verify:4,
// depth:1 } = 21 bytes, padded to the same 24-byte align-8 entry (no cache cost).
//
// EVICTION is depth-only (evict the shallowest entry in a full cluster). A
// `walls_total`-primary policy was MEASURED and REJECTED for perft: it regressed
// d5 by ~10% (positions with more walls remaining are shallower in the tree but
// carry the most plies below — evicting them first tanks the hit rate).
#[derive(Clone, Copy, Default)]
struct Entry {
    key: u64,
    nodes: u64,
    verify: u32,
    depth: u8,
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
    /// Non-empty slots currently held (grow trigger).
    filled: usize,
    max_bits: usize,
    /// False when `TT_BITS` was pinned (A/B baseline, matches static behaviour).
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
        .filter(|&b| (8..=27).contains(&b))
}

impl TranspositionTable {
    pub fn new() -> Self {
        // `TT_BITS` pins a static size (disables adaptive growth).
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

    /// Grow at load factor ≥ 50% (half of all slots filled).
    #[inline]
    fn should_grow(&self) -> bool {
        self.adaptive
            && self.bits < self.max_bits
            && self.filled * 2 >= self.clusters.len() * TT_CLUSTER
    }

    #[inline]
    pub fn probe(&self, key: u64, verify: u32, depth: u8) -> Option<u64> {
        let cluster = &self.clusters[(key as usize) & self.mask];
        for entry in &cluster.entries {
            // Both the 64-bit key AND the independent 32-bit verify must match —
            // guards against a Zobrist collision serving the wrong node count.
            if entry.key == key && entry.verify == verify && entry.depth == depth {
                return Some(entry.nodes);
            }
        }
        None
    }

    #[inline]
    pub fn store(&mut self, key: u64, verify: u32, depth: u8, nodes: u64) {
        if Self::insert_into(&mut self.clusters, self.mask, key, verify, depth, nodes) {
            self.filled += 1;
            if self.should_grow() {
                self.grow();
            }
        }
    }

    /// Insert into a (clusters, mask) table. Returns `true` iff a previously
    /// EMPTY slot was consumed (caller tracks occupancy for adaptive growth).
    /// Matching-key updates and depth-evictions return `false`.
    #[inline]
    fn insert_into(
        clusters: &mut [Cluster],
        mask: usize,
        key: u64,
        verify: u32,
        depth: u8,
        nodes: u64,
    ) -> bool {
        let cluster = &mut clusters[(key as usize) & mask];
        let mut replace = 0usize;
        let mut shallowest = u8::MAX;

        for (i, entry) in cluster.entries.iter().enumerate() {
            if entry.key == key && entry.verify == verify {
                if entry.depth <= depth {
                    cluster.entries[i] = Entry { key, nodes, verify, depth };
                }
                return false;
            }
            if entry.key == 0 {
                cluster.entries[i] = Entry { key, nodes, verify, depth };
                return true;
            }
            if entry.depth < shallowest {
                shallowest = entry.depth;
                replace = i;
            }
        }

        cluster.entries[replace] = Entry { key, nodes, verify, depth };
        false
    }

    /// How many bits to add on the next grow.
    /// Large jumps when small (cheap rehash, need to reach working set fast),
    /// single-bit steps when large (avoid overshooting and wasting RAM).
    ///   bits <  14 → +3  (11 → 14 → ... saves 4 rehashes vs +1 to reach d4=18)
    ///   bits < 18  → +2  (14 → 16 → 18, d4 sweet spot in 3 total grows)
    ///   bits ≥ 18  → +1  (careful steps for d5 and beyond)
    #[inline]
    fn grow_step(&self) -> usize {
        if self.bits < 14 { 3 } else if self.bits < 18 { 2 } else { 1 }
    }

    /// Grow the table by `grow_step()` bits, rehashing all live entries into the
    /// new allocation so no computed results are lost. O(old slots); the
    /// aggressive early steps mean total rehash work to reach d4=18 is ~3× less
    /// than single-bit steps (84K clusters vs 260K).
    fn grow(&mut self) {
        let new_bits = (self.bits + self.grow_step()).min(self.max_bits);
        let new_size = 1usize << new_bits;
        let new_mask = new_size - 1;
        let mut next = vec![Cluster::default(); new_size];
        for cluster in &self.clusters {
            for e in &cluster.entries {
                if e.key != 0 {
                    Self::insert_into(&mut next, new_mask, e.key, e.verify, e.depth, e.nodes);
                }
            }
        }
        self.clusters = next;
        self.mask = new_mask;
        self.bits = new_bits;
    }
}
