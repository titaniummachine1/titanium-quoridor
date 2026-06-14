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
/// deeper still. Default stays 22. (An adaptive table that starts small — sized
/// to the d3/d4 working set — and grows a bit at a time while preserving stored
/// entries is prototyped on the `adaptive-tt` branch for A/B testing vs this.)
const DEFAULT_TT_BITS: usize = 22;

// NOTE: a 16-byte packed layout (`key` + `depth<<56 | nodes`, cluster = one
// 64-byte cache line) was tried and measured at BOTH perft(4) and perft(5) — no
// speedup at either (d4 ~0.32s, d5 ~22s, indistinguishable from 24-byte). Even
// at depth 5, where the TT thrashes, halving the cluster cache footprint did
// nothing: the engine is compute-bound on TT-miss nodes, not TT-memory-bound.
// Kept the clear 24-byte struct. See `benches/tt_speedup.rs`.
//
// COLLISION SAFETY: the 64-bit `key` alone can't prove two boards are identical
// (distinct positions can share a Zobrist key → a wrong stored `nodes` would be
// served as if correct). `verify` is a SECOND, independent 32-bit hash of the
// board (`Board::tt_verify`); a false hit now needs BOTH `key` (64) and `verify`
// (32) to collide (~2^-96/pair — negligible even at game-solve scale). It is
// FREE: { key:8, nodes:8, verify:4, depth:1 } = 21 bytes, padded to the same
// 24-byte align-8 entry, so the cluster stays 96 B (no cache cost).
//
// EVICTION is depth-only (evict the shallowest entry in a full cluster). A
// `walls_total`-primary policy ("drop entries from unreachable game phases
// first") was MEASURED and REJECTED for perft: it regressed d5 by ~10% (15.1s →
// 16.7s). In perft's DFS, positions with more walls remaining are SHALLOWER in
// the game tree and carry the MOST plies below them — the most expensive
// subtrees to recompute — so evicting them first tanked the hit rate. The idea
// is sound for FORWARD-only game search (walls never un-placed), but that path
// uses a different table (`search_tt::SearchTt`), so it belongs there if wanted.
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
}

impl Default for TranspositionTable {
    fn default() -> Self {
        Self::new()
    }
}

impl TranspositionTable {
    pub fn new() -> Self {
        let bits = std::env::var("TT_BITS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&b| (10..=27).contains(&b))
            .unwrap_or(DEFAULT_TT_BITS);
        let size = 1usize << bits;
        Self {
            clusters: vec![Cluster::default(); size],
            mask: size - 1,
        }
    }

    pub fn clear(&mut self) {
        self.clusters.fill(Cluster::default());
    }

    #[inline]
    pub fn probe(&self, key: u64, verify: u32, depth: u8) -> Option<u64> {
        let cluster = &self.clusters[(key as usize) & self.mask];
        for entry in &cluster.entries {
            // Both the 64-bit key AND the independent 32-bit verify must match —
            // guards against a Zobrist collision serving the wrong node count.
            // (Measured: 0 such collisions at perft d4/d5; this is insurance that
            // only bites at far deeper / game-solve scale.)
            if entry.key == key && entry.verify == verify && entry.depth == depth {
                return Some(entry.nodes);
            }
        }
        None
    }

    #[inline]
    pub fn store(&mut self, key: u64, verify: u32, depth: u8, nodes: u64) {
        let cluster = &mut self.clusters[(key as usize) & self.mask];
        let mut replace = 0usize;
        let mut shallowest = u8::MAX;

        for (i, entry) in cluster.entries.iter().enumerate() {
            if entry.key == key && entry.verify == verify {
                if entry.depth <= depth {
                    cluster.entries[i] = Entry { key, nodes, verify, depth };
                }
                return;
            }
            if entry.key == 0 {
                cluster.entries[i] = Entry { key, nodes, verify, depth };
                return;
            }
            if entry.depth < shallowest {
                shallowest = entry.depth;
                replace = i;
            }
        }

        cluster.entries[replace] = Entry { key, nodes, verify, depth };
    }
}
