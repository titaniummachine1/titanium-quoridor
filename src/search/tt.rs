//! Transposition table — perft node cache now, αβ search later.
//!
//! Stockfish-style **clustered buckets** (4 slots per index) to cut collisions.

const TT_CLUSTER: usize = 4;
/// Default index bits → `2^bits` clusters of [`TT_CLUSTER`] slots (~24 MB at 18).
/// Overridable at construction via the `TT_BITS` env var (10..=27).
///
/// Size is a depth-dependent tradeoff, measured (native, perft, `tt_speedup` bench):
///   | bits | RAM | perft(4) TT-on | perft(5) TT-on |
///   | 18 | 24 MB | **0.30 s** | 20.7 s |
///   | 20 | 96 MB | 0.33 s | 16.1 s |
///   | 22 | 384 MB | 0.41 s | **12.6 s** |
///   | 24 | 1.5 GB | 0.76 s | 13.4 s |
/// A bigger table raises the hit rate (big win once it fills at depth 5+: ~1.6×),
/// but at shallow depth the tiny working set can't amortise the scattered-probe
/// page-fault/TLB cost, so it *regresses* perft(4). Default stays small to
/// protect the common case; deep perft can opt in with `TT_BITS=22`. (Beyond 22
/// the table's own cache pressure outweighs the marginal hit-rate gain.)
const DEFAULT_TT_BITS: usize = 18;

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
        let cluster = &mut self.clusters[(key as usize) & self.mask];
        let mut replace = 0usize;
        let mut shallowest = u8::MAX;

        for (i, entry) in cluster.entries.iter().enumerate() {
            if entry.key == key {
                if entry.depth <= depth {
                    cluster.entries[i] = Entry { key, depth, nodes };
                }
                return;
            }
            if entry.key == 0 {
                cluster.entries[i] = Entry { key, depth, nodes };
                return;
            }
            if entry.depth < shallowest {
                shallowest = entry.depth;
                replace = i;
            }
        }

        cluster.entries[replace] = Entry { key, depth, nodes };
    }
}
