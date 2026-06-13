//! Cold-start pawn lookup tables — built in memory at first use instead of
//! shipping ~1.85MB of baked-in data in the binary/wasm.
//!
//! The hot path reads `tables()` → `&'static PawnTables` (one `OnceLock` load,
//! hoisted to the top of each generate call). Initialization:
//!
//! - **default build**: `super::gen::discover_all_pawn_tables()` recomputes the
//!   tables (~1-2s). Lazy on first use; call [`prewarm`] up front (e.g. on UCI
//!   `isready`) so search/perft timing never includes the build.
//! - **`embed-tables` feature**: copies the committed `generated_*` consts in
//!   (no compute) — the prewarmed build for the website / latency-sensitive
//!   targets. A parity test asserts the two paths produce identical tables.

use std::sync::OnceLock;

use super::gen::{discover_all_pawn_tables, PawnSquareMeta, ENEMY_LAYERS, WALL_KEYS};

/// Physical wall-combo index width per (sq, enemy_key): `2^MAX_WALL_SLOTS`.
pub use super::gen::PHYS_WALL_COMBOS;

/// In-memory pawn lookup tables (layout identical to the embedded consts).
pub struct PawnTables {
    pub catalog: Box<[[u8; 12]; 81]>,
    pub legal: Box<[[[u16; WALL_KEYS]; ENEMY_LAYERS]; 81]>,
    pub layer_valid: Box<[[u8; ENEMY_LAYERS]; 81]>,
    pub wall_combo_count: Box<[[u16; ENEMY_LAYERS]; 81]>,
    pub wall_slot_count: Box<[[u8; ENEMY_LAYERS]; 81]>,
    pub desc_row: Box<[[[u8; super::gen::MAX_WALL_SLOTS]; ENEMY_LAYERS]; 81]>,
    pub desc_col: Box<[[[u8; super::gen::MAX_WALL_SLOTS]; ENEMY_LAYERS]; 81]>,
    pub desc_h: Box<[[[u8; super::gen::MAX_WALL_SLOTS]; ENEMY_LAYERS]; 81]>,
    pub h_pext_mask: Box<[[u64; ENEMY_LAYERS]; 81]>,
    pub v_pext_mask: Box<[[u64; ENEMY_LAYERS]; 81]>,
    pub h_slot_count: Box<[[u8; ENEMY_LAYERS]; 81]>,
    /// `[(sq*ENEMY_LAYERS + enemy_key)*PHYS_WALL_COMBOS + phys] -> semantic key`.
    pub remap: Box<[u8]>,
}

impl PawnTables {
    /// Physical combo → semantic wall key. Mirrors the embedded `wall_remap_byte`.
    #[inline]
    pub fn wall_remap_byte(&self, sq: u8, enemy_key: u8, phys_combo: usize) -> u8 {
        let idx = (sq as usize * ENEMY_LAYERS + enemy_key as usize) * PHYS_WALL_COMBOS + phys_combo;
        self.remap[idx]
    }
}

fn boxed<const N: usize, T>(v: Vec<T>) -> Box<[T; N]>
where
    T: std::fmt::Debug,
{
    v.into_boxed_slice()
        .try_into()
        .expect("boxed array length")
}

/// Assemble `PawnTables` from discovered metadata. Mirrors `build/movegen_o1/emit.rs`
/// exactly so the runtime build is byte-identical to the embedded consts.
fn assemble(pawn: &[PawnSquareMeta]) -> PawnTables {
    const MWS: usize = super::gen::MAX_WALL_SLOTS;
    let mut catalog: Box<[[u8; 12]; 81]> = boxed(vec![[0u8; 12]; 81]);
    let mut legal: Box<[[[u16; WALL_KEYS]; ENEMY_LAYERS]; 81]> =
        boxed(vec![[[0u16; WALL_KEYS]; ENEMY_LAYERS]; 81]);
    let mut layer_valid: Box<[[u8; ENEMY_LAYERS]; 81]> = boxed(vec![[0u8; ENEMY_LAYERS]; 81]);
    let mut wall_combo_count: Box<[[u16; ENEMY_LAYERS]; 81]> =
        boxed(vec![[0u16; ENEMY_LAYERS]; 81]);
    let mut wall_slot_count: Box<[[u8; ENEMY_LAYERS]; 81]> = boxed(vec![[0u8; ENEMY_LAYERS]; 81]);
    let mut desc_row: Box<[[[u8; MWS]; ENEMY_LAYERS]; 81]> =
        boxed(vec![[[255u8; MWS]; ENEMY_LAYERS]; 81]);
    let mut desc_col: Box<[[[u8; MWS]; ENEMY_LAYERS]; 81]> =
        boxed(vec![[[255u8; MWS]; ENEMY_LAYERS]; 81]);
    let mut desc_h: Box<[[[u8; MWS]; ENEMY_LAYERS]; 81]> =
        boxed(vec![[[255u8; MWS]; ENEMY_LAYERS]; 81]);
    let mut h_pext_mask: Box<[[u64; ENEMY_LAYERS]; 81]> = boxed(vec![[0u64; ENEMY_LAYERS]; 81]);
    let mut v_pext_mask: Box<[[u64; ENEMY_LAYERS]; 81]> = boxed(vec![[0u64; ENEMY_LAYERS]; 81]);
    let mut h_slot_count: Box<[[u8; ENEMY_LAYERS]; 81]> = boxed(vec![[0u8; ENEMY_LAYERS]; 81]);
    let mut remap = vec![0u8; 81 * ENEMY_LAYERS * PHYS_WALL_COMBOS];

    for (sq, p) in pawn.iter().enumerate() {
        catalog[sq] = p.catalog;
        for (k, layer) in p.layers.iter().enumerate() {
            layer_valid[sq][k] = u8::from(layer.valid);
            wall_combo_count[sq][k] = layer.wall_combo_count;
            wall_slot_count[sq][k] = layer.wall_bits.len() as u8;
            for i in 0..MWS {
                if let Some(&(r, c, h)) = layer.wall_bits.get(i) {
                    desc_row[sq][k][i] = r;
                    desc_col[sq][k][i] = c;
                    desc_h[sq][k][i] = u8::from(h);
                } // else stays 255 (off-slot sentinel)
            }
            let mut hm = 0u64;
            let mut vm = 0u64;
            let mut hsc = 0u8;
            for &(r, c, h) in &layer.wall_bits {
                let bit = 1u64 << (r as u64 * 8 + c as u64);
                if h {
                    hm |= bit;
                    hsc += 1;
                } else {
                    vm |= bit;
                }
            }
            h_pext_mask[sq][k] = hm;
            v_pext_mask[sq][k] = vm;
            h_slot_count[sq][k] = hsc;
            legal[sq][k] = layer.table;
            let base = (sq * ENEMY_LAYERS + k) * PHYS_WALL_COMBOS;
            remap[base..base + layer.wall_remap.len()].copy_from_slice(&layer.wall_remap);
        }
    }

    PawnTables {
        catalog,
        legal,
        layer_valid,
        wall_combo_count,
        wall_slot_count,
        desc_row,
        desc_col,
        desc_h,
        h_pext_mask,
        v_pext_mask,
        h_slot_count,
        remap: remap.into_boxed_slice(),
    }
}

/// Recompute the tables from scratch (cold start). ~1-2s.
pub fn build_from_discovery() -> PawnTables {
    assemble(&discover_all_pawn_tables())
}

/// Copy the committed embedded consts into a `PawnTables` (no compute).
#[cfg(feature = "embed-tables")]
pub fn build_from_embedded() -> PawnTables {
    use super::embedded as e;
    // `legal` is ~414KB: assign the const into a heap box (rodata→heap memcpy)
    // rather than `Box::new(const)`, which would stage it on the stack first.
    let mut legal: Box<[[[u16; WALL_KEYS]; ENEMY_LAYERS]; 81]> =
        boxed(vec![[[0u16; WALL_KEYS]; ENEMY_LAYERS]; 81]);
    *legal = e::PAWN_LEGAL;
    PawnTables {
        catalog: Box::new(e::PAWN_CATALOG),
        legal,
        layer_valid: Box::new(e::PAWN_LAYER_VALID),
        wall_combo_count: Box::new(e::PAWN_WALL_COMBO_COUNT),
        wall_slot_count: Box::new(e::PAWN_WALL_SLOT_COUNT),
        desc_row: Box::new(e::PAWN_WALL_DESC_ROW),
        desc_col: Box::new(e::PAWN_WALL_DESC_COL),
        desc_h: Box::new(e::PAWN_WALL_DESC_H),
        h_pext_mask: Box::new(e::PAWN_H_PEXT_MASK),
        v_pext_mask: Box::new(e::PAWN_V_PEXT_MASK),
        h_slot_count: Box::new(e::PAWN_H_SLOT_COUNT),
        remap: Box::from(e::PAWN_WALL_REMAP_BYTES),
    }
}

static TABLES: OnceLock<PawnTables> = OnceLock::new();

/// The pawn lookup tables, built on first call. Hot path: hoist to a local.
#[inline]
pub fn tables() -> &'static PawnTables {
    TABLES.get_or_init(|| {
        #[cfg(feature = "embed-tables")]
        {
            build_from_embedded()
        }
        #[cfg(not(feature = "embed-tables"))]
        {
            build_from_discovery()
        }
    })
}

/// Force table construction now (so later search/perft timing excludes the
/// cold-start build). Idempotent; safe to call from any thread.
pub fn prewarm() {
    let _ = tables();
}

/// Mirrors the embedded `wall_remap_byte` free function (kept for call-site parity).
#[inline]
pub fn wall_remap_byte(sq: u8, enemy_key: u8, phys_combo: usize) -> u8 {
    tables().wall_remap_byte(sq, enemy_key, phys_combo)
}

#[cfg(all(test, feature = "embed-tables"))]
mod parity_tests {
    use super::*;

    /// The cold-start build MUST be byte-identical to the committed embedded
    /// consts — otherwise the runtime path could silently corrupt movegen.
    /// Run with: `cargo test --features embed-tables runtime_tables_match_embedded`.
    #[test]
    fn runtime_tables_match_embedded() {
        let r = build_from_discovery();
        let e = build_from_embedded();
        assert_eq!(r.catalog, e.catalog, "catalog");
        assert_eq!(r.layer_valid, e.layer_valid, "layer_valid");
        assert_eq!(r.wall_combo_count, e.wall_combo_count, "wall_combo_count");
        assert_eq!(r.wall_slot_count, e.wall_slot_count, "wall_slot_count");
        assert_eq!(r.desc_row, e.desc_row, "desc_row");
        assert_eq!(r.desc_col, e.desc_col, "desc_col");
        assert_eq!(r.desc_h, e.desc_h, "desc_h");
        assert_eq!(r.h_pext_mask, e.h_pext_mask, "h_pext_mask");
        assert_eq!(r.v_pext_mask, e.v_pext_mask, "v_pext_mask");
        assert_eq!(r.h_slot_count, e.h_slot_count, "h_slot_count");
        assert_eq!(r.remap, e.remap, "remap");
        // `legal` is large; compare per square for a useful failure locus.
        for sq in 0..81 {
            assert_eq!(r.legal[sq], e.legal[sq], "legal[{sq}]");
        }
    }
}
