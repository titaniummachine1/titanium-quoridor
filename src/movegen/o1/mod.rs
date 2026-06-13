//! O(1) pawn lookup (production default) + shift-based wall masks.
//!
//! ```text
//! Pawns: PAWN_LEGAL tables — `PawnGenMode::O1Lookup`, the PRODUCTION DEFAULT
//!        (fastest at perft 4/5 in both default and BMI2/PEXT builds, verified
//!        against the oracle). Shift/scalar retained as bench/test alternatives.
//! Walls: L1 empty → L2 shift collision → topo shift flood-skip → L3 flood (legal.rs)
//! ```
//!
//! Pawn tables are built at COLD START (`runtime`) by default, so the binary/
//! wasm ship ~1.85MB lighter. The `gen` module is the single source of truth
//! (shared with the `movegen-o1-gen` emitter). The `embed-tables` feature bakes
//! the precomputed consts in instead (prewarmed build); a parity test asserts
//! the runtime build matches them byte-for-byte.

pub mod gen;
mod lookup;
mod runtime;

#[cfg(feature = "embed-tables")]
mod embedded;

pub use lookup::{
    encode_enemy_key, generate_pawn_moves_lean_lut, generate_pawn_moves_o1,
    generate_wall_candidates_o1, legal_pawn_move_mask, pack_wall_key,
    wall_collision_clear_h_mask, wall_collision_clear_v_mask, wall_l12_h_mask, wall_l12_v_mask,
    wall_masks, wall_needs_flood_h_mask, wall_needs_flood_v_mask, wall_physically_legal_o1,
    WallMasks,
};
pub use runtime::{prewarm, tables, PawnTables};
