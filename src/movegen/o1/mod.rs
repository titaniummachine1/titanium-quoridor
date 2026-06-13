//! O(1) pawn lookup (production default) + shift-based wall masks.
//!
//! ```text
//! Pawns: PAWN_LEGAL tables — `PawnGenMode::O1Lookup`, the PRODUCTION DEFAULT
//!        (fastest at perft 4/5 in both default and BMI2/PEXT builds, verified
//!        against the oracle). Shift/scalar retained as bench/test alternatives.
//! Walls: L1 empty → L2 shift collision → topo shift flood-skip → L3 flood (legal.rs)
//! ```

mod lookup;
mod tables;

pub use lookup::{
    encode_enemy_key, generate_pawn_moves_lean_lut, generate_pawn_moves_o1,
    generate_wall_candidates_o1, legal_pawn_move_mask,
    pack_wall_key, wall_collision_clear_h_mask, wall_collision_clear_v_mask, wall_l12_h_mask,
    wall_l12_v_mask, wall_masks, wall_needs_flood_h_mask, wall_needs_flood_v_mask,
    wall_physically_legal_o1, WallMasks,
};
