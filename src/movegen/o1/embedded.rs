//! Baked-in pawn tables — compiled ONLY under `--features embed-tables`.
//!
//! The default build does NOT include these (~1.85MB) and computes the
//! equivalent tables at cold start via `super::runtime`. This module exists for
//! the prewarmed build (website / latency-sensitive targets) and as the
//! reference the parity test checks the runtime build against.

include!("generated_tables_data.rs");

/// Physical wall combo → semantic key, flat `[(sq*5+ek)*PHYS_WALL_COMBOS + phys]`.
pub const PAWN_WALL_REMAP_BYTES: &[u8] = include_bytes!("generated_remap.bin");
