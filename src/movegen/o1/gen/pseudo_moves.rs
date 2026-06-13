//! Pseudo-legal pawn catalog + localized key ingredients.
//!
//! Pipeline (no Monte Carlo):
//! 1. **Catalog** — 12 theoretical destinations per source square (on-board slots only).
//! 2. **Enemy** — at most 4 cardinal adjacencies that can change jump/slide; else "far".
//! 3. **Walls** — only slots read by `can_step` on our steps, jumps, and diagonal slides.
//! 4. **Table** — `key(walls, enemy) → bitmask ⊆ catalog` of truly legal moves.

pub const PSEUDO_SLOTS: usize = 12;
pub const OFF_BOARD: u8 = 255;

/// Semantic pseudo-legal destinations from `(sr, sc)`.
/// 0–3 steps, 4–7 jump-through, 8–11 diagonal slide when jump blocked.
pub fn pseudo_catalog(sr: u8, sc: u8) -> [u8; PSEUDO_SLOTS] {
    let sq = |r: i16, c: i16| -> u8 {
        if (0..=8).contains(&r) && (0..=8).contains(&c) {
            super::geometry::square_index(r as u8, c as u8)
        } else {
            OFF_BOARD
        }
    };
    let r = sr as i16;
    let c = sc as i16;
    [
        sq(r + 1, c),
        sq(r - 1, c),
        sq(r, c + 1),
        sq(r, c - 1),
        sq(r + 2, c),
        sq(r - 2, c),
        sq(r, c + 2),
        sq(r, c - 2),
        sq(r - 1, c + 1),
        sq(r - 1, c - 1),
        sq(r + 1, c + 1),
        sq(r + 1, c - 1),
    ]
}

/// Map true legal destination squares → subset bitmask over the pseudo catalog.
pub fn legal_subset_mask(catalog: &[u8; PSEUDO_SLOTS], legal_dests: &[u8], n: usize) -> u16 {
    let mut mask = 0u16;
    for &d in &legal_dests[..n] {
        for (slot, &sq) in catalog.iter().enumerate() {
            if sq != OFF_BOARD && sq == d {
                mask |= 1 << slot;
                break;
            }
        }
    }
    mask
}

pub const CARDINAL_OFFSETS: [(i8, i8); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
