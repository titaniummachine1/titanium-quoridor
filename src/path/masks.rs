//! Direction bitmasks for bitwise flood fill on the pawn grid.

use crate::core::board::Board;
use crate::titanium::game::{GameState, BORDER};
use crate::util::grid::{can_step, flood_bit_sq, square_index, FLOOD_BIT_BY_SQ};

/// Bit `sq` set iff a pawn on `sq` may step in that direction.
#[derive(Clone, Copy, Default)]
pub struct DirMasks {
    pub north: u128,
    pub south: u128,
    pub east: u128,
    pub west: u128,
}

impl DirMasks {
    pub fn from_board(board: &Board) -> Self {
        let mut m = Self::default();
        for r in 0..=8u8 {
            for c in 0..=8u8 {
                let sq = square_index(r, c);
                let bit = flood_bit_sq(sq);
                if can_step(board, r, c, -1, 0) {
                    m.north |= bit;
                }
                if can_step(board, r, c, 1, 0) {
                    m.south |= bit;
                }
                if can_step(board, r, c, 0, 1) {
                    m.east |= bit;
                }
                if can_step(board, r, c, 0, -1) {
                    m.west |= bit;
                }
            }
        }
        m
    }

    /// Wall topology in Titanium internal cell order (row 0 = top) — no Board rebuild or row remap.
    pub fn from_ace_game(g: &GameState) -> Self {
        crate::bench_instr::record(
            |b| &mut b.dir_masks_from_ace,
            || Self::from_ace_game_inner(g),
        )
    }

    fn from_ace_game_inner(g: &GameState) -> Self {
        let mut m = Self::default();
        for sq in 0..81usize {
            let bit = FLOOD_BIT_BY_SQ[sq];
            let blocked = g.blocked[sq] | BORDER[sq];
            // ACE `can_step`: 0=N(-9), 1=S(+9), 2=W(-1), 3=E(+1)
            if blocked & 1 == 0 {
                m.north |= bit;
            }
            if blocked & 2 == 0 {
                m.south |= bit;
            }
            if blocked & 4 == 0 {
                m.west |= bit;
            }
            if blocked & 8 == 0 {
                m.east |= bit;
            }
        }
        m
    }

    #[inline]
    pub fn with_ace_wall(mut self, wall_type: usize, slot: usize) -> Self {
        let r = slot / 8;
        let c = slot % 8;
        let a = r * 9 + c;
        if wall_type == 0 {
            let b = a + 1;
            let cc = a + 9;
            let dd = b + 9;
            self.south &= !(FLOOD_BIT_BY_SQ[a] | FLOOD_BIT_BY_SQ[b]);
            self.north &= !(FLOOD_BIT_BY_SQ[cc] | FLOOD_BIT_BY_SQ[dd]);
        } else {
            let b = a + 9;
            let cc = a + 1;
            let dd = b + 1;
            self.east &= !(FLOOD_BIT_BY_SQ[a] | FLOOD_BIT_BY_SQ[b]);
            self.west &= !(FLOOD_BIT_BY_SQ[cc] | FLOOD_BIT_BY_SQ[dd]);
        }
        self
    }
}
