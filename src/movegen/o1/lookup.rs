//! Runtime pawn lookup + shift-based wall masks.
//!
//! Pawns: offline `PAWN_LEGAL[sq][enemy_key][wall_key]` tables — `PawnGenMode::O1Lookup`,
//! the production default (perft-proven fastest; see `legal.rs`).
//!
//! Walls — all shift algebra, no runtime tables:
//! - L1: empty slot (`!horizontal_walls` / `!vertical_walls`)
//! - L2: collision (overlap / cross / neighbor) — whole-board shifts
//! - Topo: `can_wall_block_topology` — two-of-three anchor shifts (flood-skip opt)
//! - L3: parallel flood + bit theft — `legal.rs`, lazy `WallTrialCtx`

use crate::core::board::{Board, Move, WallOrientation};
use crate::util::grid::{has_wall, square_index};

use super::runtime::tables;

/// Layer 1: 0=opponent absent, 1=up, 2=down, 3=left, 4=right (edge-invalid → 0).
pub fn encode_enemy_key(board: &Board, side: usize, sq: u8) -> u8 {
    let sr = sq / 9;
    let sc = sq % 9;
    let (or, oc) = board.pawns[1 - side];
    let dr = or as i8 - sr as i8;
    let dc = oc as i8 - sc as i8;
    let ek = match (dr, dc) {
        (-1, 0) => 1,
        (1, 0) => 2,
        (0, -1) => 3,
        (0, 1) => 4,
        _ => return 0,
    };
    if tables().layer_valid[sq as usize][ek as usize] == 0 {
        0
    } else {
        ek
    }
}

/// Pack physical wall combo (up to 12 local slots), remap to 8-bit semantic key.
///
/// Fast path: 2× PEXT instructions (BMI2, enabled by `-C target-feature=+bmi2`).
/// Fallback: scalar loop over per-slot descriptor tables.
pub fn pack_wall_key(board: &Board, sq: u8, enemy_key: u8) -> u8 {
    #[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
    return unsafe { pack_wall_key_pext(board, sq, enemy_key) };
    #[cfg(not(all(target_arch = "x86_64", target_feature = "bmi2")))]
    pack_wall_key_scalar(board, sq, enemy_key)
}

#[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
#[target_feature(enable = "bmi2")]
unsafe fn pack_wall_key_pext(board: &Board, sq: u8, enemy_key: u8) -> u8 {
    use std::arch::x86_64::_pext_u64;
    let t = tables();
    let si = sq as usize;
    let ei = enemy_key as usize;
    let h_bits = _pext_u64(board.horizontal_walls, t.h_pext_mask[si][ei]) as usize;
    let v_bits = _pext_u64(board.vertical_walls, t.v_pext_mask[si][ei]) as usize;
    let phys = h_bits | (v_bits << t.h_slot_count[si][ei]);
    t.wall_remap_byte(sq, enemy_key, phys)
}

fn pack_wall_key_scalar(board: &Board, sq: u8, enemy_key: u8) -> u8 {
    let t = tables();
    let nw = t.wall_slot_count[sq as usize][enemy_key as usize] as usize;
    let mut phys = 0usize;
    for i in 0..nw {
        let r = t.desc_row[sq as usize][enemy_key as usize][i];
        let c = t.desc_col[sq as usize][enemy_key as usize][i];
        let h = t.desc_h[sq as usize][enemy_key as usize][i] != 0;
        let orient = if h {
            WallOrientation::Horizontal
        } else {
            WallOrientation::Vertical
        };
        if has_wall(board, r, c, orient) {
            phys |= 1 << i;
        }
    }
    t.wall_remap_byte(sq, enemy_key, phys)
}

#[inline]
pub fn legal_pawn_move_mask(board: &Board, side: usize, sq: u8) -> u16 {
    let enemy_key = encode_enemy_key(board, side, sq);
    let t = tables();
    if t.layer_valid[sq as usize][enemy_key as usize] == 0 {
        return 0;
    }
    let wall_key = pack_wall_key(board, sq, enemy_key);
    let max = t.wall_combo_count[sq as usize][enemy_key as usize] as usize;
    if wall_key as usize >= max {
        return 0;
    }
    t.legal[sq as usize][enemy_key as usize][wall_key as usize]
}

/// Lean LUT: O1 table only when enemy is adjacent (ek≠0); plain `can_step` otherwise.
///
/// Rationale: ek=0 (no adjacent enemy) covers ~95 % of positions and needs only
/// ≤4 `can_step` calls. Skipping the PEXT + remap + table lookup for that common
/// case should be faster than the full O1 path.
pub fn generate_pawn_moves_lean_lut(board: &Board, out: &mut [Move]) -> usize {
    let side = board.side_to_move as usize;
    let (fr, fc) = board.pawns[side];
    let sq = square_index(fr, fc);
    let enemy_key = encode_enemy_key(board, side, sq);
    if enemy_key == 0 {
        crate::movegen::pawn_bits::generate_pawn_moves_shift_slice(board, out)
    } else {
        let t = tables();
        let wall_key = pack_wall_key(board, sq, enemy_key);
        let max = t.wall_combo_count[sq as usize][enemy_key as usize] as usize;
        if wall_key as usize >= max {
            return 0;
        }
        let mask = t.legal[sq as usize][enemy_key as usize][wall_key as usize];
        let catalog = &t.catalog[sq as usize];
        let mut n = 0usize;
        let mut bits = mask;
        while bits != 0 {
            let slot = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            let dest = catalog[slot];
            if dest == 255 {
                continue;
            }
            out[n] = Move::Pawn {
                row: dest / 9,
                col: dest % 9,
            };
            n += 1;
        }
        n
    }
}

pub fn generate_pawn_moves_o1(board: &Board, out: &mut [Move]) -> usize {
    let side = board.side_to_move as usize;
    let (fr, fc) = board.pawns[side];
    let sq = square_index(fr, fc);
    let mask = legal_pawn_move_mask(board, side, sq);
    let catalog = &tables().catalog[sq as usize];
    let mut n = 0usize;
    let mut bits = mask;
    while bits != 0 {
        let slot = bits.trailing_zeros() as usize;
        bits &= bits - 1;
        let dest = catalog[slot];
        if dest == 255 {
            continue;
        }
        out[n] = Move::Pawn {
            row: dest / 9,
            col: dest % 9,
        };
        n += 1;
    }
    n
}

/// L2: passes overlap / cross / neighbor collision rules (`wall_collides` inverse).
#[inline]
pub fn wall_physically_legal_o1(board: &Board, row: u8, col: u8, horizontal: bool) -> bool {
    let masks = wall_masks(board);
    let mask = if horizontal { masks.l12_h } else { masks.l12_v };
    (mask >> ((row as u64) * 8 + col as u64)) & 1 != 0
}

// --- shift helpers (L2 collision + topo flood-skip) ---

const COL_0: u64 = 0x0101_0101_0101_0101;
const COL_7: u64 = COL_0 << 7;
const ROW_0: u64 = 0xFF;
const ROW_7: u64 = 0xFF << 56;

#[inline]
fn east1(x: u64) -> u64 {
    (x << 1) & !COL_0
}

#[inline]
fn east2(x: u64) -> u64 {
    (x << 2) & !(COL_0 | COL_0 << 1)
}

#[inline]
fn west1(x: u64) -> u64 {
    (x >> 1) & !COL_7
}

#[inline]
fn west2(x: u64) -> u64 {
    (x >> 2) & !(COL_7 | COL_7 >> 1)
}

#[inline]
fn two_of_three(a: u64, b: u64, m: u64) -> u64 {
    (a & b) | (m & (a | b))
}

#[inline]
fn topo_h_from(h: u64, v: u64) -> u64 {
    let side_a = COL_0 | east1(v) | east1(v >> 8) | east1(v << 8) | east2(h);
    let side_b = COL_7 | west1(v) | west1(v >> 8) | west1(v << 8) | west2(h);
    let middle = (v >> 8) | (v << 8);
    two_of_three(side_a, side_b, middle)
}

#[inline]
fn topo_v_from(h: u64, v: u64) -> u64 {
    let side_a = ROW_7 | (h >> 8) | east1(h >> 8) | west1(h >> 8) | (v >> 16);
    let side_b = ROW_0 | (h << 8) | east1(h << 8) | west1(h << 8) | (v << 16);
    let middle = east1(h) | west1(h);
    two_of_three(side_a, side_b, middle)
}

/// All wall candidate masks for one node — single read of the two wall bitboards.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WallMasks {
    pub l12_h: u64,
    pub l12_v: u64,
    pub topo_h: u64,
    pub topo_v: u64,
}

#[inline]
pub fn wall_masks(board: &Board) -> WallMasks {
    let h = board.horizontal_walls;
    let v = board.vertical_walls;
    let coll_h = !(h | v | east1(h) | west1(h));
    let coll_v = !(v | h | (v << 8) | (v >> 8));
    WallMasks {
        l12_h: !h & coll_h,
        l12_v: !v & coll_v,
        topo_h: topo_h_from(h, v),
        topo_v: topo_v_from(h, v),
    }
}

#[inline]
pub fn wall_collision_clear_h_mask(board: &Board) -> u64 {
    let h = board.horizontal_walls;
    !(h | board.vertical_walls | east1(h) | west1(h))
}

#[inline]
pub fn wall_collision_clear_v_mask(board: &Board) -> u64 {
    let v = board.vertical_walls;
    !(v | board.horizontal_walls | (v << 8) | (v >> 8))
}

pub fn wall_l12_h_mask(board: &Board) -> u64 {
    wall_masks(board).l12_h
}

pub fn wall_l12_v_mask(board: &Board) -> u64 {
    wall_masks(board).l12_v
}

#[inline]
pub fn wall_needs_flood_h_mask(board: &Board) -> u64 {
    topo_h_from(board.horizontal_walls, board.vertical_walls)
}

#[inline]
pub fn wall_needs_flood_v_mask(board: &Board) -> u64 {
    topo_v_from(board.horizontal_walls, board.vertical_walls)
}

pub fn generate_wall_candidates_o1(board: &Board, horizontal: bool, out: &mut [(u8, u8)]) -> usize {
    let bits = if horizontal {
        wall_l12_h_mask(board)
    } else {
        wall_l12_v_mask(board)
    };
    let mut n = 0usize;
    let mut free = bits;
    while free != 0 {
        let bit = free.trailing_zeros();
        free &= free - 1;
        out[n] = ((bit / 8) as u8, (bit % 8) as u8);
        n += 1;
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::board::{Board, Player};
    use crate::movegen::legal::generate_pawn_moves_scalar_for;

    fn scalar_mask(board: &Board, player: Player, sq: u8) -> u16 {
        let mut moves = [Move::Pawn { row: 0, col: 0 }; 8];
        let n = generate_pawn_moves_scalar_for(board, player, &mut moves);
        let catalog = &tables().catalog[sq as usize];
        let mut mask = 0u16;
        for m in &moves[..n] {
            if let Move::Pawn { row, col } = m {
                let d = square_index(*row, *col);
                for (slot, &sq_id) in catalog.iter().enumerate() {
                    if sq_id != 255 && sq_id == d {
                        mask |= 1 << slot;
                        break;
                    }
                }
            }
        }
        mask
    }

    #[test]
    fn o1_pawn_matches_scalar_startpos() {
        let b = Board::new();
        for player in [Player::One, Player::Two] {
            let side = player as usize;
            let sq = square_index(b.pawns[side].0, b.pawns[side].1);
            assert_eq!(
                legal_pawn_move_mask(&b, side, sq),
                scalar_mask(&b, player, sq),
                "{player:?}"
            );
        }
    }

    #[test]
    fn o1_pawn_matches_scalar_walls() {
        let mut b = Board::new();
        b.horizontal_walls = 0x00_00_0A_00_14_00;
        b.vertical_walls = 0x01_02_04_00;
        for player in [Player::One, Player::Two] {
            let side = player as usize;
            let sq = square_index(b.pawns[side].0, b.pawns[side].1);
            assert_eq!(
                legal_pawn_move_mask(&b, side, sq),
                scalar_mask(&b, player, sq)
            );
        }
    }

    #[test]
    fn all_wall_slots_fit_8_bits() {
        let t = tables();
        for sq in 0u8..81 {
            for ek in 0u8..5 {
                if t.layer_valid[sq as usize][ek as usize] == 0 {
                    continue;
                }
                let max = t.wall_combo_count[sq as usize][ek as usize];
                assert!(
                    max <= 256,
                    "sq {sq} enemy {ek}: {max} combos need >8 wall bits"
                );
                let nw = t.wall_slot_count[sq as usize][ek as usize];
                assert!(nw <= 12, "sq {sq} enemy {ek}: {nw} wall slots");
            }
        }
    }

    #[test]
    fn wall_physical_matches_scalar_collides() {
        let b = Board::new();
        for hr in 0..8u8 {
            for hc in 0..8u8 {
                let o1 = wall_physically_legal_o1(&b, hr, hc, true);
                let scalar = !crate::movegen::legal::wall_collides_test(
                    &b,
                    hr,
                    hc,
                    WallOrientation::Horizontal,
                );
                assert_eq!(o1, scalar, "h {hr},{hc}");
            }
        }
    }

    fn scalar_collision_clear_h_mask(board: &Board) -> u64 {
        let mut m = 0u64;
        for r in 0..8u8 {
            for c in 0..8u8 {
                if !crate::movegen::legal::wall_collides_test(
                    board,
                    r,
                    c,
                    WallOrientation::Horizontal,
                ) {
                    m |= 1 << ((r as u64) * 8 + c as u64);
                }
            }
        }
        m
    }

    #[test]
    fn wall_masks_agrees_with_split_masks() {
        let boards = [
            Board::new(),
            {
                let mut b = Board::new();
                b.horizontal_walls = 0x00_00_0A_00_14_00;
                b.vertical_walls = 0x01_02_04_00;
                b
            },
        ];
        for b in &boards {
            let m = wall_masks(b);
            assert_eq!(m.l12_h, !b.horizontal_walls & wall_collision_clear_h_mask(b));
            assert_eq!(m.l12_v, !b.vertical_walls & wall_collision_clear_v_mask(b));
            assert_eq!(m.topo_h, wall_needs_flood_h_mask(b));
            assert_eq!(m.topo_v, wall_needs_flood_v_mask(b));
        }
    }

    #[test]
    fn collision_clear_mask_matches_scalar() {
        let boards = [
            Board::new(),
            {
                let mut b = Board::new();
                b.horizontal_walls = 0x00_00_0A_00_14_00;
                b.vertical_walls = 0x01_02_04_00;
                b
            },
        ];
        for b in &boards {
            assert_eq!(
                wall_collision_clear_h_mask(b),
                scalar_collision_clear_h_mask(b)
            );
        }
    }

    fn scalar_topo_h_mask(board: &Board) -> u64 {
        let mut m = 0u64;
        for r in 0..8u8 {
            for c in 0..8u8 {
                if crate::movegen::legal::can_wall_block_topology(
                    board,
                    r,
                    c,
                    WallOrientation::Horizontal,
                ) {
                    m |= 1 << ((r as u64) * 8 + c as u64);
                }
            }
        }
        m
    }

    fn scalar_topo_v_mask(board: &Board) -> u64 {
        let mut m = 0u64;
        for r in 0..8u8 {
            for c in 0..8u8 {
                if crate::movegen::legal::can_wall_block_topology(
                    board,
                    r,
                    c,
                    WallOrientation::Vertical,
                ) {
                    m |= 1 << ((r as u64) * 8 + c as u64);
                }
            }
        }
        m
    }

    #[test]
    fn topo_needs_flood_matches_scalar() {
        let boards = [
            Board::new(),
            {
                let mut b = Board::new();
                b.horizontal_walls = 0x00_00_0A_00_14_00;
                b.vertical_walls = 0x01_02_04_00;
                b
            },
            {
                let mut b = Board::new();
                b.horizontal_walls = 0xFF_FF_FF_FF_FF_FF;
                b.vertical_walls = 0xFF_FF_FF_FF_FF_FF;
                b
            },
        ];
        for b in &boards {
            assert_eq!(wall_needs_flood_h_mask(b), scalar_topo_h_mask(b), "h topo");
            assert_eq!(wall_needs_flood_v_mask(b), scalar_topo_v_mask(b), "v topo");
        }
    }

    #[test]
    fn topo_needs_flood_exhaustive_low_wall_count() {
        for hw in 0u64..64 {
            for vw in 0u64..64 {
                if hw.count_ones() + vw.count_ones() > 6 {
                    continue;
                }
                let mut b = Board::new();
                b.horizontal_walls = hw;
                b.vertical_walls = vw;
                assert_eq!(
                    wall_needs_flood_h_mask(&b),
                    scalar_topo_h_mask(&b),
                    "hw={hw:#x} vw={vw:#x} h"
                );
                assert_eq!(
                    wall_needs_flood_v_mask(&b),
                    scalar_topo_v_mask(&b),
                    "hw={hw:#x} vw={vw:#x} v"
                );
            }
        }
    }

    /// Verify PEXT and scalar fallback agree on every (sq, enemy_key) for two board
    /// positions. Runs on BMI2 builds only — skips silently on scalar-only builds.
    #[test]
    fn pext_pack_wall_key_matches_scalar() {
        #[cfg(not(all(target_arch = "x86_64", target_feature = "bmi2")))]
        {
            return;
        }
        #[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
        {
            let boards = [
                Board::new(),
                {
                    let mut b = Board::new();
                    b.horizontal_walls = 0x00_00_0A_00_14_00;
                    b.vertical_walls = 0x01_02_04_00;
                    b
                },
                {
                    let mut b = Board::new();
                    b.horizontal_walls = 0xFF_FF_FF_00_00_00;
                    b.vertical_walls = 0x00_00_00_FF_FF_FF;
                    b
                },
            ];
            let t = tables();
            for b in &boards {
                for sq in 0u8..81 {
                    for ek in 0u8..5 {
                        if t.layer_valid[sq as usize][ek as usize] == 0 {
                            continue;
                        }
                        let pext = unsafe { pack_wall_key_pext(b, sq, ek) };
                        let scalar = pack_wall_key_scalar(b, sq, ek);
                        assert_eq!(
                            pext, scalar,
                            "sq={sq} ek={ek} hw={:#x} vw={:#x}",
                            b.horizontal_walls, b.vertical_walls
                        );
                    }
                }
            }
        }
    }
}
