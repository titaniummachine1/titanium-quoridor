//! Two-layer pawn tables: `[sq][enemy_key][wall_key] → legal subset of 12 pseudo-moves`.
//!
//! Layer 1 — `enemy_key` (5): opponent absent or on one cardinal (edge-aware).
//! Layer 2 — `wall_key` (u8): semantic key (≤256 distinct move masks).
//! Up to 12 physical wall slots pack into a combo index, then remap → 8-bit key.

use super::geometry::{pawn_move_dests, set_wall, square_index, MiniBoard};
use super::pseudo_moves::{legal_subset_mask, pseudo_catalog, CARDINAL_OFFSETS, PSEUDO_SLOTS};
use std::collections::HashMap;

pub const ENEMY_LAYERS: usize = 5;
pub const WALL_KEYS: usize = 256;
/// Physical wall slots read from the board per layer (can exceed 8).
pub const MAX_WALL_SLOTS: usize = 12;
pub const PHYS_WALL_COMBOS: usize = 1 << MAX_WALL_SLOTS;

pub struct PawnSquareMeta {
    pub catalog: [u8; PSEUDO_SLOTS],
    pub layers: [EnemyLayerMeta; ENEMY_LAYERS],
}

pub struct EnemyLayerMeta {
    pub enemy_key: u8,
    pub valid: bool,
    pub wall_bits: Vec<(u8, u8, bool)>,
    /// Distinct semantic wall keys (≤256).
    pub wall_combo_count: u16,
    pub table: [u16; WALL_KEYS],
    /// Physical combo (2^nw) → semantic `wall_key` (length = 2^nw, padded to 4096 in emit).
    pub wall_remap: Vec<u8>,
}

/// Discover all pawn lookup tables in memory (silent — no progress bar).
/// Single source of truth for both the offline emitter (`movegen-o1-gen`) and
/// the runtime cold-start builder (`super::runtime`).
pub fn discover_all_pawn_tables() -> Vec<PawnSquareMeta> {
    let mut result = Vec::with_capacity(81);
    for sq in 0..81u8 {
        result.push(discover_pawn_square(sq));
    }
    result
}

pub fn discover_pawn_square(sq: u8) -> PawnSquareMeta {
    let (sr, sc) = (sq / 9, sq % 9);
    let catalog = pseudo_catalog(sr, sc);
    let layers = std::array::from_fn(|k| discover_enemy_layer(sr, sc, sq, k as u8, &catalog));
    PawnSquareMeta { catalog, layers }
}

fn discover_enemy_layer(
    sr: u8,
    sc: u8,
    sq: u8,
    enemy_key: u8,
    catalog: &[u8; PSEUDO_SLOTS],
) -> EnemyLayerMeta {
    if enemy_key > 0 && !cardinal_valid(sr, sc, enemy_key) {
        return EnemyLayerMeta {
            enemy_key,
            valid: false,
            wall_bits: Vec::new(),
            wall_combo_count: 0,
            table: [0u16; WALL_KEYS],
            wall_remap: Vec::new(),
        };
    }

    // Sort H walls before V walls so PEXT extraction order (H mask then V mask)
    // matches the bit-index ordering used by the remap table.
    let mut wall_bits = essential_wall_slots(sr, sc, sq, enemy_key, catalog);
    wall_bits.sort_by_key(|&(r, c, h)| (!h, r, c));
    let nw = wall_bits.len();
    assert!(
        nw <= MAX_WALL_SLOTS,
        "sq {sq} enemy_key {enemy_key}: {nw} wall slots > {MAX_WALL_SLOTS}"
    );

    let phys_combos = 1usize << nw;
    let mut table = [0u16; WALL_KEYS];
    let mut wall_remap = vec![0u8; phys_combos];
    let mut mask_to_key: HashMap<u16, u8> = HashMap::new();
    let mut next_key = 0u8;

    for phys in 0..phys_combos {
        let mask = mask_for_wall_part(sr, sc, sq, enemy_key, catalog, &wall_bits, phys);
        let key = *mask_to_key.entry(mask).or_insert_with(|| {
            assert!(
                (next_key as usize) < WALL_KEYS,
                "sq {sq} enemy_key {enemy_key}: >{WALL_KEYS} distinct wall masks (nw={nw})"
            );
            let k = next_key;
            table[k as usize] = mask;
            next_key += 1;
            k
        });
        wall_remap[phys] = key;
    }

    EnemyLayerMeta {
        enemy_key,
        valid: true,
        wall_bits,
        wall_combo_count: next_key as u16,
        table,
        wall_remap,
    }
}

/// 0=absent, 1=up(row-), 2=down(row+), 3=left, 4=right.
pub fn cardinal_valid(sr: u8, sc: u8, enemy_key: u8) -> bool {
    enemy_offset(enemy_key).is_some_and(|(dr, dc)| {
        let nr = sr as i16 + dr as i16;
        let nc = sc as i16 + dc as i16;
        (0..=8).contains(&nr) && (0..=8).contains(&nc)
    })
}

pub fn cardinal_offset(enemy_key: u8) -> Option<(i8, i8)> {
    match enemy_key {
        0 => None,
        1 => Some(CARDINAL_OFFSETS[1]), // (-1, 0) up
        2 => Some(CARDINAL_OFFSETS[0]), // (1, 0) down
        3 => Some(CARDINAL_OFFSETS[3]), // (0, -1) left
        4 => Some(CARDINAL_OFFSETS[2]), // (0, 1) right
        _ => None,
    }
}

fn enemy_offset(enemy_key: u8) -> Option<(i8, i8)> {
    cardinal_offset(enemy_key)
}

/// Minimal wall set: drop slots that never change the legal mask.
fn essential_wall_slots(
    sr: u8,
    sc: u8,
    sq: u8,
    enemy_key: u8,
    catalog: &[u8; PSEUDO_SLOTS],
) -> Vec<(u8, u8, bool)> {
    let mut walls = wall_candidates(sr, sc, enemy_key);
    loop {
        let before = walls.len();
        for idx in (0..walls.len()).rev() {
            if !wall_is_essential(sr, sc, sq, enemy_key, catalog, &walls, idx) {
                walls.remove(idx);
            }
        }
        if walls.len() == before {
            break;
        }
    }
    walls
}

fn wall_is_essential(
    sr: u8,
    sc: u8,
    sq: u8,
    enemy_key: u8,
    catalog: &[u8; PSEUDO_SLOTS],
    walls: &[(u8, u8, bool)],
    wall_idx: usize,
) -> bool {
    let nw = walls.len();
    let combos = 1usize << (nw - 1);
    for part in 0..combos {
        let mut off = 0usize;
        let mut o = 0usize;
        for i in 0..nw {
            if i == wall_idx {
                continue;
            }
            if (part >> o) & 1 != 0 {
                off |= 1 << i;
            }
            o += 1;
        }
        let on = off | (1 << wall_idx);
        let m0 = mask_for_wall_part(sr, sc, sq, enemy_key, catalog, walls, off);
        let m1 = mask_for_wall_part(sr, sc, sq, enemy_key, catalog, walls, on);
        if m0 != m1 {
            return true;
        }
    }
    false
}

fn mask_for_wall_part(
    sr: u8,
    sc: u8,
    sq: u8,
    enemy_key: u8,
    catalog: &[u8; PSEUDO_SLOTS],
    walls: &[(u8, u8, bool)],
    wall_part: usize,
) -> u16 {
    let board = board_for_layer(sr, sc, enemy_key, walls, wall_part);
    let (dests, dn) = pawn_move_dests(&board, 0, sq);
    legal_subset_mask(catalog, &dests[..dn], dn)
}

/// Local wall coords that `can_step` may consult for this mover + relative opponent.
fn wall_candidates(sr: u8, sc: u8, enemy_key: u8) -> Vec<(u8, u8, bool)> {
    let mut set = std::collections::BTreeSet::new();
    for &(dr, dc) in &CARDINAL_OFFSETS {
        let nr = sr as i16 + dr as i16;
        let nc = sc as i16 + dc as i16;
        if (0..=8).contains(&nr) && (0..=8).contains(&nc) {
            push_step_walls(sr, sc, dr, dc, &mut set);
        }
    }
    if let Some((dr, dc)) = enemy_offset(enemy_key) {
        let or = (sr as i8 + dr) as u8;
        let oc = (sc as i8 + dc) as u8;
        push_step_walls(or, oc, dr, dc, &mut set);
        let perp = if dr != 0 {
            [(0i8, 1i8), (0, -1)]
        } else {
            [(1, 0), (-1, 0)]
        };
        for (pdr, pdc) in perp {
            let nr = or as i16 + pdr as i16;
            let nc = oc as i16 + pdc as i16;
            if (0..=8).contains(&nr) && (0..=8).contains(&nc) {
                push_step_walls(or, oc, pdr, pdc, &mut set);
            }
        }
    }
    set.into_iter().collect()
}

fn push_step_walls(
    row: u8,
    col: u8,
    dr: i8,
    dc: i8,
    out: &mut std::collections::BTreeSet<(u8, u8, bool)>,
) {
    let nr = row as i16 + dr as i16;
    let nc = col as i16 + dc as i16;
    if !(0..=8).contains(&nr) || !(0..=8).contains(&nc) {
        return;
    }
    let nr = nr as u8;
    let nc = nc as u8;
    let js_from = row + 1;
    let js_to = nr + 1;

    match (dr, dc) {
        (1, 0) => {
            insert_h(out, js_from, col);
            if col > 0 {
                insert_h(out, js_from, col - 1);
            }
        }
        (-1, 0) => {
            insert_h(out, js_to, col);
            if col > 0 {
                insert_h(out, js_to, col - 1);
            }
        }
        (0, 1) => {
            insert_v(out, js_from, col);
            insert_v(out, row, col);
        }
        (0, -1) => {
            insert_v(out, js_to, nc);
            insert_v(out, nr, nc);
        }
        _ => {}
    }
}

fn insert_h(set: &mut std::collections::BTreeSet<(u8, u8, bool)>, js_row: u8, col: u8) {
    if (1..=8).contains(&js_row) && col < 8 {
        set.insert((js_row - 1, col, true));
    }
}

fn insert_v(set: &mut std::collections::BTreeSet<(u8, u8, bool)>, js_row: u8, col: u8) {
    if (1..=8).contains(&js_row) && col < 8 {
        set.insert((js_row - 1, col, false));
    }
}

/// Mover at `(sr,sc)` (always `pawns[0]`); opponent from `enemy_key` (always `pawns[1]`).
/// Side at runtime is irrelevant — only relative geometry matters.
fn board_for_layer(
    sr: u8,
    sc: u8,
    enemy_key: u8,
    wall_bits: &[(u8, u8, bool)],
    wall_part: usize,
) -> MiniBoard {
    let mut b = MiniBoard::default();
    b.pawns[0] = (sr, sc);

    for (i, &(r, c, h)) in wall_bits.iter().enumerate() {
        if (wall_part >> i) & 1 != 0 {
            set_wall(&mut b, r, c, h);
        }
    }

    b.pawns[1] = match enemy_offset(enemy_key) {
        None => far_opponent(sr, sc),
        Some((dr, dc)) => ((sr as i8 + dr) as u8, (sc as i8 + dc) as u8),
    };
    b
}

fn far_opponent(sr: u8, sc: u8) -> (u8, u8) {
    let candidates = [(8u8, 4u8), (0, 4), (4, 0), (4, 8), (0, 0), (8, 8)];
    for (or, oc) in candidates {
        if (or, oc) == (sr, sc) {
            continue;
        }
        let mut adj = false;
        for &(dr, dc) in &CARDINAL_OFFSETS {
            if (sr as i8 + dr, sc as i8 + dc) == (or as i8, oc as i8) {
                adj = true;
                break;
            }
        }
        if !adj {
            return (or, oc);
        }
    }
    if square_index(sr, sc) < 40 {
        (8, 8)
    } else {
        (0, 0)
    }
}
