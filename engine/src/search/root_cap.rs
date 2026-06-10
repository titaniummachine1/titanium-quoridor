//! Root wall cap — keep all pawns, hottest `max_walls` by CAT edge heat.

use crate::cat::CorridorAttention;
use crate::core::board::Move;
use crate::movegen::MAX_LEGAL_MOVES;

/// Indices of walls kept when capping to `max_walls` hottest by CAT (pawns always kept).
pub fn root_wall_keep_mask(
    buf: &[Move],
    n: usize,
    cat: &CorridorAttention,
    max_walls: usize,
) -> [bool; MAX_LEGAL_MOVES] {
    let mut keep = [true; MAX_LEGAL_MOVES];
    let mut ranked = [(0usize, 0u16); MAX_LEGAL_MOVES];
    let mut wall_count = 0usize;
    for i in 0..n {
        if let Move::Wall {
            row,
            col,
            orientation,
        } = buf[i]
        {
            ranked[wall_count] = (i, cat.wall_edge_heat(row, col, orientation));
            wall_count += 1;
            keep[i] = false;
        }
    }
    if wall_count <= max_walls {
        return keep;
    }
    ranked[..wall_count].sort_by(|a, b| b.1.cmp(&a.1));
    for i in 0..n {
        if matches!(buf[i], Move::Pawn { .. }) {
            keep[i] = true;
        }
    }
    for &(i, _) in &ranked[..max_walls] {
        keep[i] = true;
    }
    keep
}

/// Keep every pawn; retain only the hottest `max_walls` walls by CAT edge heat.
pub fn cap_root_wall_moves(buf: &mut [Move], n: &mut usize, cat: &CorridorAttention, max_walls: usize) {
    if *n == 0 {
        return;
    }
    let mut ranked = [(0usize, 0u16); MAX_LEGAL_MOVES];
    let mut wall_count = 0usize;
    for i in 0..*n {
        if let Move::Wall {
            row,
            col,
            orientation,
        } = buf[i]
        {
            ranked[wall_count] = (i, cat.wall_edge_heat(row, col, orientation));
            wall_count += 1;
        }
    }
    if wall_count <= max_walls {
        return;
    }
    ranked[..wall_count].sort_by(|a, b| b.1.cmp(&a.1));
    let mut keep = [false; MAX_LEGAL_MOVES];
    for &(i, _) in &ranked[..max_walls] {
        keep[i] = true;
    }
    let mut out = 0usize;
    for i in 0..*n {
        if matches!(buf[i], Move::Pawn { .. }) || keep[i] {
            buf[out] = buf[i];
            out += 1;
        }
    }
    *n = out;
}
