//! Minimal Quoridor geometry for offline table discovery (mirrors `util::grid`).

#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub struct MiniBoard {
    pub pawns: [(u8, u8); 2],
    pub horizontal_walls: u64,
    pub vertical_walls: u64,
}

#[inline]
pub fn square_index(row: u8, col: u8) -> u8 {
    row * 9 + col
}

#[inline]
fn has_horizontal(b: &MiniBoard, js_row: u8, col: u8) -> bool {
    if !(1..=8).contains(&js_row) || col >= 8 {
        return false;
    }
    let bit = ((js_row - 1) as u32) * 8 + col as u32;
    (b.horizontal_walls >> bit) & 1 != 0
}

#[inline]
fn has_vertical(b: &MiniBoard, js_row: u8, col: u8) -> bool {
    if !(1..=8).contains(&js_row) || col >= 8 {
        return false;
    }
    let bit = ((js_row - 1) as u32) * 8 + col as u32;
    (b.vertical_walls >> bit) & 1 != 0
}

#[inline]
pub fn can_step(b: &MiniBoard, row: u8, col: u8, dr: i8, dc: i8) -> bool {
    let nr = row as i16 + dr as i16;
    let nc = col as i16 + dc as i16;
    if !(0..=8).contains(&nr) || !(0..=8).contains(&nc) {
        return false;
    }
    let nr = nr as u8;
    let nc = nc as u8;
    let js_from = row + 1;
    let js_to = nr + 1;

    match (dr, dc) {
        (1, 0) => {
            !has_horizontal(b, js_from, col)
                && (col == 0 || !has_horizontal(b, js_from, col - 1))
        }
        (-1, 0) => {
            !has_horizontal(b, js_to, col)
                && (col == 0 || !has_horizontal(b, js_to, col - 1))
        }
        (0, 1) => !has_vertical(b, js_from, col) && !has_vertical(b, row, col),
        (0, -1) => !has_vertical(b, js_to, nc) && !has_vertical(b, nr, nc),
        _ => false,
    }
}

const DIRS: [(i8, i8); 4] = [(1, 0), (0, 1), (-1, 0), (0, -1)];

/// Destination squares (0..80) for legal pawn moves from `from_sq`.
pub fn pawn_move_dests(b: &MiniBoard, side: usize, from_sq: u8) -> ([u8; 8], usize) {
    let (fr, fc) = b.pawns[side];
    if square_index(fr, fc) != from_sq {
        return ([0; 8], 0);
    }
    let (or, oc) = b.pawns[1 - side];
    let mut dests = [0u8; 8];
    let mut n = 0usize;

    for (dr, dc) in DIRS {
        if !can_step(b, fr, fc, dr, dc) {
            continue;
        }
        let nr = (fr as i8 + dr) as u8;
        let nc = (fc as i8 + dc) as u8;

        if (nr, nc) != (or, oc) {
            dests[n] = square_index(nr, nc);
            n += 1;
            continue;
        }

        if can_step(b, nr, nc, dr, dc) {
            let jr = (nr as i8 + dr) as u8;
            let jc = (nc as i8 + dc) as u8;
            dests[n] = square_index(jr, jc);
            n += 1;
            continue;
        }

        let perp = if dr != 0 {
            [(0i8, 1i8), (0, -1)]
        } else {
            [(1, 0), (-1, 0)]
        };
        for (pdr, pdc) in perp {
            if can_step(b, nr, nc, pdr, pdc) {
                let sr = (nr as i8 + pdr) as u8;
                let sc = (nc as i8 + pdc) as u8;
                dests[n] = square_index(sr, sc);
                n += 1;
            }
        }
    }
    (dests, n)
}

pub fn set_wall(b: &mut MiniBoard, row: u8, col: u8, horizontal: bool) {
    let bit = (row as u64) * 8 + col as u64;
    if horizontal {
        b.horizontal_walls |= 1 << bit;
    } else {
        b.vertical_walls |= 1 << bit;
    }
}

