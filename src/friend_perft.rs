//! Validate the engine's movegen against canta's reference perft suite.
//!
//! canta sent 15 games of 15 random `Turn` bytes each plus the perft node
//! counts (depths 1..5) of each resulting position. A `Turn` byte is a C
//! bitfield `{ place_walls:1, orientation:1, index:6 }`. The pawn `index`
//! (0..16) decodes to a destination via `get_new_position` (offsets below);
//! a wall `index` is 0..64 with orientation 0=horizontal, 1=vertical.
//!
//! The friend's coordinate system / wall numbering / move order are unknown,
//! so we brute-force the board symmetry and a few bit conventions, using
//! perft(1) (legal-move count) across all 15 games as the discriminator, then
//! confirm the unique winner on depths 1..5.
//!
//! Run: `titanium friend-perft`

use crate::ace::game::AceGame;

#[rustfmt::skip]
const TURN_BYTES: [[u8; 15]; 15] = [
  [0x8d,0x9d,0x0b,0x05,0x5f,0xd5,0x19,0x81,0x33,0xc3,0x27,0x9b,0x49,0xcd,0xbd],
  [0xc5,0x87,0xbb,0x23,0xf9,0x95,0x69,0x0f,0x6f,0x61,0x7b,0x59,0xd9,0xd3,0x35],
  [0x2d,0x77,0x85,0x69,0x01,0x43,0xd1,0x9f,0x91,0xdd,0x45,0x1d,0xc1,0xf5,0x3d],
  [0xfb,0xa1,0x67,0xfd,0x3f,0x73,0x45,0xbb,0xc5,0xb7,0x39,0xcd,0x75,0x03,0x69],
  [0x3d,0x7d,0x17,0x07,0x73,0xc1,0x3b,0x97,0x99,0x67,0xf3,0xcd,0x5f,0x6d,0xfb],
  [0x95,0x45,0x81,0xd9,0x8d,0xb9,0x13,0x2b,0x9f,0xa1,0xa9,0x07,0x3d,0x3c,0x19],
  [0x9f,0xcb,0x29,0xfd,0xe7,0xdd,0x55,0x7d,0xd5,0x05,0x99,0x87,0x03,0xc5,0x61],
  [0x0c,0xaf,0xa5,0xe9,0xc9,0xdf,0x2f,0x55,0x07,0xd1,0x01,0x2c,0x7f,0x8d,0x5d],
  [0x33,0xcb,0xcd,0xfb,0x23,0x83,0x0d,0x89,0x51,0x2c,0x95,0x65,0x35,0x71,0xef],
  [0x8b,0xbd,0xf9,0x03,0xc7,0x9f,0x35,0x77,0x87,0x2b,0x83,0x55,0xbb,0x8d,0x49],
  [0xe1,0xcd,0x9f,0x03,0x2d,0x77,0x73,0x5d,0xdb,0x41,0x05,0x1b,0x9b,0x0d,0xf5],
  [0x6f,0x9f,0xcb,0x09,0xf1,0x21,0x27,0xcf,0x1d,0x11,0x53,0x7b,0x85,0xd1,0x77],
  [0x3b,0x1f,0xef,0x87,0xb9,0x01,0x4d,0xd1,0x75,0xf5,0xfd,0xdf,0x8d,0x3d,0x57],
  [0x29,0x41,0xfb,0x23,0xd1,0x13,0x17,0x4f,0x7d,0x81,0x5d,0xf7,0x3c,0x1d,0x6d],
  [0x2c,0x33,0x83,0xb1,0x49,0x5f,0x7d,0x51,0xd3,0xa7,0x25,0xa1,0x69,0xab,0x93],
];

#[rustfmt::skip]
const PERFT_REF: [[u64; 5]; 15] = [
  [79,5978,432338,29835465,1959259574],
  [78,5745,409363,27430339,1772319314],
  [77,5697,404581,27529945,1792641073],
  [82,6451,486291,35073417,2415164834],
  [80,6229,460083,32978838,2235598323],
  [82,6365,478510,33921163,2320343986],
  [82,6454,487137,35198588,2431768874],
  [87,7272,583286,44803099,3291865563],
  [80,6064,445703,30975275,2079841087],
  [79,6005,438600,30718458,2058463292],
  [77,5612,396652,26467777,1706612665],
  [74,5259,358646,23414524,1463417984],
  [76,5612,391949,26500883,1688342695],
  [85,6903,535126,39534849,2781386936],
  [84,6794,528318,39429961,2818535671],
];

/// `get_new_position` offsets in the friend's (x, y) frame, index 0..16.
#[rustfmt::skip]
const OFFS: [(i32, i32); 16] = [
  (0,2),(1,1),(-1,1),(0,1),(0,-2),(1,-1),(-1,-1),(0,-1),
  (2,0),(1,1),(1,-1),(1,0),(-2,0),(-1,1),(-1,-1),(-1,0),
];

/// A candidate convention to brute-force.
#[derive(Clone, Copy)]
struct Mapping {
    sym: usize,        // 0..8 board symmetry
    swap_orient: bool, // friend H/V matches engine, or swapped
    msb_index: bool,   // bitfield: index in high bits (true) or low bits (false)
}

/// Decode a Turn byte. Returns (is_wall, orientation, index).
fn decode_byte(byte: u8, msb_index: bool) -> (bool, u8, u8) {
    if msb_index {
        // place_walls = bit7, orientation = bit6, index = bits0..5
        ((byte >> 7) & 1 == 1, (byte >> 6) & 1, byte & 0x3F)
    } else {
        // place_walls = bit0, orientation = bit1, index = bits2..7
        (byte & 1 == 1, (byte >> 1) & 1, (byte >> 2) & 0x3F)
    }
}

/// Map a friend cell (x,y) in 0..9 to an engine cell r*9+c under symmetry `sym`.
fn map_cell(sym: usize, x: i32, y: i32) -> Option<usize> {
    let (r, c) = match sym {
        0 => (8 - y, x),
        1 => (y, x),
        2 => (8 - y, 8 - x),
        3 => (y, 8 - x),
        4 => (8 - x, y),
        5 => (x, y),
        6 => (8 - x, 8 - y),
        7 => (x, 8 - y),
        _ => return None,
    };
    if (0..9).contains(&r) && (0..9).contains(&c) {
        Some((r * 9 + c) as usize)
    } else {
        None
    }
}

/// Map a friend 8x8 wall coordinate (wx,wy) and orientation under `sym`.
/// Returns engine (wall_type, slot) where slot = wr*8+wc, type 0=H 1=V.
fn map_wall(sym: usize, swap_orient: bool, wx: i32, wy: i32, orient: u8) -> Option<(usize, usize)> {
    // Same linear symmetry, but on the 8x8 wall grid (flip uses 7-*).
    let (wr, wc, swaps_axes) = match sym {
        0 => (7 - wy, wx, false),
        1 => (wy, wx, false),
        2 => (7 - wy, 7 - wx, false),
        3 => (wy, 7 - wx, false),
        4 => (7 - wx, wy, true),
        5 => (wx, wy, true),
        6 => (7 - wx, 7 - wy, true),
        7 => (wx, 7 - wy, true),
        _ => return None,
    };
    if !(0..8).contains(&wr) || !(0..8).contains(&wc) {
        return None;
    }
    // axis-swapping symmetries turn horizontal walls into vertical and vice versa
    let mut wt = orient as usize; // 0=H,1=V
    if swaps_axes {
        wt ^= 1;
    }
    if swap_orient {
        wt ^= 1;
    }
    Some((wt, (wr * 8 + wc) as usize))
}

/// Convert a friend turn byte into an engine move code (0..80 pawn target,
/// 100+slot hw, 200+slot vw), validating it is legal in `g`. None = illegal /
/// undecodable under this mapping.
fn decode_move(byte: u8, g: &mut AceGame, m: Mapping) -> Option<i16> {
    let (is_wall, orient, index) = decode_byte(byte, m.msb_index);
    let code: i16 = if is_wall {
        if index >= 64 {
            return None;
        }
        let (wx, wy) = ((index % 8) as i32, (index / 8) as i32);
        let (wt, slot) = map_wall(m.sym, m.swap_orient, wx, wy, orient)?;
        (if wt == 0 { 100 } else { 200 } + slot) as i16
    } else {
        if index >= 16 {
            return None;
        }
        // current pawn (engine cell) -> friend (x,y) is not needed; we apply the
        // offset in friend space then map. Recover friend (x,y) of the mover by
        // inverting is messy, so instead transform the *offset* and add to the
        // engine cell directly (the linear part of the symmetry).
        let (dx, dy) = OFFS[index as usize];
        let (dr, dc) = lin(m.sym, dx, dy);
        let cur = g.pawn[g.turn] as i32;
        let (r, c) = (cur / 9 + dr, cur % 9 + dc);
        if !(0..9).contains(&r) || !(0..9).contains(&c) {
            return None;
        }
        (r * 9 + c) as i16
    };
    // legality check: must be in the generated legal set
    let mut buf = [0i16; 160];
    let n = g.gen_legal_moves(&mut buf);
    if buf[..n].contains(&code) {
        Some(code)
    } else {
        None
    }
}

/// Linear part of the symmetry applied to an offset (dx,dy) -> (dr,dc).
fn lin(sym: usize, dx: i32, dy: i32) -> (i32, i32) {
    match sym {
        0 => (-dy, dx),
        1 => (dy, dx),
        2 => (-dy, -dx),
        3 => (dy, -dx),
        4 => (-dx, dy),
        5 => (dx, dy),
        6 => (-dx, -dy),
        7 => (dx, -dy),
        _ => (dx, dy),
    }
}

fn perft(g: &mut AceGame, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }
    let mut buf = [0i16; 160];
    let n = g.gen_legal_moves(&mut buf);
    if depth == 1 {
        return n as u64;
    }
    let mut nodes = 0u64;
    for i in 0..n {
        g.make_move(buf[i]);
        nodes += perft(g, depth - 1);
        g.unmake_move();
    }
    nodes
}

/// Apply a game's 15 turns under a mapping. Returns the set-up game, or None if
/// any turn is illegal/undecodable.
fn setup(game: usize, m: Mapping) -> Option<AceGame> {
    let mut g = AceGame::new();
    for &byte in &TURN_BYTES[game] {
        let code = decode_move(byte, &mut g, m)?;
        g.make_move(code);
    }
    Some(g)
}

pub fn run() {
    // Brute-force: find mapping(s) where every game's perft(1) matches the ref.
    let mut winners = Vec::new();
    for sym in 0..8 {
        for swap_orient in [false, true] {
            for msb_index in [false, true] {
                let m = Mapping {
                    sym,
                    swap_orient,
                    msb_index,
                };
                let mut all_ok = true;
                for game in 0..15 {
                    match setup(game, m) {
                        Some(mut g) => {
                            if perft(&mut g, 1) != PERFT_REF[game][0] {
                                all_ok = false;
                                break;
                            }
                        }
                        None => {
                            all_ok = false;
                            break;
                        }
                    }
                }
                if all_ok {
                    winners.push(m);
                }
            }
        }
    }

    if winners.is_empty() {
        println!("NO MAPPING matched perft(1) across all 15 games.");
        println!("Per-mapping diagnostics (how many games' d1 matched):");
        for sym in 0..8 {
            for msb_index in [false, true] {
                for swap_orient in [false, true] {
                    let m = Mapping {
                        sym,
                        swap_orient,
                        msb_index,
                    };
                    let mut ok = 0;
                    let mut applied = 0;
                    for game in 0..15 {
                        if let Some(mut g) = setup(game, m) {
                            applied += 1;
                            if perft(&mut g, 1) == PERFT_REF[game][0] {
                                ok += 1;
                            }
                        }
                    }
                    println!(
                        "  sym={sym} swapV={swap_orient} msbIdx={msb_index}: applied={applied}/15 d1ok={ok}/15"
                    );
                }
            }
        }
        return;
    }

    let m = winners[0];
    println!(
        "MAPPING FOUND: sym={} swap_orient={} msb_index={}  ({} candidate(s))",
        m.sym,
        m.swap_orient,
        m.msb_index,
        winners.len()
    );
    // Validate depths 1..5 against the reference.
    let max_depth = std::env::args()
        .nth(2)
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(4); // d5 is ~2B/game (slow); default to d4, pass arg for d5
    println!("Validating depths 1..={max_depth}:");
    let mut all_match = true;
    for game in 0..15 {
        let mut line = format!("  game {game:2}: ");
        for d in 1..=max_depth {
            let mut g = setup(game, m).unwrap();
            let got = perft(&mut g, d);
            let exp = PERFT_REF[game][(d - 1) as usize];
            let ok = got == exp;
            all_match &= ok;
            line.push_str(&format!("d{d}{} ", if ok { "✓" } else { "✗" }));
            if !ok {
                line.push_str(&format!("(got {got} exp {exp}) "));
            }
        }
        println!("{line}");
    }
    println!(
        "{}",
        if all_match {
            "ALL MATCH — engine movegen reproduces canta's perft suite."
        } else {
            "MISMATCH at some depth (see above)."
        }
    );
}
