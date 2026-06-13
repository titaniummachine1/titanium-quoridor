//! L3 flood fraction study — how often does production movegen actually flood?
//!
//! Walks perft subtrees from startpos + the 15 Canta replay roots (wall-heavy,
//! 15 random plies each) and reports, per root:
//!   - wall candidates per node (L1∧L2)
//!   - fraction needing L3 flood (topo bit set)
//!   - fraction of floods that REJECT (an actual caging wall caught)
//!   - movegen time share spent in L3 (production vs no-flood variant)
//!
//! Decision input for HANDOFF §C (incremental L3): run with
//!   cargo run --release --bin l3-study [depth]

use std::time::Instant;

use titanium::movegen::legal::wall_path_ok_after_place;
use titanium::movegen::wall_masks::wall_masks;
use titanium::movegen::{
    generate_legal_moves_slice, generate_pawn_moves_slice_mode, PawnGenMode, MAX_LEGAL_MOVES,
};
use titanium::oracle::canta::board_after_canta_game;
use titanium::{BfsScratch, Board, Move, WallOrientation};

#[derive(Default)]
struct Stats {
    nodes: u64,
    wall_nodes: u64,
    candidates: u64,
    flood_needed: u64,
    flood_rejected: u64,
    full_ns: u128,
    no_flood_ns: u128,
}

fn count_flood_outcomes(board: &mut Board, candidates: u64, topo: u64, orientation: WallOrientation, s: &mut Stats) {
    let mut heavy = candidates & topo;
    while heavy != 0 {
        let bit = heavy.trailing_zeros();
        heavy &= heavy - 1;
        if !wall_path_ok_after_place(board, (bit / 8) as u8, (bit % 8) as u8, orientation) {
            s.flood_rejected += 1;
        }
    }
}

fn walk(board: &mut Board, depth: u32, bfs: &mut BfsScratch, s: &mut Stats) {
    if depth == 0 {
        return;
    }
    s.nodes += 1;

    let side = board.side_to_move as usize;
    if board.walls_remaining[side] > 0 && board.is_terminal().is_none() {
        s.wall_nodes += 1;
        let m = wall_masks(board);
        s.candidates += (m.l12_h.count_ones() + m.l12_v.count_ones()) as u64;
        s.flood_needed += ((m.l12_h & m.topo_h).count_ones() + (m.l12_v & m.topo_v).count_ones()) as u64;
        count_flood_outcomes(board, m.l12_h, m.topo_h, WallOrientation::Horizontal, s);
        count_flood_outcomes(board, m.l12_v, m.topo_v, WallOrientation::Vertical, s);

        // No-flood variant: pawn gen + masks + full move emission, L3 free.
        let mut buf = [Move::Pawn { row: 0, col: 0 }; MAX_LEGAL_MOVES];
        let t0 = Instant::now();
        let mut n = generate_pawn_moves_slice_mode(board, &mut buf, bfs, PawnGenMode::default());
        let m2 = wall_masks(board);
        for (bits, orientation) in [
            (m2.l12_h, WallOrientation::Horizontal),
            (m2.l12_v, WallOrientation::Vertical),
        ] {
            let mut b = bits;
            while b != 0 {
                let bit = b.trailing_zeros();
                b &= b - 1;
                buf[n] = Move::Wall {
                    row: (bit / 8) as u8,
                    col: (bit % 8) as u8,
                    orientation,
                };
                n += 1;
            }
        }
        std::hint::black_box(&buf[..n]);
        s.no_flood_ns += t0.elapsed().as_nanos();
    }

    let mut buf = [Move::Pawn { row: 0, col: 0 }; MAX_LEGAL_MOVES];
    let t0 = Instant::now();
    let n = generate_legal_moves_slice(board, &mut buf, bfs);
    s.full_ns += t0.elapsed().as_nanos();

    for &mv in &buf[..n] {
        let undo = board.make_move(mv);
        walk(board, depth - 1, bfs, s);
        board.unmake_move(undo);
    }
}

fn report(label: &str, board: &Board, depth: u32) {
    let mut b = board.clone();
    let mut bfs = BfsScratch::new();
    let mut s = Stats::default();
    walk(&mut b, depth, &mut bfs, &mut s);

    let walls_on = board.horizontal_walls.count_ones() + board.vertical_walls.count_ones();
    let cand_per_node = s.candidates as f64 / s.wall_nodes.max(1) as f64;
    let flood_pct = 100.0 * s.flood_needed as f64 / s.candidates.max(1) as f64;
    let reject_pct = 100.0 * s.flood_rejected as f64 / s.flood_needed.max(1) as f64;
    let l3_share = 100.0 * (s.full_ns.saturating_sub(s.no_flood_ns)) as f64 / s.full_ns.max(1) as f64;
    println!(
        "| {label} | {walls_on} | {} | {cand_per_node:.1} | {flood_pct:.1}% | {reject_pct:.1}% | {l3_share:.0}% |",
        s.nodes
    );
}

fn main() {
    let depth: u32 = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(3);

    println!("L3 flood fraction study — depth {depth} subtrees");
    println!("| root | walls | nodes | cand/node | flood-needed | flood-reject | L3 time share |");
    println!("|------|------:|------:|----------:|-------------:|-------------:|--------------:|");

    report("startpos", &Board::new(), depth);
    for game in 0..15 {
        report(&format!("canta{game:02}"), &board_after_canta_game(game), depth);
    }
}
