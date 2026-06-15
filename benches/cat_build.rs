//! Benchmark `build_corridor_attention` — the per-node CAT cost in wall search.
//!
//! Two positions: the open startpos (full-board flood, most squares cold) and a
//! mid-game position with several walls (corridors narrowed, more hot squares).

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use titanium::{BfsScratch, Board};

fn midgame_board() -> Board {
    // Pawns marched toward the center, then central walls narrowing both
    // corridors — representative of positions where wall search runs CAT.
    let mut b = Board::new();
    for mv in ["e2", "e8", "e3", "e7", "d3h", "f5v", "c2h"] {
        b.apply_algebraic(mv);
    }
    b
}

fn bench_cat_startpos(c: &mut Criterion) {
    let board = Board::new();
    let mut bfs = BfsScratch::new();
    c.bench_function("cat_build_startpos", |b| {
        b.iter(|| black_box(bfs.build_corridor_attention(black_box(&board))));
    });
}

fn bench_cat_midgame(c: &mut Criterion) {
    let board = midgame_board();
    let mut bfs = BfsScratch::new();
    c.bench_function("cat_build_midgame", |b| {
        b.iter(|| black_box(bfs.build_corridor_attention(black_box(&board))));
    });
}

criterion_group!(benches, bench_cat_startpos, bench_cat_midgame);
criterion_main!(benches);
