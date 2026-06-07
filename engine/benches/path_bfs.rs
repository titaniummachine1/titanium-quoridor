use criterion::{black_box, criterion_group, criterion_main, Criterion};
use titanium::{both_players_reach_goals, generate_legal_moves, perft, Board, Player};

fn bench_bfs_reach(c: &mut Criterion) {
    let board = Board::new();
    c.bench_function("bfs_both_reach_start", |b| {
        b.iter(|| black_box(both_players_reach_goals(black_box(&board))));
    });
}

fn bench_shortest_distance(c: &mut Criterion) {
    let board = Board::new();
    c.bench_function("shortest_distance_p1", |b| {
        b.iter(|| black_box(titanium::shortest_distance(black_box(&board), Player::One)));
    });
}

fn bench_legal_moves(c: &mut Criterion) {
    let board = Board::new();
    c.bench_function("legal_moves_start", |b| {
        b.iter(|| black_box(generate_legal_moves(black_box(&board))));
    });
}

fn bench_perft(c: &mut Criterion) {
    let board = Board::new();
    c.bench_function("perft_depth2", |b| {
        b.iter(|| black_box(perft(black_box(&board), 2)));
    });
}

criterion_group!(
    benches,
    bench_bfs_reach,
    bench_shortest_distance,
    bench_legal_moves,
    bench_perft
);
criterion_main!(benches);
