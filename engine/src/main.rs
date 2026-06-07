//! Checkpoint 02 CLI — list legal moves.

use titanium::{generate_legal_moves, Board};

fn main() {
    let board = Board::new();
    let moves = generate_legal_moves(&board);
    println!("{} legal moves at startpos", moves.len());
}
