//! Quoridor board — internal 0..8 rows/cols, wall bitboards, Zobrist hash.

pub type Row = u8;
pub type Column = u8;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Player {
    One = 0,
    Two = 1,
}

impl Player {
    pub fn opposite(self) -> Self {
        match self {
            Player::One => Player::Two,
            Player::Two => Player::One,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WallOrientation {
    Horizontal,
    Vertical,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Move {
    Pawn {
        row: Row,
        col: Column,
    },
    Wall {
        row: Row,
        col: Column,
        orientation: WallOrientation,
    },
}

/// Saved state for `unmake_move` — everything else is recomputed by inverse ops.
#[derive(Debug, Clone, Copy)]
pub struct Undo {
    pub mv: Move,
    pub pawn_from: (Row, Column),
    pub hash: u64,
    pub verify: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Board {
    pub pawns: [(Row, Column); 2],
    pub walls_remaining: [u8; 2],
    pub horizontal_walls: u64,
    pub vertical_walls: u64,
    pub side_to_move: Player,
    pub move_number: u32,
    pub hash: u64,
    /// Cached `compute_verify()` result — updated by `make_move`, restored by
    /// `unmake_move`. Makes `tt_verify()` a free register read on the probe hot path.
    pub verify: u32,
}

impl Default for Board {
    fn default() -> Self {
        Self::new()
    }
}

impl Board {
    /// Independent 32-bit board signature for TT collision verification.
    ///
    /// Returns the cached `self.verify` field — a free register read on the TT
    /// probe hot path. The value is computed once by `compute_verify()` and kept
    /// current by `make_move` / `unmake_move`.
    #[inline]
    pub fn tt_verify(&self) -> u32 {
        self.verify
    }

    /// Full computation of the 32-bit verify hash — called only in `Board::new()`
    /// and `make_move`. Never call this on the TT probe hot path; use `tt_verify()`.
    ///
    /// Pawns + side-to-move packed into one word, XOR-mixed with both wall
    /// bitboards under distinct odd multipliers, folded to 32 bits. Effectively
    /// independent of the Zobrist key (different inputs + mixing constants), so
    /// a false TT hit needs both `hash` (64-bit) and `verify` (32-bit) to collide
    /// (~2^-96/pair — negligible even at game-solve scale).
    #[inline]
    fn compute_verify(&self) -> u32 {
        let p = ((self.pawns[0].0 as u64) << 21)
            | ((self.pawns[0].1 as u64) << 14)
            | ((self.pawns[1].0 as u64) << 7)
            | (self.pawns[1].1 as u64)
            | ((self.side_to_move as u64) << 28);
        let x = self.horizontal_walls.wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ self.vertical_walls.wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
            ^ (p << 32);
        ((x >> 32) as u32) ^ (x as u32)
    }

    pub fn new() -> Self {
        let mut board = Self {
            pawns: [(0, 4), (8, 4)],
            walls_remaining: [10, 10],
            horizontal_walls: 0,
            vertical_walls: 0,
            side_to_move: Player::One,
            move_number: 1,
            hash: 0,
            verify: 0,
        };
        board.hash = crate::core::zobrist::hash_board(&board);
        board.verify = board.compute_verify();
        board
    }

    #[inline]
    pub fn pawn(&self, player: Player) -> (Row, Column) {
        self.pawns[player as usize]
    }

    #[inline]
    pub fn side(&self) -> Player {
        self.side_to_move
    }

    pub fn column_char(col: Column) -> char {
        (b'a' + col) as char
    }

    pub fn format_square(row: Row, col: Column) -> String {
        format!("{}{}", Self::column_char(col), row + 1)
    }

    pub fn is_terminal(&self) -> Option<Player> {
        if self.pawns[0].0 == 8 {
            return Some(Player::One);
        }
        if self.pawns[1].0 == 0 {
            return Some(Player::Two);
        }
        None
    }

    pub fn apply_algebraic(&mut self, text: &str) {
        let bytes = text.as_bytes();
        let col = bytes[0] - b'a';
        let row = bytes[1] - b'0' - 1;
        let mv = if bytes.len() > 2 {
            let orientation = match bytes[2] {
                b'h' => WallOrientation::Horizontal,
                b'v' => WallOrientation::Vertical,
                _ => panic!("bad wall suffix in {text}"),
            };
            Move::Wall {
                row,
                col,
                orientation,
            }
        } else {
            Move::Pawn { row, col }
        };
        let _ = self.make_move(mv);
    }

    /// In-place move for search/perft — pair with `unmake_move`.
    pub fn make_move(&mut self, mv: Move) -> Undo {
        let side = self.side_to_move as usize;
        let undo = Undo {
            mv,
            pawn_from: self.pawns[side],
            hash: self.hash,
            verify: self.verify,
        };

        match mv {
            Move::Pawn { row, col } => {
                self.hash ^= crate::core::zobrist::pawn_move_delta(side, undo.pawn_from, (row, col));
                self.pawns[side] = (row, col);
            }
            Move::Wall {
                row,
                col,
                orientation,
            } => {
                self.hash ^= crate::core::zobrist::wall_move_delta(
                    orientation,
                    row,
                    col,
                    side,
                    self.walls_remaining[side],
                );
                crate::util::grid::set_wall(self, row, col, orientation, true);
                self.walls_remaining[side] -= 1;
            }
        }

        self.side_to_move = self.side_to_move.opposite();
        if self.side_to_move == Player::One {
            self.move_number += 1;
        }

        self.verify = self.compute_verify();
        undo
    }

    pub fn unmake_move(&mut self, undo: Undo) {
        if self.side_to_move == Player::One {
            self.move_number -= 1;
        }
        self.side_to_move = self.side_to_move.opposite();

        let side = self.side_to_move as usize;
        match undo.mv {
            Move::Pawn { .. } => {
                self.pawns[side] = undo.pawn_from;
            }
            Move::Wall {
                row,
                col,
                orientation,
            } => {
                self.walls_remaining[side] += 1;
                crate::util::grid::set_wall(self, row, col, orientation, false);
            }
        }

        self.hash = undo.hash;
        self.verify = undo.verify;
    }

    /// Convenience API — allocates nothing but cannot unmake.
    pub fn apply_move(&mut self, mv: Move) {
        let _ = self.make_move(mv);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn starting_position_matches_scraped_ui() {
        let board = Board::new();
        assert_eq!(board.pawns[0], (0, 4));
        assert_eq!(board.pawns[1], (8, 4));
        assert_eq!(board.walls_remaining, [10, 10]);
        assert_eq!(board.side_to_move, Player::One);
        assert_eq!(Board::format_square(0, 4), "e1");
        assert_eq!(Board::format_square(8, 4), "e9");
    }

    #[test]
    fn make_unmake_restores_board_and_hash() {
        let mut board = Board::new();
        let before = board.clone();
        let mv = Move::Pawn { row: 1, col: 4 };
        let undo = board.make_move(mv);
        board.unmake_move(undo);
        assert_eq!(board, before);
        assert_eq!(board.hash, before.hash);
    }
}
