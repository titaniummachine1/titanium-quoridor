//! Dataset-format position state (24-byte packed) — move u8 codec for the opening DAG.
//! Layout matches `training/titanium_training/store/state.py`.

use crate::titanium::packed_state::{decode_packed_state, PackedFields};

pub const PAWN_NORTH: u8 = 128;
pub const PAWN_SOUTH: u8 = 129;
pub const PAWN_EAST: u8 = 130;
pub const PAWN_WEST: u8 = 131;
pub const PAWN_NORTHEAST: u8 = 132;
pub const PAWN_NORTHWEST: u8 = 133;
pub const PAWN_SOUTHEAST: u8 = 134;
pub const PAWN_SOUTHWEST: u8 = 135;

const BOARD_SIZE: u8 = 9;
const WALL_GRID_SIZE: u8 = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DatasetState {
    pub player0_cell: u8,
    pub player1_cell: u8,
    pub player0_walls: u8,
    pub player1_walls: u8,
    pub horizontal_walls: u64,
    pub vertical_walls: u64,
    pub side_to_move: u8,
}

impl DatasetState {
    pub fn from_packed(packed: &[u8]) -> Result<Self, String> {
        let f = decode_packed_state(packed)?;
        Ok(Self::from_fields(f))
    }

    pub fn from_fields(f: PackedFields) -> Self {
        Self {
            player0_cell: f.player0_cell,
            player1_cell: f.player1_cell,
            player0_walls: f.player0_walls,
            player1_walls: f.player1_walls,
            horizontal_walls: f.horizontal_walls,
            vertical_walls: f.vertical_walls,
            side_to_move: f.side_to_move,
        }
    }

    pub fn current_cell(&self) -> u8 {
        if self.side_to_move == 0 {
            self.player0_cell
        } else {
            self.player1_cell
        }
    }

    pub fn decode_move_code(&self, code: u8) -> Result<String, String> {
        if code < 128 {
            return wall_code_to_notation(code);
        }
        if code > PAWN_SOUTHWEST {
            return Err(format!("unsupported move code: {code}"));
        }
        let from_cell = self.current_cell();
        let (from_row, from_col) = cell_to_coords(from_cell)?;
        let mut candidates = Vec::new();
        for target in valid_pawn_destinations(self) {
            let (to_row, to_col) = cell_to_coords(target)?;
            if direction_code_from_delta(
                to_row as i8 - from_row as i8,
                to_col as i8 - from_col as i8,
            ) == code
            {
                candidates.push(target);
            }
        }
        if candidates.len() != 1 {
            return Err(format!(
                "pawn code {code} ambiguous/illegal: {} candidates",
                candidates.len()
            ));
        }
        cell_to_notation(candidates[0])
    }
}

fn cell_to_coords(cell: u8) -> Result<(u8, u8), String> {
    if cell >= 81 {
        return Err(format!("cell out of range: {cell}"));
    }
    Ok((cell / BOARD_SIZE, cell % BOARD_SIZE))
}

fn cell_to_notation(cell: u8) -> Result<String, String> {
    let (row, col) = cell_to_coords(cell)?;
    Ok(format!("{}{}", (b'a' + col) as char, row + 1))
}

fn wall_slot_to_notation(slot: u8, horizontal: bool) -> Result<String, String> {
    if slot >= 64 {
        return Err(format!("wall slot out of range: {slot}"));
    }
    let row = slot / WALL_GRID_SIZE;
    let col = slot % WALL_GRID_SIZE;
    let suffix = if horizontal { 'h' } else { 'v' };
    Ok(format!("{}{}{}", (b'a' + col) as char, row + 1, suffix))
}

fn wall_code_to_notation(code: u8) -> Result<String, String> {
    if code < 64 {
        return wall_slot_to_notation(code, true);
    }
    if code < 128 {
        return wall_slot_to_notation(code - 64, false);
    }
    Err(format!("wall code out of range: {code}"))
}

fn direction_code_from_delta(dr: i8, dc: i8) -> u8 {
    match (dr, dc) {
        (d, 0) if d > 0 => PAWN_NORTH,
        (d, 0) if d < 0 => PAWN_SOUTH,
        (0, d) if d > 0 => PAWN_EAST,
        (0, d) if d < 0 => PAWN_WEST,
        (d, c) if d > 0 && c > 0 => PAWN_NORTHEAST,
        (d, c) if d > 0 && c < 0 => PAWN_NORTHWEST,
        (d, c) if d < 0 && c > 0 => PAWN_SOUTHEAST,
        (d, c) if d < 0 && c < 0 => PAWN_SOUTHWEST,
        _ => 255,
    }
}

fn horizontal_wall_present(state: &DatasetState, js_row: u8, col0: u8) -> bool {
    if js_row < 1 || js_row > 8 || col0 >= 8 {
        return false;
    }
    let bit = (js_row - 1) * 8 + col0;
    (state.horizontal_walls >> bit) & 1 != 0
}

fn vertical_wall_present(state: &DatasetState, js_row: u8, col0: u8) -> bool {
    if js_row < 1 || js_row > 8 || col0 >= 8 {
        return false;
    }
    let bit = (js_row - 1) * 8 + col0;
    (state.vertical_walls >> bit) & 1 != 0
}

fn pawn_can_move(state: &DatasetState, cell: u8, dr: i8, dc: i8) -> bool {
    let Ok((row, col)) = cell_to_coords(cell) else {
        return false;
    };
    let nr = row as i8 + dr;
    let nc = col as i8 + dc;
    if nr < 0 || nr > 8 || nc < 0 || nc > 8 {
        return false;
    }
    let js_from = row + 1;
    let js_to = (nr as u8) + 1;
    let col_u = col;
    let nc_u = nc as u8;
    match (dr, dc) {
        (1, 0) => {
            !horizontal_wall_present(state, js_from, col_u)
                && (col_u == 0 || !horizontal_wall_present(state, js_from, col_u - 1))
        }
        (-1, 0) => {
            !horizontal_wall_present(state, js_to, col_u)
                && (col_u == 0 || !horizontal_wall_present(state, js_to, col_u - 1))
        }
        (0, 1) => {
            !vertical_wall_present(state, js_from, col_u) && !vertical_wall_present(state, row, col_u)
        }
        (0, -1) => {
            !vertical_wall_present(state, js_to, nc_u) && !vertical_wall_present(state, nr as u8, nc_u)
        }
        _ => false,
    }
}

fn valid_pawn_destinations(state: &DatasetState) -> Vec<u8> {
    let me = state.side_to_move;
    let current = state.current_cell();
    let opponent = if me == 0 {
        state.player1_cell
    } else {
        state.player0_cell
    };
    let mut moves = Vec::new();
    for (dr, dc) in [(1i8, 0), (0, 1), (-1, 0), (0, -1)] {
        if !pawn_can_move(state, current, dr, dc) {
            continue;
        }
        let Ok((row, col)) = cell_to_coords(current) else {
            continue;
        };
        let step = row as i8 + dr;
        let scol = col as i8 + dc;
        let target = step as u8 * BOARD_SIZE + scol as u8;
        if target != opponent {
            moves.push(target);
            continue;
        }
        if pawn_can_move(state, opponent, dr, dc) {
            let Ok((orow, ocol)) = cell_to_coords(opponent) else {
                continue;
            };
            moves.push((orow as i8 + dr) as u8 * BOARD_SIZE + (ocol as i8 + dc) as u8);
            continue;
        }
        let side_steps: &[(i8, i8)] = if dr != 0 {
            &[(0, -1), (0, 1)]
        } else {
            &[(1, 0), (-1, 0)]
        };
        for (sdr, sdc) in side_steps {
            if pawn_can_move(state, opponent, *sdr, *sdc) {
                let Ok((orow, ocol)) = cell_to_coords(opponent) else {
                    continue;
                };
                let diag =
                    (orow as i8 + sdr) as u8 * BOARD_SIZE + (ocol as i8 + sdc) as u8;
                if diag != current {
                    moves.push(diag);
                }
            }
        }
    }
    moves
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::titanium::{game::GameState, pack_state_dag};

    #[test]
    fn root_u8_128_decodes_to_e2() {
        let packed = pack_state_dag(&GameState::new());
        let state = DatasetState::from_packed(&packed).unwrap();
        assert_eq!(state.decode_move_code(128).unwrap(), "e2");
    }

    #[test]
    fn decode_matches_algebraic_at_root() {
        let packed = pack_state_dag(&GameState::new());
        let state = DatasetState::from_packed(&packed).unwrap();
        assert_eq!(state.decode_move_code(128).unwrap(), "e2");
    }
}
