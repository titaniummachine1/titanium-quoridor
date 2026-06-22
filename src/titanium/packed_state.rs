//! Canonical 24-byte packed position state (POSITION_SCHEMA_VERSION = 1).
//! Byte layout matches `training/titanium_training/store/state.py` and
//! `tools/position_store_importer/src/position_state.rs`.

use crate::titanium::game::{GameState, ZOBRIST};

pub const POSITION_SCHEMA_VERSION: u8 = 1;
pub const PACKED_STATE_LEN: usize = 24;
pub const FEATURE_SCHEMA: &str = "halfpw-sparse-route5-ws14-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackedFields {
    pub player0_cell: u8,
    pub player1_cell: u8,
    pub player0_walls: u8,
    pub player1_walls: u8,
    pub side_to_move: u8,
    pub horizontal_walls: u64,
    pub vertical_walls: u64,
}

pub fn decode_packed_state(data: &[u8]) -> Result<PackedFields, String> {
    if data.len() != PACKED_STATE_LEN {
        return Err(format!(
            "packed state must be {} bytes, got {}",
            PACKED_STATE_LEN,
            data.len()
        ));
    }
    let version = data[0];
    if version != POSITION_SCHEMA_VERSION {
        return Err(format!("unsupported position schema version: {version}"));
    }
    let player0_cell = data[1];
    let player1_cell = data[2];
    let player0_walls = data[3];
    let player1_walls = data[4];
    let side_to_move = data[5];
    if data[6] != 0 || data[7] != 0 {
        return Err("reserved packed-state bytes must be zero".into());
    }
    if player0_cell >= 81 || player1_cell >= 81 {
        return Err("pawn cell out of range".into());
    }
    if player0_cell == player1_cell {
        return Err("both pawns occupy the same cell".into());
    }
    if player0_walls > 10 || player1_walls > 10 {
        return Err("wall count out of range".into());
    }
    if side_to_move > 1 {
        return Err("side_to_move must be 0 or 1".into());
    }
    let horizontal_walls = u64::from_le_bytes(data[8..16].try_into().unwrap());
    let vertical_walls = u64::from_le_bytes(data[16..24].try_into().unwrap());
    Ok(PackedFields {
        player0_cell,
        player1_cell,
        player0_walls,
        player1_walls,
        side_to_move,
        horizontal_walls,
        vertical_walls,
    })
}

pub fn pack_state(g: &GameState) -> [u8; PACKED_STATE_LEN] {
    let mut hw_mask: u64 = 0;
    let mut vw_mask: u64 = 0;
    for slot in 0..64 {
        if g.hw[slot] != 0 {
            hw_mask |= 1u64 << slot;
        }
        if g.vw[slot] != 0 {
            vw_mask |= 1u64 << slot;
        }
    }
    let mut out = [0u8; PACKED_STATE_LEN];
    out[0] = POSITION_SCHEMA_VERSION;
    // Write in canonical dataset format so titanium_game_from_packed is the exact inverse:
    //   dataset player0 = engine pawn[1] (starts at top, goal = row 8)
    //   dataset player1 = engine pawn[0] (starts at bottom, goal = row 0)
    //   side_to_move = 0 means pawn[1]'s turn (engine turn == 1)
    out[1] = g.pawn[1] as u8;
    out[2] = g.pawn[0] as u8;
    out[3] = g.wl[1] as u8;
    out[4] = g.wl[0] as u8;
    out[5] = (1 - g.turn) as u8;
    out[8..16].copy_from_slice(&hw_mask.to_le_bytes());
    out[16..24].copy_from_slice(&vw_mask.to_le_bytes());
    out
}

fn recompute_hash(g: &mut GameState) {
    let z = &ZOBRIST;
    let mut lo = z.pawn_lo[0][g.pawn[0]] ^ z.pawn_lo[1][g.pawn[1]];
    let mut hi = z.pawn_hi[0][g.pawn[0]] ^ z.pawn_hi[1][g.pawn[1]];
    for slot in 0..64 {
        if g.hw[slot] != 0 {
            lo ^= z.hw_lo[slot];
            hi ^= z.hw_hi[slot];
        }
        if g.vw[slot] != 0 {
            lo ^= z.vw_lo[slot];
            hi ^= z.vw_hi[slot];
        }
    }
    if g.turn == 1 {
        lo ^= z.turn_lo;
        hi ^= z.turn_hi;
    }
    g.hash_lo = lo;
    g.hash_hi = hi;
}

pub fn titanium_game_from_packed(data: &[u8]) -> Result<GameState, String> {
    let fields = decode_packed_state(data)?;
    let mut g = GameState::new();
    // Dataset player0 → Titanium internal pawn[1]; dataset player1 → pawn[0].
    g.pawn[0] = fields.player1_cell as usize;
    g.pawn[1] = fields.player0_cell as usize;
    g.wl[0] = fields.player1_walls as i32;
    g.wl[1] = fields.player0_walls as i32;
    g.turn = if fields.side_to_move == 0 { 1 } else { 0 };
    g.hw = [0; 64];
    g.vw = [0; 64];
    g.blocked = [0; 81];
    for slot in 0..64 {
        if (fields.horizontal_walls >> slot) & 1 != 0 {
            g.hw[slot] = 1;
            g.set_wall_bits(0, slot, true);
        }
        if (fields.vertical_walls >> slot) & 1 != 0 {
            g.vw[slot] = 1;
            g.set_wall_bits(1, slot, true);
        }
    }
    g.hist_len = 0;
    g.last_wall_ply = 0;
    recompute_hash(&mut g);
    Ok(g)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::titanium::{algebraic_to_move_id, GameState};

    fn game_from_moves(moves: &[&str]) -> GameState {
        let mut g = GameState::new();
        for mv in moves {
            g.make_move(algebraic_to_move_id(mv));
        }
        g
    }

    #[test]
    fn pack_round_trip_startpos() {
        let g = GameState::new();
        let packed = pack_state(&g);
        let g2 = titanium_game_from_packed(&packed).unwrap();
        // pack_state writes dataset format; titanium_game_from_packed is its exact inverse.
        assert_eq!(g.pawn, g2.pawn);
        assert_eq!(g.wl, g2.wl);
        assert_eq!(g.turn, g2.turn);
        assert_eq!(g.hw, g2.hw);
        assert_eq!(g.vw, g2.vw);
    }

    #[test]
    fn packed_matches_move_prefix_positions() {
        let cases: &[&[&str]] = &[
            &[],
            &["e2", "e8", "e3", "e7", "d3h", "f5v"],
            &["e2", "e8", "e3", "e7", "e4", "e6", "a3h", "d4v"],
            &["e2", "e8", "d2", "f8", "c4h", "g5h"],
        ];
        for moves in cases {
            let g = game_from_moves(moves);
            let packed = pack_state(&g);
            let g2 = titanium_game_from_packed(&packed).unwrap();
            assert_eq!(g.pawn, g2.pawn, "moves={moves:?}");
            assert_eq!(g.hw, g2.hw, "moves={moves:?}");
            assert_eq!(g.vw, g2.vw, "moves={moves:?}");
            assert_eq!(g.turn, g2.turn, "moves={moves:?}");
        }
    }

    #[test]
    fn rejects_bad_version_and_length() {
        assert!(decode_packed_state(&[0u8; 8]).is_err());
        let mut bad = pack_state(&GameState::new());
        bad[0] = 2;
        assert!(decode_packed_state(&bad).is_err());
    }

    #[test]
    fn side_to_move_preserved() {
        // After 2 moves: engine turn=0 (pawn[0]'s turn).
        // Dataset side_to_move=0 means pawn[1]'s turn (engine turn=1), so packed[5] = 1-0 = 1.
        let g = game_from_moves(&["e2", "e8"]);
        let packed = pack_state(&g);
        assert_eq!(g.turn, 0);
        assert_eq!(packed[5], 1); // 1 - engine.turn
        // After 3 moves: engine turn=1 (pawn[1]'s turn).
        // Dataset side_to_move = 1-1 = 0.
        let g2 = game_from_moves(&["e2", "e8", "e3"]);
        let packed2 = pack_state(&g2);
        assert_eq!(g2.turn, 1);
        assert_eq!(packed2[5], 0); // 1 - engine.turn
        let restored = titanium_game_from_packed(&packed2).unwrap();
        assert_eq!(restored.turn, g2.turn);
    }
}
