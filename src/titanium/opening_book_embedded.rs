//! Compact in-memory opening DAG (QBKG v1) — WASM-safe, no SQLite.

use std::sync::OnceLock;

use crate::titanium::opening_book::{
    consult_from_edge_rows, BookEdgeRow, OpeningBookConsult, OpeningBookDiagnostics,
    OpeningBookMode, OPENING_BOOK_EXTENDED_MAX_PLIES,
};
use crate::titanium::game::GameState;
use crate::titanium::packed_state::pack_state_dag;

const EMBED: &[u8] = include_bytes!("data/non_titanium_opening_dag.bin");
const MAGIC: &[u8; 4] = b"QBKG";

#[derive(Clone, Copy)]
struct EdgeRec {
    code: u8,
    visits: u32,
    wins: u32,
    losses: u32,
    draws: u32,
}

pub struct EmbeddedOpeningBook {
    positions: Vec<[u8; 24]>,
    edges: Vec<EdgeRec>,
    edge_start: Vec<u32>,
}

impl EmbeddedOpeningBook {
    fn parse(data: &[u8]) -> Result<Self, String> {
        if data.len() < 12 {
            return Err("embedded opening book too small".into());
        }
        if &data[0..4] != MAGIC {
            return Err("embedded opening book bad magic".into());
        }
        if data[4] != 1 {
            return Err(format!("embedded opening book unsupported version: {}", data[4]));
        }
        let n_pos = u16::from_le_bytes([data[8], data[9]]) as usize;
        let n_edges = u16::from_le_bytes([data[10], data[11]]) as usize;
        let mut off = 12usize;
        let mut positions = Vec::with_capacity(n_pos);
        let mut edges = Vec::with_capacity(n_edges);
        let mut edge_start = Vec::with_capacity(n_pos + 1);
        edge_start.push(0);
        for _ in 0..n_pos {
            if off + 26 > data.len() {
                return Err("truncated position header".into());
            }
            let mut packed = [0u8; 24];
            packed.copy_from_slice(&data[off..off + 24]);
            let count = u16::from_le_bytes([data[off + 24], data[off + 25]]) as usize;
            off += 26;
            positions.push(packed);
            for _ in 0..count {
                if off + 17 > data.len() {
                    return Err("truncated edge record".into());
                }
                edges.push(EdgeRec {
                    code: data[off],
                    visits: u32::from_le_bytes(data[off + 1..off + 5].try_into().unwrap()),
                    wins: u32::from_le_bytes(data[off + 5..off + 9].try_into().unwrap()),
                    losses: u32::from_le_bytes(data[off + 9..off + 13].try_into().unwrap()),
                    draws: u32::from_le_bytes(data[off + 13..off + 17].try_into().unwrap()),
                });
                off += 17;
            }
            edge_start.push(edges.len() as u32);
        }
        if edges.len() != n_edges {
            return Err(format!(
                "edge count mismatch: header {n_edges} parsed {}",
                edges.len()
            ));
        }
        Ok(Self {
            positions,
            edges,
            edge_start,
        })
    }

    fn lookup_index(&self, packed: &[u8; 24]) -> Option<usize> {
        self.positions
            .binary_search_by(|key| key.cmp(packed))
            .ok()
    }

    pub fn consult(
        &self,
        g: &GameState,
        mode: OpeningBookMode,
        legal_moves: &[i16],
    ) -> OpeningBookConsult {
        let mut diag = OpeningBookDiagnostics {
            mode,
            ply_from_start: g.hist_len,
            db_path: "embedded:non_titanium_opening_dag.bin".into(),
            ..Default::default()
        };
        if mode == OpeningBookMode::Off || g.hist_len >= OPENING_BOOK_EXTENDED_MAX_PLIES {
            diag.effective_mode = OpeningBookMode::Off;
            return OpeningBookConsult {
                diagnostics: diag,
                order: Vec::new(),
                direct_play: None,
            };
        }
        let packed = pack_state_dag(g);
        let Some(idx) = self.lookup_index(&packed) else {
            diag.effective_mode = OpeningBookMode::Off;
            return OpeningBookConsult {
                diagnostics: diag,
                order: Vec::new(),
                direct_play: None,
            };
        };
        diag.position_hit = true;
        let start = self.edge_start[idx] as usize;
        let end = self.edge_start[idx + 1] as usize;
        let rows: Vec<BookEdgeRow> = self.edges[start..end]
            .iter()
            .map(|e| BookEdgeRow {
                code: e.code,
                visits: e.visits,
                wins: e.wins,
                losses: e.losses,
                draws: e.draws,
            })
            .collect();
        consult_from_edge_rows(&mut diag, mode, &packed, legal_moves, &rows)
    }
}

static EMBEDDED: OnceLock<EmbeddedOpeningBook> = OnceLock::new();

pub fn embedded_opening_book() -> &'static EmbeddedOpeningBook {
    EMBEDDED.get_or_init(|| {
        EmbeddedOpeningBook::parse(EMBED).expect("embedded opening DAG must parse")
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::titanium::{algebraic_to_move_id, game::GameState};

    #[test]
    fn embedded_root_hit() {
        let book = embedded_opening_book();
        let g = GameState::new();
        let consult = book.consult(
            &g,
            OpeningBookMode::Order,
            &[algebraic_to_move_id("e2"), algebraic_to_move_id("d2")],
        );
        assert!(consult.diagnostics.position_hit);
    }
}
