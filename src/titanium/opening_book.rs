//! Root opening book backed by the non-Titanium DAG SQLite database.

#[cfg(target_arch = "wasm32")]
use std::path::Path;
#[cfg(not(target_arch = "wasm32"))]
use std::path::{Path, PathBuf};
use std::sync::Arc;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::Mutex;

#[cfg(not(target_arch = "wasm32"))]
use rusqlite::{Connection, OpenFlags};

use crate::titanium::dataset_state::DatasetState;
use crate::titanium::game::GameState;
use crate::titanium::opening_book_embedded::embedded_opening_book;
use crate::titanium::packed_state::pack_state_dag;
use crate::titanium::{algebraic_to_move_id, move_id_to_algebraic};

/// Default opening-book horizon (order + play).
pub const OPENING_BOOK_MAX_PLIES: usize = 12;
/// Extended horizon when the top book move has a strong win rate.
pub const OPENING_BOOK_EXTENDED_MAX_PLIES: usize = 15;
/// Minimum raw win rate (decided games) to keep book active past [`OPENING_BOOK_MAX_PLIES`].
pub const OPENING_BOOK_EXTENDED_MIN_WIN_RATE: f64 = 0.55;
pub const PLAY_MIN_VISITS: u32 = 12;
pub const PLAY_MIN_SHARE: f64 = 0.60;
pub const PLAY_WILSON_GAP: f64 = 0.02;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OpeningBookMode {
    #[default]
    Off,
    Order,
    Play,
}

impl OpeningBookMode {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "off" | "none" | "0" => Some(Self::Off),
            "order" | "sort" => Some(Self::Order),
            "play" | "direct" => Some(Self::Play),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Order => "order",
            Self::Play => "play",
        }
    }
}

#[derive(Debug, Clone)]
pub struct BookCandidate {
    pub move_code_u8: u8,
    pub algebraic: String,
    pub move_id: i16,
    pub visits: u32,
    pub wins_stm: u32,
    pub losses_stm: u32,
    pub draws: u32,
    pub raw_win_rate: f64,
    pub wilson_lower: f64,
}

#[derive(Debug, Clone, Default)]
pub struct OpeningBookDiagnostics {
    pub mode: OpeningBookMode,
    pub ply_from_start: usize,
    pub position_hit: bool,
    pub effective_mode: OpeningBookMode,
    pub played_directly: bool,
    pub ordered_only: bool,
    pub candidates: Vec<BookCandidate>,
    pub selected_move: Option<i16>,
    pub db_path: String,
}

#[derive(Clone, Copy)]
pub struct BookEdgeRow {
    pub code: u8,
    pub visits: u32,
    pub wins: u32,
    pub losses: u32,
    pub draws: u32,
}

enum OpeningBookBackend {
    #[cfg(not(target_arch = "wasm32"))]
    Sqlite {
        conn: Mutex<Connection>,
        path: PathBuf,
    },
    Embedded,
}

pub struct OpeningBook {
    backend: OpeningBookBackend,
}

impl OpeningBook {
    pub fn open(path: Option<&Path>) -> Result<Arc<Self>, String> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let path = path
                .map(Path::to_path_buf)
                .unwrap_or_else(Self::default_path);
            if path.is_file() {
                let conn = Connection::open_with_flags(&path, OpenFlags::SQLITE_OPEN_READ_ONLY)
                    .map_err(|e| format!("opening book open failed: {e}"))?;
                return Ok(Arc::new(Self {
                    backend: OpeningBookBackend::Sqlite {
                        conn: Mutex::new(conn),
                        path,
                    },
                }));
            }
        }
        let _ = path;
        Ok(Arc::new(Self {
            backend: OpeningBookBackend::Embedded,
        }))
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn default_path() -> PathBuf {
        if let Ok(raw) = std::env::var("TITANIUM_BOOK_DB") {
            return PathBuf::from(raw);
        }
        PathBuf::from("training/data/opening_book/non_titanium_opening_dag.db")
    }

    fn db_path_label(&self) -> String {
        match &self.backend {
            #[cfg(not(target_arch = "wasm32"))]
            OpeningBookBackend::Sqlite { path, .. } => path.display().to_string(),
            OpeningBookBackend::Embedded => "embedded:non_titanium_opening_dag.bin".into(),
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn lookup_position_id(&self, packed: &[u8; 24]) -> Option<i64> {
        let OpeningBookBackend::Sqlite { conn, .. } = &self.backend else {
            return None;
        };
        let conn = conn.lock().ok()?;
        conn.query_row(
            "SELECT position_id FROM positions WHERE packed_state = ?1",
            rusqlite::params![packed.as_slice()],
            |row| row.get(0),
        )
        .ok()
    }

    pub fn consult(
        &self,
        g: &GameState,
        mode: OpeningBookMode,
        legal_moves: &[i16],
    ) -> OpeningBookConsult {
        match &self.backend {
            OpeningBookBackend::Embedded => embedded_opening_book().consult(g, mode, legal_moves),
            #[cfg(not(target_arch = "wasm32"))]
            OpeningBookBackend::Sqlite { .. } => self.consult_sqlite(g, mode, legal_moves),
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn consult_sqlite(
        &self,
        g: &GameState,
        mode: OpeningBookMode,
        legal_moves: &[i16],
    ) -> OpeningBookConsult {
        let mut diag = OpeningBookDiagnostics {
            mode,
            ply_from_start: g.hist_len,
            db_path: self.db_path_label(),
            ..Default::default()
        };
        if mode == OpeningBookMode::Off {
            return OpeningBookConsult {
                diagnostics: diag,
                order: Vec::new(),
                direct_play: None,
            };
        }
        if g.hist_len >= OPENING_BOOK_EXTENDED_MAX_PLIES {
            diag.effective_mode = OpeningBookMode::Off;
            return OpeningBookConsult {
                diagnostics: diag,
                order: Vec::new(),
                direct_play: None,
            };
        }

        let packed = pack_state_dag(g);
        let Some(position_id) = self.lookup_position_id(&packed) else {
            diag.effective_mode = OpeningBookMode::Off;
            return OpeningBookConsult {
                diagnostics: diag,
                order: Vec::new(),
                direct_play: None,
            };
        };
        diag.position_hit = true;

        let conn = match self.sqlite_conn() {
            Some(c) => c,
            None => {
                diag.effective_mode = OpeningBookMode::Off;
                return OpeningBookConsult {
                    diagnostics: diag,
                    order: Vec::new(),
                    direct_play: None,
                };
            }
        };
        let mut stmt = match conn.prepare(
            "SELECT move_code_u8, visit_count, wins_stm, losses_stm, draws \
             FROM edges WHERE parent_position_id = ?1",
        ) {
            Ok(s) => s,
            Err(_) => {
                diag.effective_mode = OpeningBookMode::Off;
                return OpeningBookConsult {
                    diagnostics: diag,
                    order: Vec::new(),
                    direct_play: None,
                };
            }
        };
        let rows = stmt.query_map([position_id], |row| {
            Ok(BookEdgeRow {
                code: row.get::<_, i64>(0)? as u8,
                visits: row.get::<_, i64>(1)? as u32,
                wins: row.get::<_, i64>(2)? as u32,
                losses: row.get::<_, i64>(3)? as u32,
                draws: row.get::<_, i64>(4)? as u32,
            })
        });
        let Ok(rows) = rows else {
            diag.effective_mode = OpeningBookMode::Off;
            return OpeningBookConsult {
                diagnostics: diag,
                order: Vec::new(),
                direct_play: None,
            };
        };
        let edge_rows: Vec<BookEdgeRow> = rows.flatten().collect();
        consult_from_edge_rows(&mut diag, mode, &packed, legal_moves, &edge_rows)
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn sqlite_conn(&self) -> Option<std::sync::MutexGuard<'_, Connection>> {
        let OpeningBookBackend::Sqlite { conn, .. } = &self.backend else {
            return None;
        };
        conn.lock().ok()
    }
}

pub fn consult_from_edge_rows(
    diag: &mut OpeningBookDiagnostics,
    mode: OpeningBookMode,
    packed: &[u8; 24],
    legal_moves: &[i16],
    edge_rows: &[BookEdgeRow],
) -> OpeningBookConsult {
    let state = match DatasetState::from_packed(packed) {
        Ok(s) => s,
        Err(_) => {
            diag.effective_mode = OpeningBookMode::Off;
            return OpeningBookConsult {
                diagnostics: diag.clone(),
                order: Vec::new(),
                direct_play: None,
            };
        }
    };

    let legal_set: std::collections::HashSet<i16> = legal_moves.iter().copied().collect();
    let mut candidates = Vec::new();
    for row in edge_rows {
        let Ok(alg) = state.decode_move_code(row.code) else {
            continue;
        };
        let mv = algebraic_to_move_id(&alg);
        if !legal_set.contains(&mv) {
            continue;
        }
        let raw_wr = raw_win_rate(row.wins, row.losses);
        candidates.push(BookCandidate {
            move_code_u8: row.code,
            algebraic: alg,
            move_id: mv,
            visits: row.visits,
            wins_stm: row.wins,
            losses_stm: row.losses,
            draws: row.draws,
            raw_win_rate: raw_wr,
            wilson_lower: wilson_lower_bound(row.wins, row.losses),
        });
    }
    candidates.sort_by(|a, b| {
        b.wilson_lower
            .partial_cmp(&a.wilson_lower)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.visits.cmp(&a.visits))
            .then_with(|| a.algebraic.cmp(&b.algebraic))
    });
    diag.candidates = candidates.clone();

    if candidates.is_empty() {
        diag.effective_mode = OpeningBookMode::Off;
        return OpeningBookConsult {
            diagnostics: diag.clone(),
            order: Vec::new(),
            direct_play: None,
        };
    }

    if diag.ply_from_start >= OPENING_BOOK_MAX_PLIES
        && !qualifies_for_extended_opening(&candidates)
    {
        diag.effective_mode = OpeningBookMode::Off;
        return OpeningBookConsult {
            diagnostics: diag.clone(),
            order: Vec::new(),
            direct_play: None,
        };
    }

    let order: Vec<i16> = candidates.iter().map(|c| c.move_id).collect();
    diag.effective_mode = mode;

    let direct_play = if mode == OpeningBookMode::Play {
        if should_play_direct(&candidates) {
            diag.played_directly = true;
            diag.selected_move = Some(candidates[0].move_id);
            Some(candidates[0].move_id)
        } else {
            diag.ordered_only = true;
            diag.effective_mode = OpeningBookMode::Order;
            None
        }
    } else {
        diag.ordered_only = true;
        None
    };

    OpeningBookConsult {
        diagnostics: diag.clone(),
        order,
        direct_play,
    }
}

#[derive(Debug, Clone)]
pub struct OpeningBookConsult {
    pub diagnostics: OpeningBookDiagnostics,
    pub order: Vec<i16>,
    pub direct_play: Option<i16>,
}

pub fn raw_win_rate(wins: u32, losses: u32) -> f64 {
    let decided = wins + losses;
    if decided == 0 {
        0.5
    } else {
        wins as f64 / decided as f64
    }
}

/// Wilson score lower confidence bound (95%, z = 1.96).
pub fn wilson_lower_bound(wins: u32, losses: u32) -> f64 {
    wilson_lower_bound_z(wins, losses, 1.96)
}

pub fn wilson_lower_bound_z(wins: u32, losses: u32, z: f64) -> f64 {
    let n = (wins + losses) as f64;
    if n <= 0.0 {
        return 0.5;
    }
    let p = wins as f64 / n;
    let z2 = z * z;
    let denom = 1.0 + z2 / n;
    let center = p + z2 / (2.0 * n);
    let margin = z * ((p * (1.0 - p) / n + z2 / (4.0 * n * n)).max(0.0)).sqrt();
    ((center - margin) / denom).clamp(0.0, 1.0)
}

/// Past the base ply cap, keep book order/play only when the top line is statistically strong.
pub fn qualifies_for_extended_opening(candidates: &[BookCandidate]) -> bool {
    let Some(top) = candidates.first() else {
        return false;
    };
    if top.visits < PLAY_MIN_VISITS {
        return false;
    }
    if top.raw_win_rate >= OPENING_BOOK_EXTENDED_MIN_WIN_RATE {
        return true;
    }
    should_play_direct(candidates)
}

pub fn should_play_direct(candidates: &[BookCandidate]) -> bool {
    let Some(top) = candidates.first() else {
        return false;
    };
    if top.visits < PLAY_MIN_VISITS {
        return false;
    }
    let total: u32 = candidates.iter().map(|c| c.visits).sum();
    if total > 0 && top.visits as f64 / total as f64 >= PLAY_MIN_SHARE {
        return true;
    }
    if let Some(second) = candidates.get(1) {
        return top.wilson_lower > second.wilson_lower + PLAY_WILSON_GAP;
    }
    true
}

pub fn diagnostics_json(diag: &OpeningBookDiagnostics) -> String {
    let mut out = String::from("{\"book\":{");
    out.push_str(&format!(
        "\"mode\":\"{}\",\"effectiveMode\":\"{}\",\"positionHit\":{},\"ply\":{},\"playedDirectly\":{},\"orderedOnly\":{},\"db\":",
        diag.mode.as_str(),
        diag.effective_mode.as_str(),
        diag.position_hit,
        diag.ply_from_start,
        diag.played_directly,
        diag.ordered_only,
    ));
    out.push_str(&json_str(&diag.db_path));
    out.push_str(",\"candidates\":[");
    for (i, c) in diag.candidates.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&format!(
            "{{\"move\":\"{}\",\"u8\":{},\"visits\":{},\"wins\":{},\"losses\":{},\"draws\":{},\"rawWinRate\":{:.4},\"wilsonLower\":{:.4}}}",
            json_escape(&c.algebraic),
            c.move_code_u8,
            c.visits,
            c.wins_stm,
            c.losses_stm,
            c.draws,
            c.raw_win_rate,
            c.wilson_lower,
        ));
    }
    out.push_str("],\"selectedMove\":");
    match diag.selected_move {
        Some(mv) => out.push_str(&format!("\"{}\"", json_escape(&move_id_to_algebraic(mv)))),
        None => out.push_str("null"),
    }
    out.push_str("}}");
    out
}

fn json_escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn json_str(s: &str) -> String {
    format!("\"{}\"", json_escape(s))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::titanium::game::GameState;
    use std::path::PathBuf;

    fn book_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("training")
            .join("data")
            .join("opening_book")
            .join("non_titanium_opening_dag.db")
    }

    #[test]
    fn root_packed_state_lookup() {
        let book = match OpeningBook::open(Some(&book_path())) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skip: {e}");
                return;
            }
        };
        let g = GameState::new();
        let _packed = crate::titanium::pack_state(&g);
        let consult = book.consult(
            &g,
            OpeningBookMode::Order,
            &[algebraic_to_move_id("e2"), algebraic_to_move_id("d2")],
        );
        assert!(consult.diagnostics.position_hit);
    }

    #[test]
    fn wilson_ranks_e2_above_f1_at_root() {
        let book = match OpeningBook::open(Some(&book_path())) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skip: {e}");
                return;
            }
        };
        let g = GameState::new();
        let legal = vec![
            algebraic_to_move_id("e2"),
            algebraic_to_move_id("d2"),
            algebraic_to_move_id("f2"),
            algebraic_to_move_id("f1"),
            algebraic_to_move_id("d1"),
        ];
        let consult = book.consult(&g, OpeningBookMode::Order, &legal);
        assert!(consult.diagnostics.position_hit);
        let e2 = consult
            .diagnostics
            .candidates
            .iter()
            .find(|c| c.algebraic == "e2")
            .expect("e2 in book");
        let f1 = consult
            .diagnostics
            .candidates
            .iter()
            .find(|c| c.algebraic == "f1")
            .expect("f1 in book");
        assert!(e2.visits > f1.visits);
        assert_eq!(consult.order[0], e2.move_id);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn book_off_matches_default_search_on_startpos() {
        use crate::titanium::opening_book::OpeningBookMode;
        use crate::titanium::TitaniumSearch;
        use std::path::PathBuf;

        let db = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("training")
            .join("data")
            .join("opening_book")
            .join("non_titanium_opening_dag.db");
        if !db.is_file() {
            eprintln!("skip: opening dag missing");
            return;
        }
        let mut off = TitaniumSearch::grafted(GameState::new(), Some(18));
        off.set_opening_book(OpeningBookMode::Off, Some(db.clone()));
        let r_off = off.think(50, 8, true, false, "book-test-off");

        let mut on = TitaniumSearch::grafted(GameState::new(), Some(18));
        on.set_opening_book(OpeningBookMode::Off, None);
        let r_on = on.think(50, 8, true, false, "book-test-off");

        assert_eq!(r_off.mv, r_on.mv);
        assert_eq!(r_off.nodes, r_on.nodes);
    }

    #[test]
    fn missing_position_is_miss() {
        let book = match OpeningBook::open(Some(&book_path())) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skip: {e}");
                return;
            }
        };
        let mut g = GameState::new();
        for mv in ["e2", "e8", "e3", "e7", "e4", "e6", "a3h", "d4v", "e5", "f6"] {
            g.make_move(algebraic_to_move_id(mv));
        }
        let consult = book.consult(&g, OpeningBookMode::Order, &[algebraic_to_move_id("e7")]);
        assert!(!consult.diagnostics.position_hit);
        assert!(consult.order.is_empty());
    }

    #[test]
    fn illegal_db_move_rejected() {
        let book = match OpeningBook::open(Some(&book_path())) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skip: {e}");
                return;
            }
        };
        let g = GameState::new();
        // only d2 legal in this fake set — e2 from book should be dropped
        let consult = book.consult(&g, OpeningBookMode::Order, &[algebraic_to_move_id("d2")]);
        assert!(consult
            .diagnostics
            .candidates
            .iter()
            .all(|c| c.move_id == algebraic_to_move_id("d2")));
    }

    #[test]
    fn play_mode_falls_back_without_confidence() {
        let candidates = vec![
            BookCandidate {
                move_code_u8: 130,
                algebraic: "f1".into(),
                move_id: algebraic_to_move_id("f1"),
                visits: 6,
                wins_stm: 4,
                losses_stm: 2,
                draws: 0,
                raw_win_rate: 0.667,
                wilson_lower: wilson_lower_bound(4, 2),
            },
            BookCandidate {
                move_code_u8: 128,
                algebraic: "e2".into(),
                move_id: algebraic_to_move_id("e2"),
                visits: 328,
                wins_stm: 137,
                losses_stm: 191,
                draws: 0,
                raw_win_rate: 0.418,
                wilson_lower: wilson_lower_bound(137, 191),
            },
        ];
        assert!(!should_play_direct(&candidates));
    }

    #[test]
    fn extended_opening_requires_strong_top_line() {
        let strong = BookCandidate {
            move_code_u8: 128,
            algebraic: "e2".into(),
            move_id: algebraic_to_move_id("e2"),
            visits: 20,
            wins_stm: 12,
            losses_stm: 8,
            draws: 0,
            raw_win_rate: 0.6,
            wilson_lower: wilson_lower_bound(12, 8),
        };
        assert!(qualifies_for_extended_opening(&[strong.clone()]));
        let weak_top = BookCandidate {
            visits: 20,
            wins_stm: 8,
            losses_stm: 12,
            raw_win_rate: 0.4,
            wilson_lower: wilson_lower_bound(8, 12),
            ..strong.clone()
        };
        let alt = BookCandidate {
            move_code_u8: 130,
            algebraic: "f1".into(),
            move_id: algebraic_to_move_id("f1"),
            visits: 18,
            wins_stm: 9,
            losses_stm: 9,
            draws: 0,
            raw_win_rate: 0.5,
            wilson_lower: wilson_lower_bound(9, 9),
        };
        assert!(!qualifies_for_extended_opening(&[weak_top, alt]));
    }

    #[test]
    fn consult_from_edge_rows_blocks_ply_12_without_strong_line() {
        let mut diag = OpeningBookDiagnostics {
            mode: OpeningBookMode::Order,
            ply_from_start: OPENING_BOOK_MAX_PLIES,
            ..Default::default()
        };
        let packed = pack_state_dag(&GameState::new());
        let rows = vec![BookEdgeRow {
            code: 128,
            visits: 8,
            wins: 3,
            losses: 5,
            draws: 0,
        }];
        let consult = consult_from_edge_rows(
            &mut diag,
            OpeningBookMode::Order,
            &packed,
            &[algebraic_to_move_id("e2")],
            &rows,
        );
        assert!(consult.order.is_empty());
        assert_eq!(diag.effective_mode, OpeningBookMode::Off);
    }
}
