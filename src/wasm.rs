//! wasm-bindgen bindings for the website (GitHub Pages + static hosting).
//!
//! Build (from repo root):
//!   cd site/web && npm run build:wasm

use wasm_bindgen::prelude::*;

use crate::cat::cat_snapshot_json;
use crate::core::board::Board;
use crate::titanium::net::live_weights_sha256;
use crate::titanium::search::think_result_progress_json;
use crate::titanium::{
    algebraic_to_move_id, move_id_to_algebraic, GameState, TitaniumSearch, TITANIUM_NO_MOVE,
};

const ENGINE_VERSION: &str = "titanium-v16";

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn error(msg: &str);
}

/// Lock-free capture of the last panic message into shared linear memory, so a
/// rayon helper-thread panic (whose own worker `console` is not surfaced to the
/// page) can be read back from the main thread via `last_panic()` after the
/// trap. Lock-free on purpose: a panic hook must not risk blocking on a poisoned
/// lock.
#[cfg(target_arch = "wasm32")]
mod panic_capture {
    use core::sync::atomic::{AtomicU8, AtomicUsize, Ordering};
    pub static LEN: AtomicUsize = AtomicUsize::new(0);
    pub static BUF: [AtomicU8; 2048] = [const { AtomicU8::new(0) }; 2048];
    pub fn record(msg: &str) {
        let b = msg.as_bytes();
        let n = b.len().min(BUF.len());
        for (i, &byte) in b.iter().take(n).enumerate() {
            BUF[i].store(byte, Ordering::Relaxed);
        }
        LEN.store(n, Ordering::Relaxed);
    }
    pub fn read() -> String {
        let n = LEN.load(Ordering::Relaxed);
        let bytes: Vec<u8> = (0..n).map(|i| BUF[i].load(Ordering::Relaxed)).collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

/// Read the last captured panic message (see `panic_capture`). Returns "" if no
/// panic has occurred. Safe to call after a trapped threaded search.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn last_panic() -> String {
    panic_capture::read()
}

/// Count how many lazy-SMP helper closures have started in this WASM module.
/// The site uses this as telemetry to prove browser searches are using the
/// internal Rayon pool rather than JavaScript-distributed fake workers.
#[cfg(target_arch = "wasm32")]
pub static HELPER_STARTS: core::sync::atomic::AtomicUsize =
    core::sync::atomic::AtomicUsize::new(0);
#[cfg(target_arch = "wasm32")]
pub fn note_helper_start() {
    HELPER_STARTS.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
}
#[cfg(not(target_arch = "wasm32"))]
pub fn note_helper_start() {}
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn helper_starts() -> usize {
    HELPER_STARTS.load(core::sync::atomic::Ordering::Relaxed)
}

/// Route Rust panics (including in rayon helper threads) to `console` AND a
/// shared buffer before `panic=abort` turns them into a bare wasm `unreachable`.
/// Without this a panic in the threaded search surfaces only as "unreachable",
/// hiding the real message and location.
#[cfg(target_arch = "wasm32")]
fn install_panic_hook() {
    use std::sync::Once;
    static HOOK: Once = Once::new();
    HOOK.call_once(|| {
        std::panic::set_hook(Box::new(|info| {
            let msg = format!("[titanium-panic] {info}");
            panic_capture::record(&msg);
            error(&msg);
        }));
    });
}

#[cfg(not(target_arch = "wasm32"))]
fn install_panic_hook() {}
#[cfg(feature = "wasm-threads")]
const WASM_FEATURES: &str = "wasm-threads,embed-tables";
#[cfg(not(feature = "wasm-threads"))]
const WASM_FEATURES: &str = "wasm,embed-tables";

fn ace_params_from_mode(
    engine_mode: &str,
    movetime_ms: u32,
    max_depth: i32,
) -> crate::ace::AceParams {
    let ti_movegen = engine_mode.contains("-ti");
    let eme = engine_mode.contains("pmc");
    crate::ace::AceParams {
        time_ms: (movetime_ms as u64).max(1),
        max_depth: if max_depth > 0 { max_depth } else { 30 },
        full: false,
        cat: false,
        ti_movegen,
        log: false,
        eme,
    }
}

fn replay_moves(moves: &str) -> Result<GameState, JsError> {
    let mut g = GameState::new();
    for text in moves.split_whitespace().filter(|s| !s.is_empty()) {
        if g.winner() >= 0 {
            return Err(JsError::new(&format!(
                "illegal replay past terminal: {text}"
            )));
        }
        g.make_move(algebraic_to_move_id(text));
    }
    Ok(g)
}

fn hex32(bytes: &[u8; 32]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn json_string(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn replay_board_from_moves(moves: &str) -> Board {
    let mut board = Board::new();
    for text in moves.split_whitespace().filter(|s| !s.is_empty()) {
        board.apply_algebraic(text);
    }
    board
}

/// CAT v3 heatmap JSON for the website overlay (`catHeatmap.js`).
///
/// Stateless: rebuilds the board from the full move list each call. Prefer
/// `WasmCatEngine` (below), which keeps the board warm and only applies the new
/// move — the overlay was re-replaying the whole game on every ply.
#[wasm_bindgen]
pub fn cat_snapshot(moves: &str) -> String {
    let mut board = replay_board_from_moves(moves);
    cat_snapshot_json(&mut board)
}

/// Warm, single-purpose CAT instance for the overlay worker. Holds the board
/// across plies: forward play applies only the appended move(s); undo/jump
/// rebuilds from the longest common prefix. No search, no thread pool — its only
/// job is to return the CAT snapshot for the current node fast.
#[wasm_bindgen]
pub struct WasmCatEngine {
    board: Board,
    applied: Vec<String>,
}

#[wasm_bindgen]
impl WasmCatEngine {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmCatEngine {
        WasmCatEngine { board: Board::new(), applied: Vec::new() }
    }

    /// CAT JSON for `moves` (space-separated algebraic), reusing the warm board.
    pub fn snapshot(&mut self, moves: &str) -> String {
        self.sync_to(moves);
        cat_snapshot_json(&mut self.board)
    }

    /// LMR plan JSON for `moves` at aggressiveness `max_extra` — same warm board,
    /// single-thread. Per-move `childDepthUsed`/`childDepthFull` give the search-
    /// depth %. Mirrors the live Titanium LMR (base index reduction + CAT modifier),
    /// so the overlay shows what the engine actually does.
    pub fn lmr_snapshot(
        &mut self,
        moves: &str,
        time_ms: u32,
        id_depth: u32,
        max_extra: f64,
    ) -> String {
        self.sync_to(moves);
        crate::search::lmr_viz::lmr_snapshot_json(
            &mut self.board,
            u64::from(time_ms),
            id_depth,
            max_extra,
        )
    }
}

impl WasmCatEngine {
    /// Advance the warm board to `moves`: forward play applies only the appended
    /// moves; divergence/undo rewinds to the longest common prefix.
    fn sync_to(&mut self, moves: &str) {
        let want: Vec<&str> = moves.split_whitespace().filter(|s| !s.is_empty()).collect();
        let mut common = 0usize;
        while common < self.applied.len()
            && common < want.len()
            && self.applied[common] == want[common]
        {
            common += 1;
        }
        if common < self.applied.len() {
            self.board = Board::new();
            for m in &want[..common] {
                self.board.apply_algebraic(m);
            }
            self.applied.truncate(common);
        }
        for m in &want[common..] {
            self.board.apply_algebraic(m);
            self.applied.push((*m).to_string());
        }
    }
}

impl Default for WasmCatEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// JSON build identity for browser debug panel / console.
#[wasm_bindgen]
pub fn wasm_build_identity_json() -> String {
    let git = option_env!("GIT_COMMIT_HASH").unwrap_or("unknown");
    let built_at = option_env!("WASM_BUILD_TIMESTAMP").unwrap_or("unknown");
    format!(
        r#"{{"engine_version":"{ENGINE_VERSION}","git_commit":"{git}","build_timestamp":"{built_at}","features":"{WASM_FEATURES}","weights_live_sha256":"{live}"}}"#,
        live = hex32(&live_weights_sha256()),
    )
}

/// ACE Rust port in WASM — one-shot genmove from a move list (GitHub Pages; no native binary).
#[wasm_bindgen]
pub struct WasmAceEngine;

#[wasm_bindgen]
impl WasmAceEngine {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmAceEngine {
        WasmAceEngine
    }

    pub fn genmove(
        &self,
        moves: &str,
        movetime_ms: u32,
        max_depth: i32,
        engine_mode: &str,
        on_progress: Option<js_sys::Function>,
    ) -> String {
        let _ = on_progress;
        let list: Vec<String> = moves
            .split_whitespace()
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .collect();
        let params = ace_params_from_mode(engine_mode, movetime_ms, max_depth);
        match crate::ace::ace_genmove(&list, params, engine_mode) {
            Some((alg, _)) => alg,
            None => "(none)".to_string(),
        }
    }
}

/// Warm Titanium v16 session. TT and history persist between plies.
#[wasm_bindgen]
pub struct WasmEngine {
    search: TitaniumSearch,
    engine_label: String,
    last_depth: i32,
    last_nodes: u64,
    last_stop_reason: &'static str,
}

#[wasm_bindgen]
impl WasmEngine {
    /// `tier`: 3 = CAT 500, 4 = CAT 800, 5 = CAT 1000. Other values use CAT 800.
    #[wasm_bindgen(constructor)]
    pub fn new(tier: u8) -> WasmEngine {
        install_panic_hook();
        let g = GameState::new();
        let ceiling = match tier {
            3 => 500,
            5 => 1000,
            _ => 800,
        };
        let search = *TitaniumSearch::grafted_v16_with_ceiling(g, None, ceiling);
        let engine_label = "titanium-v16".to_string();
        WasmEngine {
            search,
            engine_label,
            last_depth: 0,
            last_nodes: 0,
            last_stop_reason: "none",
        }
    }

    pub fn reset(&mut self) {
        self.search.set_position(GameState::new());
    }

    pub fn position(&mut self, moves: &str) -> Result<usize, JsError> {
        let g = replay_moves(moves)?;
        let n = moves.split_whitespace().filter(|s| !s.is_empty()).count();
        self.search.set_position(g);
        Ok(n)
    }

    pub fn make_move(&mut self, mv: &str) -> bool {
        if self.search.g.winner() >= 0 {
            return false;
        }
        self.search.apply_move(algebraic_to_move_id(mv));
        true
    }

    pub fn go(
        &mut self,
        movetime_ms: u32,
        _max_nodes: u32,
        on_progress: Option<js_sys::Function>,
    ) -> String {
        self.go_threads(movetime_ms, _max_nodes, 1, on_progress)
    }

    pub fn go_threads(
        &mut self,
        movetime_ms: u32,
        _max_nodes: u32,
        threads: u32,
        on_progress: Option<js_sys::Function>,
    ) -> String {
        let stream = on_progress.is_some();
        self.search.set_wasm_progress(on_progress.clone());
        if self.search.g.winner() >= 0 {
            self.last_depth = 0;
            self.last_nodes = 0;
            self.last_stop_reason = "terminal";
            return "(none)".to_string();
        }
        let thread_count = usize::try_from(threads.max(1)).unwrap_or(1);
        #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
        let result = self.search.think_with_threads(
            (movetime_ms as u64).max(1),
            30,
            true,
            stream,
            &self.engine_label,
            thread_count,
        );
        #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
        let result = {
            // Static web builds currently compile without wasm atomics/SharedArrayBuffer,
            // so the browser worker hosts one standalone Titanium instance. The thread
            // count stays in the WASM API contract for the future threaded build.
            let _ = thread_count;
            self.search.set_cat_lmr_worker_profile(0);
            self.search.think(
                (movetime_ms as u64).max(1),
                30,
                true,
                stream,
                &self.engine_label,
            )
        };
        self.search.set_wasm_progress(None);
        self.last_depth = result.depth;
        self.last_nodes = result.nodes;
        self.last_stop_reason = result.stop_reason;
        if stream {
            if let Some(f) = on_progress.as_ref() {
                let json = think_result_progress_json(&self.engine_label, &result);
                let _ = f.call1(&JsValue::NULL, &JsValue::from_str(&json));
            }
        }
        if result.mv == TITANIUM_NO_MOVE {
            "(none)".to_string()
        } else {
            move_id_to_algebraic(result.mv)
        }
    }

    pub fn go_threads_json(
        &mut self,
        movetime_ms: u32,
        _max_nodes: u32,
        threads: u32,
        on_progress: Option<js_sys::Function>,
    ) -> String {
        let best = self.go_threads(movetime_ms, _max_nodes, threads, on_progress);
        format!(
            "{{\"move\":{},\"depth\":{},\"nodes\":{},\"stopReason\":{}}}",
            json_string(&best),
            self.last_depth,
            self.last_nodes,
            json_string(self.last_stop_reason)
        )
    }

    pub fn last_search_depth(&self) -> i32 {
        self.last_depth
    }

    pub fn last_search_nodes(&self) -> u64 {
        self.last_nodes
    }

    pub fn last_stop_reason(&self) -> String {
        self.last_stop_reason.to_string()
    }

    pub fn engine_mode(&self) -> String {
        self.engine_label.clone()
    }

    pub fn legal_moves(&self) -> String {
        String::new()
    }

    pub fn winner(&self) -> i32 {
        let w = self.search.g.winner();
        if w < 0 {
            -1
        } else {
            w
        }
    }
}
