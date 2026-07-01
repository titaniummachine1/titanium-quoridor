//! Benchmark-only counters and phase timers (`bench-instrument` feature).

use std::time::{Duration, Instant};

#[derive(Clone, Copy, Default, Debug)]
pub struct OpStat {
    pub calls: u64,
    pub ns: u128,
}

impl OpStat {
    #[inline]
    pub fn record(&mut self, dt: Duration) {
        self.calls += 1;
        self.ns += dt.as_nanos();
    }

    pub fn ns_per_call(&self) -> f64 {
        if self.calls == 0 {
            0.0
        } else {
            self.ns as f64 / self.calls as f64
        }
    }
}

#[derive(Clone, Default, Debug)]
pub struct BenchInstr {
    pub search_nodes: u64,
    pub stop_reason: &'static str,
    pub evaluate: OpStat,
    pub eval_race_bound: OpStat,
    pub race_gate_cached: OpStat,
    pub race_winner_table: OpStat,
    pub eval_route_features: OpStat,
    pub eval_nnue_prep: OpStat,
    pub eval_nnue_infer: OpStat,
    pub eval_wall_cross: OpStat,
    pub eval_legal_wall_count: OpStat,
    pub eval_misc_scalar: OpStat,
    pub refresh_dist: OpStat,
    pub shortest_path: OpStat,
    pub dir_masks_from_ace: OpStat,
    pub flood_bit_sq: OpStat,
    pub flood_bit_index: OpStat,
    pub flood_sq_from_bit: OpStat,
    pub flood_scatter: OpStat,
    pub unpack_square: OpStat,
    pub wall_crossing_count: OpStat,
    pub collect_wall_orientation: OpStat,
    pub wall_legality: OpStat,
    pub can_step: OpStat,
    pub gen_moves: OpStat,
    pub tt_probe: OpStat,
    pub tt_hit: OpStat,
    pub tt_cutoff: OpStat,
    pub tt_store: OpStat,
    pub make_move: OpStat,
    pub unmake_move: OpStat,
    pub nnue_full_refresh: OpStat,
    pub nnue_incr_update: OpStat,
    search_t0: Option<Instant>,
    measured_ns: u128,
}

impl BenchInstr {
    pub fn begin_search(&mut self) {
        *self = Self {
            search_t0: Some(Instant::now()),
            ..Default::default()
        };
    }

    pub fn end_search(&mut self, nodes: u64) {
        self.search_nodes = nodes;
        if let Some(t0) = self.search_t0.take() {
            self.measured_ns = t0.elapsed().as_nanos();
        }
    }

    pub fn set_stop_reason(&mut self, reason: &'static str) {
        self.stop_reason = reason;
    }

    pub fn to_json(&self) -> String {
        fn row(name: &str, s: &OpStat, nodes: u64, total_ns: u128) -> String {
            let cpn = if nodes == 0 {
                0.0
            } else {
                s.calls as f64 / nodes as f64
            };
            let pct = if total_ns == 0 {
                0.0
            } else {
                100.0 * s.ns as f64 / total_ns as f64
            };
            format!(
                r#"{{"op":"{name}","calls":{calls},"calls_per_node":{cpn:.4},"total_ns":{ns},"ns_per_call":{npc:.1},"pct_measured":{pct:.2}}}"#,
                name = name,
                calls = s.calls,
                cpn = cpn,
                ns = s.ns,
                npc = s.ns_per_call(),
                pct = pct,
            )
        }
        let nodes = self.search_nodes;
        let total_ns = self.measured_ns;
        let ops: [(&str, &OpStat); 31] = [
            ("evaluate", &self.evaluate),
            ("eval_race_bound", &self.eval_race_bound),
            ("race_gate_cached", &self.race_gate_cached),
            ("race_winner_table", &self.race_winner_table),
            ("eval_route_features", &self.eval_route_features),
            ("eval_nnue_prep", &self.eval_nnue_prep),
            ("eval_nnue_infer", &self.eval_nnue_infer),
            ("eval_wall_cross", &self.eval_wall_cross),
            ("eval_legal_wall_count", &self.eval_legal_wall_count),
            ("eval_misc_scalar", &self.eval_misc_scalar),
            ("refresh_dist", &self.refresh_dist),
            ("shortest_path", &self.shortest_path),
            ("dir_masks_from_ace", &self.dir_masks_from_ace),
            ("flood_bit_sq", &self.flood_bit_sq),
            ("flood_bit_index", &self.flood_bit_index),
            ("flood_sq_from_bit", &self.flood_sq_from_bit),
            ("flood_scatter", &self.flood_scatter),
            ("unpack_square", &self.unpack_square),
            ("wall_crossing_count", &self.wall_crossing_count),
            ("collect_wall_orientation", &self.collect_wall_orientation),
            ("wall_legality", &self.wall_legality),
            ("can_step", &self.can_step),
            ("gen_moves", &self.gen_moves),
            ("tt_probe", &self.tt_probe),
            ("tt_hit", &self.tt_hit),
            ("tt_cutoff", &self.tt_cutoff),
            ("tt_store", &self.tt_store),
            ("make_move", &self.make_move),
            ("unmake_move", &self.unmake_move),
            ("nnue_full_refresh", &self.nnue_full_refresh),
            ("nnue_incr_update", &self.nnue_incr_update),
        ];
        let parts: Vec<String> = ops
            .iter()
            .map(|(name, s)| row(name, s, nodes, total_ns))
            .collect();
        format!(
            r#"{{"search_nodes":{nodes},"measured_ns":{total_ns},"stop_reason":"{}","ops":[{}]}}"#,
            self.stop_reason,
            parts.join(",")
        )
    }
}

thread_local! {
    static BENCH: std::cell::RefCell<BenchInstr> = std::cell::RefCell::new(BenchInstr::default());
}

pub fn with_bench<F, R>(f: F) -> R
where
    F: FnOnce(&mut BenchInstr) -> R,
{
    BENCH.with(|c| f(&mut c.borrow_mut()))
}

#[inline(always)]
pub fn record<F, R>(pick: fn(&mut BenchInstr) -> &mut OpStat, body: F) -> R
where
    F: FnOnce() -> R,
{
    #[cfg(feature = "bench-instrument")]
    {
        let t0 = Instant::now();
        let out = body();
        with_bench(|b| pick(b).record(t0.elapsed()));
        out
    }
    #[cfg(not(feature = "bench-instrument"))]
    body()
}

/// Count-only on nanosecond-hot paths (timing omitted to avoid probe skew).
#[inline(always)]
pub fn count<F, R>(pick: fn(&mut BenchInstr) -> &mut OpStat, body: F) -> R
where
    F: FnOnce() -> R,
{
    #[cfg(feature = "bench-instrument")]
    {
        let out = body();
        with_bench(|b| pick(b).calls += 1);
        out
    }
    #[cfg(not(feature = "bench-instrument"))]
    body()
}

#[inline(always)]
pub fn bump(pick: fn(&mut BenchInstr) -> &mut OpStat) {
    #[cfg(feature = "bench-instrument")]
    with_bench(|b| {
        pick(b).calls += 1;
    });
}

pub fn begin_search() {
    #[cfg(feature = "bench-instrument")]
    with_bench(|b| b.begin_search());
}

pub fn end_search(nodes: u64) {
    #[cfg(feature = "bench-instrument")]
    with_bench(|b| b.end_search(nodes));
}

pub fn set_stop_reason(reason: &'static str) {
    #[cfg(feature = "bench-instrument")]
    with_bench(|b| b.set_stop_reason(reason));
}

pub fn take_json_report() -> Option<String> {
    #[cfg(feature = "bench-instrument")]
    {
        return Some(with_bench(|b| b.to_json()));
    }
    #[cfg(not(feature = "bench-instrument"))]
    None
}

/// RAII timer for multi-statement regions (e.g. `evaluate`).
pub struct OpTimer {
    #[cfg(feature = "bench-instrument")]
    pick: fn(&mut BenchInstr) -> &mut OpStat,
    #[cfg(feature = "bench-instrument")]
    t0: Instant,
}

impl OpTimer {
    #[inline(always)]
    pub fn start(pick: fn(&mut BenchInstr) -> &mut OpStat) -> Self {
        #[cfg(feature = "bench-instrument")]
        {
            Self {
                pick,
                t0: Instant::now(),
            }
        }
        #[cfg(not(feature = "bench-instrument"))]
        {
            let _ = pick;
            Self {}
        }
    }
}

impl Drop for OpTimer {
    fn drop(&mut self) {
        #[cfg(feature = "bench-instrument")]
        {
            let dt = self.t0.elapsed();
            with_bench(|b| (self.pick)(b).record(dt));
        }
    }
}
