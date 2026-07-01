//! CAT × move-index LMR — normalized against max legal-move impact at this node.

/// Attention at or below this fraction of `Hmax` → dead tail (maximum CAT pressure).
pub const CAT_ATTENTION_TAIL_CUTOFF: f64 = 0.10;

/// Lazy-SMP worker aggression schedule (UI percent). Thread 3+ caps at 350 (not 400).
pub fn lmr_aggression_percent(thread_id: usize) -> i32 {
    match thread_id {
        0 => 177,
        1 => 200,
        2 => 277,
        _ => 350,
    }
}

/// When unset, every worker uses thread-0 aggression until path correction is validated.
pub fn lmr_thread_aggression_enabled() -> bool {
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::env::var("TITANIUM_LMR_THREAD_AGGRESSION")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    }
    #[cfg(target_arch = "wasm32")]
    {
        false
    }
}

/// Signed tuning percent for [`lmr_tuning_to_aggression_g`].
pub fn lmr_aggression_tuning_percent(thread_id: usize) -> i32 {
    let id = if lmr_thread_aggression_enabled() {
        thread_id
    } else {
        0
    };
    -lmr_aggression_percent(id)
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LmrPathDiagnostics {
    pub self_gain: i32,
    pub opponent_delay: i32,
    pub race_gain: i32,
    pub attention_ratio: f64,
    pub base_reduction: u32,
    pub path_adjustment: i32,
    pub final_reduction: u32,
    pub thread_aggression_percent: i32,
}

/// Cached d0/d1 pawn scalars before/after `make_move` + child `refresh_dist`.
#[inline]
pub fn compute_race_gain(pre_our: u8, pre_opp: u8, post_our: u8, post_opp: u8) -> (i32, i32, i32) {
    let self_gain = i32::from(pre_our) - i32::from(post_our);
    let opponent_delay = i32::from(post_opp) - i32::from(pre_opp);
    let race_gain = self_gain + opponent_delay;
    (self_gain, opponent_delay, race_gain)
}

/// Path-aware final reduction after base CAT/index LMR. Does not alter CAT cm or attention.
pub fn apply_lmr_path_correction(
    base_reduction: u32,
    full_child_depth: u32,
    race_gain: i32,
    attention_ratio: f64,
    skip_reduction: bool,
) -> LmrPathDiagnostics {
    let max_safe = full_child_depth.saturating_sub(1);
    if skip_reduction || full_child_depth <= 1 {
        return LmrPathDiagnostics {
            self_gain: 0,
            opponent_delay: 0,
            race_gain,
            attention_ratio,
            base_reduction,
            path_adjustment: 0,
            final_reduction: base_reduction.min(max_safe),
            thread_aggression_percent: 0,
        };
    }

    let mut path_adjustment = 0i32;
    let mut final_reduction = base_reduction;

    if race_gain > 0 && final_reduction > 0 {
        path_adjustment = -1;
        final_reduction = final_reduction.saturating_sub(1);
    } else if race_gain == 0 && attention_ratio <= CAT_ATTENTION_TAIL_CUTOFF {
        let delta = max_safe as i32 - final_reduction as i32;
        path_adjustment = delta;
        final_reduction = max_safe;
    }

    final_reduction = final_reduction.min(max_safe);

    LmrPathDiagnostics {
        self_gain: 0,
        opponent_delay: 0,
        race_gain,
        attention_ratio,
        base_reduction,
        path_adjustment,
        final_reduction,
        thread_aggression_percent: 0,
    }
}

/// Full LMR plan: base CAT/index reduction, cached path correction, clamp.
pub fn cat_index_lmr_with_path(
    full_child_depth: u32,
    move_rank: usize,
    move_count: usize,
    move_impact: i32,
    max_move_impact: u32,
    thread_id: usize,
    skip_reduction: bool,
    first_reducible_rank: usize,
    pre_our: u8,
    pre_opp: u8,
    post_our: u8,
    post_opp: u8,
) -> LmrPathDiagnostics {
    let thread_aggression_percent = lmr_aggression_percent(if lmr_thread_aggression_enabled() {
        thread_id
    } else {
        0
    });
    let attention_ratio = cat_attention(move_impact, max_move_impact);
    let base_reduction = cat_index_lmr_reduction(
        full_child_depth,
        move_rank,
        move_count,
        move_impact,
        max_move_impact,
        lmr_tuning_to_aggression_g(lmr_aggression_tuning_percent(thread_id)),
        skip_reduction,
        first_reducible_rank,
    );
    let (self_gain, opponent_delay, race_gain) =
        compute_race_gain(pre_our, pre_opp, post_our, post_opp);
    let mut diag = apply_lmr_path_correction(
        base_reduction,
        full_child_depth,
        race_gain,
        attention_ratio,
        skip_reduction,
    );
    diag.self_gain = self_gain;
    diag.opponent_delay = opponent_delay;
    diag.race_gain = race_gain;
    diag.thread_aggression_percent = thread_aggression_percent;
    diag
}

/// Map UI / viz tuning percent (−500..150) to aggression multiplier `g`.
pub fn lmr_tuning_to_aggression_g(tuning_percent: i32) -> f64 {
    let t = tuning_percent.clamp(-500, 150) as f64;
    if t >= 150.0 {
        return 0.0;
    }
    if t >= 100.0 {
        return (150.0 - t) / 50.0;
    }
    if t <= -500.0 {
        return 1.0;
    }
    if t < 0.0 {
        // More negative → slightly hotter cuts at the same attention (still clamped in P).
        return 1.0 + (-t / 500.0) * 0.35;
    }
    1.0
}

#[inline]
pub fn cat_attention(move_impact: i32, max_move_impact: u32) -> f64 {
    if max_move_impact == 0 {
        return 0.0;
    }
    (move_impact.max(0) as f64 / max_move_impact as f64).clamp(0.0, 1.0)
}

/// CAT reduction pressure in `[0, 1]` from normalized attention.
#[inline]
pub fn cat_pressure(attention: f64) -> f64 {
    if attention <= CAT_ATTENTION_TAIL_CUTOFF {
        1.0
    } else {
        let num = 1.0 - attention;
        let den = 1.0 - CAT_ATTENTION_TAIL_CUTOFF;
        (num / den).powi(2)
    }
}

/// Logarithmic move-rank pressure: first move → 0, last → 1.
#[inline]
pub fn move_index_pressure(move_rank: usize, move_count: usize) -> f64 {
    if move_count <= 1 || move_rank <= 1 {
        return 0.0;
    }
    let k = move_rank.min(move_count) as f64;
    let n = move_count as f64;
    (k.ln() / n.ln()).clamp(0.0, 1.0)
}

/// Combined CAT × move-index LMR reduction in plies.
pub fn cat_index_lmr_reduction(
    full_child_depth: u32,
    move_rank: usize,
    move_count: usize,
    move_impact: i32,
    max_move_impact: u32,
    aggression_g: f64,
    skip_reduction: bool,
    first_reducible_rank: usize,
) -> u32 {
    if skip_reduction
        || full_child_depth <= 1
        || move_count <= 1
        || move_rank < first_reducible_rank
        || aggression_g <= 0.0
    {
        return 0;
    }

    let max_reduction = full_child_depth.saturating_sub(1);
    let index_pressure = move_index_pressure(move_rank, move_count);

    if max_move_impact > 0 {
        let attention = cat_attention(move_impact, max_move_impact);
        if attention <= CAT_ATTENTION_TAIL_CUTOFF {
            // Dead tail → leaf eval only; do not spend even 1 child ply.
            return full_child_depth;
        }
    }

    let total_pressure = if max_move_impact == 0 {
        aggression_g * index_pressure
    } else {
        let attention = cat_attention(move_impact, max_move_impact);
        if attention <= CAT_ATTENTION_TAIL_CUTOFF {
            aggression_g
        } else {
            aggression_g * cat_pressure(attention) * index_pressure
        }
    }
    .clamp(0.0, 1.0);

    ((max_reduction as f64 * total_pressure).round() as u32).min(max_reduction)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tail_cutoff_uses_max_legal_move_impact() {
        let hmax = 621u32;
        let d = 10u32;
        let rank = 8usize;
        let n = 20usize;
        let g = 1.0;

        let dead = cat_index_lmr_reduction(d, rank, n, 40, hmax, g, false, 2);
        let fringe = cat_index_lmr_reduction(d, rank, n, 77, hmax, g, false, 2);
        let core = cat_index_lmr_reduction(d, rank, n, 400, hmax, g, false, 2);

        assert_eq!(dead, 10, "40/621 ≈ 6.4% → dead tail → leaf (depth 0)");
        assert!(
            dead > fringe,
            "77/621 ≈ 12.4% should survive hard tail with less reduction, dead={dead} fringe={fringe}"
        );
        assert!(
            fringe > core,
            "400/621 should keep more depth than 77, got core={core} fringe={fringe}"
        );
    }

    #[test]
    fn hottest_move_keeps_full_depth_at_front() {
        let d = 10u32;
        let g = 1.0;
        let hmax = 621u32;
        let r = cat_index_lmr_reduction(d, 1, 20, 621, hmax, g, false, 2);
        assert_eq!(r, 0, "first move has zero index pressure");
    }

    #[test]
    fn skipped_and_shallow_skip_reduction() {
        assert_eq!(cat_index_lmr_reduction(1, 5, 10, 10, 100, 1.0, false, 2), 0);
        assert_eq!(cat_index_lmr_reduction(10, 1, 10, 10, 100, 1.0, true, 2), 0);
        assert_eq!(
            cat_index_lmr_reduction(10, 3, 10, 10, 100, 1.0, false, 5),
            0
        );
    }

    #[test]
    fn tuning_150_disables_reduction() {
        let g = lmr_tuning_to_aggression_g(150);
        assert_eq!(g, 0.0);
        let r = cat_index_lmr_reduction(10, 8, 20, 40, 621, g, false, 2);
        assert_eq!(r, 0);
    }

    #[test]
    fn lmr_aggression_schedule_matches_spec() {
        assert_eq!(lmr_aggression_percent(0), 177);
        assert_eq!(lmr_aggression_percent(1), 200);
        assert_eq!(lmr_aggression_percent(2), 277);
        assert_eq!(lmr_aggression_percent(3), 350);
        assert_eq!(lmr_aggression_percent(9), 350);
    }

    #[test]
    fn path_correction_reduces_cut_when_race_improves() {
        let base = cat_index_lmr_reduction(10, 8, 20, 400, 621, 1.0, false, 2);
        assert!(base > 0);
        let diag = apply_lmr_path_correction(base, 10, 1, 0.5, false);
        assert_eq!(diag.path_adjustment, -1);
        assert_eq!(diag.final_reduction, base - 1);
    }

    #[test]
    fn path_correction_dead_tail_when_no_race_and_cold_attention() {
        let base = cat_index_lmr_reduction(10, 8, 20, 40, 621, 1.0, false, 2);
        assert_eq!(base, 10);
        let diag = apply_lmr_path_correction(base, 10, 0, 0.06, false);
        assert_eq!(diag.final_reduction, 9);
        assert_eq!(diag.path_adjustment, -1);
    }

    #[test]
    fn thread_aggression_gated_until_env_enabled() {
        assert_eq!(lmr_aggression_tuning_percent(3), -177);
    }

    #[test]
    fn with_path_wires_cached_scalars() {
        let diag = cat_index_lmr_with_path(10, 8, 20, 400, 621, 0, false, 2, 8, 6, 7, 7);
        assert_eq!(diag.self_gain, 1);
        assert_eq!(diag.opponent_delay, 1);
        assert_eq!(diag.race_gain, 2);
        assert!(diag.base_reduction > 0);
        assert_eq!(diag.final_reduction, diag.base_reduction - 1);
        assert_eq!(diag.thread_aggression_percent, 177);
    }
}
