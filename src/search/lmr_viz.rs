//! Root LMR plan snapshot — mirrors alphabeta root move list + planned reductions.

use crate::cat::build::build_impact_heatmap;
use crate::cat::constants::DIST_PENALTY;
use crate::cat::prune::{
    get_shortest_path, is_lmr_heat_hot, is_tactical_move, move_impact_heat, order_moves,
    path_distance,
};
use crate::cat::CorridorAttention;
use crate::core::board::{Board, Move};
use crate::movegen::{generate_legal_moves_slice, MAX_LEGAL_MOVES};
use crate::path::BfsScratch;
use crate::search::v16_lmr::{
    plan_v16_pawn_lmr, plan_v16_wall_lmr, ACE_LMR_AFTER_MOVE, ACE_LMR_MIN_DEPTH, V16HardOverride,
};
use crate::search::lmr_profile::{compute_stage_t, LmrProfile};
use crate::util::perft::format_move;

const LMR_MIN_DEPTH: u32 = 2;

#[derive(Debug, Clone)]
pub struct RootLmrPlan {
    pub mv: String,
    pub is_pawn: bool,
    pub order: usize,
    pub cat_cm: i32,
    pub tactical: bool,
    pub hot: bool,
    pub cold: bool,
    pub pruned: bool,
    pub baseline_reduction_fp: f64,
    pub baseline_reduction: u32,
    pub baseline_child_depth_full: u32,
    pub baseline_child_depth_used: u32,
    pub requested_reduction_fp: f64,
    pub reduction: u32,
    pub child_depth_full: u32,
    pub child_depth_used: u32,
    pub reduction_clamped: bool,
    pub in_full_window: bool,
    pub attention_ratio: f64,
    pub dead_tail: bool,
    pub ace_base_reduction: u32,
    pub hard_override: &'static str,
    pub final_reduction: u32,
}

#[derive(Debug, Clone, Default)]
pub struct LmrPlanSummary {
    pub moves_more_reduction: u32,
    pub avg_baseline_reduction_fp: f64,
    pub avg_adjusted_reduction_fp: f64,
    pub max_baseline_reduction: u32,
    pub max_adjusted_reduction: u32,
    pub hot_count: u32,
    pub cold_count: u32,
}

fn plan_v16_root_move(
    move_index: usize,
    depth: u32,
    child_depth_full: u32,
    attention_ratio: f64,
    wall_opponent_delay: Option<i32>,
    wall_self_delay: Option<i32>,
    pawn_self_gain: Option<i32>,
    is_wall: bool,
) -> (u32, &'static str, u32, u32) {
    if is_wall
        && move_index >= ACE_LMR_AFTER_MOVE
        && depth >= ACE_LMR_MIN_DEPTH as u32
        && child_depth_full > 1
    {
        let p = plan_v16_wall_lmr(
            move_index,
            depth as i32,
            child_depth_full as i32,
            attention_ratio,
            wall_opponent_delay.unwrap_or(1),
            wall_self_delay.unwrap_or(0),
        );
        return (
            p.ace_base_reduction as u32,
            p.hard_override.as_str(),
            p.final_reduction as u32,
            p.child_depth_used as u32,
        );
    }
    if !is_wall && move_index > 0 && depth >= ACE_LMR_MIN_DEPTH as u32 && child_depth_full > 1 {
        if let Some(gain) = pawn_self_gain {
            if let Some(p) = plan_v16_pawn_lmr(
                move_index,
                depth as i32,
                child_depth_full as i32,
                gain,
            ) {
                return (
                    p.ace_base_reduction as u32,
                    p.hard_override.as_str(),
                    p.final_reduction as u32,
                    p.child_depth_used as u32,
                );
            }
        }
    }
    (
        0,
        V16HardOverride::None.as_str(),
        0,
        child_depth_full,
    )
}

fn root_cat_heat_stats(moves: &[Move], n: usize, cat: &CorridorAttention) -> (u16, u16) {
    let mut heats = Vec::with_capacity(n);
    for mv in &moves[..n] {
        heats.push(move_impact_heat(*mv, cat).max(0) as u16);
    }
    if heats.is_empty() {
        return (0, 0);
    }
    heats.sort_by(|a, b| b.cmp(a));
    let max = heats[0];
    let p75_idx = (heats.len() * 3 / 4).min(heats.len() - 1);
    (max, heats[p75_idx])
}

fn summarize_plans(plans: &[RootLmrPlan]) -> LmrPlanSummary {
    let mut summary = LmrPlanSummary::default();
    let mut baseline_sum = 0.0;
    let mut adjusted_sum = 0.0;
    let mut n_eligible = 0u32;
    for p in plans {
        if p.hot {
            summary.hot_count += 1;
        }
        if p.cold {
            summary.cold_count += 1;
        }
        n_eligible += 1;
        baseline_sum += p.baseline_reduction_fp;
        adjusted_sum += p.requested_reduction_fp;
        summary.max_baseline_reduction = summary.max_baseline_reduction.max(p.baseline_reduction);
        summary.max_adjusted_reduction = summary.max_adjusted_reduction.max(p.reduction);
        if p.reduction > p.baseline_reduction {
            summary.moves_more_reduction += 1;
        }
    }
    if n_eligible > 0 {
        let n = f64::from(n_eligible);
        summary.avg_baseline_reduction_fp = baseline_sum / n;
        summary.avg_adjusted_reduction_fp = adjusted_sum / n;
    }
    summary
}

/// Planned root LMR for `id_depth` at `pierce_fraction` elapsed (0 = pierce peak).
pub fn plan_root_lmr(
    board: &mut Board,
    bfs: &mut BfsScratch,
    id_depth: u32,
    time_ms: u64,
    pierce_fraction: f32,
    depth_kept_percent: i32,
) -> (LmrProfile, Vec<RootLmrPlan>) {
    let _depth_kept_percent = depth_kept_percent;
    let root_side = board.side();
    let opp_side = root_side.opposite();
    let our_dist = bfs
        .shortest_distance(board, root_side)
        .unwrap_or(DIST_PENALTY);
    let opp_dist = bfs
        .shortest_distance(board, opp_side)
        .unwrap_or(DIST_PENALTY);
    let endgame_race = our_dist.min(opp_dist) <= 4;

    let mut opp_path = [0u8; 81];
    let opp_path_len = get_shortest_path(board, opp_side, bfs, &mut opp_path);
    let opp_dist_path = path_distance(opp_side, &opp_path, opp_path_len);

    // Symmetric view: both players' corridors visible, no STM rear-zeroing.
    // Walls that block the opponent's path register even when they're "behind" us.
    let cat = build_impact_heatmap(board);

    let mut buf = [Move::Pawn { row: 1, col: 4 }; MAX_LEGAL_MOVES];
    let n = generate_legal_moves_slice(board, &mut buf, bfs);
    if n == 0 {
        return (LmrProfile::from_stage(0.5, endgame_race, false), Vec::new());
    }

    let (cat_max_seed, cat_p75) = root_cat_heat_stats(&buf, n, &cat);
    let stage_t = compute_stage_t(board, our_dist, opp_dist, cat_max_seed, cat_p75);

    let mut profile = LmrProfile::from_stage(stage_t, endgame_race, false);
    profile.apply_time_budget(time_ms);
    profile.apply_pierce_schedule(pierce_fraction, time_ms);

    let mut scores = [0i32; MAX_LEGAL_MOVES];
    order_moves(
        board,
        &mut buf,
        n,
        None,
        None,
        &mut scores,
        our_dist,
        opp_dist_path,
        &opp_path,
        opp_path_len,
        bfs,
        &cat,
        &crate::cat::prune::OrderExtras::default(),
        |_| 0,
    );

    let mut cat_values = Vec::with_capacity(n);
    let mut max_move_impact = 0u32;
    for i in 0..n {
        let mv = buf[i];
        let cm = move_impact_heat(mv, &cat);
        cat_values.push(cm);
        max_move_impact = max_move_impact.max(cm.max(0) as u32);
    }

    let depth = id_depth.max(1);
    let child_depth_full = depth.saturating_sub(1);

    let mut plans = Vec::with_capacity(n);

    for i in 0..n {
        let mv = buf[i];
        let cat_cm = cat_values[i];
        let heat_ratio_hot = is_lmr_heat_hot(
            cat_cm,
            max_move_impact as u16,
            profile.cold_cm,
            profile.hot_ratio_pct,
        );
        let cold = cat_cm < i32::from(profile.cold_cm);
        let is_wall = matches!(mv, Move::Wall { .. });
        let is_tactical = if i == 0 || depth < LMR_MIN_DEPTH {
            true
        } else if is_wall && !crate::cat::prune::wall_intersects_path(mv, &opp_path, opp_path_len)
        {
            false
        } else {
            is_tactical_move(board, mv, our_dist, opp_dist_path, bfs)
        };
        let attention = if max_move_impact > 0 {
            cat_cm.max(0) as f64 / max_move_impact as f64
        } else {
            0.0
        };

        let (wall_opponent_delay, wall_self_delay) = if is_wall && i >= ACE_LMR_AFTER_MOVE {
            let mut trial = board.clone();
            trial.apply_move(mv);
            let opp_after = bfs
                .shortest_distance(&trial, opp_side)
                .unwrap_or(DIST_PENALTY);
            let our_after = bfs
                .shortest_distance(&trial, root_side)
                .unwrap_or(DIST_PENALTY);
            (
                Some(i32::from(opp_after) - i32::from(opp_dist_path)),
                Some(i32::from(our_after) - i32::from(our_dist)),
            )
        } else {
            (None, None)
        };

        let pawn_self_gain = if !is_wall && i > 0 {
            let mut trial = board.clone();
            trial.apply_move(mv);
            let after = bfs
                .shortest_distance(&trial, root_side)
                .unwrap_or(DIST_PENALTY);
            Some(i32::from(our_dist) - i32::from(after))
        } else {
            None
        };

        let (ace_base, hard_override, final_reduction, child_used) = plan_v16_root_move(
            i,
            depth,
            child_depth_full,
            attention,
            wall_opponent_delay,
            wall_self_delay,
            pawn_self_gain,
            is_wall,
        );

        let in_full_window = child_used >= child_depth_full.saturating_sub(1);
        let dead_tail = hard_override == V16HardOverride::DeadTail.as_str();

        plans.push(RootLmrPlan {
            mv: format_move(mv),
            is_pawn: !is_wall,
            order: i,
            cat_cm,
            tactical: is_tactical,
            hot: heat_ratio_hot || attention >= 0.72,
            cold,
            pruned: false,
            baseline_reduction_fp: ace_base as f64,
            baseline_reduction: ace_base,
            baseline_child_depth_full: child_depth_full,
            baseline_child_depth_used: child_depth_full.saturating_sub(ace_base),
            requested_reduction_fp: final_reduction as f64,
            reduction: final_reduction,
            child_depth_full,
            child_depth_used: child_used,
            reduction_clamped: final_reduction > child_depth_full.saturating_sub(1),
            in_full_window,
            attention_ratio: attention,
            dead_tail,
            ace_base_reduction: ace_base,
            hard_override,
            final_reduction,
        });
    }

    (profile, plans)
}

pub fn lmr_profile_fields(profile: &LmrProfile, id_depth: u32) -> String {
    format!(
        "{{\"stageT\":{:.3},\"aggression\":{:.2},\"pierceT\":{:.3},\"moveWindow\":{},\"lmrAfter\":{},\"hotPct\":{},\"coldCm\":{},\"idDepth\":{}}}",
        profile.stage_t,
        profile.aggression,
        profile.pierce_t,
        profile.move_window,
        profile.lmr_after_move,
        profile.hot_ratio_pct,
        profile.cold_cm,
        id_depth,
    )
}

pub fn format_lmr_plans_json(plans: &[RootLmrPlan]) -> String {
    let summary = summarize_plans(plans);
    let mut out = String::new();
    for (i, p) in plans.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&format!(
            "{{\"move\":\"{}\",\"kind\":\"{}\",\"order\":{},\"catCm\":{},\"tactical\":{},\"hot\":{},\"cold\":{},\"pruned\":{},\
\"baselineReductionFp\":{:.4},\"baselineReduction\":{},\"baselineChildDepthFull\":{},\"baselineChildDepthUsed\":{},\
\"requestedReductionFp\":{:.4},\"reduction\":{},\"childDepthFull\":{},\"childDepthUsed\":{},\"reductionClamped\":{},\"inFullWindow\":{},\"attentionRatio\":{:.4},\"deadTail\":{},\
\"aceBaseReduction\":{},\"hardOverride\":\"{}\",\"finalReduction\":{}}}",
            p.mv,
            if p.is_pawn { "pawn" } else { "wall" },
            p.order,
            p.cat_cm,
            p.tactical,
            p.hot,
            p.cold,
            p.pruned,
            p.baseline_reduction_fp,
            p.baseline_reduction,
            p.baseline_child_depth_full,
            p.baseline_child_depth_used,
            p.requested_reduction_fp,
            p.reduction,
            p.child_depth_full,
            p.child_depth_used,
            p.reduction_clamped,
            p.in_full_window,
            p.attention_ratio,
            p.dead_tail,
            p.ace_base_reduction,
            p.hard_override,
            p.final_reduction,
        ));
    }
    format!(
        "\"summary\":{{\"movesMoreReduction\":{},\"avgBaselineReductionFp\":{:.4},\"avgAdjustedReductionFp\":{:.4},\"maxBaselineReduction\":{},\"maxAdjustedReduction\":{},\"hotCount\":{},\"coldCount\":{}}},\"moves\":[{}]",
        summary.moves_more_reduction,
        summary.avg_baseline_reduction_fp,
        summary.avg_adjusted_reduction_fp,
        summary.max_baseline_reduction,
        summary.max_adjusted_reduction,
        summary.hot_count,
        summary.cold_count,
        out,
    )
}

/// Merge the opponent's plan into the side-to-move plan so the display shows
/// BOTH players' perspectives: a move keeps the deeper of the two plans
/// (a wall matters if it matters for either side); opponent-only moves
/// (their pawn steps, walls the mover can't place) are appended.
fn merge_combined_plans(plans: &mut Vec<RootLmrPlan>, opp_plans: Vec<RootLmrPlan>) {
    use std::collections::HashMap;
    let idx: HashMap<String, usize> = plans
        .iter()
        .enumerate()
        .map(|(i, p)| (p.mv.clone(), i))
        .collect();
    for op in opp_plans {
        if let Some(&i) = idx.get(&op.mv) {
            let p = &mut plans[i];
            p.cat_cm = p.cat_cm.max(op.cat_cm);
            p.hot |= op.hot;
            if op.child_depth_used > p.child_depth_used {
                let order = p.order;
                *p = op;
                p.order = order;
            }
        } else {
            let mut op = op;
            op.order = plans.len();
            plans.push(op);
        }
    }
}

/// Pre-search LMR plan — static profile at pierce peak, combined for both players.
pub fn lmr_snapshot_json(
    board: &mut Board,
    time_ms: u64,
    id_depth: u32,
    depth_kept_percent: i32,
) -> String {
    let mut bfs = BfsScratch::new();
    let depth = id_depth.clamp(4, 32);
    let (profile, mut plans) =
        plan_root_lmr(board, &mut bfs, depth, time_ms, 0.0, depth_kept_percent);
    // Display is not turn-based: plan the opponent's move set too and merge.
    // The search itself stays per-mover — this only widens the visualization.
    board.side_to_move = board.side_to_move.opposite();
    let (_, opp_plans) = plan_root_lmr(board, &mut bfs, depth, time_ms, 0.0, depth_kept_percent);
    board.side_to_move = board.side_to_move.opposite();
    merge_combined_plans(&mut plans, opp_plans);
    let plans = plans;
    format!(
        "{{\"source\":\"shallow\",\"idDepth\":{},\"timeMs\":{},\"lmrAggressionPercent\":{},\"lmrTuningPercent\":{},\"lmrProfile\":{},{}}}",
        depth,
        time_ms,
        depth_kept_percent.clamp(-500, 150),
        depth_kept_percent.clamp(-500, 150),
        lmr_profile_fields(&profile, depth),
        format_lmr_plans_json(&plans),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashMap, HashSet};

    use crate::cat::constants::CAT_COLD_CM;
    use crate::cat::prune::is_cat_hot_corridor;
    use crate::search::lmr_profile::TIME_REFERENCE_MS;

    #[test]
    fn cat_hot_cold_threshold_semantics() {
        assert!(is_cat_hot_corridor(160));
        assert!(!is_cat_hot_corridor(159));
        assert!(59 < i32::from(CAT_COLD_CM));
        assert!(!(60 < i32::from(CAT_COLD_CM)));
    }

    #[test]
    fn shallow_snapshot_has_legal_moves() {
        let mut board = Board::new();
        let mut bfs = BfsScratch::new();
        let (_, plans) = plan_root_lmr(&mut board, &mut bfs, 8, TIME_REFERENCE_MS, 0.0, 100);
        assert!(plans.len() >= 4);
        assert!(plans[0].tactical);
    }

    #[test]
    fn lmr_vision_cat_cm_matches_impact_heatmap() {
        let mut board = Board::new();
        for m in ["e2", "e8", "e3", "e7", "e4", "e6"] {
            board.apply_algebraic(m);
        }
        let fixture = board.clone();
        let mut bfs = BfsScratch::new();
        let cat = build_impact_heatmap(&fixture);
        let mut work = board;
        let mut legal_buf = [Move::Pawn { row: 0, col: 0 }; MAX_LEGAL_MOVES];
        let mut legal_board = fixture.clone();
        let legal_n = generate_legal_moves_slice(&mut legal_board, &mut legal_buf, &mut bfs);
        let (_, plans) = plan_root_lmr(&mut work, &mut bfs, 8, TIME_REFERENCE_MS, 0.0, 100);
        assert_eq!(
            plans.len(),
            plans.iter().map(|p| &p.mv).collect::<HashSet<_>>().len(),
            "duplicate moves"
        );
        let by_mv: HashMap<String, i32> = plans.iter().map(|p| (p.mv.clone(), p.cat_cm)).collect();
        assert_eq!(by_mv.len(), legal_n, "one plan per legal move");
        for mv in &legal_buf[..legal_n] {
            let key = format_move(*mv);
            let reported = *by_mv.get(&key).unwrap_or_else(|| panic!("missing {key}"));
            let expected = move_impact_heat(*mv, &cat);
            assert_eq!(reported, expected, "LMR vision mismatch for {key}");
        }
    }

    #[test]
    fn dead_tail_forces_depth_one_at_ten_percent() {
        let child_full = 7i32;
        let dead = plan_v16_wall_lmr(8, 8, child_full, 0.05, 1, 0);
        let fringe = plan_v16_wall_lmr(8, 8, child_full, 0.12, 1, 0);
        let side = plan_v16_wall_lmr(8, 8, child_full, 0.65, 1, 0);
        assert_eq!(dead.child_depth_used, 1);
        assert_eq!(dead.hard_override, V16HardOverride::DeadTail);
        assert!(fringe.child_depth_used > dead.child_depth_used);
        assert!(side.child_depth_used >= fringe.child_depth_used);
    }

    #[test]
    fn ace_baseline_ignores_cat_ratio_above_tail() {
        let child_full = 7i32;
        let mid = plan_v16_wall_lmr(12, 8, child_full, 0.50, 1, 0);
        let hot = plan_v16_wall_lmr(12, 8, child_full, 1.0, 1, 0);
        assert_eq!(mid.ace_base_reduction, hot.ace_base_reduction);
        assert_eq!(mid.final_reduction, hot.final_reduction);
        assert_eq!(mid.child_depth_used, hot.child_depth_used);
        assert_eq!(mid.hard_override, V16HardOverride::None);
    }

    #[test]
    fn backward_move_override_forces_depth_one() {
        let p = plan_v16_pawn_lmr(3, 8, 7, -1).expect("backwards");
        assert_eq!(p.hard_override, V16HardOverride::BackwardMove);
        assert_eq!(p.child_depth_used, 1);
    }

    #[test]
    fn ace_graduated_increases_with_move_index() {
        let child_full = 10i32;
        let early = plan_v16_wall_lmr(4, 8, child_full, 0.5, 1, 0);
        let late = plan_v16_wall_lmr(12, 8, child_full, 0.5, 1, 0);
        let very_late = plan_v16_wall_lmr(24, 8, child_full, 0.5, 1, 0);
        assert!(late.final_reduction > early.final_reduction);
        assert!(very_late.final_reduction > late.final_reduction);
    }

    #[test]
    fn tuning_slider_no_longer_changes_planned_reduction() {
        let mut board = Board::new();
        let mut bfs = BfsScratch::new();
        let (_, at_zero) = plan_root_lmr(&mut board, &mut bfs, 8, TIME_REFERENCE_MS, 0.0, 0);
        let (_, at_full) = plan_root_lmr(&mut board, &mut bfs, 8, TIME_REFERENCE_MS, 0.0, 150);
        for (a, b) in at_zero.iter().zip(at_full.iter()) {
            assert_eq!(a.mv, b.mv);
            assert_eq!(
                a.final_reduction, b.final_reduction,
                "{} reduction must not depend on tuning slider",
                a.mv
            );
            assert_eq!(a.child_depth_used, b.child_depth_used);
        }
    }

    #[test]
    fn engine_default_snapshot_shows_full_depth_and_hard_wall_reductions() {
        let mut board = Board::new();
        board.apply_algebraic("e3");
        board.apply_algebraic("e8");
        let mut bfs = BfsScratch::new();
        let (_, plans) = plan_root_lmr(
            &mut board,
            &mut bfs,
            11,
            TIME_REFERENCE_MS,
            0.0,
            100,
        );
        assert!(
            plans
                .iter()
                .any(|p| p.reduction == 0 && p.child_depth_used == 10),
            "hot/PV moves should still keep full depth"
        );
        assert!(
            plans
                .iter()
                .any(|p| p.hard_override == V16HardOverride::BackwardMove.as_str()
                    && p.child_depth_used <= 1),
            "backward/useless walls should be visibly reduced to child depth 1"
        );
    }
}
