//! Root LMR plan snapshot — mirrors alphabeta root move list + planned reductions.

use crate::cat::constants::DIST_PENALTY;
use crate::cat::prune::{
    collect_search_moves, get_shortest_path, is_tactical_move, move_corridor_attention,
    order_moves, path_distance,
};
use crate::cat::CorridorAttention;
use crate::core::board::{Board, Move};
use crate::movegen::MAX_LEGAL_MOVES;
use crate::path::BfsScratch;
use crate::search::lmr_profile::{build_lmr_table, compute_stage_t, LmrProfile};
use crate::search::root_cap::root_wall_keep_mask;
use crate::util::perft::format_move;

const LMR_MIN_DEPTH: u32 = 2;
const ROOT_WALL_CAP_OPENING: usize = 26;
const ROOT_WALL_CAP_MID: usize = 38;

#[derive(Debug, Clone)]
pub struct RootLmrPlan {
    pub mv: String,
    pub is_pawn: bool,
    pub order: usize,
    pub cat_cm: i32,
    pub tactical: bool,
    pub hot: bool,
    pub pruned: bool,
    pub reduction: u32,
    pub child_depth_full: u32,
    pub child_depth_used: u32,
    pub in_full_window: bool,
}

fn root_cat_heat_stats(
    board: &Board,
    moves: &[Move],
    n: usize,
    cat: &CorridorAttention,
) -> (u16, u16) {
    let mut heats = Vec::with_capacity(n);
    for mv in &moves[..n] {
        heats.push(move_corridor_attention(board, *mv, cat).max(0) as u16);
    }
    if heats.is_empty() {
        return (0, 0);
    }
    heats.sort_by(|a, b| b.cmp(a));
    let max = heats[0];
    let p75_idx = (heats.len() * 3 / 4).min(heats.len() - 1);
    (max, heats[p75_idx])
}

/// Planned root LMR for `id_depth` at `pierce_fraction` elapsed (0 = pierce peak).
pub fn plan_root_lmr(
    board: &mut Board,
    bfs: &mut BfsScratch,
    id_depth: u32,
    time_ms: u64,
    pierce_fraction: f32,
) -> (LmrProfile, Vec<RootLmrPlan>) {
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

    let cat = bfs.build_corridor_attention(board);

    let mut buf = [Move::Pawn { row: 1, col: 4 }; MAX_LEGAL_MOVES];
    let mut n = collect_search_moves(
        board,
        &mut buf,
        bfs,
        &cat,
        &opp_path,
        opp_path_len,
        our_dist,
        opp_dist_path,
        false,
        true,
    );
    if n == 0 {
        return (LmrProfile::from_stage(0.5, endgame_race, false), Vec::new());
    }

    let (cat_max_seed, cat_p75) = root_cat_heat_stats(board, &buf, n, &cat);
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

    let mut cat_max = 0u16;
    for j in 0..n {
        let cm = move_corridor_attention(board, buf[j], &cat).max(0) as u16;
        cat_max = cat_max.max(cm);
    }

    let wall_keep = if id_depth >= 3 {
        let cap = profile.root_wall_cap().min(if profile.stage_t < 0.40 {
            ROOT_WALL_CAP_OPENING
        } else {
            ROOT_WALL_CAP_MID
        });
        root_wall_keep_mask(&buf, n, &cat, cap)
    } else {
        [true; MAX_LEGAL_MOVES]
    };

    let lmr_table = build_lmr_table(profile.aggression);
    let depth = id_depth.max(1);
    let child_depth_full = depth.saturating_sub(1);
    let full_depth_slots = profile.move_window.max(profile.lmr_after_move);

    let mut plans = Vec::with_capacity(n);
    let mut moves_searched = 0usize;

    for i in 0..n {
        let mv = buf[i];
        let cat_cm = move_corridor_attention(board, mv, &cat);
        let heat_ratio_hot = cat_max > 0
            && (cat_cm.max(0) as u32) * 100 >= (cat_max as u32) * u32::from(profile.hot_ratio_pct);
        let corridor_relevant = cat_cm >= i32::from(profile.cold_cm);
        let in_full_window = moves_searched < full_depth_slots;
        let is_tactical = if moves_searched == 0
            || depth < LMR_MIN_DEPTH
            || heat_ratio_hot
        {
            true
        } else if matches!(mv, Move::Wall { .. })
            && !crate::cat::prune::wall_intersects_path(mv, &opp_path, opp_path_len)
        {
            false
        } else {
            is_tactical_move(board, mv, our_dist, opp_dist_path, bfs)
        };

        let pruned = matches!(mv, Move::Wall { .. }) && !wall_keep[i];

        let reduction = if pruned {
            child_depth_full
        } else if moves_searched == 0
            || depth < LMR_MIN_DEPTH
            || heat_ratio_hot
        {
            0u32
        } else {
            let d = (depth as usize).min(63);
            let m = (i + 1).min(63);
            let base_r = lmr_table[d][m];
            let gap = cat_max.saturating_sub(cat_cm.max(0) as u16);
            let cat_extra = (gap as f32 * profile.cat_heat_lmr_slope).round() as u32;
            let wall_extra = if matches!(mv, Move::Wall { .. }) && cat_cm == 0 {
                4u32
            } else if matches!(mv, Move::Wall { .. })
                && !crate::cat::prune::wall_intersects_path(mv, &opp_path, opp_path_len)
                && !corridor_relevant
            {
                3u32
            } else if cat_cm < i32::from(profile.cold_cm) {
                if profile.stage_t < 0.35 {
                    3u32
                } else {
                    1u32
                }
            } else {
                0u32
            };
            (base_r + wall_extra + cat_extra).min(depth.saturating_sub(1))
        };

        let child_depth_used = child_depth_full.saturating_sub(reduction);

        plans.push(RootLmrPlan {
            mv: format_move(mv),
            is_pawn: matches!(mv, Move::Pawn { .. }),
            order: i,
            cat_cm,
            tactical: is_tactical,
            hot: heat_ratio_hot,
            pruned,
            reduction,
            child_depth_full,
            child_depth_used,
            in_full_window,
        });

        if !pruned {
            moves_searched += 1;
        }
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
    let mut out = String::new();
    for (i, p) in plans.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&format!(
            "{{\"move\":\"{}\",\"kind\":\"{}\",\"order\":{},\"catCm\":{},\"tactical\":{},\"hot\":{},\"pruned\":{},\"reduction\":{},\"childDepthFull\":{},\"childDepthUsed\":{},\"inFullWindow\":{}}}",
            p.mv,
            if p.is_pawn { "pawn" } else { "wall" },
            p.order,
            p.cat_cm,
            p.tactical,
            p.hot,
            p.pruned,
            p.reduction,
            p.child_depth_full,
            p.child_depth_used,
            p.in_full_window,
        ));
    }
    out
}

/// Pre-search LMR plan — static profile at pierce peak.
pub fn lmr_snapshot_json(board: &mut Board, time_ms: u64, id_depth: u32) -> String {
    let mut bfs = BfsScratch::new();
    let depth = id_depth.clamp(4, 32);
    let (profile, plans) = plan_root_lmr(board, &mut bfs, depth, time_ms, 0.0);
    format!(
        "{{\"source\":\"shallow\",\"idDepth\":{},\"timeMs\":{},\"lmrProfile\":{},\"moves\":[{}]}}",
        depth,
        time_ms,
        lmr_profile_fields(&profile, depth),
        format_lmr_plans_json(&plans),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::lmr_profile::TIME_REFERENCE_MS;

    #[test]
    fn shallow_snapshot_has_legal_moves() {
        let mut board = Board::new();
        let mut bfs = BfsScratch::new();
        let (_, plans) = plan_root_lmr(&mut board, &mut bfs, 8, TIME_REFERENCE_MS, 0.0);
        assert!(plans.len() >= 4);
        assert!(plans[0].tactical);
    }

    #[test]
    fn shallow_follows_cat_searchable_walls_not_raw_legal_order() {
        let mut board = Board::new();
        board.apply_algebraic("e2");
        board.apply_algebraic("e1h");
        board.apply_algebraic("e3");
        let mut bfs = BfsScratch::new();
        let (_, plans) = plan_root_lmr(&mut board, &mut bfs, 8, TIME_REFERENCE_MS, 0.0);
        assert!(
            plans.len() >= 20,
            "expected CAT-filtered root list, got {}",
            plans.len()
        );
        let top_wall = plans
            .iter()
            .filter(|p| !p.is_pawn)
            .max_by_key(|p| p.cat_cm)
            .expect("hot wall");
        assert!(
            top_wall.cat_cm >= 200,
            "hottest planned wall should be corridor-hot, got {} on {}",
            top_wall.cat_cm,
            top_wall.mv
        );
        assert!(
            !plans.iter().any(|p| p.mv == "a1h" && p.cat_cm < 40),
            "cold fringe walls should not dominate shallow plan"
        );
    }
}
