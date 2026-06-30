//! Root LMR plan snapshot — mirrors alphabeta root move list + planned reductions.

use crate::cat::constants::DIST_PENALTY;
use crate::cat::prune::{
    cat_heat_fraction, cat_heat_ref_max, cat_heat_refs, cat_v16_lmr_reduction, get_shortest_path,
    is_lmr_heat_hot, is_tactical_move, move_corridor_attention, order_moves, path_distance,
};
use crate::cat::CorridorAttention;
use crate::core::board::{Board, Move};
use crate::movegen::{generate_legal_moves_slice, MAX_LEGAL_MOVES};
use crate::path::BfsScratch;
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
    aggression: f64,
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
    let n = generate_legal_moves_slice(board, &mut buf, bfs);
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

    let cat_refs = cat_heat_refs(&buf, n, board, &cat);
    let cat_max = cat_refs.all;

    let depth = id_depth.max(1);
    let child_depth_full = depth.saturating_sub(1);

    let mut plans = Vec::with_capacity(n);

    for i in 0..n {
        let mv = buf[i];
        let cat_cm = move_corridor_attention(board, mv, &cat);
        let cat_ref = cat_heat_ref_max(mv, cat_refs);
        let heat_ratio_hot =
            is_lmr_heat_hot(cat_cm, cat_max, profile.cold_cm, profile.hot_ratio_pct);
        let corridor_relevant = cat_cm >= i32::from(profile.cold_cm);
        let is_tactical = if i == 0 || depth < LMR_MIN_DEPTH {
            true
        } else if matches!(mv, Move::Wall { .. })
            && !crate::cat::prune::wall_intersects_path(mv, &opp_path, opp_path_len)
        {
            false
        } else {
            is_tactical_move(board, mv, our_dist, opp_dist_path, bfs)
        };

        // Connected model: identical to the live search. `aggression` 0→1 maps
        // 0% reduction → everything maximally reduced; reducibility (impact + index)
        // sets how fast each move sheds depth. Mirrors what the engine actually does.
        let reduction = if i == 0 || depth < LMR_MIN_DEPTH {
            0u32
        } else {
            cat_v16_lmr_reduction(mv, cat_cm, cat_refs, i, child_depth_full, aggression)
        };
        let _ = corridor_relevant;

        let child_depth_used = child_depth_full.saturating_sub(reduction);
        let in_full_window = child_depth_used >= child_depth_full.saturating_sub(1);

        plans.push(RootLmrPlan {
            mv: format_move(mv),
            is_pawn: matches!(mv, Move::Pawn { .. }),
            order: i,
            cat_cm,
            tactical: is_tactical,
            hot: heat_ratio_hot || cat_heat_fraction(cat_cm, cat_ref, profile.cold_cm) >= 0.72,
            pruned: false,
            reduction,
            child_depth_full,
            child_depth_used,
            in_full_window,
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
pub fn lmr_snapshot_json(board: &mut Board, time_ms: u64, id_depth: u32, aggression: f64) -> String {
    let mut bfs = BfsScratch::new();
    let depth = id_depth.clamp(4, 32);
    let (profile, plans) = plan_root_lmr(board, &mut bfs, depth, time_ms, 0.0, aggression);
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
        let (_, plans) = plan_root_lmr(&mut board, &mut bfs, 8, TIME_REFERENCE_MS, 0.0, 0.5);
        assert!(plans.len() >= 4);
        assert!(plans[0].tactical);
    }

    #[test]
    fn shallow_lists_raw_legal_walls_before_visual_reduction() {
        let mut board = Board::new();
        board.apply_algebraic("e2");
        board.apply_algebraic("e1h");
        board.apply_algebraic("e3");
        let mut bfs = BfsScratch::new();
        let (_, plans) = plan_root_lmr(&mut board, &mut bfs, 8, TIME_REFERENCE_MS, 0.0, 0.5);
        assert!(
            plans.len() >= 100,
            "expected full legal root list, got {}",
            plans.len()
        );
        assert!(
            plans.iter().any(|p| p.mv == "a1h"),
            "cold legal walls should still be visible in LMR vision"
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
    }

    #[test]
    fn shallow_zero_cap_disables_visual_reduction() {
        let mut board = Board::new();
        let mut bfs = BfsScratch::new();
        let (_, plans) = plan_root_lmr(&mut board, &mut bfs, 8, TIME_REFERENCE_MS, 0.0, 0.0);
        assert!(plans.len() >= 100);
        assert!(
            plans.iter().all(|p| p.reduction == 0 && p.child_depth_used == p.child_depth_full),
            "0 slider should show every legal move at 100% depth"
        );
    }

    #[test]
    fn shallow_includes_e_file_corridor_walls_at_e4_e6() {
        let mut board = Board::new();
        for m in ["e2", "e8", "e3", "e7", "e4", "e6"] {
            board.apply_algebraic(m);
        }
        let mut bfs = BfsScratch::new();
        let (_, plans) = plan_root_lmr(&mut board, &mut bfs, 8, TIME_REFERENCE_MS, 0.0, 0.5);
        assert!(
            plans.len() >= 15,
            "expected CAT-filtered root list at e4/e6, got {}",
            plans.len()
        );
        let corridor = ["d4h", "e4h", "d5h", "e5h", "d6h", "e6h"];
        for wall in corridor {
            let p = plans
                .iter()
                .find(|p| p.mv == wall)
                .unwrap_or_else(|| panic!("missing corridor wall {wall} in shallow plan"));
            assert!(
                p.cat_cm >= 160,
                "{wall} should be CAT-hot, got {}cm",
                p.cat_cm
            );
            assert!(!p.pruned, "CAT-hot wall {wall} must stay in plan");
        }
        let d5 = plans.iter().find(|p| p.mv == "d5h").expect("d5h");
        let d6 = plans.iter().find(|p| p.mv == "d6h").expect("d6h");
        assert!(
            d6.reduction >= d5.reduction,
            "cooler wall should not get less cut than hotter peak: d6={} d5={}",
            d6.reduction,
            d5.reduction
        );
        assert!(
            d5.child_depth_used >= d6.child_depth_used,
            "hotter wall should not search shallower than cooler wall: d5=d{} d6=d{}",
            d5.child_depth_used,
            d6.child_depth_used
        );
        let hot = plans
            .iter()
            .filter(|p| !p.is_pawn)
            .max_by_key(|p| p.cat_cm)
            .expect("hot wall");
        assert!(
            hot.cat_cm >= 200,
            "hottest wall should be corridor-hot, got {} on {}",
            hot.cat_cm,
            hot.mv
        );
    }
}
