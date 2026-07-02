//! Titanium v16 LMR — ACE v13 graduated baseline + two hard CAT overrides only.

use crate::search::cat_index_lmr::CAT_ATTENTION_TAIL_CUTOFF;

pub const ACE_LMR_AFTER_MOVE: usize = 4;
pub const ACE_LMR_MIN_DEPTH: i32 = 3;

/// Late-move reduction plies — same formula as ACE v13 / JS graduated LMR.
#[inline]
pub fn ace_graduated_lmr_reduction(move_index: usize, depth: i32) -> i32 {
    let mut red = 1;
    if move_index >= 12 {
        red += 1;
    }
    if depth >= 6 && move_index >= 24 {
        red += 1;
    }
    red
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum V16HardOverride {
    None,
    DeadTail,
    BackwardMove,
}

impl V16HardOverride {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::DeadTail => "deadTail",
            Self::BackwardMove => "backwardMove",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct V16LmrPlan {
    pub ace_base_reduction: i32,
    pub hard_override: V16HardOverride,
    pub final_reduction: i32,
    pub child_depth_used: i32,
}

#[inline]
fn max_safe_reduction(child_depth_full: i32) -> i32 {
    (child_depth_full - 1).max(0)
}

/// Wall move: ACE baseline, unless CAT attention ≤ 10% of max legal impact.
/// The rear-zeroed CAT heatmap IS the backward test: a wall touching squares a
/// pawn can reach without moving backwards carries heat, so only true rear /
/// no-op walls fall in the tail. Path-delay is NOT used to hard-cut — a wall
/// spanning into a forward corridor can leave the current BFS distance
/// unchanged (equal-length detour) and still be tactically critical.
pub fn plan_v16_wall_lmr(
    move_index: usize,
    depth: i32,
    child_depth_full: i32,
    attention_ratio: f64,
    opponent_delay: i32,
    self_delay: i32,
) -> V16LmrPlan {
    let ace_base = ace_graduated_lmr_reduction(move_index, depth);
    let max_safe = max_safe_reduction(child_depth_full);
    if attention_ratio <= CAT_ATTENTION_TAIL_CUTOFF {
        let label = if opponent_delay <= 0 && self_delay <= 0 {
            V16HardOverride::BackwardMove
        } else {
            V16HardOverride::DeadTail
        };
        return V16LmrPlan {
            ace_base_reduction: ace_base,
            hard_override: label,
            final_reduction: max_safe,
            child_depth_used: 1,
        };
    }
    let final_reduction = ace_base.min(max_safe);
    V16LmrPlan {
        ace_base_reduction: ace_base,
        hard_override: V16HardOverride::None,
        final_reduction,
        child_depth_used: (child_depth_full - final_reduction).max(0),
    }
}

/// Pawn moved farther from goal (`self_gain < 0`) → depth-1 leaf search.
/// Sideways moves (`self_gain == 0`) are NOT backwards — pockets and detours
/// around tunnels stay fully searched.
pub fn plan_v16_pawn_lmr(
    move_index: usize,
    depth: i32,
    child_depth_full: i32,
    self_gain: i32,
) -> Option<V16LmrPlan> {
    if self_gain >= 0 {
        return None;
    }
    let ace_base = ace_graduated_lmr_reduction(move_index, depth);
    let max_safe = max_safe_reduction(child_depth_full);
    Some(V16LmrPlan {
        ace_base_reduction: ace_base,
        hard_override: V16HardOverride::BackwardMove,
        final_reduction: max_safe,
        child_depth_used: 1,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ace_baseline_matches_v13_schedule() {
        assert_eq!(ace_graduated_lmr_reduction(4, 5), 1);
        assert_eq!(ace_graduated_lmr_reduction(12, 5), 2);
        assert_eq!(ace_graduated_lmr_reduction(24, 6), 3);
    }

    #[test]
    fn dead_tail_forces_depth_one() {
        let p = plan_v16_wall_lmr(8, 10, 9, 0.05, 1, 0);
        assert_eq!(p.hard_override, V16HardOverride::DeadTail);
        assert_eq!(p.child_depth_used, 1);
        assert_eq!(p.final_reduction, 8);
    }

    #[test]
    fn rear_no_delay_wall_forces_depth_one() {
        // True backward wall: rear-zeroed CAT → tail attention, no path delay.
        let p = plan_v16_wall_lmr(8, 10, 9, 0.05, 0, 0);
        assert_eq!(p.hard_override, V16HardOverride::BackwardMove);
        assert_eq!(p.child_depth_used, 1);
        assert_eq!(p.final_reduction, 8);
    }

    #[test]
    fn forward_touching_wall_never_backward_culled() {
        // Wall spans into a forward corridor (hot CAT) but leaves the current
        // BFS distance unchanged (delay 0) — must stay graded, not depth 1.
        let p = plan_v16_wall_lmr(8, 10, 9, 0.5, 0, 0);
        assert_eq!(p.hard_override, V16HardOverride::None);
        assert!(p.child_depth_used > 1);
    }

    #[test]
    fn self_path_wall_stays_graded() {
        let p = plan_v16_wall_lmr(8, 10, 9, 0.5, 0, 2);
        assert_eq!(p.hard_override, V16HardOverride::None);
        assert!(p.child_depth_used > 1);
    }

    #[test]
    fn backward_pawn_forces_depth_one() {
        let p = plan_v16_pawn_lmr(3, 8, 7, -1).expect("backwards");
        assert_eq!(p.hard_override, V16HardOverride::BackwardMove);
        assert_eq!(p.child_depth_used, 1);
    }

    #[test]
    fn sideways_pawn_not_reduced() {
        assert!(plan_v16_pawn_lmr(3, 8, 7, 0).is_none());
    }
}
