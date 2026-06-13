//! Reachability — bitwise flood fill and `BfsScratch` (no CAT logic here; see `cat`).

pub mod bfs;
pub mod distance;
pub mod flood;
pub mod masks;
pub mod parallel;

pub use bfs::{
    both_players_reach_goals, both_players_reach_goals_with_masks, can_reach_goal,
    shortest_distance, BfsScratch,
};
pub use masks::DirMasks;
pub use parallel::{
    both_players_reach_goals_grids, both_players_reach_goals_grids_ks,
    both_players_reach_goals_parallel, flood_to_goal_grids, flood_to_goal_grids_ks,
    flood_to_goal_with_cache, flood_to_goal_with_cache_ks, wall_delta, WallGrids,
};

#[cfg(test)]
mod tests;
