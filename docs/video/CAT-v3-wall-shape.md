# CAT v3 - Half-protruding wall shape attention

**Episode hook:** Search was pruning walls that looked geometrically irrelevant but were tactically sharp.

---

## Problem

Tactical wall pruning (`wall_should_search`) correctly dropped enclosed T-walls and dead-zone junk, but it also dropped **half-protruding corridor walls**: perpendicular placements at the junction of two aligned same-orientation chain segments. These walls often sit on or beside hot CAT squares yet had weak `wall_edge_heat` because only one edge of the segment touches the corridor.

The old fallback ("hot touched square") was implicit and easy to miss when the protrusion anchor was one step off the wall's four touch squares.

---

## Shape patterns (explicit geometry)

### Half-protrusion

A candidate wall is **half-protruding** when it is perpendicular to two adjacent chain walls:

- **Vertical** at `(r,c)`: horizontal chain `H(r,c-1)` + `H(r,c)` (also checked one row below the slot).
- **Horizontal** at `(r,c)`: vertical chain `V(r-1,c)` + `V(r,c)`.

This is the classic "half T" sticking out of a wall line.

### Prophylactic block

A candidate is a **prevention** move when it sits **one step left/right (or up/down along the chain)** from a would-be half-protrusion site, blocking the opponent from completing the pattern next turn.

---

## Search integration (subtle, not eval)

| Mechanism | Effect |
|-----------|--------|
| `wall_shape_attention_bonus` | +60 cm for protrusion or +50 cm for prevention in move ordering |
| `wall_should_search` | Keeps shape-relevant walls when local CAT heat >= `CAT_COLD_CM` |
| `move_corridor_attention` | Feeds LMR / futility "corridor relevant" without touching static eval |

**Gating:** Bonus and pruning rescue only fire when `max(wall_edge_heat, touched_square_heat) >= 60 cm`. Delta > 3 corridors stay cold; no board-wide bleed.

---

## Why it works

Pure distance eval cannot see "this wall completes a corridor mouth." CAT already knows *where* routes compete; v3 adds a cheap geometric filter so search **notices** T-junction tactics near those routes without turning them into a strategic heuristic.

---

## Files

| Symbol | Location |
|--------|----------|
| `is_half_protruding_wall` | `engine/src/search.rs` |
| `prevents_half_protruding_wall` | `engine/src/search.rs` |
| `wall_shape_attention_bonus` | `engine/src/search.rs` |
| Tests | `half_protruding_wall_*`, `prevents_half_protruding_wall_*`, `wall_shape_bonus_*` |
