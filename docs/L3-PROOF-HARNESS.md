# L3 incremental-flood proof harness — spec (HANDOFF §C)

**Status:** spec only. Per the contact protocol, no §C implementation starts
until this harness exists and the spec is signed off.

---

## Why §C is worth it (measured)

`cargo run --release --bin l3_study` (depth-3 subtrees, production movegen,
i7-4900MQ, main @ post-Zobrist):

| root | walls | cand/node | flood-needed | flood-reject | L3 time share |
|------|------:|----------:|-------------:|-------------:|--------------:|
| startpos | 0 | 120.7 | 2.4% | 0.1% | 48% |
| canta00–14 (median) | 15 | ~72 | **~41%** | **~2.5%** | **~94%** |
| canta12 (worst reject) | 15 | 71.9 | 39.9% | 14.1% | 94% |

Read:

- In real wall-heavy positions, **L3 flood is ~94% of movegen time**.
- **97%+ of floods confirm the wall is legal** — almost all flood work is
  spent re-proving connectivity that a cheaper sound certificate could prove.
- Upper bound: a perfect skip would make wall-heavy movegen ~15× cheaper.
  Even a certificate that covers 80% of floods is a ~4× movegen win there.

So the prize is real. The risk is also real — the topo shortcut family has
produced two soundness bugs already (`a1h`/`a5h` false negatives from V10's
partial-component shortcut; right-edge `sideOnEdge` bug, see BUG-DIARY).

---

## Contract under test

Any §C scheme is a function

```text
fast_verdict(board, wall) ∈ { SafeSkip, Unknown }
```

used as: `SafeSkip` → accept wall without flood; `Unknown` → run the current
full flood (`both_players_reach_goals_grids` after `grids.place(delta)`).

**Soundness obligation (the only hard one):**
`fast_verdict = SafeSkip` ⇒ full flood would accept.
One-sided. `Unknown` is always allowed; a false `Unknown` costs time, never
correctness. Any scheme that can return a *rejection* without flooding takes
on a second obligation (rejects ⇒ flood rejects) and must pass the same
tiers for it.

**Reference oracles, strongest first:**

1. `reach_goal_naive` per player (scalar BFS, `path/parallel.rs` tests) —
   ground truth.
2. Current production flood `both_players_reach_goals_grids` — already
   differential-tested against (1) on 500 random boards.

The harness always compares against (1); (2) is the production fallback.

---

## Test tiers (all must pass before merge)

### T1 — exhaustive small

All wall sets of size ≤ 2 (`C(128,1) + C(128,2)` ≈ 8.3K boards, skipping
physically-colliding pairs) × every L1∧L2 candidate wall × 9 pawn placements
per board (4 corners-ish, 2 center, P1/P2 near own goal, adjacent pair,
the a5h row-4 shape). For every `SafeSkip`, assert naive BFS accepts.
Deterministic, no sampling. Target runtime: < 60 s release.

### T2 — LCG fuzz at real densities

Deterministic splitmix/LCG stream (no `rand` dep, same style as
`random_walls_match_naive_reference`). ≥ 10⁶ (board, candidate) trials at
3–18 walls, pawn positions uniform, **biased so ≥ 30% of boards contain a
near-cage** (a pawn with exactly one escape edge — these are where false
`SafeSkip` hides). Assert the soundness obligation per trial.

### T3 — regression family (named bugs stay dead)

- `a1h`, `a5h` replay prefixes (`test_replay.rs`) — the V10 false-negative
  family; assert exact legality sets.
- Right-edge H-wall family (canta game 0 depth-2 `5980 ≠ 5978` bug):
  every js_col-8 H slot in every canta root.
- Full-row barriers with 1-gap at each of the 9 columns, both players on
  both sides of the barrier.

### T4 — differential perft (the scale gate)

For each of the 15 canta roots **and** startpos: perft 1–3 with the §C
scheme enabled vs disabled must produce identical node counts
(`PERFT_VALUES` table stays the oracle). Then the standard d3/d4 startpos
gates. Any mismatch = unsound, full stop — this is the test that caught
canta game 0.

### T5 — skip-rate report (justification, not correctness)

`l3_study` extended with the scheme: report % of floods converted to
`SafeSkip` per root. **Merge bar: ≥ 60% median skip on canta roots**,
otherwise the added complexity isn't paying for itself (measured ceiling
is ~94% of movegen time; a sub-60% skip leaves most of it).

---

## Residual characterization (required in PR)

If the scheme is partial, the PR must name the family it does *not* cover
(e.g. "walls whose both ends touch independent components that each touch
the same edge"), with one concrete board per uncovered family, so the next
person extends coverage instead of rediscovering it.

## Candidate scheme notes (for whoever implements)

The data says optimize for *certifying legality cheaply*, not for fast
rejection (rejects are ~2.5% of floods). Plausible sound certificates, in
increasing power:

1. **Component-count certificate** — wall's two ends touch ≤ 1 existing
   wall-chain component ⇒ placing it cannot close a cycle/separate the
   grid beyond what topo already allows. Needs a union-find over wall
   endpoints maintained incrementally in make/unmake.
2. **Goal-corridor witness** — cache one shortest path per player per node;
   wall not intersecting either cached path ⇒ SafeSkip (path still valid).
   Cache invalidation on make/unmake is the risk point; the a1h/a5h family
   is exactly stale-witness territory. T2's near-cage bias targets this.
3. Full incremental reachability (dynamic connectivity) — only if 1+2
   under-deliver on T5.

Recommendation: implement (2) first — witness-path reuse — because each
flood already computes a reachability set; storing one path is nearly free
and skip-rate should be high (most walls don't touch the current shortest
paths). It is also exactly the scheme the harness's regression tier was
built to break.

---

## Sign-off checklist

- [ ] T1–T4 implemented and green on main
- [ ] T5 report committed with the PR
- [ ] Residual families documented
- [ ] Perft d3/d4 startpos gates unchanged
- [ ] BUG-DIARY entry written *before* merge (what could break, how T1–T4 cover it)
