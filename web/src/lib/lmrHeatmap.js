/** LMR vision — root move depth / reduction overlays from engine JSON. */

/**
 * @param {string[]} algebraicMoves
 * @param {number} [timeSec]
 */
export async function fetchLmrSnapshot(algebraicMoves, timeSec = 10, idDepth = 8) {
  const res = await fetch('/api/titanium/lmr', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ moves: algebraicMoves, timeSec, idDepth }),
  });
  const data = await res.json();
  if (!res.ok || data.error) {
    throw new Error(data.error ?? `LMR request failed (${res.status})`);
  }
  return data;
}

function normalizeLmrEntry(entry) {
  const reduction = Number(entry.reduction ?? 0);
  const childFull = Number(entry.childDepthFull ?? entry.child_depth_full ?? 0);
  const childUsed = Number(entry.childDepthUsed ?? entry.child_depth_used ?? childFull);
  return {
    move: entry.move ?? entry.mv,
    kind: entry.kind ?? (entry.is_pawn || entry.isPawn ? 'pawn' : 'wall'),
    order: entry.order ?? 0,
    catCm: entry.catCm ?? entry.cat_cm ?? 0,
    tactical: Boolean(entry.tactical),
    hot: Boolean(entry.hot),
    pruned: Boolean(entry.pruned),
    reduction,
    childDepthFull: childFull,
    childDepthUsed: childUsed,
    reSearched: Boolean(entry.reSearched ?? entry.re_searched),
    inFullWindow: Boolean(entry.inFullWindow ?? entry.in_full_window),
    score: entry.score ?? null,
    nodes: Number(entry.nodes ?? 0),
    sharePct: 0,
    searched: entry.searched !== false,
    unsearched: Boolean(entry.unsearched),
  };
}

function attachNodeShares(moves) {
  const total = moves.reduce((sum, m) => sum + (m.nodes > 0 ? m.nodes : 0), 0);
  if (total <= 0) {
    return moves;
  }
  return moves.map((m) => ({
    ...m,
    sharePct: m.nodes > 0 ? Math.round((m.nodes / total) * 100) : 0,
  }));
}

/**
 * Fill gaps in search rootMoves with the static pre-search plan (same legal list).
 * Search behaviour unchanged — viz only.
 *
 * @param {object[]} planMoves
 * @param {object[]} searchMoves
 */
export function mergeLmrPlanWithSearch(planMoves, searchMoves) {
  if (!planMoves?.length) {
    return searchMoves ?? [];
  }
  if (!searchMoves?.length) {
    return planMoves.map((m) => ({ ...m, unsearched: true, searched: false, nodes: 0 }));
  }
  const planByKey = indexLmrMoves(planMoves);
  const searchByKey = indexLmrMoves(searchMoves);
  const keys = new Set([...planByKey.keys(), ...searchByKey.keys()]);
  const merged = [];
  for (const key of keys) {
    const plan = planByKey.get(key);
    const search = searchByKey.get(key);
    if (search) {
      merged.push({
        ...plan,
        ...search,
        catCm: search.catCm ?? plan?.catCm ?? 0,
        searched: true,
        unsearched: false,
      });
    } else if (plan) {
      merged.push({
        ...plan,
        searched: false,
        unsearched: true,
        nodes: 0,
        sharePct: 0,
      });
    }
  }
  merged.sort((a, b) => a.order - b.order);
  return merged;
}

/**
 * @param {Array<Record<string, unknown>>} moves
 * @returns {Map<string, object>}
 */
export function indexLmrMoves(moves) {
  const map = new Map();
  for (const entry of moves ?? []) {
    const alg = entry.move ?? entry.mv;
    if (!alg) {
      continue;
    }
    map.set(String(alg), entry);
  }
  return map;
}

function coldCmThreshold(viz) {
  return Number(viz?.lmrProfile?.coldCm ?? 60);
}

function fmtDepth(used) {
  const d = Number(used ?? 0);
  return d > 0 ? `d${d}` : '';
}

/** Minimum ply reduction before we paint a slot in live search (shallow is sparser). */
function minCutToShow(viz) {
  return viz?.shallow ? 1 : 2;
}

/**
 * Skip pruned / noise — only draw moves with a meaningful cut, corridor heat, or search share.
 * `−1` in the UI means "1 ply LMR cut", not a leaf-node flag; we hide lone 1-ply plan noise.
 */
export function lmrEntryWorthShowing(entry, viz) {
  if (!entry) {
    return false;
  }
  // Pierce cap dropout — still paint in shallow when CAT says the wall matters.
  if (entry.pruned) {
    return Boolean(viz?.shallow && entry.catCm > 0);
  }
  const cold = coldCmThreshold(viz);
  const minCut = minCutToShow(viz);

  if (entry.reSearched) {
    return true;
  }

  // Actually searched at root — always interesting.
  if (!viz?.shallow && entry.searched && (entry.nodes > 0 || entry.sharePct > 0)) {
    return true;
  }

  // Significant planned or actual cut.
  if (entry.reduction >= minCut) {
    return true;
  }

  // Corridor-hot — LMR treats as tactical.
  if (entry.catCm >= cold) {
    return true;
  }

  // First root slot with a real signal only.
  if (entry.order === 0 && (entry.tactical || entry.inFullWindow)) {
    return (
      entry.catCm > 0 ||
      entry.reduction >= minCut ||
      (!viz?.shallow && entry.searched && entry.nodes > 0)
    );
  }

  // Pre-search plan slots.
  if (viz?.shallow) {
    if (entry.reduction >= minCut) {
      return true;
    }
    if (entry.inFullWindow || entry.tactical) {
      return true;
    }
    if (entry.catCm > 0) {
      return true;
    }
    return false;
  }

  if (entry.unsearched && entry.reduction >= minCut) {
    return true;
  }

  return false;
}

/** Map value into 0..1 using this view's min–max (zeros are not drawn). */
function proportionalT(value, min, max) {
  const v = Number(value);
  if (!Number.isFinite(v) || v <= 0) {
    return 0;
  }
  if (max <= min) {
    return 1;
  }
  return Math.min(1, Math.max(0, (v - min) / (max - min)));
}

function computeLmrRanges(visibleMoves) {
  const catValues = visibleMoves.map((m) => Number(m.catCm) || 0).filter((v) => v > 0);
  const cutValues = visibleMoves.map((m) => Number(m.reduction) || 0).filter((v) => v > 0);
  const shareValues = visibleMoves.map((m) => Number(m.sharePct) || 0).filter((v) => v > 0);
  const minCat = catValues.length ? Math.min(...catValues) : 0;
  const maxCat = catValues.length ? Math.max(...catValues) : 1;
  const maxCut = cutValues.length ? Math.max(...cutValues) : 1;
  const maxShare = shareValues.length ? Math.max(...shareValues) : 1;
  return {
    catCm: { min: minCat, max: maxCat },
    reduction: { min: 0, max: maxCut },
    sharePct: { min: 0, max: maxShare },
  };
}

/** Corridor cm — yellow → orange → red, scaled to visible min..max. */
function corridorFill(t, alpha = 0.8) {
  const hue = Math.round(52 * (1 - t));
  const sat = Math.round(86 + 10 * t);
  const light = Math.round(58 - 12 * t);
  return {
    fill: `hsla(${hue}, ${sat}%, ${light}%, ${alpha})`,
    textLight: light < 48 || t > 0.72,
  };
}

/** Ply reduction — teal → amber → crimson, scaled to visible max cut. */
function cutFill(t, alpha = 0.82) {
  const hue = Math.round(168 * (1 - t));
  const sat = Math.round(62 + 30 * t);
  const light = Math.round(54 - 14 * t);
  return {
    fill: `hsla(${hue}, ${sat}%, ${light}%, ${alpha})`,
    textLight: t > 0.55,
  };
}

/** Search node share — slate → indigo → violet, scaled to visible max %. */
function shareFill(t, alpha = 0.82) {
  const hue = Math.round(215 - 55 * t);
  const sat = Math.round(42 + 38 * t);
  const light = Math.round(64 - 20 * t);
  return {
    fill: `hsla(${hue}, ${sat}%, ${light}%, ${alpha})`,
    textLight: t > 0.45,
  };
}

/**
 * @param {object} payload
 * @param {object[]} [payload.planMoves] — pre-search plan to pad search gaps
 */
export function buildLmrViz(payload) {
  const shallow = payload.source === 'shallow';
  const profile = payload.lmrProfile ?? {};
  const depthLog = payload.depthLog ?? [];
  const deepFromLog = depthLog.length
    ? depthLog.reduce((best, e) => ((e.depth ?? 0) > (best?.depth ?? 0) ? e : best))
    : null;
  const searchDepth =
    payload.searchDepth ??
    profile.idDepth ??
    deepFromLog?.depth ??
    payload.idDepth ??
    1;

  let raw = payload?.moves ?? payload?.rootMoves ?? [];
  if (!shallow && payload.planMoves?.length) {
    const normalizedSearch = raw.map(normalizeLmrEntry);
    const normalizedPlan = payload.planMoves.map(normalizeLmrEntry);
    raw = mergeLmrPlanWithSearch(normalizedPlan, normalizedSearch);
  }
  if (!raw.length) {
    return null;
  }

  let moves = raw.map(normalizeLmrEntry);
  if (!shallow) {
    moves = attachNodeShares(moves);
  }
  const vizDraft = { shallow, searchDepth, lmrProfile: profile };
  let visibleMoves = moves.filter((m) => lmrEntryWorthShowing(m, vizDraft));
  if (shallow && visibleMoves.length === 0 && moves.length > 0) {
    visibleMoves = moves.filter((m) => !m.pruned).slice(0, 48);
  }
  const moveIndex = indexLmrMoves(visibleMoves);
  const ranges = computeLmrRanges(visibleMoves);
  return {
    source: payload.source ?? 'search',
    shallow,
    idDepth: searchDepth,
    searchDepth,
    ranges,
    maxCatCm: ranges.catCm.max,
    maxSharePct: ranges.sharePct.max,
    maxReduction: ranges.reduction.max,
    lmrProfile: profile,
    lmrReSearches: payload.lmrReSearches ?? null,
    totalNodes: moves.reduce((s, m) => s + m.nodes, 0),
    searchedCount: moves.filter((m) => m.searched).length,
    visibleCount: visibleMoves.length,
    moveIndex,
    moves,
    visibleMoves,
    label: shallow ? 'pre-search plan' : `search d${searchDepth}`,
  };
}

/**
 * @returns {{ fill: string, label: string, mode: string, textLight: boolean }}
 */
export function lmrDepthStyle(entry, viz) {
  if (!entry) {
    return { fill: 'transparent', label: '', mode: '', textLight: false };
  }
  const alpha = entry.unsearched ? 0.42 : 0.84;
  const used = entry.childDepthUsed;
  const ranges = viz?.ranges ?? computeLmrRanges([entry]);
  let painted;
  let mode;
  if (entry.reduction > 0) {
    painted = cutFill(
      proportionalT(entry.reduction, ranges.reduction.min, ranges.reduction.max),
      alpha,
    );
    mode = 'cut';
  } else if (!viz?.shallow && entry.searched && entry.sharePct > 0) {
    painted = shareFill(
      proportionalT(entry.sharePct, ranges.sharePct.min, ranges.sharePct.max),
      alpha,
    );
    mode = 'share';
  } else if (entry.catCm > 0) {
    painted = corridorFill(
      proportionalT(entry.catCm, ranges.catCm.min, ranges.catCm.max),
      alpha,
    );
    mode = 'corridor';
  } else {
    painted = cutFill(0, alpha * 0.75);
    mode = 'full';
  }
  const label = entry.unsearched
    ? `plan only · −${entry.reduction} ply${used > 0 ? ` · child d${used}` : ''}`
    : entry.reduction > 0
      ? `LMR cut −${entry.reduction} ply${used > 0 ? ` · searched d${used}` : ''}`
      : mode === 'share'
        ? `${entry.sharePct}% nodes`
        : entry.catCm > 0
          ? `corridor ${entry.catCm}cm`
          : used > 0
            ? `d${used} full`
            : 'full depth';
  return { fill: painted.fill, label, mode, textLight: painted.textLight };
}

export function lmrWallOutlineColor(entry, viz) {
  const style = lmrDepthStyle(entry, viz);
  return style.fill.replace(/,\s*[\d.]+%?\)$/, ', 0.95)');
}

export function lmrDisplayText(entry, viz) {
  if (!entry || !lmrEntryWorthShowing(entry, viz)) {
    return '';
  }
  if (!viz?.shallow && entry.searched && entry.sharePct > 0) {
    return `${entry.sharePct}%`;
  }
  if (entry.reduction >= 2) {
    return `−${entry.reduction}`;
  }
  if (entry.catCm > 0) {
    return String(entry.catCm);
  }
  const depth = fmtDepth(entry.childDepthUsed);
  if (depth) {
    return depth;
  }
  if (entry.reduction === 1) {
    return '−1';
  }
  return '';
}

export function lmrSubLabel(entry, viz) {
  if (!entry || !lmrEntryWorthShowing(entry, viz)) {
    return '';
  }
  const parts = [];
  const depth = fmtDepth(entry.childDepthUsed);
  if (entry.reduction > 0 && entry.catCm > 0) {
    parts.push(String(entry.catCm));
  } else if (depth && entry.reduction > 0) {
    parts.push(depth);
  } else if (!viz?.shallow && depth && entry.searched) {
    parts.push(depth);
  }
  if (entry.reSearched) {
    parts.push('↺');
  }
  return parts.join(' ');
}
