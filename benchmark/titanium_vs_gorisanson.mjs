#!/usr/bin/env node
/**
 * Rust Titanium vs Gorisanson MCTS — equal budget (10s / 2B sims, whichever stops first).
 *
 *   node benchmark/titanium_vs_gorisanson.mjs
 *   node benchmark/titanium_vs_gorisanson.mjs --games 10
 */

import { eloFromMatch, playMatch } from './lib/match_engine.mjs';
import { RUST_TITANIUM_ID, GORISANSON_ID } from './lib/engine_ids.mjs';
import { BENCH_TIME_SEC, BENCH_MAX_SIMULATIONS, formatThinkBudget } from './lib/bench_limits.mjs';
import { encodeReplayFromAlgebraic, formatReplayBlock } from './lib/replay_code.mjs';

function parseArgs(argv) {
  const opts = {
    games: 4,
    verbose: false,
  };

  for (let i = 2; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--games' && argv[i + 1]) {
      opts.games = Number(argv[++i]);
    } else if (arg === '--verbose' || arg === '-v') {
      opts.verbose = true;
    }
  }

  return opts;
}

async function main() {
  const opts = parseArgs(process.argv);
  const budget = { timeSec: BENCH_TIME_SEC, maxSimulations: BENCH_MAX_SIMULATIONS };

  const titanium = { id: RUST_TITANIUM_ID, engine: 'minimax' };
  const gorisanson = { id: GORISANSON_ID };

  const live = {
    gameIndex: 0,
    whiteId: titanium.id,
    blackId: gorisanson.id,
    algebraicHistory: [],
  };
  process.on('SIGINT', () => {
    const replay = encodeReplayFromAlgebraic(live.algebraicHistory, {
      a: live.whiteId,
      b: live.blackId,
      plies: live.algebraicHistory.length,
      winner: 'draw',
    });
    console.log('');
    console.log('Interrupted. Partial replay from current position:');
    console.log(formatReplayBlock(replay, { label: `REPLAY interrupted game ${live.gameIndex || 1}` }));
    process.exit(130);
  });

  console.log('Rust Titanium vs Gorisanson MCTS');
  console.log(`games=${opts.games}  budget=${formatThinkBudget(budget)} (both sides)`);

  const started = performance.now();
  const match = await playMatch(titanium, gorisanson, opts.games, {
    ...budget,
    verbose: opts.verbose,
    onGameStart: ({ gameIndex, whiteId, blackId }) => {
      live.gameIndex = gameIndex;
      live.whiteId = whiteId;
      live.blackId = blackId;
      live.algebraicHistory = [];
    },
    onPly: ({ algebraicHistory }) => {
      live.algebraicHistory = algebraicHistory;
    },
  });
  const elapsed = (performance.now() - started) / 1000;

  const totals = {
    [RUST_TITANIUM_ID]: { plies: 0, simulations: 0, nodes: 0 },
    [GORISANSON_ID]: { plies: 0, simulations: 0, nodes: 0 },
  };
  for (const game of match.results) {
    const byEngine = game.stats?.byEngine ?? {};
    for (const id of [RUST_TITANIUM_ID, GORISANSON_ID]) {
      const src = byEngine[id];
      if (!src) {
        continue;
      }
      totals[id].plies += src.plies ?? 0;
      totals[id].simulations += src.simulations ?? 0;
      totals[id].nodes += src.nodes ?? 0;
    }
  }

  function fmtInt(n) {
    return n.toLocaleString('en-US');
  }

  function perMove(total, plies) {
    if (!plies) {
      return 'n/a';
    }
    return fmtInt(Math.round(total / plies));
  }

  const { ratingA, ratingB, expectedA } = eloFromMatch(match.scoreA, match.scoreB, opts.games, 1400, 1600);

  console.log('');
  console.log(`Score: titanium ${match.scoreA} — gorisanson ${match.scoreB}  (draws ${match.draws})`);
  console.log('Search totals:');
  console.log(
    `  titanium   sims=${fmtInt(totals[RUST_TITANIUM_ID].simulations)}  avg/move=${perMove(totals[RUST_TITANIUM_ID].simulations, totals[RUST_TITANIUM_ID].plies)}  plies=${fmtInt(totals[RUST_TITANIUM_ID].plies)}`,
  );
  console.log(
    `  gorisanson sims=${fmtInt(totals[GORISANSON_ID].simulations)}  avg/move=${perMove(totals[GORISANSON_ID].simulations, totals[GORISANSON_ID].plies)}  plies=${fmtInt(totals[GORISANSON_ID].plies)}`,
  );
  if (totals[RUST_TITANIUM_ID].nodes > 0 || totals[GORISANSON_ID].nodes > 0) {
    console.log('Node totals:');
    console.log(
      `  titanium   nodes=${fmtInt(totals[RUST_TITANIUM_ID].nodes)}  avg/move=${perMove(totals[RUST_TITANIUM_ID].nodes, totals[RUST_TITANIUM_ID].plies)}`,
    );
    console.log(
      `  gorisanson nodes=${fmtInt(totals[GORISANSON_ID].nodes)}  avg/move=${perMove(totals[GORISANSON_ID].nodes, totals[GORISANSON_ID].plies)}`,
    );
  }
  console.log(`Time:  ${elapsed.toFixed(1)}s`);
  console.log(`Elo: titanium ${ratingA.toFixed(0)} · gorisanson ${ratingB.toFixed(0)} · expected ${(expectedA * 100).toFixed(1)}%`);

  process.exit(0);
}

main().catch((err) => {
  console.error(err?.stack || String(err));
  process.exit(1);
});
