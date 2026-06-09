#!/usr/bin/env node
/**
 * Mine opening lines: Ishtar vs Ishtar at scraped max strength (Long).
 * Pure line collection — no eval engine. Built for overnight runs.
 *
 *   node benchmark/ishtar_opening_mine.mjs
 *   node benchmark/ishtar_opening_mine.mjs --games 48 --workers 4
 *   node benchmark/ishtar_opening_mine.mjs --resume
 */

import { existsSync, readFileSync, writeFileSync, appendFileSync } from 'node:fs';
import { parseAlgebraic, toAlgebraic } from '../web/src/lib/gameLogic.js';
import { actionToGorisansonMove, gorisansonMoveToAction } from './lib/gorisanson_bridge.mjs';
import {
  applyGorisansonMove,
  createGorisansonGame,
  winnerIndex,
} from './lib/gorisanson_ai.mjs';
import { IshtarGameSession, resolveIshtarOptions } from './lib/ishtar_remote.mjs';
import { encodeReplayFromAlgebraic, formatReplayBlock } from './lib/replay_code.mjs';

const DEFAULT_MAX_PLIES = 10;
/** Server drops 4+ concurrent WS — 2 is the stable ceiling. */
const DEFAULT_WORKERS = 2;
/** ~13 min/game at Medium × 2 workers → ~40 games in ~4 hr. */
const DEFAULT_GAMES = 40;
const WORKER_STAGGER_MS = 8_000;
const GAME_RETRY_ATTEMPTS = 3;
const GAME_RETRY_DELAY_MS = 20_000;
const ISHTAR_MOVE_RETRIES = 5;
const OUT_PATH = new URL('./ishtar_opening_mine_out.json', import.meta.url);
const LIVE_PATH = new URL('./ishtar_opening_mine_live.txt', import.meta.url);

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

function parseArgs(argv) {
  const opts = {
    games: DEFAULT_GAMES,
    workers: DEFAULT_WORKERS,
    maxPlies: DEFAULT_MAX_PLIES,
    ishtarPreset: 'medium',
    resume: false,
    quiet: false,
  };
  for (let i = 2; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--games' && argv[i + 1]) opts.games = Number(argv[++i]);
    else if (arg === '--workers' && argv[i + 1]) opts.workers = Number(argv[++i]);
    else if (arg === '--max-plies' && argv[i + 1]) opts.maxPlies = Number(argv[++i]);
    else if (arg === '--ishtar' && argv[i + 1]) opts.ishtarPreset = argv[++i];
    else if (arg === '--resume') opts.resume = true;
    else if (arg === '--quiet' || arg === '-q') opts.quiet = true;
  }
  return opts;
}

function log(msg) {
  const line = `[${new Date().toISOString()}] ${msg}`;
  console.log(line);
  try {
    appendFileSync(LIVE_PATH, `${line}\n`);
  } catch {
    // ignore live-log failures
  }
}

function loadCheckpoint() {
  if (!existsSync(OUT_PATH)) {
    return { lines: [], failures: [] };
  }
  try {
    const data = JSON.parse(readFileSync(OUT_PATH, 'utf8'));
    const lines = (data.lines ?? []).filter((l) => !l.failed && l.algebraicHistory?.length > 0);
    const failures = data.failures ?? [];
    return { lines, failures };
  } catch {
    return { lines: [], failures: [] };
  }
}

async function playOpeningGame(gameIndex, opts, session) {
  const tag = `[g${gameIndex}]`;
  const ishtarOpts = resolveIshtarOptions(opts.ishtarPreset);
  let game = createGorisansonGame();
  const algebraicHistory = [];
  let plies = 0;
  let stopReason = 'max-plies';

  log(`${tag} start · ${ishtarOpts.label} · ${opts.maxPlies} plies`);

  while (winnerIndex(game) === null && plies < opts.maxPlies) {
    const side = game.pawnOfTurn.index;
    const ply = plies + 1;
    const started = performance.now();

    const algebraic = await session.requestMove(algebraicHistory);
    const ishtarMs = performance.now() - started;

    const move = actionToGorisansonMove(parseAlgebraic(algebraic));
    applyGorisansonMove(game, move);
    algebraicHistory.push(toAlgebraic(gorisansonMoveToAction(move)));
    plies += 1;

    log(`${tag} ply ${ply} P${side + 1}: ${algebraic} (${(ishtarMs / 1000).toFixed(1)}s)`);

    if (winnerIndex(game) !== null) {
      stopReason = 'terminal';
      break;
    }
  }

  return {
    gameIndex,
    plies,
    stopReason,
    algebraicHistory,
    replayCode: encodeReplayFromAlgebraic(algebraicHistory, {
      game: 'ishtar-opening-mine',
      visits: ishtarOpts.visits,
      parallelism: ishtarOpts.parallelism,
      stopReason,
      plies,
    }),
  };
}

async function playOpeningGameWithRetry(gameIndex, opts, session) {
  const tag = `[g${gameIndex}]`;
  let lastErr;
  for (let attempt = 1; attempt <= GAME_RETRY_ATTEMPTS; attempt += 1) {
    try {
      return await playOpeningGame(gameIndex, opts, session);
    } catch (err) {
      lastErr = err;
      log(`${tag} attempt ${attempt}/${GAME_RETRY_ATTEMPTS} failed: ${err?.message ?? err}`);
      session.close();
      if (attempt < GAME_RETRY_ATTEMPTS) {
        await sleep(GAME_RETRY_DELAY_MS * attempt);
        await session.connect();
      }
    }
  }
  throw lastErr;
}

function saveCheckpoint(opts, lines, failures, startedMs, partial = true) {
  const okLines = lines.filter((l) => !l.failed && l.algebraicHistory?.length > 0);
  const candidates = buildBookCandidates(okLines);
  const payload = {
    opts,
    lines: okLines,
    failures,
    candidates,
    elapsedSec: (performance.now() - startedMs) / 1000,
    completedGames: okLines.length,
    failedGames: failures.length,
    partial,
  };
  writeFileSync(OUT_PATH, JSON.stringify(payload, null, 2));
}

async function runPool(opts, startedMs, existingLines, existingFailures) {
  const results = [...existingLines];
  const failures = [...existingFailures];
  const startIdx = existingLines.length;
  let nextGame = startIdx;

  if (startIdx >= opts.games) {
    log(`Already have ${startIdx} games — nothing to do (target ${opts.games})`);
    return { results, failures };
  }

  if (startIdx > 0) {
    log(`Resuming from game ${startIdx + 1}/${opts.games} (${startIdx} already done)`);
  }

  async function worker(workerId) {
    await sleep((workerId - 1) * WORKER_STAGGER_MS);
    const session = new IshtarGameSession({ preset: opts.ishtarPreset });

    while (true) {
      const idx = nextGame;
      nextGame += 1;
      if (idx >= opts.games) {
        session.close();
        return;
      }

      const gameIndex = idx + 1;
      log(`[pool] worker ${workerId} → game ${gameIndex}/${opts.games}`);

      try {
        session.close();
        await session.connect();
        const line = await playOpeningGameWithRetry(gameIndex, opts, session);
        results.push(line);
        saveCheckpoint(opts, results, failures, startedMs, true);
        log(`[done] game ${gameIndex}: ${line.plies} plies · ${line.algebraicHistory.join(' ')}`);
      } catch (err) {
        session.close();
        const fail = {
          gameIndex,
          error: String(err?.message ?? err),
          at: new Date().toISOString(),
        };
        failures.push(fail);
        saveCheckpoint(opts, results, failures, startedMs, true);
        log(`[fail] game ${gameIndex}: ${fail.error}`);
        await sleep(30_000);
      }
    }
  }

  const n = Math.min(opts.workers, opts.games - startIdx);
  await Promise.all(Array.from({ length: n }, (_, i) => worker(i + 1)));
  return { results, failures };
}

function buildBookCandidates(lines) {
  const seen = new Map();
  for (const line of lines) {
    const hist = line.algebraicHistory;
    for (let i = 1; i < hist.length; i += 1) {
      const prefix = hist.slice(0, i);
      const reply = hist[i];
      const key = `${prefix.join(' ')}|${reply}`;
      const replySide = i % 2;
      const existing = seen.get(key);
      if (existing) {
        existing.hits += 1;
      } else {
        seen.set(key, { prefix, reply, replySide, hits: 1 });
      }
    }
  }
  return [...seen.values()]
    .filter((c) => c.hits >= 2 || c.prefix.length <= 2)
    .sort((a, b) => b.hits - a.hits || b.prefix.length - a.prefix.length);
}

async function main() {
  const opts = parseArgs(process.argv);
  const ishtarOpts = resolveIshtarOptions(opts.ishtarPreset);

  let checkpoint = { lines: [], failures: [] };
  if (existsSync(OUT_PATH)) {
    try {
      const prior = JSON.parse(readFileSync(OUT_PATH, 'utf8'));
      if (opts.resume || prior.partial) {
        checkpoint = loadCheckpoint();
        log(`Resuming: ${checkpoint.lines.length} games already saved`);
      } else if ((prior.lines ?? []).length > 0) {
        log(`Prior complete run (${prior.lines.length} games) — starting fresh (delete out.json or pass --resume to append)`);
      }
    } catch {
      // corrupt file — start fresh
    }
  }

  log('Ishtar opening miner (pure lines — overnight mode)');
  log(`  games=${opts.games}  workers=${opts.workers}  maxPlies=${opts.maxPlies}`);
  log(`  ishtar=${ishtarOpts.label} (${ishtarOpts.visits.toLocaleString()} visits · ${ishtarOpts.parallelism} threads)`);
  log(`  stagger=${WORKER_STAGGER_MS}ms  gameRetries=${GAME_RETRY_ATTEMPTS}  moveRetries=${ISHTAR_MOVE_RETRIES}`);
  log(`  output → ${OUT_PATH.pathname}`);
  log(`  live log → ${LIVE_PATH.pathname}`);

  const started = performance.now();
  const { results, failures } = await runPool(opts, started, checkpoint.lines, checkpoint.failures);
  const elapsed = (performance.now() - started) / 1000;

  const okLines = results.filter((l) => !l.failed && l.algebraicHistory?.length > 0);
  for (const line of okLines) {
    log(formatReplayBlock(line.replayCode, { label: `line g${line.gameIndex}` }));
  }

  const candidates = buildBookCandidates(okLines);
  log(`\n── Book candidates (${candidates.length} entries from ${okLines.length} games, ${failures.length} failed) ──`);
  for (const cand of candidates) {
    const sideLabel = cand.replySide === 0 ? 'White' : 'Black';
    log(`  // ${sideLabel} · seen ${cand.hits}x`);
    log(`  prefix: [${cand.prefix.map((m) => `"${m}"`).join(', ')}] → "${cand.reply}"`);
  }

  saveCheckpoint(opts, okLines, failures, started, false);
  log(`\nDone in ${elapsed.toFixed(0)}s · ${okLines.length}/${opts.games} games · ${failures.length} failed`);
  log(`→ ${OUT_PATH.pathname}`);
}

main().catch((err) => {
  console.error(err?.stack || err);
  process.exit(1);
});
