#!/usr/bin/env node
/**
 * Parallel Titanium minimax vs Gorisanson — one game per worker process.
 *
 *   node benchmark/parallel_gorisanson.mjs --workers 4 --games 4 --time 10
 *   node benchmark/parallel_gorisanson.mjs --workers 4 --games 8 --time 10 -v
 */

import { spawn } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const WORKER = path.join(ROOT, 'benchmark', 'tune_minimax.mjs');

function parseArgs(argv) {
  const opts = {
    workers: 4,
    games: 4,
    timeSec: 10,
    gorisansonTimeSec: 3,
    verbose: true,
    label: 'parallel',
  };

  for (let i = 2; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--workers' && argv[i + 1]) opts.workers = Number(argv[++i]);
    else if (arg === '--games' && argv[i + 1]) opts.games = Number(argv[++i]);
    else if (arg === '--time' && argv[i + 1]) opts.timeSec = Number(argv[++i]);
    else if (arg === '--gorisanson-time' && argv[i + 1]) {
      opts.gorisansonTimeSec = Number(argv[++i]);
    } else if (arg === '--label' && argv[i + 1]) opts.label = argv[++i];
    else if (arg === '--verbose' || arg === '-v') opts.verbose = true;
  }

  return opts;
}

function prefixLines(stream, tag, onLine) {
  let carry = '';
  stream.setEncoding('utf8');
  stream.on('data', (chunk) => {
    carry += chunk;
    const parts = carry.split(/\r?\n/);
    carry = parts.pop() ?? '';
    for (const line of parts) {
      if (!line) {
        continue;
      }
      process.stdout.write(`${tag} ${line}\n`);
      onLine?.(line);
    }
  });
}

function runOneGame(gameIndex, opts) {
  return new Promise((resolve, reject) => {
    const tag = `[g${gameIndex}]`;
    const args = [
      WORKER,
      '--games',
      '1',
      '--time',
      String(opts.timeSec),
      '--gorisanson-time',
      String(opts.gorisansonTimeSec),
      '--label',
      `${opts.label}-g${gameIndex}`,
    ];
    if (opts.verbose) {
      args.push('-v');
    }

    const env = {
      ...process.env,
      TITANIUM_ENGINE: 'minimax',
    };

    const child = spawn(process.execPath, args, {
      cwd: ROOT,
      env,
      stdio: ['ignore', 'pipe', 'pipe'],
    });

    let jsonLine = null;

    prefixLines(child.stdout, tag, (line) => {
      if (line.startsWith('{')) {
        jsonLine = line;
      }
    });
    prefixLines(child.stderr, tag);

    child.on('error', reject);
    child.on('close', (code) => {
      if (!jsonLine) {
        reject(new Error(`game ${gameIndex}: no JSON output`));
        return;
      }

      try {
        resolve({ gameIndex, code: code ?? 1, summary: JSON.parse(jsonLine) });
      } catch (err) {
        reject(new Error(`game ${gameIndex}: bad JSON — ${err.message}\n${jsonLine}`));
      }
    });
  });
}

async function runPool(opts) {
  const results = [];
  let nextGame = 1;
  let inFlight = 0;
  const started = performance.now();

  return new Promise((resolve, reject) => {
    function launch() {
      while (inFlight < opts.workers && nextGame <= opts.games) {
        const gameIndex = nextGame++;
        inFlight += 1;

        process.stdout.write(`[start] game ${gameIndex}/${opts.games}\n`);

        runOneGame(gameIndex, opts)
          .then((result) => {
            inFlight -= 1;
            results.push(result);

            if (result.summary) {
              const s = result.summary;
              process.stdout.write(
                `[done]  game ${gameIndex}: score ${s.score} · ${s.avgPlies} plies · ${s.wallSec}s wall · avg ${s.avgNodesPerMove} nodes/move\n`,
              );
            }

            if (nextGame > opts.games && inFlight === 0) {
              resolve({ results, wallSec: (performance.now() - started) / 1000 });
              return;
            }

            launch();
          })
          .catch(reject);
      }
    }

    launch();
  });
}

function aggregate(results, wallSec, opts) {
  let titaniumWins = 0;
  let gorisansonWins = 0;
  let draws = 0;
  let totalPlies = 0;
  let totalNodes = 0;
  let nodeSamples = 0;

  for (const { summary } of results) {
    if (!summary) {
      continue;
    }
    const [a, b] = summary.score.split('-').map(Number);
    titaniumWins += a;
    gorisansonWins += b;
    draws += summary.draws ?? 0;
    totalPlies += summary.avgPlies ?? 0;
    if (summary.avgNodesPerMove) {
      totalNodes += summary.avgNodesPerMove * (summary.avgPlies ?? 0);
      nodeSamples += summary.avgPlies ?? 0;
    }
  }

  const games = results.length;
  return {
    label: opts.label,
    workers: opts.workers,
    games,
    timeSec: opts.timeSec,
    engine: 'minimax',
    score: `${titaniumWins}-${gorisansonWins}`,
    draws,
    winRate: games ? titaniumWins / games : 0,
    wallSec: Number(wallSec.toFixed(1)),
    avgPlies: games ? Number((totalPlies / games).toFixed(1)) : 0,
    avgNodesPerMove: nodeSamples ? Math.round(totalNodes / nodeSamples) : 0,
    games_detail: results.map((r) => r.summary).filter(Boolean),
  };
}

async function main() {
  const opts = parseArgs(process.argv);

  console.log(
    `Parallel minimax vs Gorisanson — ${opts.games} games · ${opts.workers} workers · Ti ${opts.timeSec}s / Go ${opts.gorisansonTimeSec}s`,
  );
  console.log('');

  const { results, wallSec } = await runPool(opts);
  const summary = aggregate(results, wallSec, opts);

  console.log('');
  console.log(JSON.stringify(summary, null, 2));
  process.exit(summary.winRate > 0.5 ? 0 : 1);
}

main().catch((err) => {
  console.error(err?.stack || String(err));
  process.exit(2);
});
