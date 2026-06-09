#!/usr/bin/env node
/**
 * Compare Titanium modes vs Gorisanson (equal 10s budget both sides).
 *
 *   node benchmark/sweep_vs_gorisanson.mjs
 *   node benchmark/sweep_vs_gorisanson.mjs --games 4 --workers 2
 */

import { spawn } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { writeFileSync } from 'node:fs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const WORKER = path.join(ROOT, 'benchmark', 'lib', 'sweep_worker.mjs');

const VARIANTS = [
  {
    id: 'mcts-nobook',
    label: 'MCTS only (no book)',
    engine: 'mcts',
    disableBook: true,
    disableBridge: true,
  },
  {
    id: 'hybrid',
    label: 'Hybrid (MCTS opening + minimax)',
    engine: 'minimax',
    disableBook: false,
    disableBridge: true,
  },
  {
    id: 'hybrid-nobook',
    label: 'Hybrid no book (MCTS opening + minimax)',
    engine: 'minimax',
    disableBook: true,
    disableBridge: true,
  },
  {
    id: 'minimax-pure',
    label: 'Minimax only (no book, no bridge)',
    engine: 'minimax',
    disableBook: true,
    disableBridge: true,
  },
];

function parseArgs(argv) {
  const opts = { games: 4, workers: 2, timeSec: 10 };
  for (let i = 2; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--games' && argv[i + 1]) opts.games = Number(argv[++i]);
    else if (arg === '--workers' && argv[i + 1]) opts.workers = Number(argv[++i]);
    else if (arg === '--time' && argv[i + 1]) opts.timeSec = Number(argv[++i]);
  }
  return opts;
}

function runWorker(variant, gameIndex, opts) {
  return new Promise((resolve, reject) => {
    const env = { ...process.env, TITANIUM_ENGINE: variant.engine };
    if (variant.disableBook) {
      env.TITANIUM_DISABLE_BOOK = '1';
    } else {
      delete env.TITANIUM_DISABLE_BOOK;
    }
    if (variant.disableBridge) {
      env.TITANIUM_BRIDGE = '0';
    } else {
      env.TITANIUM_BRIDGE = '1';
    }

    const child = spawn(
      process.execPath,
      [
        WORKER,
        '--variant',
        variant.id,
        '--game',
        String(gameIndex),
        '--time',
        String(opts.timeSec),
      ],
      {
        cwd: ROOT,
        env,
        stdio: ['ignore', 'pipe', 'pipe'],
      },
    );
    let out = '';
    child.stdout.on('data', (c) => {
      out += c;
    });
    child.on('close', (code) => {
      try {
        const line = out.trim().split(/\r?\n/).pop();
        resolve(JSON.parse(line));
      } catch (err) {
        reject(new Error(out || `worker exit ${code}`));
      }
    });
    child.on('error', reject);
  });
}

async function runVariant(variant, opts) {
  const results = [];
  let next = 0;

  async function poolWorker() {
    while (true) {
      const idx = next++;
      if (idx >= opts.games) return;
      const r = await runWorker(variant, idx + 1, opts);
      results.push(r);
      console.log(
        `  [${variant.id}] game ${idx + 1}/${opts.games}: ${r.result} (${r.plies} plies)`,
      );
    }
  }

  const n = Math.min(opts.workers, opts.games);
  await Promise.all(Array.from({ length: n }, () => poolWorker()));

  const wins = results.filter((r) => r.result === 'win').length;
  const losses = results.filter((r) => r.result === 'loss').length;
  const aborted = results.filter((r) => r.result === 'aborted').length;
  return { variant: variant.id, label: variant.label, wins, losses, aborted, games: opts.games, results };
}

async function main() {
  const opts = parseArgs(process.argv);
  console.log(`Titanium sweep vs Gorisanson · ${opts.games} games/variant · ${opts.workers} parallel · ${opts.timeSec}s`);
  console.log('');

  const summary = [];
  for (const variant of VARIANTS) {
    console.log(`── ${variant.label} ──`);
    const row = await runVariant(variant, opts);
    summary.push(row);
    const pct = ((row.wins / row.games) * 100).toFixed(0);
    const abortNote = row.aborted ? ` · ${row.aborted} aborted (no goal by ply 250)` : '';
    console.log(`  → ${row.wins}-${row.losses} W-L (${pct}% win)${abortNote}\n`);
  }

  console.log('══ SUMMARY ══');
  for (const row of summary) {
    const pct = ((row.wins / row.games) * 100).toFixed(0);
    console.log(`  ${row.label.padEnd(36)} ${row.wins}-${row.losses} W-L  (${pct}% win)`);
  }

  const outPath = path.join(ROOT, 'benchmark', 'sweep_vs_gorisanson_out.json');
  writeFileSync(outPath, JSON.stringify({ opts, summary, at: new Date().toISOString() }, null, 2));
  console.log(`\n→ ${outPath}`);
}

main().catch((err) => {
  console.error(err?.stack || err);
  process.exit(1);
});
