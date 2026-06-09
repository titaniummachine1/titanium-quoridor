/**
 * RUST TITANIUM ONLY — release CLI (`titanium genmove`).
 * Default: Gorisanson-style MCTS in Rust (`--engine mcts`).
 */

import { spawn } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { BENCH_MAX_SIMULATIONS, BENCH_TIME_SEC } from './bench_limits.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const BIN_NAME = process.platform === 'win32' ? 'titanium.exe' : 'titanium';
const DEFAULT_BIN = path.join(ROOT, 'engine', 'target', 'release', BIN_NAME);

function resolveBinary() {
  if (process.env.TITANIUM_BIN) {
    return process.env.TITANIUM_BIN;
  }
  return DEFAULT_BIN;
}

function parseSearchInfo(stderr) {
  const text = stderr || '';
  const jsonLine = text
    .split(/\r?\n/)
    .reverse()
    .find((line) => line.startsWith('info json '));
  if (!jsonLine) {
    return null;
  }
  try {
    return JSON.parse(jsonLine.slice('info json '.length));
  } catch {
    return null;
  }
}

function metaFromInfo(info) {
  return {
    stoppedBy: info?.stoppedBy ?? 'mcts',
    simulations: info?.simulations ?? 0,
    nodes: info?.nodes ?? 0,
    whiteDist: info?.whiteDist,
    blackDist: info?.blackDist,
    rootWinRate: info?.rootWinRate,
    rootScore: info?.rootScore,
    searchDepth: info?.searchDepth,
    depthLog: info?.depthLog,
    aspirationFails: info?.aspirationFails,
    lmrReSearches: info?.lmrReSearches,
    mateExtensions: info?.mateExtensions,
    pvMateFailures: info?.pvMateFailures,
    elapsedMs: info?.elapsedMs,
  };
}

function parseDepthLine(line) {
  const m = /^info depth (\d+) score (-?\d+) nodes (\d+) asp (\d+) lmr (\d+)/.exec(line.trim());
  if (!m) {
    return null;
  }
  return {
    depth: Number(m[1]),
    score: Number(m[2]),
    nodes: Number(m[3]),
    aspirationFails: Number(m[4]),
    lmrReSearches: Number(m[5]),
  };
}

function parseProgressLine(line) {
  // Example: info progress sims 12345 elapsed_ms 789 winrate 0.512
  const m = /^info\s+progress\s+sims\s+(\d+)\s+elapsed_ms\s+(\d+)\s+winrate\s+([0-9.]+)/.exec(
    line.trim(),
  );
  if (!m) {
    return null;
  }
  return {
    simulations: Number(m[1]),
    elapsedMs: Number(m[2]),
    rootWinRate: Number(m[3]),
  };
}

/**
 * @param {string[]} algebraicMoves
 * @param {{ timeSec?: number, maxSims?: number, uct?: number, engine?: string, log?: boolean, useCatGuidance?: boolean, onProgress?: (p: { simulations: number, elapsedMs: number, rootWinRate?: number }) => void }} [opts]
 */
export async function chooseTitaniumMove(algebraicMoves = [], opts = {}) {
  const bin = resolveBinary();
  const timeSec = opts.timeSec ?? (Number(process.env.TITANIUM_TIME_SEC) || BENCH_TIME_SEC);
  const maxSims =
    opts.maxSims ?? (Number(process.env.TITANIUM_MAX_SIMS) || BENCH_MAX_SIMULATIONS);
  const uct = opts.uct ?? 0.2;
  const engine = opts.engine ?? process.env.TITANIUM_ENGINE ?? 'mcts';
  const log = opts.log ?? process.env.TITANIUM_LOG === '1';

  const args = [
    'genmove',
    ...algebraicMoves,
    '--engine',
    engine,
    '--time',
    String(timeSec),
    '--sims',
    String(maxSims),
    '--uct',
    String(uct),
  ];
  if (log) {
    args.push('--log');
  }
  if (opts.useCatGuidance) {
    args.push('--cat');
  }

  const childEnv = { ...process.env };
  if (opts.disableBridge) {
    childEnv.TITANIUM_BRIDGE = '0';
  }
  if (opts.disableBook) {
    childEnv.TITANIUM_DISABLE_BOOK = '1';
  }

  const child = spawn(bin, args, {
    cwd: ROOT,
    stdio: ['ignore', 'pipe', 'pipe'],
    env: childEnv,
  });

  let stdoutText = '';
  let stderrText = '';
  let stderrCarry = '';

  child.stdout.setEncoding('utf8');
  child.stderr.setEncoding('utf8');

  child.stdout.on('data', (chunk) => {
    stdoutText += chunk;
  });

  child.stderr.on('data', (chunk) => {
    stderrText += chunk;
    stderrCarry += chunk;

    const parts = stderrCarry.split(/\r?\n/);
    stderrCarry = parts.pop() ?? '';
    for (const line of parts) {
      const progress = parseProgressLine(line);
      if (progress && typeof opts.onProgress === 'function') {
        opts.onProgress(progress);
      }
      const depth = parseDepthLine(line);
      if (depth && typeof opts.onDepth === 'function') {
        opts.onDepth(depth);
      }
    }
  });

  const exitResult = await new Promise((resolve, reject) => {
    child.on('error', reject);
    child.on('close', (code, signal) => resolve({ code, signal }));
  });

  if (exitResult.code !== 0) {
    throw new Error(stderrText.trim() || `titanium genmove exited ${exitResult.code}`);
  }

  const line = stdoutText.trim().split(/\r?\n/).pop() || '';
  const match = /^bestmove\s+(\S+)/.exec(line);
  if (!match || match[1] === '(none)') {
    throw new Error(`no legal move from titanium: ${line}`);
  }

  const info = parseSearchInfo(stderrText);
  return { move: match[1], meta: metaFromInfo(info) };
}
