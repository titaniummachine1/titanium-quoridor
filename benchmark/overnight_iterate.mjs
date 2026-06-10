#!/usr/bin/env node
/**
 * Chunked overnight iteration — one small run at a time, verify, resume.
 *
 *   node benchmark/overnight_iterate.mjs --steps 1          # one probe round (~5–15 min)
 *   node benchmark/overnight_iterate.mjs --steps 4 --resume   # four rounds, then stop
 *   node benchmark/overnight_iterate.mjs --resume --pierce-sweep   # pierce grid (6×2g) only when asked
 *
 * Do NOT use a single 10h process. Loop manually or via scheduler:
 *   for ($i=0; $i -lt 20; $i++) { node benchmark/overnight_iterate.mjs --resume --steps 1 }
 */

import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const OUT_DIR = path.join(ROOT, 'benchmark', 'overnight');
const CHECKPOINT_DIR = path.join(OUT_DIR, 'checkpoints');
const PARALLEL_GORI = path.join(ROOT, 'benchmark', 'parallel_gorisanson.mjs');
const PARALLEL_SELF = path.join(ROOT, 'benchmark', 'parallel_selfplay.mjs');
const PARALLEL_ISHTAR = path.join(ROOT, 'benchmark', 'parallel_ishtar.mjs');
const KEEP_AWAKE = path.join(ROOT, 'benchmark', 'keep_awake.ps1');
const LOG_PATH = path.join(OUT_DIR, 'overnight.log');
const CHECKPOINT_PATH = path.join(OUT_DIR, 'checkpoint.json');
const STATUS_PATH = path.join(OUT_DIR, 'STATUS.md');
const BEST_PIERCE_PATH = path.join(OUT_DIR, 'best_pierce.json');

/** Pierce-first LMR presets — env overrides read in engine/src/search/lmr_profile.rs */
const PIERCE_PRESETS = [
  { name: 'default', env: {} },
  { name: 'deep-pierce', env: { TITANIUM_PIERCE_RELAX: '0.38', TITANIUM_PIERCE_HOT: '7', TITANIUM_PIERCE_AGGR: '0.65' } },
  { name: 'rapid-pierce', env: { TITANIUM_PIERCE_RELAX: '0.42', TITANIUM_PIERCE_HOT: '6', TITANIUM_PIERCE_AGGR: '0.60', TITANIUM_PIERCE_POW: '1.25' } },
  { name: 'late-relax', env: { TITANIUM_PIERCE_RELAX: '0.55', TITANIUM_PIERCE_HOT: '4', TITANIUM_PIERCE_AGGR: '0.48' } },
  { name: 'narrow-open', env: { TITANIUM_PIERCE_RELAX: '0.45', TITANIUM_PIERCE_HOT: '8', TITANIUM_PIERCE_AGGR: '0.70', TITANIUM_PIERCE_POW: '1.30' } },
  { name: 'balanced', env: { TITANIUM_PIERCE_RELAX: '0.48', TITANIUM_PIERCE_HOT: '5', TITANIUM_PIERCE_AGGR: '0.55' } },
];

const STRESS_PROBE = { label: 'stress-8v12', timeSec: 8, gorisansonTimeSec: 12 };

/** Fast probes — many per night. */
const PROBES = [
  { label: 'probe-10v10', timeSec: 10, gorisansonTimeSec: 10 },
  { label: 'probe-5v10', timeSec: 5, gorisansonTimeSec: 10 },
  { label: 'probe-10v5', timeSec: 10, gorisansonTimeSec: 5 },
  { label: 'probe-3v10', timeSec: 3, gorisansonTimeSec: 10 },
  { label: 'probe-8v12', timeSec: 8, gorisansonTimeSec: 12 },
  { label: 'probe-10v15', timeSec: 10, gorisansonTimeSec: 15 },
  { label: 'probe-10v20', timeSec: 10, gorisansonTimeSec: 20 },
  { label: 'probe-2v10', timeSec: 2, gorisansonTimeSec: 10 },
];

function parseArgs(argv) {
  const opts = {
    steps: 1,
    hours: 0,
    workers: 6,
    probeGames: 2,
    confirmGames: 4,
    resume: false,
    skipPerft: false,
    pierceSweep: false,
    pierceEvery: 0,
    noBuild: false,
    noConfirm: false,
  };
  for (let i = 2; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--steps' && argv[i + 1]) opts.steps = Number(argv[++i]);
    else if (arg === '--hours' && argv[i + 1]) opts.hours = Number(argv[++i]);
    else if (arg === '--workers' && argv[i + 1]) opts.workers = Number(argv[++i]);
    else if (arg === '--probe-games' && argv[i + 1]) opts.probeGames = Number(argv[++i]);
    else if (arg === '--confirm-games' && argv[i + 1]) opts.confirmGames = Number(argv[++i]);
    else if (arg === '--pierce-every' && argv[i + 1]) opts.pierceEvery = Number(argv[++i]);
    else if (arg === '--resume') opts.resume = true;
    else if (arg === '--skip-perft') opts.skipPerft = true;
    else if (arg === '--pierce-sweep') opts.pierceSweep = true;
    else if (arg === '--no-build') opts.noBuild = true;
    else if (arg === '--no-confirm') opts.noConfirm = true;
  }
  return opts;
}

function log(msg) {
  const line = `[${new Date().toISOString()}] ${msg}`;
  console.log(line);
  fs.mkdirSync(OUT_DIR, { recursive: true });
  fs.appendFileSync(LOG_PATH, `${line}\n`, 'utf8');
}

function writeStatus(state) {
  const lines = [
    '# Overnight tournament',
    '',
    `Updated: ${new Date().toISOString()}`,
    '',
    `| Metric | Value |`,
    `|--------|-------|`,
    `| Step | ${state.stepIndex ?? 0} |`,
    `| Chunks this run | ${state.chunksThisRun ?? 0} / ${state.opts?.steps ?? 1} |`,
    `| Last | ${state.lastLabel ?? '—'} |`,
    `| Last score | ${state.lastScore ?? '—'} |`,
    `| Probes run | ${state.probesRun ?? 0} |`,
    `| Confirms run | ${state.confirmsRun ?? 0} |`,
    `| Self-play | ${state.selfRuns ?? 0} |`,
    `| Ishtar | ${state.ishtarRuns ?? 0} |`,
    `| Pierce preset | ${state.bestPierce?.name ?? 'default'} |`,
    `| Workers | ${state.opts?.workers ?? '?'} |`,
    `| Deadline | ${state.deadline ? new Date(state.deadline).toISOString() : '—'} |`,
    '',
    'Pierce-first LMR via `TITANIUM_PIERCE_*` env + `apply_pierce_schedule` in engine.',
    '',
    '## Next chunk',
    '```',
    'node benchmark/overnight_iterate.mjs --resume --steps 1',
    '```',
  ];
  fs.writeFileSync(STATUS_PATH, lines.join('\n'), 'utf8');
}

function saveCheckpoint(state) {
  fs.mkdirSync(CHECKPOINT_DIR, { recursive: true });
  fs.writeFileSync(CHECKPOINT_PATH, JSON.stringify(state, null, 2), 'utf8');
}

function loadCheckpoint() {
  if (!fs.existsSync(CHECKPOINT_PATH)) return null;
  return JSON.parse(fs.readFileSync(CHECKPOINT_PATH, 'utf8'));
}

function parseSummary(stdout, reportDir, label) {
  const marker = stdout
    .split(/\r?\n/)
    .find((l) => l.startsWith('OVERNIGHT_JSON:'));
  if (marker) {
    return JSON.parse(marker.slice('OVERNIGHT_JSON:'.length));
  }
  const agg = path.join(reportDir, `${label}-aggregate.json`);
  if (fs.existsSync(agg)) {
    return JSON.parse(fs.readFileSync(agg, 'utf8'));
  }
  const lines = stdout.split(/\r?\n/).filter((l) => l.startsWith('{'));
  if (lines.length) {
    try {
      return JSON.parse(lines.join('\n'));
    } catch {
      /* fall through */
    }
  }
  return null;
}

function runCmd(cmd, args, { cwd = ROOT, env = process.env, timeoutMs = 0 } = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(cmd, args, { cwd, env, stdio: ['ignore', 'pipe', 'pipe'] });
    let stdout = '';
    let stderr = '';
    let timer;
    if (timeoutMs > 0) {
      timer = setTimeout(() => {
        child.kill('SIGTERM');
        reject(new Error(`timeout ${timeoutMs}ms`));
      }, timeoutMs);
    }
    child.stdout.on('data', (c) => {
      stdout += c;
    });
    child.stderr.on('data', (c) => {
      stderr += c;
    });
    child.on('error', (e) => {
      if (timer) clearTimeout(timer);
      reject(e);
    });
    child.on('close', (code) => {
      if (timer) clearTimeout(timer);
      resolve({ code: code ?? 1, stdout, stderr });
    });
  });
}

let awakeChild = null;

function startKeepAwake() {
  if (awakeChild) return;
  awakeChild = spawn(
    'powershell',
    ['-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', KEEP_AWAKE],
    { cwd: ROOT, stdio: 'ignore', detached: true },
  );
  awakeChild.unref();
  log(`keep_awake started pid=${awakeChild.pid}`);
}

function stopKeepAwake() {
  if (awakeChild) {
    try {
      process.kill(awakeChild.pid);
    } catch {
      /* ignore */
    }
    awakeChild = null;
  }
}

async function buildEngine() {
  log('cargo build --release');
  const { code, stderr } = await runCmd('cargo', ['build', '--release'], {
    cwd: path.join(ROOT, 'engine'),
    timeoutMs: 600_000,
  });
  if (code !== 0) throw new Error(stderr.slice(-2000));
}

async function runPerftGate() {
  log('perft d4 gate');
  const { code, stderr, stdout } = await runCmd(
    'cargo',
    ['test', '--release', 'perft_depth4', '--', '--ignored', '--nocapture'],
    { cwd: path.join(ROOT, 'engine'), timeoutMs: 120_000 },
  );
  if (code !== 0) throw new Error(`perft failed:\n${stderr}\n${stdout}`);
  log('perft PASS');
}

function analyzeSummary(summary) {
  let minMargin = Infinity;
  let losses = 0;
  for (const g of summary.games_detail ?? []) {
    if (g.winner !== 'rust-titanium') losses += 1;
    if (g.finalMargin != null && g.finalMargin < 200) {
      minMargin = Math.min(minMargin, g.finalMargin);
    }
  }
  return {
    winRate: summary.winRate,
    losses,
    minMargin: Number.isFinite(minMargin) ? minMargin : null,
    illegalMoveCount: summary.illegalMoveCount ?? 0,
  };
}

function needsConfirm(analysis) {
  if (analysis.illegalMoveCount > 0) return true;
  if (analysis.losses > 0) return true;
  if (analysis.winRate < 1) return true;
  if (analysis.minMargin != null && analysis.minMargin <= 5) return true;
  return false;
}

function pierceEnv(state) {
  const fromFile = loadBestPierce();
  const base = { ...fromFile.env, ...(state.bestPierce?.env ?? {}) };
  return {
    TITANIUM_ENGINE: 'minimax',
    TITANIUM_MAX_NODES: '10000000000',
    GORISANSON_MAX_VISITS: '66000',
    ...base,
  };
}

function loadBestPierce() {
  if (!fs.existsSync(BEST_PIERCE_PATH)) {
    return { name: 'default', env: {}, score: -Infinity };
  }
  return JSON.parse(fs.readFileSync(BEST_PIERCE_PATH, 'utf8'));
}

function saveBestPierce(preset, score) {
  const data = { ...preset, score, updated: new Date().toISOString() };
  fs.writeFileSync(BEST_PIERCE_PATH, JSON.stringify(data, null, 2), 'utf8');
}

function scoreGori(analysis) {
  if (analysis.illegalMoveCount > 0) return -10_000;
  return analysis.winRate * 1000 + (analysis.minMargin ?? 0) * 8 - analysis.losses * 250;
}

function scoreSelf(summary) {
  if (summary.illegalMoveCount > 0) return -10_000;
  const sym = summary.symmetryDelta ?? 0;
  return 500 - sym * 120 + (summary.avgPlies ?? 0);
}

function scoreIshtar(analysis) {
  if (analysis.illegalMoveCount > 0) return -10_000;
  return analysis.winRate * 1500 + (analysis.minMargin ?? 0) * 12;
}

function goriDominates(analysis) {
  return analysis.winRate >= 1 && (analysis.minMargin ?? 0) > 5 && analysis.illegalMoveCount === 0;
}

function goriTooEasy(analysis) {
  return analysis.winRate >= 0.75 && analysis.losses === 0;
}

async function runParallel(args, { label, reportDir, env, timeoutMs }) {
  fs.mkdirSync(reportDir, { recursive: true });
  const { code, stdout, stderr } = await runCmd(process.execPath, args, {
    env: { ...process.env, ...env },
    timeoutMs,
  });
  const summary = parseSummary(stdout, reportDir, label);
  if (!summary) {
    log(`BATCH ${label} FAILED parse — exit ${code}\n${stderr.slice(-1200)}`);
    return null;
  }
  return summary;
}

function recordHistory(entry) {
  fs.appendFileSync(
    path.join(OUT_DIR, 'history.jsonl'),
    `${JSON.stringify({ ts: new Date().toISOString(), ...entry })}\n`,
  );
}

async function runGoriBatch(round, { games, workers, suffix = '', env }) {
  const label = suffix ? `${round.label}${suffix}` : round.label;
  const reportDir = path.join(OUT_DIR, label);
  log(`BATCH ${label} · Ti ${round.timeSec}s Go ${round.gorisansonTimeSec}s · ${games}g ${workers}w pierce=${env.TITANIUM_PIERCE_RELAX ?? 'def'}`);

  const timeoutMs = Math.max(2_700_000, games * (round.timeSec + round.gorisansonTimeSec) * 50_000);
  const summary = await runParallel(
    [
      PARALLEL_GORI,
      '--workers', String(Math.min(workers, games)),
      '--games', String(games),
      '--time', String(round.timeSec),
      '--gorisanson-time', String(round.gorisansonTimeSec),
      '--label', label,
      '--report-dir', reportDir,
    ],
    { label, reportDir, env, timeoutMs },
  );
  if (!summary) return null;

  const analysis = analyzeSummary(summary);
  log(
    `BATCH ${label}: ${summary.score} WR=${(summary.winRate * 100).toFixed(0)}% ` +
      `minMargin=${analysis.minMargin} illegal=${analysis.illegalMoveCount} wall=${summary.wallSec}s`,
  );
  recordHistory({ label, opponent: 'gorisanson', games, workers, summary, analysis, pierce: env });
  return { summary, analysis, label };
}

async function runSelfBatch({ games, workers, timeSec, label, env }) {
  const reportDir = path.join(OUT_DIR, label);
  log(`SELF ${label} · ${games}g ${workers}w · ${timeSec}s`);
  const timeoutMs = Math.max(1_800_000, games * timeSec * 40_000);
  const summary = await runParallel(
    [
      PARALLEL_SELF,
      '--workers', String(Math.min(workers, games)),
      '--games', String(games),
      '--time', String(timeSec),
      '--label', label,
      '--report-dir', reportDir,
    ],
    { label, reportDir, env, timeoutMs },
  );
  if (!summary) return null;
  log(`SELF ${label}: ${summary.score} symΔ=${summary.symmetryDelta} illegal=${summary.illegalMoveCount}`);
  recordHistory({ label, opponent: 'self', games, workers, summary, pierce: env });
  return { summary, label };
}

async function runIshtarBatch({ games, workers, timeSec, label, ishtarPreset, env }) {
  const reportDir = path.join(OUT_DIR, label);
  log(`ISHTAR ${label} · ${games}g ${workers}w · Ti ${timeSec}s · ${ishtarPreset}`);
  const timeoutMs = Math.max(2_400_000, games * (timeSec + 30) * 60_000);
  const summary = await runParallel(
    [
      PARALLEL_ISHTAR,
      '--workers', String(Math.min(workers, games, 2)),
      '--games', String(games),
      '--time', String(timeSec),
      '--ishtar', ishtarPreset,
      '--label', label,
      '--report-dir', reportDir,
    ],
    { label, reportDir, env, timeoutMs },
  );
  if (!summary) return null;
  const analysis = analyzeSummary(summary);
  log(`ISHTAR ${label}: ${summary.score} WR=${(summary.winRate * 100).toFixed(0)}%`);
  recordHistory({ label, opponent: 'ishtar', games, workers, summary, analysis, pierce: env });
  return { summary, analysis, label };
}

async function sweepPiercePresets(state, opts) {
  log(`PIERCE sweep (${PIERCE_PRESETS.length} presets × ${Math.min(2, opts.probeGames)}g on stress-8v12)`);
  let best = loadBestPierce();
  for (const preset of PIERCE_PRESETS) {
    if (opts.hours > 0 && Date.now() >= state.deadline) break;
    const env = { ...pierceEnv({ bestPierce: preset }), ...preset.env };
    const result = await runGoriBatch(STRESS_PROBE, {
      games: Math.min(2, opts.probeGames),
      workers: 2,
      suffix: `-pierce-${preset.name}`,
      env,
    });
    if (!result) continue;
    const s = scoreGori(result.analysis);
    log(`  pierce ${preset.name}: score=${s.toFixed(0)} (${result.summary.score})`);
    if (s > (best.score ?? -Infinity)) {
      best = { name: preset.name, env: preset.env, score: s };
      state.bestPierce = { name: preset.name, env: preset.env };
      saveBestPierce(best, s);
      log(`  → new best pierce: ${preset.name} score=${s.toFixed(0)}`);
    }
  }
}

async function gitCheckpoint(message) {
  const paths = [
    'benchmark/overnight/history.jsonl',
    'benchmark/overnight/checkpoint.json',
    'benchmark/overnight/STATUS.md',
    'engine/src/search/lmr_profile.rs',
    'engine/src/search/alphabeta.rs',
    'benchmark/overnight_iterate.mjs',
    'benchmark/overnight/best_pierce.json',
    'benchmark/parallel_gorisanson.mjs',
    'benchmark/parallel_selfplay.mjs',
    'benchmark/parallel_ishtar.mjs',
    'benchmark/tune_selfplay.mjs',
    'benchmark/tune_ishtar.mjs',
    'benchmark/lib/match_engine.mjs',
    'benchmark/lib/bench_limits.mjs',
  ];
  for (const p of paths) {
    const full = path.join(ROOT, p);
    if (fs.existsSync(full)) {
      await runCmd('git', ['add', p]);
    }
  }
  const { code, stderr, stdout } = await runCmd('git', ['commit', '-m', message]);
  if (code === 0) {
    const hash = stdout.match(/\[[\w/-]+ ([0-9a-f]+)\]/)?.[1] ?? 'ok';
    log(`git checkpoint ${hash}: ${message.split('\n')[0]}`);
  } else if (!`${stderr}${stdout}`.includes('nothing to commit')) {
    log(`git skip: ${stderr.slice(0, 200)}`);
  }
}

function ingestExistingResults(state) {
  const hist = path.join(OUT_DIR, 'history.jsonl');
  const existing = fs.existsSync(hist) ? fs.readFileSync(hist, 'utf8') : '';

  for (const [label, file] of [
    ['fair-10v10', 'fair-10v10-aggregate.json'],
    ['ti8-go12', 'ti8-go12-aggregate.json'],
  ]) {
    const agg = path.join(OUT_DIR, label, file);
    if (!fs.existsSync(agg) || existing.includes(`"${label}"`)) continue;
    const summary = JSON.parse(fs.readFileSync(agg, 'utf8'));
    const analysis = analyzeSummary(summary);
    fs.appendFileSync(
      hist,
      `${JSON.stringify({ ts: new Date().toISOString(), label, summary, analysis, note: 'recovered' })}\n`,
    );
    log(`Recovered ${label}: ${summary.score} WR=${summary.winRate}`);
    state.probesRun = (state.probesRun ?? 0) + 1;
    state.lastScore = summary.score;
    state.lastLabel = `${label} (recovered)`;
  }
}

function shouldRunPierceSweep(opts, stepIndex) {
  if (opts.pierceSweep) return true;
  if (opts.pierceEvery > 0 && stepIndex > 0 && stepIndex % opts.pierceEvery === 0) return true;
  return false;
}

async function main() {
  const opts = parseArgs(process.argv);
  let stepIndex = 0;
  const deadline = opts.hours > 0 ? Date.now() + opts.hours * 3600 * 1000 : null;
  const state = {
    stepIndex: 0,
    chunksThisRun: 0,
    probesRun: 0,
    confirmsRun: 0,
    selfRuns: 0,
    ishtarRuns: 0,
    deadline,
    bestPierce: loadBestPierce(),
    opts: {
      steps: opts.steps,
      workers: opts.workers,
      probeGames: opts.probeGames,
      confirmGames: opts.confirmGames,
    },
  };

  if (opts.resume) {
    const cp = loadCheckpoint();
    if (cp) {
      stepIndex = cp.stepIndex ?? 0;
      state.stepIndex = stepIndex;
      state.probesRun = cp.probesRun ?? 0;
      state.confirmsRun = cp.confirmsRun ?? 0;
      state.selfRuns = cp.selfRuns ?? 0;
      state.ishtarRuns = cp.ishtarRuns ?? 0;
      state.bestPierce = cp.bestPierce ?? loadBestPierce();
      if (cp.deadline && opts.hours > 0) state.deadline = cp.deadline;
      log(`Resume step ${stepIndex} pierce=${state.bestPierce?.name ?? 'default'}`);
    }
  }

  fs.mkdirSync(OUT_DIR, { recursive: true });
  ingestExistingResults(state);
  if (opts.hours > 0) startKeepAwake();

  log(
    `Chunk run · steps=${opts.steps} workers=${opts.workers} ` +
      `probe=${opts.probeGames} confirm=${opts.confirmGames} pierce=${state.bestPierce?.name ?? 'default'}`,
  );
  if (!opts.noBuild) await buildEngine();
  if (!opts.skipPerft) await runPerftGate();

  writeStatus(state);
  saveCheckpoint(state);

  if (opts.pierceSweep && opts.steps === 0) {
    await sweepPiercePresets(state, opts);
    saveCheckpoint(state);
    writeStatus(state);
    log('Pierce sweep only — done');
    stopKeepAwake();
    return;
  }

  let chunksDone = 0;
  while (chunksDone < opts.steps) {
    if (deadline != null && Date.now() >= deadline) {
      log('deadline hit');
      break;
    }

    const probe = PROBES[stepIndex % PROBES.length];
    stepIndex += 1;
    state.stepIndex = stepIndex;
    chunksDone += 1;
    state.chunksThisRun = chunksDone;
    const env = pierceEnv(state);

    log(`── chunk ${chunksDone}/${opts.steps} · step ${stepIndex} · ${probe.label} ──`);

    try {
      if (shouldRunPierceSweep(opts, stepIndex)) {
        await sweepPiercePresets(state, opts);
      }

      const probeResult = await runGoriBatch(probe, {
        games: opts.probeGames,
        workers: opts.workers,
        env,
      });
      state.probesRun = (state.probesRun ?? 0) + 1;

      if (probeResult) {
        state.lastLabel = probeResult.label;
        state.lastScore = probeResult.summary.score;

        if (!opts.noConfirm && needsConfirm(probeResult.analysis)) {
          log(`  → confirm (${probeResult.analysis.losses}L margin=${probeResult.analysis.minMargin})`);
          const confirmResult = await runGoriBatch(probe, {
            games: opts.confirmGames,
            workers: opts.workers,
            suffix: '-confirm',
            env,
          });
          state.confirmsRun = (state.confirmsRun ?? 0) + 1;
          if (confirmResult) {
            state.lastLabel = confirmResult.label;
            state.lastScore = confirmResult.summary.score;
            if (confirmResult.analysis.illegalMoveCount > 0 && !opts.skipPerft) {
              await runPerftGate();
            }
          }
        } else if (opts.noConfirm && needsConfirm(probeResult.analysis)) {
          log(`  → confirm deferred (manual review): ${probeResult.analysis.losses}L margin=${probeResult.analysis.minMargin}`);
        } else {
          log('  → skip confirm (clean blowout)');
        }

        const analysis = probeResult.analysis;
        if (goriDominates(analysis) || goriTooEasy(analysis)) {
          const selfResult = await runSelfBatch({
            games: Math.min(2, opts.probeGames),
            workers: Math.min(2, opts.workers),
            timeSec: probe.timeSec,
            label: `${probe.label}-self`,
            env,
          });
          state.selfRuns = (state.selfRuns ?? 0) + 1;
          if (selfResult) {
            const ss = scoreSelf(selfResult.summary);
            log(`  self-play score=${ss.toFixed(0)} symΔ=${selfResult.summary.symmetryDelta}`);
          }
        }

        if (goriTooEasy(analysis) && probe.gorisansonTimeSec >= 10) {
          try {
            const ishtarResult = await runIshtarBatch({
              games: Math.min(2, opts.probeGames),
              workers: 2,
              timeSec: probe.timeSec,
              ishtarPreset: 'short',
              label: `${probe.label}-ishtar`,
              env,
            });
            state.ishtarRuns = (state.ishtarRuns ?? 0) + 1;
            if (ishtarResult) {
              log(`  ishtar score=${scoreIshtar(ishtarResult.analysis).toFixed(0)}`);
            }
          } catch (ishtarErr) {
            log(`  ishtar skip: ${ishtarErr?.message ?? ishtarErr}`);
          }
        }
      }

      if (probeResult) {
        await gitCheckpoint(
          `overnight: ${probeResult.label} ${probeResult.summary.score} pierce=${state.bestPierce?.name ?? 'default'}`,
        );
      }
    } catch (err) {
      log(`STEP error: ${err?.message ?? err}`);
    }

    saveCheckpoint(state);
    writeStatus(state);
  }

  log(
    `Chunk done (${chunksDone}/${opts.steps}) · total probes=${state.probesRun} ` +
      `last=${state.lastLabel} ${state.lastScore} pierce=${state.bestPierce?.name ?? 'default'}`,
  );
  log('Next: node benchmark/overnight_iterate.mjs --resume --steps 1');
  stopKeepAwake();
  saveCheckpoint({ ...state, chunksThisRun: 0 });
  writeStatus(state);
}

process.on('SIGINT', () => {
  stopKeepAwake();
  process.exit(130);
});

main().catch((err) => {
  log(`FATAL: ${err?.stack || err}`);
  stopKeepAwake();
  process.exit(2);
});
