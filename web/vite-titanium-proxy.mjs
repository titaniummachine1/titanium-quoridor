/**
 * Dev-server proxy — browser calls /api/titanium/genmove → Rust titanium binary.
 * Supports SSE progress stream + wall-clock / visit budget from UI sliders.
 */

import { spawn, spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const binName = process.platform === 'win32' ? 'titanium.exe' : 'titanium';
const defaultBin = path.join(repoRoot, 'engine', 'target', 'release', binName);

function resolveBinary() {
  const bin = process.env.TITANIUM_BIN ?? defaultBin;
  if (!existsSync(bin)) {
    throw new Error(
      `Titanium binary missing at ${bin} — run: cd engine && cargo build --release`,
    );
  }
  return bin;
}

function parseProgressLine(line) {
  const progress = /^info progress sims (\d+) elapsed_ms (\d+) winrate ([\d.]+)/.exec(line);
  if (progress) {
    return {
      type: 'progress',
      simulations: Number(progress[1]),
      elapsedMs: Number(progress[2]),
      winRate: Number(progress[3]),
      stoppedBy: 'mcts',
    };
  }
  if (line.startsWith('info json ')) {
    try {
      return { type: 'info', ...JSON.parse(line.slice('info json '.length)) };
    } catch {
      return null;
    }
  }
  return null;
}

function runGenmoveStreaming(moves, options, res) {
  const bin = resolveBinary();
  const timeSec = Math.max(0.1, Number(options.timeSec) || 10);
  const maxSims = Math.max(1, Number(options.maxSimulations) || 2_000_000_000);
  const maxNodes = Math.max(1, Number(options.maxNodes) || maxSims);
  const uct = Number(options.uct) || 0.2;
  const engine = options.engine === 'minimax' ? 'minimax' : 'mcts';

  const args = [
    'genmove',
    ...moves,
    '--engine',
    engine,
    '--time',
    String(timeSec),
    '--log',
  ];
  if (engine === 'minimax') {
    args.push('--nodes', String(maxNodes));
  } else {
    args.push('--sims', String(maxSims), '--uct', String(uct));
  }

  const childEnv = { ...process.env };
  delete childEnv.TITANIUM_DISABLE_BOOK;
  delete childEnv.TITANIUM_BRIDGE;

  const child = spawn(bin, args, { cwd: repoRoot, env: childEnv });
  let stdout = '';
  let stderrBuf = '';

  const writeEvent = (payload) => {
    res.write(`data: ${JSON.stringify(payload)}\n\n`);
  };

  child.stdout.on('data', (chunk) => {
    stdout += chunk.toString();
  });

  child.stderr.on('data', (chunk) => {
    stderrBuf += chunk.toString();
    const lines = stderrBuf.split(/\r?\n/);
    stderrBuf = lines.pop() ?? '';
    for (const line of lines) {
      const parsed = parseProgressLine(line.trim());
      if (parsed) {
        writeEvent(parsed);
      }
    }
  });

  child.on('error', (err) => {
    writeEvent({ type: 'error', error: err.message });
    res.end();
  });

  child.on('close', (code) => {
    if (stderrBuf.trim()) {
      const parsed = parseProgressLine(stderrBuf.trim());
      if (parsed) {
        writeEvent(parsed);
      }
    }

    if (code !== 0) {
      writeEvent({ type: 'error', error: `titanium exited ${code}` });
      res.end();
      return;
    }

    const line = stdout.trim().split(/\r?\n/).pop() || '';
    const match = /^bestmove\s+(\S+)/.exec(line);
    if (!match || match[1] === '(none)') {
      writeEvent({ type: 'error', error: `no legal move: ${line}` });
      res.end();
      return;
    }

    writeEvent({
      type: 'bestmove',
      algebraic: match[1],
      stoppedBy: engine,
    });
    res.end();
  });
}

function runGenmoveSync(moves, options) {
  const bin = resolveBinary();
  const timeSec = Math.max(0.1, Number(options.timeSec) || 10);
  const maxSims = Math.max(1, Number(options.maxSimulations) || 2_000_000_000);
  const maxNodes = Math.max(1, Number(options.maxNodes) || maxSims);
  const uct = Number(options.uct) || 0.2;
  const engine = options.engine === 'minimax' ? 'minimax' : 'mcts';

  const args = [
    'genmove',
    ...moves,
    '--engine',
    engine,
    '--time',
    String(timeSec),
    '--log',
  ];
  if (engine === 'minimax') {
    args.push('--nodes', String(maxNodes));
  } else {
    args.push('--sims', String(maxSims), '--uct', String(uct));
  }

  const childEnv = { ...process.env };
  delete childEnv.TITANIUM_DISABLE_BOOK;
  delete childEnv.TITANIUM_BRIDGE;

  const result = spawnSync(bin, args, {
    encoding: 'utf8',
    cwd: repoRoot,
    maxBuffer: 4 * 1024 * 1024,
    env: childEnv,
  });

  if (result.error) {
    throw new Error(`Titanium binary not found at ${bin}`);
  }
  if (result.status !== 0) {
    throw new Error(result.stderr?.trim() || `titanium genmove exited ${result.status}`);
  }

  const line = (result.stdout || '').trim().split(/\r?\n/).pop() || '';
  const match = /^bestmove\s+(\S+)/.exec(line);
  if (!match || match[1] === '(none)') {
    throw new Error(`no legal move: ${line}`);
  }

  let meta = { stoppedBy: engine, simulations: 0, nodes: 0 };
  const jsonLine = (result.stderr || '')
    .split(/\r?\n/)
    .reverse()
    .find((l) => l.startsWith('info json '));
  if (jsonLine) {
    try {
      meta = { ...meta, ...JSON.parse(jsonLine.slice('info json '.length)) };
    } catch {
      /* ignore */
    }
  }

  return { algebraic: match[1], ...meta };
}

export function titaniumProxyPlugin() {
  return {
    name: 'titanium-rust-proxy',
    configureServer(server) {
      server.middlewares.use('/api/titanium/genmove', (req, res) => {
        if (req.method !== 'POST') {
          res.statusCode = 405;
          res.end('POST only');
          return;
        }

        let body = '';
        req.on('data', (chunk) => {
          body += chunk;
        });
        req.on('end', () => {
          try {
            const payload = JSON.parse(body || '{}');
            const moves = Array.isArray(payload.moves) ? payload.moves.map(String) : [];
            const options = {
              timeSec: payload.timeSec ?? payload.timeMs / 1000,
              maxSimulations: payload.maxSimulations ?? payload.visitsBudget,
              maxNodes: payload.maxNodes,
              uct: payload.uct,
              engine: payload.engine ?? 'mcts',
            };

            const wantsStream =
              req.headers.accept?.includes('text/event-stream') || payload.stream === true;

            if (wantsStream) {
              res.writeHead(200, {
                'Content-Type': 'text/event-stream',
                'Cache-Control': 'no-cache',
                Connection: 'keep-alive',
              });
              runGenmoveStreaming(moves, options, res);
              return;
            }

            const result = runGenmoveSync(moves, options);
            res.setHeader('Content-Type', 'application/json');
            res.end(JSON.stringify(result));
          } catch (err) {
            res.statusCode = 500;
            res.setHeader('Content-Type', 'application/json');
            res.end(JSON.stringify({ error: err.message ?? String(err) }));
          }
        });
      });
    },
  };
}
