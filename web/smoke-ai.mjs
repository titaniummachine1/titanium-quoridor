/**
 * Headless smoke test — AppController + live dev-server Titanium proxy.
 * Run: node smoke-ai.mjs [baseUrl]
 */
import { AppController } from './src/game/appController.js';
import { PlayerType } from './src/lib/engineConfig.js';

const baseUrl = process.argv[2] ?? 'http://localhost:5175';

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

async function waitFor(label, fn, timeoutMs = 20_000) {
  const t0 = Date.now();
  while (Date.now() - t0 < timeoutMs) {
    if (fn()) {
      return;
    }
    await sleep(50);
  }
  throw new Error(`timeout: ${label}`);
}

async function probeTitaniumApi() {
  const res = await fetch(`${baseUrl}/api/titanium/genmove`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
    body: JSON.stringify({
      moves: [],
      timeSec: 1,
      maxNodes: 2000,
      engine: 'minimax',
      stream: true,
    }),
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Titanium API ${res.status}: ${err}`);
  }
  const text = await res.text();
  if (!text.includes('"type":"bestmove"')) {
    throw new Error(`Titanium stream missing bestmove: ${text.slice(-300)}`);
  }
  console.log('OK  Titanium API stream returns bestmove');
}

async function runControllerCase(name, setup) {
  const ctrl = new AppController();
  setup(ctrl);
  ctrl.settings.playerAiSettings[0].wallClockSeconds = 1;
  ctrl.settings.playerAiSettings[1].wallClockSeconds = 1;

  const errors = [];
  ctrl.onChange = () => {
    const e = ctrl.engineErrors;
    for (const [k, v] of Object.entries(e)) {
      if (v) {
        errors.push(`${k}:${v}`);
      }
    }
  };

  ctrl.maybeRequestAiMove();

  await waitFor(
    `${name} first ply`,
    () => ctrl.session.actions.length > 0 || errors.length > 0,
    25_000,
  );

  if (errors.length) {
    throw new Error(`${name} engine error: ${errors.join(' | ')}`);
  }
  if (!ctrl.session.actions.length) {
    throw new Error(`${name}: no move applied (aiThinking=${ctrl.aiThinking})`);
  }

  console.log(
    `OK  ${name}: ply1=${ctrl.session.actionToLabel?.(ctrl.session.actions[0]) ?? ctrl.session.actions[0]} aiThinking=${ctrl.aiThinking}`,
  );
}

async function main() {
  console.log(`Smoke test → ${baseUrl}`);
  await probeTitaniumApi();

  await runControllerCase('Titanium vs Human', (ctrl) => {
    ctrl.settings.players = [PlayerType.TitaniumMinimax, PlayerType.Human];
  });

  await runControllerCase('Gorisanson vs Human', (ctrl) => {
    ctrl.settings.players = [PlayerType.GorisansonMCTS, PlayerType.Human];
  });

  console.log('All smoke checks passed.');
}

main().catch((err) => {
  console.error('FAIL', err.message ?? err);
  process.exit(1);
});
