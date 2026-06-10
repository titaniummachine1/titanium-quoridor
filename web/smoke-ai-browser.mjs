/**
 * Browser smoke test — at least two AI plies on a fresh game.
 * Run: node smoke-ai-browser.mjs [baseUrl]
 */
import { chromium } from 'playwright';

const baseUrl = process.argv[2] ?? 'http://localhost:5175';

async function main() {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();

  const logs = [];
  page.on('console', (m) => {
    if (m.type() === 'error') {
      logs.push(m.text());
    }
  });
  page.on('pageerror', (err) => logs.push(String(err)));

  await page.goto(baseUrl, { waitUntil: 'networkidle', timeout: 30_000 });

  await page.waitForFunction(
    () => window.__controller?.session?.actions?.length >= 1,
    { timeout: 25_000 },
  );

  await page.waitForFunction(
    () => window.__controller?.session?.actions?.length >= 2,
    { timeout: 35_000 },
  );

  const state = await page.evaluate(() => ({
    actions: window.__controller.session.actions.length,
    players: [...window.__controller.settings.players],
    ptm: window.__controller.session.playerToMove,
    errors: { ...window.__controller.engineErrors },
  }));

  if (Object.values(state.errors).some(Boolean)) {
    throw new Error(`Engine errors: ${JSON.stringify(state.errors)}`);
  }

  console.log(`OK  ${state.actions} plies · players=${state.players.join(' vs ')} · ptm=${state.ptm}`);
  if (logs.length) {
    console.warn('Console errors:', logs.join(' | '));
  }

  await browser.close();
}

main().catch((err) => {
  console.error('FAIL', err.message ?? err);
  process.exit(1);
});
