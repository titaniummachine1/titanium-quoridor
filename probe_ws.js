#!/usr/bin/env node
/**
 * Live probe of quoridor-ai.com WebSocket engines.
 * Usage: node probe_ws.js [ishtar|ka]
 */
const { QuoridorEngineClient, ENGINES } = require('./extracted/engine_client');

const engineKey = process.argv[2] || 'ishtar';
const config = ENGINES[engineKey];

if (!config) {
  console.error(`Unknown engine: ${engineKey}. Use ishtar or ka.`);
  process.exit(1);
}

const engine = new QuoridorEngineClient(config);

engine.onRawMessage = (message) => console.log('<<', message);
engine.onStatus = (status) => console.log('status:', status);
engine.onInfo = (info) => console.log('info:', info);
engine.onBestMove = (action, raw) => {
  console.log('bestmove:', action, `(${raw})`);
  engine.destroy();
  process.exit(0);
};
engine.onError = (error) => console.error('error:', error.message);

console.log(`Connecting to ${config.name} at ${config.uri}...`);
engine.connect();
// Fresh games: server defaults to start — no setposition needed (matches website behavior).
engine.go('intuition');

setTimeout(() => {
  console.log('timeout');
  engine.destroy();
  process.exit(1);
}, 15000);
