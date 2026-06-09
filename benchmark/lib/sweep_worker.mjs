#!/usr/bin/env node
import { playOneGame } from './match_engine.mjs';
import { RUST_TITANIUM_ID, GORISANSON_ID } from './engine_ids.mjs';

function parseArgs(argv) {
  const opts = { variant: '?', game: 1, timeSec: 10 };
  for (let i = 2; i < argv.length; i += 1) {
    if (argv[i] === '--variant' && argv[i + 1]) opts.variant = argv[++i];
    else if (argv[i] === '--game' && argv[i + 1]) opts.game = Number(argv[++i]);
    else if (argv[i] === '--time' && argv[i + 1]) opts.timeSec = Number(argv[++i]);
  }
  return opts;
}

const opts = parseArgs(process.argv);
const engine = process.env.TITANIUM_ENGINE ?? 'minimax';
const disableBook = process.env.TITANIUM_DISABLE_BOOK === '1';
const disableBridge = process.env.TITANIUM_BRIDGE === '0';

const titanium = {
  id: RUST_TITANIUM_ID,
  engine,
  disableBook,
  disableBridge,
};
const gorisanson = { id: GORISANSON_ID };

const budget = {
  timeSec: opts.timeSec,
  timeMs: opts.timeSec * 1000,
  maxSimulations: 2_000_000_000,
};

const outcome = await playOneGame(titanium, gorisanson, {
  ...budget,
  quiet: true,
  logMoves: false,
  logSearch: false,
  disableBook,
  disableBridge,
  engine,
});

let result = 'aborted';
if (outcome.result === 'draw') {
  result = 'aborted'; // hit MAX_PLIES (250) with no goal — not a real Quoridor draw
} else if (outcome.winner === RUST_TITANIUM_ID) {
  result = 'win';
} else if (outcome.winner === GORISANSON_ID) {
  result = 'loss';
}

console.log(
  JSON.stringify({
    variant: opts.variant,
    game: opts.game,
    result,
    winner: outcome.winner ?? null,
    plies: outcome.plies,
    engine,
    disableBook,
    disableBridge,
  }),
);
