#!/usr/bin/env node
/**
 * Convert ishtar_opening_mine_out.json candidates → Rust BookLine snippets.
 *
 *   node benchmark/ingest_mined_book.mjs
 *   node benchmark/ingest_mined_book.mjs --min-hits 2 --max-plies 10
 */

import { readFileSync, existsSync } from 'node:fs';

const IN_PATH = new URL('./ishtar_opening_mine_out.json', import.meta.url);

function parseArgs(argv) {
  const opts = { minHits: 2, maxPrefixLen: 10 };
  for (let i = 2; i < argv.length; i += 1) {
    if (argv[i] === '--min-hits' && argv[i + 1]) opts.minHits = Number(argv[++i]);
    else if (argv[i] === '--max-plies' && argv[i + 1]) opts.maxPrefixLen = Number(argv[++i]);
  }
  return opts;
}

function slug(prefix, reply) {
  const base = `${prefix.join('-') || 'start'}-${reply}`.replace(/[^a-z0-9]+/gi, '-');
  return `mined-${base}`.slice(0, 48);
}

function priorityFromHits(hits, prefixLen) {
  if (hits >= 8) return 140;
  if (hits >= 5) return 130;
  if (hits >= 3) return 120;
  if (prefixLen <= 2) return 115;
  return 110;
}

function main() {
  if (!existsSync(IN_PATH)) {
    console.error(`Missing ${IN_PATH.pathname} — run the miner first.`);
    process.exit(1);
  }

  const opts = parseArgs(process.argv);
  const data = JSON.parse(readFileSync(IN_PATH, 'utf8'));
  const candidates = data.candidates ?? [];

  const filtered = candidates.filter(
    (c) => c.hits >= opts.minHits && c.prefix.length <= opts.maxPrefixLen,
  );

  console.log(`// ${filtered.length} mined lines (min hits ${opts.minHits}, from ${data.completedGames ?? data.lines?.length ?? '?'} games)`);
  console.log('// Paste into engine/src/opening.rs BOOK_LINES:\n');

  for (const cand of filtered) {
    const prefixLit = cand.prefix.map((m) => `"${m}"`).join(', ');
    const pri = priorityFromHits(cand.hits, cand.prefix.length);
    console.log(`    BookLine {`);
    console.log(`        name: "${slug(cand.prefix, cand.reply)}",`);
    console.log(`        prefix: &[${prefixLit}],`);
    console.log(`        reply: "${cand.reply}",`);
    console.log(`        priority: ${pri},`);
    console.log(`        stm_bias: 0,`);
    console.log(`    },`);
  }
}

main();
