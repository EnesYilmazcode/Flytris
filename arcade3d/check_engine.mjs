import fs from 'fs';
import { Game } from './engine.js';
const meta = JSON.parse(fs.readFileSync('games.json'));
const bin = fs.readFileSync('games.bin');
let bad = 0;
for (const g of meta.games) {
  const pieces = bin.subarray(g.off, g.off + g.n), moves = bin.subarray(meta.total + g.off, meta.total + g.off + g.n);
  const game = new Game(pieces, moves); game.advanceTo(g.n);
  if (game.lines !== g.lines) bad++;
}
console.log(`${meta.games.length} games, ${bad} line mismatches`);
