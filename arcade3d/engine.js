// Tetris replay engine, a direct port of flytris/tetris.py lock/clear.
// Every game replays its recorded placements; export_games.py already checked
// them against the Python engine, and check() re-checks lines here.

export const ROWS = 20, COLS = 10, HIDDEN = 2, H = ROWS + HIDDEN;

const BASE = [
  [[0, 0], [0, 1], [0, 2], [0, 3]], // I
  [[0, 0], [0, 1], [1, 0], [1, 1]], // O
  [[0, 1], [1, 0], [1, 1], [1, 2]], // T
  [[0, 1], [0, 2], [1, 0], [1, 1]], // S
  [[0, 0], [0, 1], [1, 1], [1, 2]], // Z
  [[0, 0], [1, 0], [1, 1], [1, 2]], // J
  [[0, 2], [1, 0], [1, 1], [1, 2]], // L
];

function rotations(cells) {
  const out = [];
  let cur = cells.map(c => c.slice());
  for (let i = 0; i < 4; i++) {
    const r0 = Math.min(...cur.map(c => c[0])), c0 = Math.min(...cur.map(c => c[1]));
    const norm = cur.map(c => [c[0] - r0, c[1] - c0]).sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    const key = JSON.stringify(norm);
    if (!out.some(o => JSON.stringify(o) === key)) out.push(norm);
    cur = cur.map(c => [c[1], -c[0]]);
  }
  return out;
}

// PLACEMENTS[piece][move] = {rot, x, cells}
export const PLACEMENTS = BASE.map(base => {
  const opts = [];
  rotations(base).forEach((cells, rot) => {
    const width = Math.max(...cells.map(c => c[1])) + 1;
    for (let x = 0; x <= COLS - width; x++) opts.push({ rot, x, cells });
  });
  return opts;
});

export function landing(board, piece, move) {
  const { x, cells } = PLACEMENTS[piece][move];
  let y = Infinity;
  for (const [r, c] of cells) {
    let top = H;
    for (let row = 0; row < H; row++) if (board[row * COLS + c + x]) { top = row; break; }
    y = Math.min(y, top - 1 - r);
  }
  return y;
}

export class Game {
  constructor(pieces, moves) {
    this.pieces = pieces;
    this.moves = moves;
    this.n = pieces.length;
    this.board = new Uint8Array(H * COLS);
    this.k = 0;
    this.lines = 0;
    this.lineSteps = []; // lines after each step, for the HUD counter
  }

  step() {
    const piece = this.pieces[this.k], move = this.moves[this.k];
    const y = landing(this.board, piece, move);
    if (y >= 0) {
      const { x, cells } = PLACEMENTS[piece][move];
      for (const [r, c] of cells) this.board[(y + r) * COLS + c + x] = piece + 1;
      this.lines += clearRows(this.board);
    }
    this.k++;
    this.lineSteps.push(this.lines);
  }

  advanceTo(k) {
    while (this.k < Math.min(k, this.n)) this.step();
  }
}

export function fullRows(board) {
  const full = [];
  for (let row = 0; row < H; row++) {
    let ok = true;
    for (let c = 0; c < COLS; c++) if (!board[row * COLS + c]) { ok = false; break; }
    if (ok) full.push(row);
  }
  return full;
}

function clearRows(board) {
  const full = fullRows(board);
  if (!full.length) return 0;
  const keep = [];
  for (let row = 0; row < H; row++) if (!full.includes(row)) keep.push(board.slice(row * COLS, row * COLS + COLS));
  board.fill(0);
  keep.forEach((r, i) => board.set(r, (H - keep.length + i) * COLS));
  return full.length;
}

// ---------- how a player makes one placement ----------
// Rotate with the buttons, slide with the joystick, then hard drop. Times are fractions
// of the step; the piece locks at 0.8 in main.js.
export const ROT_CELLS = BASE.map(rotations);
export const DROP_F = 0.5;
const widthOf = cells => Math.max(...cells.map(c => c[1])) + 1;
const spawnX = cells => Math.min(3, COLS - widthOf(cells));

export function stepPlan(piece, move) {
  const { rot, x } = PLACEMENTS[piece][move];
  // Button 0 rotates one way, button 1 the other way (three turns = one back), button 2 drops.
  const presses = rot === 3 ? [{ f: 0.05, button: 1, to: 3 }]
    : Array.from({ length: rot }, (_, i) => ({ f: 0.05 + i * 0.08, button: 0, to: i + 1 }));
  const slideStart = presses.length ? presses[presses.length - 1].f + 0.04 : 0.06;
  const x0 = spawnX(ROT_CELLS[piece][rot]);
  const slideEnd = Math.min(DROP_F - 0.05, slideStart + 0.05 * Math.abs(x - x0));
  presses.push({ f: DROP_F, button: 2, to: rot });
  return { rot, x, x0, presses, slideStart, slideEnd };
}

// Where the falling piece is drawn at fraction f of its step (board rows, hidden rows included).
export function fallingAt(plan, piece, f, landY) {
  let q = 0;
  for (const p of plan.presses) if (p.f <= f && p.button < 2) q = p.to;
  const cells = ROT_CELLS[piece][q];
  let x;
  if (q !== plan.rot) x = spawnX(cells);
  else if (f <= plan.slideStart) x = plan.x0;
  else if (f >= plan.slideEnd) x = plan.x;
  else x = plan.x0 + Math.round((plan.x - plan.x0) * (f - plan.slideStart) / (plan.slideEnd - plan.slideStart));
  const y = f >= DROP_F ? landY : Math.min(landY, HIDDEN + Math.floor(2 * f / DROP_F));
  return { cells, x, y };
}
