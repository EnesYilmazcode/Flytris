// Upright arcade cabinet and bar stool. Cabinet origin: floor, center of footprint,
// screen facing +z. The stool sits in front at STOOL_Z.
import * as THREE from 'three';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { FLY_SCALE } from './fly.js';

export const CAB_W = 1.3;
export const STOOL_Z = 1.55, SEAT_Y = 0.72;

// Side profile as (z, y).
const PROFILE = [
  [-0.5, 0], [0.42, 0], [0.42, 1.02], [0.66, 1.1], [0.66, 1.18], [0.3, 1.3],
  [0.3, 1.34], [0.14, 2.1], [0.3, 2.14], [0.3, 2.5], [-0.5, 2.56],
];
const SCREEN_BOT = new THREE.Vector2(0.3, 1.34), SCREEN_TOP = new THREE.Vector2(0.14, 2.1);
export const SCREEN_TILT = Math.atan2(SCREEN_BOT.x - SCREEN_TOP.x, SCREEN_TOP.y - SCREEN_BOT.y);
export const SCREEN_W = 0.62, SCREEN_H = 0.7;
export const SCREEN_NORMAL = new THREE.Vector3(0, Math.sin(SCREEN_TILT), Math.cos(SCREEN_TILT));
export const SCREEN_CENTER = new THREE.Vector3(0, (SCREEN_BOT.y + SCREEN_TOP.y) / 2, (SCREEN_BOT.x + SCREEN_TOP.x) / 2)
  .addScaledVector(SCREEN_NORMAL, 0.04);

const PANEL_ANGLE = Math.atan2(0.12, 0.36);
const panelPoint = (x, t) => new THREE.Vector3(x, 1.18 + 0.12 * t, 0.66 - 0.36 * t);
export const JOYSTICK = panelPoint(-0.22, 0.4);
export const BUTTONS = [0.06, 0.18, 0.3].map(x => panelPoint(x, 0.45));
const BUTTON_COLORS = ['#ff2a3a', '#ffd21e', '#2a7bff'];

function tag(geo, color, emitTrim = 0, emitSelf = 0, rough = 0.5) {
  geo = geo.index ? geo.toNonIndexed() : geo;
  const n = geo.attributes.position.count, c = new THREE.Color(color);
  const col = new Float32Array(n * 3), emit = new Float32Array(n * 2), r = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    col.set([c.r, c.g, c.b], i * 3);
    emit.set([emitTrim, emitSelf], i * 2);
    r[i] = rough;
  }
  const out = new THREE.BufferGeometry();
  out.setAttribute('position', geo.attributes.position);
  out.setAttribute('normal', geo.attributes.normal);
  out.setAttribute('color', new THREE.BufferAttribute(col, 3));
  out.setAttribute('aEmit', new THREE.BufferAttribute(emit, 2));
  out.setAttribute('aRough', new THREE.BufferAttribute(r, 1));
  return out;
}

export function cabinetGeometry(detail = 1) {
  const parts = [];
  const shape = new THREE.Shape(PROFILE.map(([z, y]) => new THREE.Vector2(z, y)));
  const body = new THREE.ExtrudeGeometry(shape, {
    depth: CAB_W - 0.04, bevelEnabled: true, bevelThickness: 0.02, bevelSize: 0.02, bevelSegments: 2,
  });
  body.rotateY(-Math.PI / 2);
  body.translate(CAB_W / 2 - 0.02, 0, 0);
  body.computeVertexNormals();
  parts.push(tag(body, '#15101f', 0, 0, 0.45));

  // Neon T-molding along the front edge of both side panels.
  const edge = PROFILE.slice(1, 10);
  for (const x of [-CAB_W / 2, CAB_W / 2]) {
    const path = new THREE.CurvePath();
    for (let i = 0; i < edge.length - 1; i++) {
      path.add(new THREE.LineCurve3(new THREE.Vector3(x, edge[i][1], edge[i][0]), new THREE.Vector3(x, edge[i + 1][1], edge[i + 1][0])));
    }
    parts.push(tag(new THREE.TubeGeometry(path, 40 * detail, 0.024, 5, false), '#ffffff', 1, 0, 0.3));
  }

  // Screen bezel: a black glossy frame slightly proud of the body.
  const bezel = new THREE.BoxGeometry(CAB_W - 0.12, Math.hypot(0.16, 0.76) - 0.03, 0.02);
  bezel.rotateX(-SCREEN_TILT);
  bezel.translate(0, (SCREEN_BOT.y + SCREEN_TOP.y) / 2, (SCREEN_BOT.x + SCREEN_TOP.x) / 2);
  bezel.translate(0, SCREEN_NORMAL.y * 0.022, SCREEN_NORMAL.z * 0.022);
  parts.push(tag(bezel, '#050507', 0, 0, 0.15));

  // Control panel: three lit buttons (the joystick is its own mesh so it can tilt).
  BUTTONS.forEach((p, i) => {
    const b = new THREE.CylinderGeometry(0.032, 0.034, 0.025, 16);
    b.rotateX(PANEL_ANGLE);
    b.translate(p.x, p.y + 0.03 * Math.cos(PANEL_ANGLE), p.z + 0.03 * Math.sin(PANEL_ANGLE));
    parts.push(tag(b, BUTTON_COLORS[i], 0, 0.9, 0.25));
  });

  // Coin door with two glowing slots.
  const door = new THREE.BoxGeometry(0.4, 0.36, 0.02);
  door.translate(0, 0.66, 0.45);
  parts.push(tag(door, '#2a2a33', 0, 0, 0.3));
  for (const x of [-0.08, 0.08]) {
    const slot = new THREE.BoxGeometry(0.045, 0.08, 0.012);
    slot.translate(x, 0.72, 0.462);
    parts.push(tag(slot, '#ff3010', 0, 3.0, 0.3));
  }
  return mergeGeometries(parts);
}

// Joystick around its base on the panel. Tilt it about JOY_AXIS (the panel's forward
// direction): a positive angle leans the ball to -x.
export const JOY_AXIS = new THREE.Vector3(0, -Math.sin(PANEL_ANGLE), Math.cos(PANEL_ANGLE));
export function joystickGeometry() {
  const stick = new THREE.CylinderGeometry(0.012, 0.014, 0.12, 8);
  stick.translate(0, 0.06, 0);
  stick.rotateX(PANEL_ANGLE);
  const ball = new THREE.SphereGeometry(0.045, 14, 10);
  ball.translate(0, 0.13 * Math.cos(PANEL_ANGLE), 0.13 * Math.sin(PANEL_ANGLE));
  return mergeGeometries([tag(stick, '#111111', 0, 0, 0.3), tag(ball, '#e01020', 0, 0.12, 0.2)]);
}

export function stoolGeometry() {
  const parts = [];
  const seat = new THREE.CylinderGeometry(0.42, 0.4, 0.08, 28);
  seat.translate(0, SEAT_Y - 0.04, 0);
  parts.push(tag(seat, '#1b2033', 0, 0, 0.35));
  const pole = new THREE.CylinderGeometry(0.035, 0.035, SEAT_Y - 0.1, 10);
  pole.translate(0, (SEAT_Y - 0.1) / 2 + 0.04, 0);
  parts.push(tag(pole, '#8d939c', 0, 0, 0.2));
  const base = new THREE.CylinderGeometry(0.2, 0.24, 0.04, 20);
  base.translate(0, 0.02, 0);
  parts.push(tag(base, '#5c6068', 0, 0, 0.25));
  const ring = new THREE.TorusGeometry(0.17, 0.012, 6, 24);
  ring.rotateX(Math.PI / 2);
  ring.translate(0, 0.34, 0);
  parts.push(tag(ring, '#8d939c', 0, 0, 0.2));
  return mergeGeometries(parts);
}

// Feet targets in fly frame (origin on the seat, unit = FLY_SCALE world units).
export function flyFeet() {
  const origin = new THREE.Vector3(0, SEAT_Y, STOOL_Z);
  const local = w => w.clone().sub(origin).divideScalar(FLY_SCALE);
  const ballTop = new THREE.Vector3(JOYSTICK.x, JOYSTICK.y + 0.15, JOYSTICK.z + 0.03);
  return {
    frontL: local(ballTop),
    frontR: local(BUTTONS[0].clone().add(new THREE.Vector3(0, 0.03, 0.0))),
    midL: new THREE.Vector3(-0.19, 0, -0.08), midR: new THREE.Vector3(0.19, 0, -0.08),
    hindL: new THREE.Vector3(-0.17, 0, 0.13), hindR: new THREE.Vector3(0.17, 0, 0.13),
  };
}

export function marqueeTexture() {
  const W = 1024, H = 280, cv = document.createElement('canvas');
  cv.width = W; cv.height = H;
  const g = cv.getContext('2d');
  const bg = g.createLinearGradient(0, 0, 0, H);
  bg.addColorStop(0, '#12051f');
  bg.addColorStop(1, '#050210');
  g.fillStyle = bg;
  g.fillRect(0, 0, W, H);
  // Title in Tetris blocks: each letter on a 3x5 grid.
  const FONT = {
    F: ['111', '100', '110', '100', '100'], L: ['100', '100', '100', '100', '111'],
    Y: ['101', '101', '010', '010', '010'], T: ['111', '010', '010', '010', '010'],
    R: ['110', '101', '110', '101', '101'], I: ['111', '010', '010', '010', '111'],
    S: ['011', '100', '010', '001', '110'],
  };
  const colors = ['#00d8ff', '#ffd21e', '#b04dff', '#39ff5a', '#ff3b3b', '#2f6bff', '#ff8c1a'];
  const word = 'FLYTRIS', cell = 30, gap = 18;
  const totalW = word.length * 3 * cell + (word.length - 1) * gap;
  let x0 = (W - totalW) / 2;
  const y0 = (H - 5 * cell) / 2;
  [...word].forEach((ch, li) => {
    FONT[ch].forEach((row, r) => [...row].forEach((bit, c) => {
      if (bit !== '1') return;
      const x = x0 + c * cell, y = y0 + r * cell;
      g.fillStyle = colors[li];
      g.fillRect(x + 2, y + 2, cell - 4, cell - 4);
      g.fillStyle = 'rgba(255,255,255,0.35)';
      g.fillRect(x + 2, y + 2, cell - 4, 5);
    }));
    x0 += 3 * cell + gap;
  });
  g.strokeStyle = '#ff2bd6';
  g.lineWidth = 6;
  g.strokeRect(10, 10, W - 20, H - 20);
  const tex = new THREE.CanvasTexture(cv);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 8;
  return tex;
}

export function carpetTexture() {
  // Blacklight arcade carpet: dark navy with scattered neon shapes.
  const S = 512, cv = document.createElement('canvas');
  cv.width = cv.height = S;
  const g = cv.getContext('2d');
  g.fillStyle = '#07061a';
  g.fillRect(0, 0, S, S);
  let a = 12345;
  const rnd = () => ((a = (a * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
  const cols = ['#1ee0ff', '#ff2bd6', '#ffe01e', '#7a3cff', '#39ff7a'];
  for (let i = 0; i < 70; i++) {
    const x = rnd() * S, y = rnd() * S, c = cols[i % cols.length], s = 6 + rnd() * 16;
    g.strokeStyle = c;
    g.fillStyle = c;
    g.globalAlpha = 0.55;
    g.lineWidth = 3;
    const kind = i % 4;
    for (const [ox, oy] of [[0, 0], [S, 0], [-S, 0], [0, S], [0, -S]]) {
      g.beginPath();
      if (kind === 0) g.arc(x + ox, y + oy, s, 0, Math.PI * 2);
      else if (kind === 1) { g.moveTo(x + ox - s, y + oy); g.quadraticCurveTo(x + ox, y + oy - s * 1.5, x + ox + s, y + oy); }
      else if (kind === 2) { g.moveTo(x + ox, y + oy - s); g.lineTo(x + ox + s * 0.8, y + oy + s * 0.6); g.lineTo(x + ox - s * 0.8, y + oy + s * 0.6); g.closePath(); }
      else { g.arc(x + ox, y + oy, s * 0.3, 0, Math.PI * 2); g.fill(); continue; }
      g.stroke();
    }
  }
  const tex = new THREE.CanvasTexture(cv);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.anisotropy = 8;
  return tex;
}
