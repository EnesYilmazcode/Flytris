// Procedural Drosophila, sized so body length is about 1 unit.
// Body frame: head toward -z, back up +y. The fly rears up by PITCH and sits
// on a stool, so fly-frame origin is the seat top, facing -z.
import * as THREE from 'three';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';

export const PITCH = THREE.MathUtils.degToRad(20);
export const THORAX = new THREE.Vector3(0, 0.3, -0.15);
export const FLY_SCALE = 1.35;
export const BODY_AXIS = new THREE.Vector3(0, 0, 1).applyAxisAngle(new THREE.Vector3(1, 0, 0), PITCH);

const bodyRot = new THREE.Matrix4().makeRotationX(PITCH);
const toFly = new THREE.Matrix4().makeTranslation(THORAX.x, THORAX.y, THORAX.z).multiply(bodyRot);
export const bodyPoint = (x, y, z) => new THREE.Vector3(x, y, z).applyMatrix4(toFly);

const C = {
  thorax: new THREE.Color('#8f5f2c'),
  thoraxDark: new THREE.Color('#5a3a1a'),
  head: new THREE.Color('#a4713a'),
  eye: new THREE.Color('#6e0608'),
  tan: new THREE.Color('#c29458'),
  band: new THREE.Color('#22150c'),
  leg: new THREE.Color('#3a2616'),
  bristle: new THREE.Color('#0d0906'),
};

function paint(geo, fn) {
  const pos = geo.attributes.position, col = new Float32Array(pos.count * 3);
  const rough = new Float32Array(pos.count);
  for (let i = 0; i < pos.count; i++) {
    const [c, r] = fn(pos.getX(i), pos.getY(i), pos.getZ(i));
    col.set([c.r, c.g, c.b], i * 3);
    rough[i] = r;
  }
  geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
  geo.setAttribute('aRough', new THREE.BufferAttribute(rough, 1));
  return geo;
}

function ellipsoid(r, at, seg, colorFn, rough = 0.55, taper = 0) {
  const g = new THREE.SphereGeometry(1, seg * 2, seg);
  paint(g, (x, y, z) => [colorFn(x, y, z), rough]);
  const pos = g.attributes.position;
  for (let i = 0; i < pos.count; i++) {
    const z = pos.getZ(i), k = 1 - taper * Math.max(z, 0) ** 1.6;
    pos.setXYZ(i, pos.getX(i) * r[0] * k, pos.getY(i) * r[1] * k, z * r[2]);
  }
  g.computeVertexNormals();
  g.translate(at[0], at[1], at[2]);
  return g;
}

// A tapered cylinder from p to q, in whatever frame p and q are in.
function segment(p, q, r0, r1, radial, color, rough = 0.6) {
  const len = p.distanceTo(q);
  const g = new THREE.CylinderGeometry(r1, r0, len, radial, 1, false);
  g.translate(0, len / 2, 0);
  const quat = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), q.clone().sub(p).normalize());
  g.applyQuaternion(quat);
  g.translate(p.x, p.y, p.z);
  return paint(g, () => [color, rough]);
}

function joint(p, r, seg, color) {
  const g = new THREE.SphereGeometry(r, seg, Math.max(3, seg >> 1));
  g.translate(p.x, p.y, p.z);
  return paint(g, () => [color, 0.6]);
}

// Two-bone IK: knee at distance l1 from a and l2 from t, bent toward pole.
function knee(a, t, l1, l2, pole) {
  const d = Math.min(a.distanceTo(t), l1 + l2 - 1e-3);
  const dir = t.clone().sub(a).normalize();
  const along = (l1 * l1 - l2 * l2 + d * d) / (2 * d);
  const h = Math.sqrt(Math.max(l1 * l1 - along * along, 0));
  const side = pole.clone().sub(dir.clone().multiplyScalar(pole.dot(dir))).normalize();
  return a.clone().add(dir.multiplyScalar(along)).add(side.multiplyScalar(h));
}

// Legs: attach points in body frame, feet in fly frame (feet come from the cabinet layout).
const LEGS = [
  { attach: [0.06, -0.10, -0.15], len: [0.24, 0.25, 0.17], pole: [1.0, -0.2, 0.3], name: 'front' },
  { attach: [0.085, -0.12, -0.01], len: [0.27, 0.29, 0.20], pole: [1.0, 0.9, 0.0], name: 'mid' },
  { attach: [0.08, -0.11, 0.11], len: [0.29, 0.31, 0.21], pole: [0.8, 1.0, 0.5], name: 'hind' },
];

export function legGeometry(feet, radial) {
  const parts = [];
  for (const side of [-1, 1]) {
    LEGS.forEach((leg, i) => {
      const a = bodyPoint(leg.attach[0] * side, leg.attach[1], leg.attach[2]);
      const foot = feet[`${leg.name}${side < 0 ? 'L' : 'R'}`];
      const back = a.clone().sub(foot).setY(0).normalize();
      const tarsusStart = foot.clone().add(back.multiplyScalar(leg.len[2] * 0.75)).add(new THREE.Vector3(0, leg.len[2] * 0.6, 0));
      const pole = new THREE.Vector3(leg.pole[0] * side, leg.pole[1], leg.pole[2]);
      const k = knee(a, tarsusStart, leg.len[0], leg.len[1], pole);
      const r = i === 0 ? 0.9 : 1;
      parts.push(segment(a, k, 0.018 * r, 0.014 * r, radial, C.leg));
      parts.push(joint(k, 0.014 * r, radial, C.leg));
      parts.push(segment(k, tarsusStart, 0.012 * r, 0.009 * r, radial, C.leg));
      parts.push(joint(tarsusStart, 0.009 * r, radial, C.leg));
      parts.push(segment(tarsusStart, foot, 0.008 * r, 0.005 * r, radial, C.leg));
    });
  }
  return parts;
}

function abdomenColor(x, y, z) {
  // Dorsal tergite bands; the last two segments are dark, as in males.
  const t = (z + 1) / 2;
  const seg = t * 5.2, s = seg - Math.floor(seg);
  const dorsal = THREE.MathUtils.smoothstep(y, -0.35, 0.1);
  let dark = s > 0.6 ? 1 : 0;
  if (seg > 3.3) dark = 1;
  return C.tan.clone().lerp(C.band, dark * dorsal * 0.92);
}

// Body parts in body frame, returned as geometries in fly frame, grouped by material.
export function flyParts({ seg = 12, radial = 6, hero = false, feet }) {
  const body = [], eyes = [], dark = [];
  const B = g => g.applyMatrix4(toFly);
  body.push(B(ellipsoid([0.16, 0.15, 0.22], [0, 0, 0], seg, (x, y) => C.thoraxDark.clone().lerp(C.thorax, 0.5 + 0.5 * Math.sin(x * 9) * (y > 0 ? 0.4 : 0) + 0.3 * (1 - y)))));
  body.push(B(ellipsoid([0.085, 0.055, 0.07], [0, 0.105, 0.19], seg >> 1, () => C.thorax)));
  body.push(B(ellipsoid([0.155, 0.135, 0.28], [0, -0.035, 0.44], seg, abdomenColor, 0.5, 0.45)));
  body.push(B(ellipsoid([0.165, 0.14, 0.105], [0, 0.03, -0.30], seg, () => C.head)));
  for (const s of [-1, 1]) {
    const eye = ellipsoid([0.075, 0.125, 0.10], [0.1 * s, 0.035, -0.315], seg, () => C.eye, 0.28);
    eyes.push(B(eye));
    // Antenna: scape, round third segment, feathery arista.
    const a0 = new THREE.Vector3(0.035 * s, 0.07, -0.395), a1 = new THREE.Vector3(0.045 * s, 0.035, -0.43);
    body.push(B(segment(a0, a1, 0.016, 0.013, radial, C.head)));
    body.push(B(ellipsoid([0.022, 0.032, 0.022], [0.048 * s, 0.015, -0.438], seg >> 1, () => C.tan)));
    dark.push(B(segment(new THREE.Vector3(0.055 * s, 0.03, -0.445), new THREE.Vector3(0.085 * s, 0.11, -0.50), 0.004, 0.002, 4, C.bristle)));
    // Haltere.
    body.push(B(segment(new THREE.Vector3(0.1 * s, 0.03, 0.12), new THREE.Vector3(0.14 * s, 0.0, 0.17), 0.008, 0.006, 4, C.tan)));
    body.push(B(ellipsoid([0.018, 0.018, 0.022], [0.143 * s, -0.004, 0.175], 5, () => C.tan)));
  }
  body.push(B(segment(new THREE.Vector3(0, -0.07, -0.34), new THREE.Vector3(0, -0.15, -0.37), 0.028, 0.02, radial, C.tan)));
  dark.push(...legGeometry(feet, radial));
  if (hero) dark.push(...bristles());
  return { body, eyes, dark };
}

function bristles() {
  // Macrochaetae: long dorsal bristles on thorax, scutellum and head, pointing back.
  const out = [], rng = mulberry(7);
  const spots = [];
  for (const s of [-1, 1]) {
    for (const [x, z] of [[0.05, -0.13], [0.09, -0.06], [0.12, 0.03], [0.1, 0.1], [0.05, 0.02], [0.13, -0.12]]) spots.push([x * s, z, 0.1]);
    spots.push([0.04 * s, 0.22, 0.13], [0.07 * s, 0.2, 0.12]);
    for (const [x, z] of [[0.04, -0.33], [0.09, -0.3], [0.12, -0.26]]) spots.push([x * s, z, 0.07]);
  }
  for (const [x, z, len] of spots) {
    const onHead = z < -0.2, onScut = z > 0.17;
    const cz = onHead ? -0.30 : onScut ? 0.19 : 0;
    const r = onHead ? [0.165, 0.14, 0.105] : onScut ? [0.085, 0.055, 0.07] : [0.17, 0.155, 0.235];
    const cy = onHead ? 0.03 : onScut ? 0.105 : 0;
    const nx = x / r[0], nz = (z - cz) / r[2];
    const y = cy + r[1] * Math.sqrt(Math.max(0.05, 1 - nx * nx - nz * nz));
    const p = new THREE.Vector3(x, y - 0.004, z);
    const dir = new THREE.Vector3(x * 0.8 + (rng() - 0.5) * 0.2, 0.55, 1).normalize();
    out.push(segment(p, p.clone().add(dir.multiplyScalar(len * (0.8 + rng() * 0.4))), 0.0055, 0.0008, 4, C.bristle).applyMatrix4(toFly));
  }
  return out;
}

// Wing outline in wing space: u along the wing (0 root, 1 tip), v across (+ leading edge).
const WING_L = 0.8, WING_W = 0.155;
function wingShape() {
  const pts = [];
  for (let i = 0; i <= 48; i++) {
    const th = (i / 48) * Math.PI * 2;
    let u = 0.55 + 0.45 * Math.cos(th), v = Math.sin(th);
    const pinch = THREE.MathUtils.smoothstep(u, 0.0, 0.35);
    v *= (0.25 + 0.75 * pinch) * (v > 0 ? 0.95 : 1.05);
    pts.push(new THREE.Vector2(u * WING_L, v * WING_W));
  }
  return new THREE.Shape(pts);
}

// Rest pose: folded back over the abdomen in a shallow V. Flight pose: spread sideways.
function wingFrame(side, spread) {
  const hinge = bodyPoint(0.07 * side, 0.14, 0.02);
  const yaw = THREE.MathUtils.degToRad(spread ? 78 : 20) * side;
  const lift = THREE.MathUtils.degToRad(spread ? 10 : 4);
  const m = new THREE.Matrix4().makeTranslation(hinge.x, hinge.y, hinge.z)
    .multiply(bodyRot)
    .multiply(new THREE.Matrix4().makeRotationY(yaw))
    .multiply(new THREE.Matrix4().makeRotationZ(-lift * side))
    // wing space (u, v) -> body frame: u along +z (backward), v outward along x.
    .multiply(new THREE.Matrix4().set(0, side, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1));
  return { m, hinge };
}

export function wingGeometry(side) {
  const g = new THREE.ShapeGeometry(wingShape(), 1);
  const pos = g.attributes.position, uv = g.attributes.uv;
  for (let i = 0; i < pos.count; i++) uv.setXY(i, pos.getX(i) / WING_L, (pos.getY(i) / WING_W + 1) / 2);
  const flight = g.clone().applyMatrix4(wingFrame(side, true).m);
  const { m, hinge } = wingFrame(side, false);
  g.applyMatrix4(m);
  g.setAttribute('aFlight', flight.attributes.position.clone());
  const n = pos.count;
  g.setAttribute('aSide', new THREE.BufferAttribute(new Float32Array(n).fill(side), 1));
  g.setAttribute('aHinge', new THREE.BufferAttribute(new Float32Array(Array.from({ length: n }, () => [hinge.x, hinge.y, hinge.z]).flat()), 3));
  return g;
}

export function wingTexture() {
  const W = 1024, Hh = 400, cv = document.createElement('canvas');
  cv.width = W; cv.height = Hh;
  const g = cv.getContext('2d');
  const grad = g.createLinearGradient(0, 0, W, 0);
  grad.addColorStop(0, 'rgba(110,100,90,0.75)');
  grad.addColorStop(0.3, 'rgba(175,170,165,0.45)');
  grad.addColorStop(1, 'rgba(200,200,205,0.38)');
  g.fillStyle = grad;
  g.fillRect(0, 0, W, Hh);
  const P = (u, v) => [u * W, (1 - v) * Hh];
  g.strokeStyle = 'rgba(60,42,26,0.85)';
  g.lineCap = 'round';
  const vein = (w, pts) => {
    g.lineWidth = w;
    g.beginPath();
    g.moveTo(...P(...pts[0]));
    g.quadraticCurveTo(...P(...pts[1]), ...P(...pts[2]));
    g.stroke();
  };
  vein(6, [[0.0, 0.62], [0.45, 0.99], [0.98, 0.72]]); // costa
  vein(3, [[0.03, 0.6], [0.4, 0.86], [0.78, 0.9]]); // L2
  vein(3, [[0.03, 0.55], [0.5, 0.64], [0.99, 0.56]]); // L3
  vein(3, [[0.03, 0.47], [0.5, 0.38], [0.97, 0.3]]); // L4
  vein(3, [[0.03, 0.42], [0.45, 0.2], [0.8, 0.1]]); // L5
  vein(2, [[0.36, 0.6], [0.37, 0.52], [0.38, 0.43]]); // anterior crossvein
  vein(2, [[0.62, 0.36], [0.63, 0.28], [0.64, 0.2]]); // posterior crossvein
  const tex = new THREE.CanvasTexture(cv);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 8;
  return tex;
}

export function eyeFacetTexture() {
  // Hexagonal ommatidia as a bump map.
  const W = 1024, Hh = 512, cv = document.createElement('canvas');
  cv.width = W; cv.height = Hh;
  const g = cv.getContext('2d');
  g.fillStyle = '#000';
  g.fillRect(0, 0, W, Hh);
  const r = 4.2, dx = r * Math.sqrt(3), dy = r * 1.5;
  const grad = (x, y) => {
    const gr = g.createRadialGradient(x, y, 0, x, y, r);
    gr.addColorStop(0, '#fff');
    gr.addColorStop(0.75, '#bbb');
    gr.addColorStop(1, '#000');
    return gr;
  };
  for (let row = 0; row * dy < Hh + r; row++) {
    for (let col = 0; col * dx < W + r; col++) {
      const x = col * dx + (row % 2 ? dx / 2 : 0), y = row * dy;
      g.fillStyle = grad(x, y);
      g.beginPath();
      for (let k = 0; k < 6; k++) {
        const a = Math.PI / 6 + (k * Math.PI) / 3;
        g.lineTo(x + Math.cos(a) * r * 0.92, y + Math.sin(a) * r * 0.92);
      }
      g.fill();
    }
  }
  const tex = new THREE.CanvasTexture(cv);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  return tex;
}

export function fuzzTexture() {
  // Fine noise for a bump map, so bodies read as hairy rather than plastic.
  const S = 256, cv = document.createElement('canvas');
  cv.width = cv.height = S;
  const g = cv.getContext('2d'), img = g.createImageData(S, S), rng = mulberry(3);
  for (let i = 0; i < S * S; i++) {
    const v = 90 + rng() * 120;
    img.data.set([v, v, v, 255], i * 4);
  }
  g.putImageData(img, 0, 0);
  const tex = new THREE.CanvasTexture(cv);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(6, 3);
  return tex;
}

export function mergeParts(list, withUv = false) {
  return mergeGeometries(list.map(g => {
    // Drop attributes the merge cannot reconcile across primitive types.
    const keep = new THREE.BufferGeometry();
    for (const k of ['position', 'normal', 'color', 'aRough', ...(withUv ? ['uv'] : [])]) keep.setAttribute(k, g.attributes[k]);
    keep.setIndex(g.index);
    return keep.index ? keep.toNonIndexed() : keep;
  }));
}

export function mulberry(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
