// Flytris arcade: every recorded fly game on its own cabinet, one fly per stool.
// Losers' screens flash red and the fly takes off; the last fly playing is the winner.
import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { RectAreaLightUniformsLib } from 'three/addons/lights/RectAreaLightUniformsLib.js';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { Game, PLACEMENTS, landing, fullRows, stepPlan, fallingAt, COLS, HIDDEN } from './engine.js';
import * as FLY from './fly.js';
import * as CAB from './cabinet.js';

const Q = new URLSearchParams(location.search);
const W = +Q.get('w') || 1080, HT = +Q.get('h') || 1350;
const FPS = 30;
const LIMIT = +Q.get('n') || 0;
const MODE = Q.get('mode') || 'preview';

const DX = 1.75, AISLE = 2.6, AISLE_EVERY = 10, DZ = 4.0;
const T = { closeEnd: 3.4, heroDie: 5.0, wide: 8.0, massEnd: 10.5, runner: 12.5, arrive: 14.2, crown: 14.8, end: 18.5 };

// ---------- data ----------
const [meta, bin, M, MB] = await Promise.all([
  fetch('games.json').then(r => r.json()),
  fetch('games.bin').then(r => r.arrayBuffer()).then(b => new Uint8Array(b)),
  fetch('fly_model.json').then(r => r.json()),
  fetch('fly_model.bin').then(r => r.arrayBuffer()),
]);
const arr = d => new (d.type === 'uint32' ? Uint32Array : Float32Array)(MB, d.offset, d.length);
const all = meta.games.map((g, i) => ({ ...g, id: i }));
const rng = FLY.mulberry(20260918);
const shuffle = a => { for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; };

const v3 = all.filter(g => g.src === 'v3');
const winner = v3.reduce((a, b) => (b.n > a.n ? b : a));
const runner = v3.filter(g => g !== winner).reduce((a, b) => (b.n > a.n ? b : a));
// The opening fly clears a couple of lines, then tops out before most of the field does.
const opener = all.filter(g => g.src !== 'v3' && g.n >= 30 && g.n <= 35 && g.lines >= 2).sort((a, b) => b.lines - a.lines || a.n - b.n || a.id - b.id)[0];
let chosen = [...v3, opener];
const rest = shuffle(all.filter(g => g.src !== 'v3' && g !== opener));
chosen = chosen.concat(rest.slice(0, (LIMIT || all.length) - chosen.length));
const N = chosen.length;

// ---------- layout ----------
const GCOLS = Math.max(8, Math.round(Math.sqrt(N / 1.55)));
const GROWS = Math.ceil(N / GCOLS);
const colX = c => c * DX + Math.floor(c / AISLE_EVERY) * AISLE;
const xMid = (colX(0) + colX(GCOLS - 1)) / 2;
const cellPos = (c, r) => new THREE.Vector3(colX(c) - xMid, 0, -r * DZ);
const cells = [];
for (let r = 0; r < GROWS; r++) for (let c = 0; c < GCOLS; c++) cells.push({ c, r });
const used = cells.slice(0, N);
const take = (c, r) => used.splice(used.findIndex(x => x.c === c && x.r === r), 1)[0];
// Hero and winner sit at the end of a block so the camera has an aisle to swing through.
const blockEnd = x => Math.min(GCOLS - 1, Math.max(AISLE_EVERY - 1, Math.round((x + 1) / AISLE_EVERY) * AISLE_EVERY - 1));
const heroCell = take(blockEnd(GCOLS / 2), 0);
const winCell = take(blockEnd(GCOLS * 0.3), Math.floor(GROWS * 0.3));
// Long survivors go where the wide shot can see them.
const visible = shuffle(used.filter(x => x.r < GROWS * 0.6 && Math.abs(x.c - GCOLS / 2) < GCOLS * 0.38 && x.r > 1));
const survivors = chosen.filter(g => g.n > 60 && g !== winner && g !== opener);
const slotOf = new Map();
slotOf.set(opener, heroCell);
slotOf.set(winner, winCell);
survivors.forEach((g, i) => { const cell = visible[i]; used.splice(used.indexOf(cell), 1); slotOf.set(g, cell); });
shuffle(used);
chosen.filter(g => !slotOf.has(g)).forEach((g, i) => slotOf.set(g, used[i]));

const slots = chosen.map((g, i) => {
  const cell = slotOf.get(g);
  return {
    g, i, cell, pos: cellPos(cell.c, cell.r), seed: rng(),
    game: new Game(bin.subarray(g.off, g.off + g.n), bin.subarray(meta.total + g.off, meta.total + g.off + g.n)),
  };
});
const heroSlot = slots.find(s => s.g === opener), winSlot = slots.find(s => s.g === winner);

// Winner's last line: the clock ends on it.
const wg = new Game(winSlot.game.pieces, winSlot.game.moves);
wg.advanceTo(wg.n);
const winLast = wg.lineSteps.indexOf(winner.lines) + 1;

// ---------- clock: video seconds -> pieces placed ----------
function monotone(xs, ys) {
  const n = xs.length, d = [], m = new Array(n);
  for (let i = 0; i < n - 1; i++) d.push((ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i]));
  m[0] = d[0]; m[n - 1] = d[n - 2];
  for (let i = 1; i < n - 1; i++) m[i] = d[i - 1] * d[i] <= 0 ? 0 : (d[i - 1] + d[i]) / 2;
  for (let i = 0; i < n - 1; i++) {
    if (!d[i]) { m[i] = m[i + 1] = 0; continue; }
    const a = m[i] / d[i], b = m[i + 1] / d[i], s = a * a + b * b;
    if (s > 9) { const t = 3 / Math.sqrt(s); m[i] = t * a * d[i]; m[i + 1] = t * b * d[i]; }
  }
  return x => {
    if (x <= xs[0]) return ys[0];
    if (x >= xs[n - 1]) return ys[n - 1];
    let i = 0;
    while (x > xs[i + 1]) i++;
    const h = xs[i + 1] - xs[i], t = (x - xs[i]) / h, t2 = t * t, t3 = t2 * t;
    return (2 * t3 - 3 * t2 + 1) * ys[i] + (t3 - 2 * t2 + t) * h * m[i] + (-2 * t3 + 3 * t2) * ys[i + 1] + (t3 - t2) * h * m[i + 1];
  };
}
const LOCK = 0.8; // fraction of a step spent falling; the piece locks after it
const clock = monotone(
  [0, T.closeEnd, T.heroDie, T.wide, 9.2, T.massEnd, T.runner, 13.6, 15.6, T.end],
  // The video opens 13 pieces before the opening fly's loss, with every board already mid-game.
  [opener.n - 13, opener.n - 13 + 13 * 0.66, opener.n - 1 + LOCK, 36.5, 44, 62, runner.n - 1 + LOCK, winLast - 6, winLast, winLast + 0.25],
);
function timeAtPiece(p) {
  if (clock(0) >= p) return -10; // already over before the video starts
  let lo = 0, hi = T.end;
  if (clock(hi) < p) return 1e9;
  for (let i = 0; i < 50; i++) { const mid = (lo + hi) / 2; if (clock(mid) < p) lo = mid; else hi = mid; }
  return hi;
}
slots.forEach(s => { s.death = s.g === winner ? 1e9 : timeAtPiece(s.g.n - 1 + LOCK); });

// ---------- renderer ----------
const renderer = new THREE.WebGLRenderer({ antialias: false, preserveDrawingBuffer: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(1);
renderer.setSize(W, HT, false);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;
document.body.appendChild(renderer.domElement);
RectAreaLightUniformsLib.init();

const scene = new THREE.Scene();
const FOG = new THREE.Color('#0a0716');
scene.background = FOG;
scene.fog = new THREE.FogExp2(FOG, 0.01);
const camera = new THREE.PerspectiveCamera(40, W / HT, 0.15, 1500);

const rt = new THREE.WebGLRenderTarget(W, HT, { type: THREE.HalfFloatType, samples: +(Q.get('msaa') ?? 4) });
const composer = new EffectComposer(renderer, rt);
composer.addPass(new RenderPass(scene, camera));
const bloom = new UnrealBloomPass(new THREE.Vector2(W, HT), 0.85, 0.55, 0.85);
if (Q.get('bloom') !== '0') composer.addPass(bloom);
composer.addPass(new OutputPass());

const U = { uTime: { value: 0 } };

// ---------- board atlas ----------
const AC = Math.ceil(Math.sqrt(N / 2)) || 1, AR = Math.ceil(N / AC);
const atlasData = new Uint8Array(AC * 10 * AR * 20 * 4);
const atlas = new THREE.DataTexture(atlasData, AC * 10, AR * 20, THREE.RGBAFormat);
atlas.magFilter = atlas.minFilter = THREE.NearestFilter;
const avgData = new Uint8Array(AC * AR * 4);
const avgTex = new THREE.DataTexture(avgData, AC, AR, THREE.RGBAFormat);
const statsData = new Uint8Array(AC * AR * 4);
const stats = new THREE.DataTexture(statsData, AC, AR, THREE.RGBAFormat);
const PAL = [[0, 220, 255], [255, 214, 30], [176, 77, 255], [57, 255, 90], [255, 59, 59], [47, 107, 255], [255, 140, 26]];

function putCell(slot, row, col, rgb, kind) {
  // row: visible row 0 (top) .. 19 (bottom)
  if (row < 0 || row >= 20 || col < 0 || col >= COLS) return;
  const bx = slot % AC, by = Math.floor(slot / AC);
  const tx = bx * 10 + col, ty = by * 20 + (19 - row);
  const o = (ty * AC * 10 + tx) * 4;
  atlasData[o] = rgb[0]; atlasData[o + 1] = rgb[1]; atlasData[o + 2] = rgb[2]; atlasData[o + 3] = kind;
}

function boardAverage(slot) {
  const bx = slot % AC, by = Math.floor(slot / AC);
  let r = 0, g = 0, b = 0;
  for (let ty = by * 20; ty < by * 20 + 20; ty++) {
    let o = (ty * AC * 10 + bx * 10) * 4;
    for (let c = 0; c < 10; c++, o += 4) { r += atlasData[o]; g += atlasData[o + 1]; b += atlasData[o + 2]; }
  }
  avgData.set([r / 200, g / 200, b / 200, 255], slot * 4);
}

let lastP = -1;
function updateBoards(p) {
  if (p < lastP) slots.forEach(s => { s.game = new Game(s.game.pieces, s.game.moves); s.done = false; });
  lastP = p;
  const k = Math.floor(p), f = p - k;
  for (const s of slots) {
    if (s.done) continue;
    const game = s.game;
    game.advanceTo(k);
    const board = game.board;
    for (let r = 0; r < 20; r++) for (let c = 0; c < COLS; c++) {
      const v = board[(r + HIDDEN) * COLS + c];
      putCell(s.i, r, c, v ? PAL[v - 1] : [0, 0, 0], v ? 255 : 0);
    }
    const so = (Math.floor(s.i / AC) * AC + (s.i % AC)) * 4;
    statsData[so] = Math.min(255, game.lines);
    if (game.k >= game.n) { s.done = true; statsData[so + 1] = 0; boardAverage(s.i); continue; }
    const piece = game.pieces[game.k], move = game.moves[game.k];
    statsData[so + 1] = game.k + 1 < game.n ? game.pieces[game.k + 1] + 1 : 0;
    const { x, cells } = PLACEMENTS[piece][move];
    const y = landing(board, piece, move);
    if (f < LOCK || y < 0) {
      const fp = fallingAt(stepPlan(piece, move), piece, f, y);
      for (const [r, c] of fp.cells) putCell(s.i, fp.y + r - HIDDEN, fp.x + c, PAL[piece], 180);
    } else {
      const tmp = board.slice();
      for (const [r, c] of cells) tmp[(y + r) * COLS + c + x] = piece + 1;
      const full = fullRows(tmp);
      for (const [r, c] of cells) putCell(s.i, y + r - HIDDEN, x + c, PAL[piece], 255);
      for (const row of full) for (let c = 0; c < COLS; c++) putCell(s.i, row - HIDDEN, c, [255, 255, 255], 120);
    }
    boardAverage(s.i);
  }
  atlas.needsUpdate = true;
  stats.needsUpdate = true;
  avgTex.needsUpdate = true;
}

// ---------- instanced world ----------
const TRIMS = ['#1ee0ff', '#ff2bd6', '#ffe01e', '#9b4dff', '#39ff7a', '#ff8c1a', '#2f7bff'].map(c => new THREE.Color(c));
const aDeath = new THREE.InstancedBufferAttribute(new Float32Array(N), 1);
const aSeed = new THREE.InstancedBufferAttribute(new Float32Array(N), 1);
const aSlot = new THREE.InstancedBufferAttribute(new Float32Array(N), 1);
const aTrim = new THREE.InstancedBufferAttribute(new Float32Array(N * 3), 3);
slots.forEach(s => {
  aDeath.array[s.i] = s.death;
  aSeed.array[s.i] = s.seed;
  aSlot.array[s.i] = s.i;
  const t = TRIMS[Math.floor(s.seed * 997) % TRIMS.length];
  aTrim.array.set([t.r * 0.75, t.g * 0.75, t.b * 0.75], s.i * 3);
});
function instanced(geo, mat, matrixFn) {
  const mesh = new THREE.InstancedMesh(geo, mat, N);
  const m = new THREE.Matrix4();
  slots.forEach(s => { matrixFn(s, m); mesh.setMatrixAt(s.i, m); });
  for (const [k, a] of Object.entries({ aDeath, aSeed, aSlot, aTrim })) geo.setAttribute(k, a);
  mesh.frustumCulled = false;
  scene.add(mesh);
  return mesh;
}

const glsl = {
  rot: `
    vec3 rotX(vec3 p, float a) { float c = cos(a), s = sin(a); return vec3(p.x, p.y * c - p.z * s, p.y * s + p.z * c); }
    vec3 rotAxis(vec3 p, vec3 k, float a) { return p * cos(a) + cross(k, p) * sin(a) + k * dot(k, p) * (1.0 - cos(a)); }
    float ign(vec2 p) { return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715)))); }`,
};

function injectCabinet(mat) {
  mat.onBeforeCompile = sh => {
    sh.uniforms.uTime = U.uTime;
    sh.vertexShader = sh.vertexShader
      .replace('#include <common>', `#include <common>
        attribute vec2 aEmit; attribute float aRough; attribute vec3 aTrim; attribute float aDeath;
        varying vec2 vEmit; varying float vRough; varying vec3 vTrim; varying float vDeathC; varying vec3 vLocal; varying vec3 vLocalN;`)
      .replace('#include <begin_vertex>', `#include <begin_vertex>
        vEmit = aEmit; vRough = aRough; vTrim = aTrim; vDeathC = aDeath; vLocal = position; vLocalN = normal;`);
    sh.fragmentShader = sh.fragmentShader
      .replace('#include <common>', `#include <common>
        uniform float uTime;
        varying vec2 vEmit; varying float vRough; varying vec3 vTrim; varying float vDeathC; varying vec3 vLocal; varying vec3 vLocalN;`)
      .replace('#include <roughnessmap_fragment>', `#include <roughnessmap_fragment>
        roughnessFactor = vRough;`)
      .replace('#include <emissivemap_fragment>', `#include <emissivemap_fragment>
        float trimK = mix(1.0, 0.16, smoothstep(vDeathC, vDeathC + 1.5, uTime));
        totalEmissiveRadiance += vEmit.x * vTrim * trimK + vEmit.y * vColor.rgb;
        if (abs(vLocalN.x) > 0.9 && vLocal.y < 2.08) {
          float s = fract((vLocal.y * 1.1 - vLocal.z * 0.9) * 1.3);
          float band = smoothstep(0.0, 0.015, s) * (1.0 - smoothstep(0.05, 0.065, s));
          totalEmissiveRadiance += band * vTrim * 0.06 * trimK;
        }`);
  };
  return mat;
}

function injectFly(mat, { hero = false, wing = false, rough = !wing }) {
  const defs = `${hero ? '#define HERO\n' : ''}${wing ? '#define WING\n' : ''}${rough ? '#define HAS_ROUGH\n' : ''}`;
  mat.onBeforeCompile = sh => {
    Object.assign(sh.uniforms, {
      uTime: U.uTime, uThorax: { value: new THREE.Vector3(...M.thorax) }, uPitch: { value: M.pitch }, uBodyAxis: { value: new THREE.Vector3(...M.bodyAxis) },
      uGlow: mat.userData.uGlow || { value: 0 }, uDeath: mat.userData.uDeath || { value: 1e9 }, uSeed: { value: 0.37 },
    });
    sh.vertexShader = defs + sh.vertexShader
      .replace('#include <common>', `#include <common>
        uniform float uTime; uniform vec3 uThorax; uniform float uPitch; uniform vec3 uBodyAxis;
        #ifdef HERO
          uniform float uDeath; uniform float uSeed;
          #define DEATH uDeath
          #define SEED uSeed
        #else
          attribute float aDeath; attribute float aSeed;
          #define DEATH aDeath
          #define SEED aSeed
        #endif
        #ifdef HAS_ROUGH
          attribute float aRough; varying float vRough;
        #endif
        #ifdef WING
          attribute vec3 aFlight; attribute vec3 aHinge; attribute float aSide;
        #endif
        varying float vFade; varying float vFly;
        ${glsl.rot}`)
      .replace('#include <beginnormal_vertex>', `#include <beginnormal_vertex>
        float dt = uTime - DEATH;
        float lvl = smoothstep(0.0, 0.45, dt);
        if (dt > 0.0) objectNormal = rotX(objectNormal, -uPitch * 0.85 * lvl);`)
      .replace('#include <begin_vertex>', `#include <begin_vertex>
        vFade = 1.0; vFly = 0.0;
        #ifdef HAS_ROUGH
          vRough = aRough;
        #endif
        #ifdef WING
          if (dt > 0.0) {
            float flap = sin(uTime * 47.0 + SEED * 20.0) * 1.0 - 0.2;
            vec3 fl = rotAxis(aFlight - aHinge, uBodyAxis, aSide * flap) + aHinge;
            transformed = mix(transformed, fl, clamp(dt / 0.12, 0.0, 1.0));
          }
        #endif
        if (dt > 0.0) {
          // A beat to spread the wings after GAME OVER, then lift off.
          float lt = max(dt - 0.3, 0.0);
          transformed = rotX(transformed - uThorax, -uPitch * 0.85 * lvl) + uThorax;
          float up = 1.1 * lt + 2.6 * lt * lt;
          transformed += vec3(sin(lt * 2.3 + SEED * 6.283) * 0.9 * lt, up,
                              0.5 * lt + 0.35 * lt * lt + cos(lt * 1.9 + SEED * 4.1) * 0.5 * lt);
          vFade = 1.0 - smoothstep(1.1, 2.6, dt);
          vFly = smoothstep(0.0, 0.3, dt);
        }`);
    sh.fragmentShader = defs + sh.fragmentShader
      .replace('#include <common>', `#include <common>
        uniform float uGlow;
        varying float vFade; varying float vFly;
        #ifdef HAS_ROUGH
          varying float vRough;
        #endif
        ${glsl.rot}`)
      .replace('#include <clipping_planes_fragment>', `#include <clipping_planes_fragment>
        if (vFade < 0.999 && vFade < ign(gl_FragCoord.xy)) discard;`)
      .replace('#include <roughnessmap_fragment>', `#include <roughnessmap_fragment>
        #ifdef HAS_ROUGH
          roughnessFactor = vRough;
        #endif`)
      .replace('#include <emissivemap_fragment>', `#include <emissivemap_fragment>
        vec3 toScreen = normalize((viewMatrix * vec4(0.0, 0.45, -1.0, 0.0)).xyz);
        totalEmissiveRadiance += diffuseColor.rgb * vec3(0.35, 0.6, 1.0) * uGlow * pow(max(dot(normal, toScreen), 0.0), 1.3);
        totalEmissiveRadiance += diffuseColor.rgb * vec3(1.6, 0.7, 0.45) * uGlow * vFly;`);
  };
  mat.customProgramCacheKey = () => defs;
  return mat;
}

const cabGeo = CAB.cabinetGeometry();
const cabMat = injectCabinet(new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.45, metalness: 0.15 }));
instanced(cabGeo, cabMat, (s, m) => m.makeTranslation(s.pos.x, 0, s.pos.z));
const joys = instanced(CAB.joystickGeometry(), cabMat, (s, m) => m.makeTranslation(s.pos.x + CAB.JOYSTICK.x, CAB.JOYSTICK.y, s.pos.z + CAB.JOYSTICK.z));
const joyM = new THREE.Matrix4();
function setJoystick(slot, tilt) {
  joyM.makeRotationAxis(CAB.JOY_AXIS, -tilt * THREE.MathUtils.degToRad(18))
    .setPosition(slot.pos.x + CAB.JOYSTICK.x, CAB.JOYSTICK.y, slot.pos.z + CAB.JOYSTICK.z);
  joys.setMatrixAt(slot.i, joyM);
  joys.instanceMatrix.needsUpdate = true;
}

const stoolGeo = CAB.stoolGeometry();
const stoolMat = injectCabinet(new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.4, metalness: 0.45 }));
instanced(stoolGeo, stoolMat, (s, m) => m.makeTranslation(s.pos.x, 0, s.pos.z + CAB.STOOL_Z));

// Screens.
const screenMat = new THREE.ShaderMaterial({
  fog: true,
  uniforms: THREE.UniformsUtils.merge([THREE.UniformsLib.fog, {
    uAC: { value: AC },
    uWinner: { value: winSlot.i }, uWinGlow: { value: 0 }, uLiveBoost: { value: 1 }, uGain: { value: 1 },
  }]),
  vertexShader: `
    attribute float aSlot; attribute float aDeath;
    varying vec2 vUv; flat varying float vSlot; flat varying float vDeath;
    #include <fog_pars_vertex>
    void main() {
      vUv = uv; vSlot = aSlot; vDeath = aDeath;
      vec4 mvPosition = modelViewMatrix * instanceMatrix * vec4(position, 1.0);
      gl_Position = projectionMatrix * mvPosition;
      #include <fog_vertex>
    }`,
  fragmentShader: `
    #define SS ${Q.get('ss') ?? 1}
    uniform sampler2D uAtlas; uniform sampler2D uStats; uniform sampler2D uAvg; uniform float uAC;
    uniform float uTime; uniform float uWinner; uniform float uWinGlow; uniform float uLiveBoost; uniform float uGain;
    varying vec2 vUv; flat varying float vSlot; flat varying float vDeath;
    #include <fog_pars_fragment>
    const int NEXTMASK[7] = int[7](0x0F, 0x66, 0x72, 0x36, 0x63, 0x71, 0x74);
    const int DIG[10] = int[10](0x3F, 0x06, 0x5B, 0x4F, 0x66, 0x6D, 0x7D, 0x07, 0x7F, 0x6F);
    // 3x5 glyphs, top row in the high bits: L I N E S X T
    const int GLYPH[13] = int[13](18727, 29847, 27501, 31143, 14478, 23213, 29842, 14699, 11245, 24557, 31599, 23402, 27565);
    const int LINES_TXT[5] = int[5](0, 1, 2, 3, 4);
    const int NEXT_TXT[4] = int[4](2, 3, 5, 6);
    const int GAME_TXT[4] = int[4](7, 8, 9, 3);
    const int OVER_TXT[4] = int[4](10, 11, 3, 12);
    const vec3 PAL[7] = vec3[7](vec3(0.0, 0.86, 1.0), vec3(1.0, 0.84, 0.12), vec3(0.69, 0.3, 1.0), vec3(0.22, 1.0, 0.35),
                                vec3(1.0, 0.23, 0.23), vec3(0.18, 0.42, 1.0), vec3(1.0, 0.55, 0.1));
    const vec2 PF_LO = vec2(0.05, 0.04), PF_HI = vec2(0.569, 0.96);
    const vec3 FRAME = vec3(0.12, 0.4, 1.0);
    vec3 lin(vec3 c) { return pow(c, vec3(2.2)); }
    float inBox(vec2 p, vec2 lo, vec2 hi) { vec2 q = step(lo, p) * step(p, hi); return q.x * q.y; }
    float seg7(vec2 p, int d) {
      int m = DIG[d]; float t = 0.17, on = 0.0;
      if ((m & 1) != 0) on += inBox(p, vec2(t, 1.0 - t), vec2(1.0 - t, 1.0));
      if ((m & 2) != 0) on += inBox(p, vec2(1.0 - t, 0.5), vec2(1.0, 1.0 - t));
      if ((m & 4) != 0) on += inBox(p, vec2(1.0 - t, t), vec2(1.0, 0.5));
      if ((m & 8) != 0) on += inBox(p, vec2(t, 0.0), vec2(1.0 - t, t));
      if ((m & 16) != 0) on += inBox(p, vec2(0.0, t), vec2(t, 0.5));
      if ((m & 32) != 0) on += inBox(p, vec2(0.0, 0.5), vec2(t, 1.0 - t));
      if ((m & 64) != 0) on += inBox(p, vec2(t, 0.5 - t * 0.5), vec2(1.0 - t, 0.5 + t * 0.5));
      return min(on, 1.0);
    }
    float glyphAt(vec2 p, int g) {
      // p in glyph pixels: x 0..3 from the left, y 0..5 from the top
      if (p.x < 0.0 || p.x >= 3.0 || p.y < 0.0 || p.y >= 5.0) return 0.0;
      int bit = (4 - int(p.y)) * 3 + (2 - int(p.x));
      return float((GLYPH[g] >> bit) & 1);
    }
    float label(vec2 uv, vec2 lo, float unitY, int which) {
      // unitY: glyph pixel height in uv; x is rescaled for the 0.62 x 0.70 screen.
      vec2 unit = vec2(unitY * 0.70 / 0.62, unitY);
      vec2 p = (uv - lo) / unit;
      p.y = 5.0 - p.y;
      int n = which == 0 ? 5 : 4;
      int i = int(floor(p.x / 4.0));
      if (i < 0 || i >= n) return 0.0;
      int g = which == 0 ? LINES_TXT[i] : (which == 1 ? NEXT_TXT[i] : (which == 2 ? GAME_TXT[i] : OVER_TXT[i]));
      return glyphAt(vec2(p.x - float(i) * 4.0, p.y), g);
    }
    vec3 block(vec3 c, vec2 f, float gain) {
      float e = min(min(f.x, f.y), min(1.0 - f.x, 1.0 - f.y));
      float hi = min(smoothstep(0.72, 0.88, f.y) + smoothstep(0.28, 0.12, f.x), 1.0);
      float lo = min(smoothstep(0.28, 0.12, f.y) + smoothstep(0.72, 0.88, f.x), 1.0);
      return c * (1.0 + 0.45 * hi - 0.35 * lo) * mix(0.12, 1.0, smoothstep(0.04, 0.09, e)) * gain;
    }
    vec3 screenColor(vec2 uv, vec2 b, vec4 st) {
      vec3 col = lin(vec3(0.02, 0.03, 0.08)) * (1.1 - 0.7 * length(uv - 0.5));
      vec2 gv = (uv - PF_LO) / (PF_HI - PF_LO) * vec2(10.0, 20.0);
      float inPF = inBox(gv, vec2(0.0), vec2(10.0, 20.0) - 1e-4);
      if (inPF > 0.5) {
        vec2 cell = floor(gv), f = fract(gv);
        vec4 t = texelFetch(uAtlas, ivec2(b * vec2(10.0, 20.0) + cell), 0);
        col = lin(vec3(0.01, 0.013, 0.035));
        if (t.a > 0.1) col = block(lin(t.rgb), f, t.a > 0.9 ? 1.0 : (t.a > 0.6 ? 1.35 : 2.4));
        else col += lin(vec3(0.08, 0.12, 0.28)) * 0.3 * (1.0 - smoothstep(0.0, 0.06, min(f.x, f.y)));
      }
      col += inBox(uv, PF_LO - 0.012, PF_HI + 0.012) * (1.0 - inPF) * FRAME * 1.1;
      // NEXT
      col += label(uv, vec2(0.628, 0.925), 0.0085, 1) * lin(vec3(0.55, 0.8, 1.0)) * 0.9;
      int nxt = int(st.g * 255.0 + 0.5) - 1;
      vec2 nlo = vec2(0.62, 0.77), nhi = vec2(0.94, 0.905);
      col += (inBox(uv, nlo - 0.008, nhi + 0.008) - inBox(uv, nlo, nhi)) * FRAME * 0.9;
      if (nxt >= 0 && inBox(uv, nlo, nhi) > 0.5) {
        vec2 q = (uv - nlo) / (nhi - nlo) * vec2(4.0, 2.0);
        if (nxt > 1) q.x -= 0.5;
        if (nxt == 0) q.y += 0.5;
        ivec2 qc = ivec2(floor(q));
        if (qc.x >= 0 && qc.x < 4 && qc.y >= 0 && qc.y < 2) {
          int bit = (1 - qc.y) * 4 + qc.x;
          if ((NEXTMASK[nxt] & (1 << bit)) != 0) col = block(PAL[nxt], fract(q), 1.1);
        }
      }
      // LINES
      col += label(uv, vec2(0.628, 0.695), 0.0085, 0) * lin(vec3(0.55, 0.8, 1.0)) * 0.9;
      int lines = int(st.r * 255.0 + 0.5);
      for (int i = 0; i < 3; i++) {
        vec2 lo = vec2(0.625 + float(i) * 0.107, 0.49), hi = lo + vec2(0.095, 0.18);
        if (inBox(uv, lo, hi) > 0.5) {
          int dig = (i == 0) ? lines / 100 : (i == 1 ? (lines / 10) % 10 : lines % 10);
          float on = seg7((uv - lo) / (hi - lo), dig);
          col = mix(lin(vec3(0.07, 0.045, 0.02)), lin(vec3(1.0, 0.78, 0.3)) * 1.5, on);
        }
      }
      // Meter toward the champion's 86 lines.
      vec2 mlo = vec2(0.64, 0.06), mhi = vec2(0.92, 0.42);
      if (inBox(uv, mlo, mhi) > 0.5) {
        float yy = (uv.y - mlo.y) / (mhi.y - mlo.y);
        float bars = step(0.3, fract(yy * 10.0));
        float fill = clamp(float(lines) / 86.0, 0.0, 1.0);
        col += bars * (yy < fill ? mix(lin(vec3(0.2, 1.0, 0.5)), lin(vec3(1.0, 0.85, 0.2)), yy) * 1.1 : lin(vec3(0.04, 0.06, 0.12)));
      }
      return col;
    }
    void main() {
      // Board index as an exact integer: an interpolated float can land on the neighbour's board.
      int slot = int(vSlot + 0.5), ac = int(uAC + 0.5);
      vec2 b = vec2(float(slot % ac), float(slot / ac));
      vec4 st = texelFetch(uStats, ivec2(b), 0);
      vec2 dx = dFdx(vUv), dy = dFdy(vUv);
      float heightPx = 1.0 / max(max(length(dx), length(dy)), 1e-5);
      float farK = 1.0 - smoothstep(24.0, 60.0, heightPx);
      vec3 col = vec3(0.0);
      if (farK < 1.0 && SS == 0) {
        col = screenColor(vUv, b, st);
      } else if (farK < 1.0) {
        // Rotated-grid supersampling so cells, digits and labels do not shimmer.
        col += screenColor(vUv + dx * -0.125 + dy * -0.375, b, st);
        col += screenColor(vUv + dx * 0.375 + dy * -0.125, b, st);
        col += screenColor(vUv + dx * 0.125 + dy * 0.375, b, st);
        col += screenColor(vUv + dx * -0.375 + dy * 0.125, b, st);
        col *= 0.25;
        float period = (1.0 / 190.0) * heightPx;
        col *= 1.0 - 0.16 * (0.5 + 0.5 * sin(vUv.y * 190.0 * 6.2832)) * smoothstep(3.0, 6.0, period);
      }
      if (farK > 0.0) {
        // Tiny screens: this board's own average colour, so nothing bleeds in from neighbours.
        vec3 far = lin(vec3(0.02, 0.03, 0.08));
        vec3 avg = lin(texelFetch(uAvg, ivec2(b), 0).rgb) * 1.3;
        far = mix(far, avg + lin(vec3(0.01, 0.013, 0.035)), inBox(vUv, PF_LO, PF_HI));
        far += (inBox(vUv, PF_LO - 0.02, PF_HI + 0.02) - inBox(vUv, PF_LO, PF_HI)) * FRAME * 0.6;
        col = mix(col, far, farK);
      }
      float dt = uTime - vDeath;
      if (dt >= 0.0) {
        float l = dot(col, vec3(0.3, 0.59, 0.11));
        col = mix(col, vec3(1.0, 0.07, 0.09) * (l * 1.3 + 0.03), smoothstep(0.0, 0.3, dt));
        col += vec3(3.0, 0.2, 0.2) * exp(-dt * 5.0);
        col *= mix(1.0, 0.2, smoothstep(0.3, 1.6, dt));
        if (farK < 1.0) {
          // GAME OVER, supersampled like the rest of the screen.
          float txt = 0.0;
          vec2 offs[4] = vec2[4](dx * -0.125 + dy * -0.375, dx * 0.375 + dy * -0.125, dx * 0.125 + dy * 0.375, dx * -0.375 + dy * 0.125);
          for (int k = 0; k < 4; k++) {
            vec2 q = vUv + offs[k];
            txt += 0.25 * max(label(q, vec2(0.125, 0.53), 0.022, 2), label(q, vec2(0.125, 0.39), 0.022, 3));
          }
          float show = smoothstep(0.0, 0.12, dt) * (1.0 - farK);
          col *= 1.0 - 0.75 * inBox(vUv, vec2(0.085, 0.35), vec2(0.535, 0.68)) * show;
          col += txt * vec3(2.6, 0.55, 0.45) * show * mix(1.0, 0.6, smoothstep(1.0, 3.0, dt));
        }
      } else col *= uLiveBoost;
      if (abs(float(slot) - uWinner) < 0.5) {
        float edge = 1.0 - inBox(vUv, vec2(0.025), vec2(0.975));
        col += edge * vec3(1.6, 1.0, 0.25) * uWinGlow;
      }
      gl_FragColor = vec4(col * uGain, 1.0);
      #include <fog_fragment>
    }`,
  polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2,
});
Object.assign(screenMat.uniforms, { uAtlas: { value: atlas }, uStats: { value: stats }, uAvg: { value: avgTex }, uTime: U.uTime });
const screenGeo = new THREE.PlaneGeometry(CAB.SCREEN_W, CAB.SCREEN_H);
instanced(screenGeo, screenMat, (s, m) => {
  m.makeRotationX(-CAB.SCREEN_TILT).setPosition(s.pos.x + CAB.SCREEN_CENTER.x, CAB.SCREEN_CENTER.y, s.pos.z + CAB.SCREEN_CENTER.z);
});

// Marquees.
const marqueeMat = new THREE.MeshBasicMaterial({ map: CAB.marqueeTexture(), color: new THREE.Color(1.6, 1.6, 1.6), polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2 });
instanced(new THREE.PlaneGeometry(1.18, 0.3), marqueeMat, (s, m) => m.makeTranslation(s.pos.x, 2.32, s.pos.z + 0.335));

// Light pools on the floor in front of each screen.
const poolMat = new THREE.ShaderMaterial({
  fog: true, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
  uniforms: THREE.UniformsUtils.merge([THREE.UniformsLib.fog, { uGain: { value: 1 } }]),
  vertexShader: `
    attribute float aDeath; varying vec2 vUv; varying float vDeath;
    #include <fog_pars_vertex>
    void main() {
      vUv = uv; vDeath = aDeath;
      vec4 mvPosition = modelViewMatrix * instanceMatrix * vec4(position, 1.0);
      gl_Position = projectionMatrix * mvPosition;
      #include <fog_vertex>
    }`,
  fragmentShader: `
    uniform float uTime; uniform float uGain; varying vec2 vUv; varying float vDeath;
    #include <fog_pars_fragment>
    void main() {
      vec2 p = (vUv - 0.5) * vec2(2.0, 2.0);
      float g = exp(-dot(p, p) * 2.6);
      float dt = uTime - vDeath;
      vec3 c = vec3(0.12, 0.3, 0.8);
      if (dt > 0.0) c = mix(c, vec3(0.5, 0.03, 0.04), smoothstep(0.0, 0.3, dt)) * mix(1.0, 0.25, smoothstep(0.3, 1.8, dt));
      gl_FragColor = vec4(c * g * uGain, 1.0);
      #include <fog_fragment>
    }`,
  polygonOffset: true, polygonOffsetFactor: -4, polygonOffsetUnits: -4,
});
poolMat.uniforms.uTime = U.uTime;
const poolGeo = new THREE.PlaneGeometry(2.6, 2.4);
poolGeo.rotateX(-Math.PI / 2);
instanced(poolGeo, poolMat, (s, m) => m.makeTranslation(s.pos.x, 0.03, s.pos.z + 1.2));

// Spotlight cones over the survivors once the field thins out.
const beamMat = new THREE.ShaderMaterial({
  transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide,
  uniforms: { uTime: U.uTime, uBeam: { value: 0 } },
  vertexShader: `
    attribute float aDeath; varying float vY; varying float vDeath; varying vec3 vN; varying vec3 vV;
    void main() {
      vY = position.y; vDeath = aDeath;
      vec4 wp = instanceMatrix * vec4(position, 1.0);
      vec4 mv = viewMatrix * modelMatrix * wp;
      vN = normalize(mat3(viewMatrix) * normal); vV = normalize(-mv.xyz);
      gl_Position = projectionMatrix * mv;
    }`,
  fragmentShader: `
    uniform float uTime; uniform float uBeam; varying float vY; varying float vDeath; varying vec3 vN; varying vec3 vV;
    void main() {
      float alive = 1.0 - smoothstep(0.0, 0.5, uTime - vDeath);
      float edge = pow(abs(dot(vN, vV)), 1.5);
      float fall = smoothstep(18.0, 2.0, vY) * smoothstep(0.0, 1.5, vY);
      gl_FragColor = vec4(vec3(1.0, 0.85, 0.55) * 0.22 * edge * fall * alive * uBeam, 1.0);
    }`,
});
const beamGeo = new THREE.CylinderGeometry(0.25, 1.3, 18, 24, 1, true);
beamGeo.translate(0, 9, 0);
const beams = instanced(beamGeo, beamMat, (s, m) => m.makeTranslation(s.pos.x, 0, s.pos.z + 0.9));

// Flies: NeuroMechFly body (micro-CT Drosophila), posed by build_fly.py.
const uGlow = { value: 1.0 };
function geoFrom(d, rough) {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(arr(d.position), 3));
  if (d.color) g.setAttribute('color', new THREE.BufferAttribute(arr(d.color), 3));
  if (d.uv) g.setAttribute('uv', new THREE.BufferAttribute(arr(d.uv), 2));
  g.setIndex(new THREE.BufferAttribute(arr(d.index), 1));
  g.computeVertexNormals();
  // Thin two-sided parts (arista, wing membrane) cancel to zero normals, which shade as NaN.
  const nrm = g.attributes.normal;
  for (let i = 0; i < nrm.count; i++) if (Math.hypot(nrm.getX(i), nrm.getY(i), nrm.getZ(i)) < 1e-4) nrm.setXYZ(i, 0, 1, 0);
  if (rough !== undefined) g.setAttribute('aRough', new THREE.Float32BufferAttribute(new Float32Array(g.attributes.position.count).fill(rough), 1));
  return g;
}
function wingGeo(d) {
  const g = geoFrom(d);
  const n = g.attributes.position.count;
  g.setAttribute('aFlight', new THREE.BufferAttribute(arr(d.flight), 3));
  g.setAttribute('aHinge', new THREE.Float32BufferAttribute(Array.from({ length: n }, () => d.hinge).flat(), 3));
  g.setAttribute('aSide', new THREE.Float32BufferAttribute(new Float32Array(n).fill(d.side), 1));
  return g;
}
function eyeUv(g) {
  // Spherical UVs around each eye's centre, for the facet bump map.
  const p = g.attributes.position, n = p.count, uv = new Float32Array(n * 2);
  const centre = side => {
    const c = new THREE.Vector3(); let k = 0;
    for (let i = 0; i < n; i++) if (Math.sign(p.getX(i)) === side) { c.add(new THREE.Vector3().fromBufferAttribute(p, i)); k++; }
    return c.divideScalar(k);
  };
  const cs = { '-1': centre(-1), '1': centre(1) };
  const v = new THREE.Vector3();
  for (let i = 0; i < n; i++) {
    v.fromBufferAttribute(p, i).sub(cs[Math.sign(p.getX(i)) || 1]).normalize();
    uv[i * 2] = Math.atan2(v.z, v.x) / (2 * Math.PI) + 0.5;
    uv[i * 2 + 1] = Math.acos(THREE.MathUtils.clamp(v.y, -1, 1)) / Math.PI;
  }
  g.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
  return g;
}
const crowdGeo = geoFrom(M.crowd, 0.5);
const flyMat = injectFly(new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.5, metalness: 0.05 }), {});
flyMat.userData.uGlow = uGlow;
const flyMatrix = (s, m) => {
  const k = s === heroSlot || s === winSlot ? 0 : FLY.FLY_SCALE;
  m.makeScale(k, k, k).setPosition(s.pos.x, CAB.SEAT_Y, s.pos.z + CAB.STOOL_Z);
};
instanced(crowdGeo, flyMat, flyMatrix);
const wingTex = FLY.wingTexture();
const crowdWings = mergeGeometries([wingGeo(M.wings.l), wingGeo(M.wings.r)]);
const wingMat = injectFly(new THREE.MeshStandardMaterial({
  map: wingTex, transparent: true, depthWrite: false, side: THREE.DoubleSide, roughness: 0.25, metalness: 0,
  emissive: new THREE.Color(0.05, 0.07, 0.1),
}), { wing: true });
wingMat.userData.uGlow = uGlow;
instanced(crowdWings, wingMat, flyMatrix);

// Hero front legs follow the game: the left foot rides the joystick, the right foot hops
// between the rotate, rotate-back and drop buttons. Forward kinematics runs on the CPU so the
// legs stay in the fly frame the take-off shader expects.
const RIG_ROOT = new THREE.Matrix4().fromArray(M.rig.root);
const lerpAngles = (A, B, k) => A.map((seg, i) => seg.map((v, j) => v + (B[i][j] - v) * k));
function legRig(leg, group, uDeath) {
  const R = M.rig.legs[leg];
  const segs = R.segments.map(sg => {
    const g = geoFrom(sg.mesh, 0.5);
    return { g, rest: new THREE.Matrix4().fromArray(sg.rest), axes: sg.axes.map(a => new THREE.Vector3(...a)),
             pos: g.attributes.position.array.slice(), nrm: g.attributes.normal.array.slice() };
  });
  const geo = mergeGeometries(segs.map(sg => sg.g));
  const mat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.5 });
  mat.userData.uGlow = { value: 0 };
  mat.userData.uDeath = uDeath;
  injectFly(mat, { hero: true });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.frustumCulled = false;
  group.add(mesh);
  const chain = new THREE.Matrix4(), rot = new THREE.Matrix4(), nm = new THREE.Matrix3();
  function apply(angles) {
    const P = geo.attributes.position.array, Nn = geo.attributes.normal.array;
    chain.copy(RIG_ROOT);
    let off = 0;
    segs.forEach((sg, i) => {
      chain.multiply(sg.rest);
      for (let a = 0; a < 3; a++) if (angles[i][a]) chain.multiply(rot.makeRotationAxis(sg.axes[a], angles[i][a]));
      nm.getNormalMatrix(chain);
      const e = chain.elements, n = nm.elements;
      for (let j = 0; j < sg.pos.length; j += 3) {
        const x = sg.pos[j], y = sg.pos[j + 1], z = sg.pos[j + 2];
        P[off + j] = e[0] * x + e[4] * y + e[8] * z + e[12];
        P[off + j + 1] = e[1] * x + e[5] * y + e[9] * z + e[13];
        P[off + j + 2] = e[2] * x + e[6] * y + e[10] * z + e[14];
        const a = sg.nrm[j], b = sg.nrm[j + 1], c = sg.nrm[j + 2];
        const nx = n[0] * a + n[3] * b + n[6] * c, ny = n[1] * a + n[4] * b + n[7] * c, nz = n[2] * a + n[5] * b + n[8] * c;
        const l = Math.hypot(nx, ny, nz) || 1;
        Nn[off + j] = nx / l; Nn[off + j + 1] = ny / l; Nn[off + j + 2] = nz / l;
      }
      off += sg.pos.length;
    });
    geo.attributes.position.needsUpdate = true;
    geo.attributes.normal.needsUpdate = true;
  }
  return { apply, poses: R.poses };
}
const leftAngles = (rig, tilt) => lerpAngles(rig.poses['joy+0'], rig.poses[tilt >= 0 ? 'joy+1' : 'joy-1'], Math.abs(tilt));
function rightAngles(rig, u, h) {
  const b = Math.min(1, Math.floor(u)), k = u - b, P = rig.poses;
  const hover = lerpAngles(P[`b${b}_hover`], P[`b${b + 1}_hover`], k);
  const press = lerpAngles(P[`b${b}_press`], P[`b${b + 1}_press`], k);
  return lerpAngles(press, hover, h);
}
// Joystick tilt, button position (0..2) and hover height (1 up, 0 pressed) at clock p.
function handState(slot, p) {
  const g = slot.game, k = Math.floor(p), f = p - k;
  if (k >= g.n) return { tilt: 0, u: 0, h: 1 };
  const plan = stepPlan(g.pieces[k], g.moves[k]);
  const dir = Math.sign(plan.x - plan.x0);
  const tilt = dir * smooth(plan.slideStart - 0.03, plan.slideStart, f) * (1 - smooth(plan.slideEnd, plan.slideEnd + 0.04, f));
  const keys = [[0, 0, 1]];
  for (const pr of plan.presses) {
    const prev = keys[keys.length - 1][0];
    keys.push([Math.max(prev + 0.005, pr.f - 0.035), pr.button, 1], [pr.f, pr.button, 0], [pr.f + 0.03, pr.button, 1]);
  }
  keys.push([0.97, 0, 1]);
  let i = 0;
  while (i < keys.length - 2 && f > keys[i + 1][0]) i++;
  const [f0, u0, h0] = keys[i], [f1, u1, h1] = keys[i + 1];
  const w = smooth(f0, f1, f);
  return { tilt, u: u0 + (u1 - u0) * w, h: h0 + (h1 - h0) * w };
}

// Hero flies: full mesh detail, lit by their own screens.
const facet = FLY.eyeFacetTexture();
facet.repeat.set(3, 2);
function heroFly(slot) {
  const group = new THREE.Group();
  const uDeath = { value: slot.death };
  const mk = (geo, mat, opts = {}, name = '') => {
    if ((Q.get('hide') || '').split(',').includes(name)) return;
    mat.userData.uGlow = { value: 0 };
    mat.userData.uDeath = uDeath;
    injectFly(mat, { hero: true, ...opts });
    group.add(new THREE.Mesh(geo, mat));
  };
  mk(geoFrom(M.hero.body, 0.5), new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.5 }), {}, 'body');
  mk(geoFrom(M.hero.bristle, 0.4), new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.4 }), {}, 'bristle');
  mk(eyeUv(geoFrom(M.hero.eye, 0.35)), new THREE.MeshPhysicalMaterial({
    vertexColors: true, roughness: 0.38, clearcoat: 0.4, clearcoatRoughness: 0.25,
    bumpMap: facet, bumpScale: 0.5, sheen: 0.3, sheenColor: new THREE.Color('#ff2010'), sheenRoughness: 0.5,
  }), {}, 'eye');
  for (const side of ['l', 'r']) {
    mk(wingGeo(M.wings[side]), new THREE.MeshPhysicalMaterial({
      map: wingTex, transparent: true, opacity: 0.7, depthWrite: false, side: THREE.DoubleSide, roughness: 0.35,
      specularIntensity: 0.5, iridescence: 0.8, iridescenceIOR: 1.4, iridescenceThicknessRange: [250, 700],
    }), { wing: true, rough: false }, 'wing');
  }
  group.scale.setScalar(FLY.FLY_SCALE);
  group.position.set(slot.pos.x, CAB.SEAT_Y, slot.pos.z + CAB.STOOL_Z);
  scene.add(group);
  const light = new THREE.RectAreaLight('#7ab8ff', 9, CAB.SCREEN_W, CAB.SCREEN_H);
  light.position.set(slot.pos.x, CAB.SCREEN_CENTER.y, slot.pos.z + CAB.SCREEN_CENTER.z + 0.02);
  light.lookAt(slot.pos.x, CAB.SEAT_Y + 0.6, slot.pos.z + CAB.STOOL_Z);
  scene.add(light);
  const key = new THREE.PointLight('#ffb070', 6, 6, 2);
  key.position.set(slot.pos.x + 1.1, 2.3, slot.pos.z + 2.6);
  scene.add(key);
  const rim = new THREE.PointLight('#b070ff', 2.5, 4, 2);
  rim.position.set(slot.pos.x - 1.0, 2.1, slot.pos.z + 0.6);
  scene.add(rim);
  const left = legRig('lf', group, uDeath), right = legRig('rf', group, uDeath);
  left.apply(leftAngles(left, 0));
  right.apply(rightAngles(right, 0, 1));
  return { group, light, key, rim, left, right };
}
const heroA = heroFly(heroSlot);
const heroB = heroFly(winSlot);
const flyPoint = (s, p) => new THREE.Vector3(...p).multiplyScalar(FLY.FLY_SCALE).add(new THREE.Vector3(s.pos.x, CAB.SEAT_Y, s.pos.z + CAB.STOOL_Z));
const flyHead = s => flyPoint(s, M.head);

// Crown for the winner.
const crown = new THREE.Group();
{
  const gold = new THREE.MeshStandardMaterial({ color: '#ffcc40', metalness: 1, roughness: 0.22, emissive: '#6a4000', emissiveIntensity: 0.6 });
  const band = new THREE.CylinderGeometry(0.1, 0.092, 0.07, 32, 1, true);
  crown.add(new THREE.Mesh(band, Object.assign(gold.clone(), { side: THREE.DoubleSide })));
  for (let i = 0; i < 8; i++) {
    const a = (i / 8) * Math.PI * 2;
    const spike = new THREE.Mesh(new THREE.ConeGeometry(0.022, 0.07, 8), gold);
    spike.position.set(Math.cos(a) * 0.097, 0.07, Math.sin(a) * 0.097);
    crown.add(spike);
    const gem = new THREE.Mesh(new THREE.SphereGeometry(0.012, 10, 8), new THREE.MeshStandardMaterial({ color: '#ff1030', emissive: '#ff1030', emissiveIntensity: 1.5 }));
    gem.position.set(Math.cos(a) * 0.1, 0.0, Math.sin(a) * 0.1);
    crown.add(gem);
  }
  crown.scale.setScalar(FLY.FLY_SCALE * 0.75);
  crown.visible = false;
  scene.add(crown);
}
const crownRest = flyPoint(winSlot, [M.headTop[0], M.headTop[1] + 0.02, M.headTop[2]]);

const winSpot = new THREE.SpotLight('#ffcf66', 0, 14, 0.32, 0.7, 1.5);
winSpot.position.set(winSlot.pos.x, 9, winSlot.pos.z + 2.5);
winSpot.target.position.copy(flyHead(winSlot));
scene.add(winSpot, winSpot.target);

// Hall.
const carpet = CAB.carpetTexture();
const fieldW = colX(GCOLS - 1) + 60, fieldD = GROWS * DZ + 120;
carpet.repeat.set(fieldW / 4, fieldD / 4);
const floor = new THREE.Mesh(new THREE.PlaneGeometry(fieldW, fieldD), new THREE.MeshStandardMaterial({
  map: carpet, emissiveMap: carpet, emissive: new THREE.Color(1, 1, 1), emissiveIntensity: 0.1, roughness: 0.92,
}));
floor.rotation.x = -Math.PI / 2;
floor.position.set(0, 0, -GROWS * DZ / 2 + 10);
scene.add(floor);
scene.add(new THREE.HemisphereLight('#6a58b0', '#0a0814', 0.55));
const sun = new THREE.DirectionalLight('#8fa0ff', 0.45);
sun.position.set(0.4, 1, 0.6);
scene.add(sun);

// ---------- camera path ----------
function spline(ts, vals) {
  // Catmull-Rom tangents on non-uniform times, per component.
  const n = ts.length;
  const tan = vals.map((v, i) => {
    if (i === 0 || i === n - 1) return v.map(() => 0);
    return v.map((_, k) => (vals[i + 1][k] - vals[i - 1][k]) / (ts[i + 1] - ts[i - 1]));
  });
  return t => {
    if (t <= ts[0]) return vals[0];
    if (t >= ts[n - 1]) return vals[n - 1];
    let i = 0;
    while (t > ts[i + 1]) i++;
    const h = ts[i + 1] - ts[i], u = (t - ts[i]) / h, u2 = u * u, u3 = u2 * u;
    return vals[i].map((a, k) => (2 * u3 - 3 * u2 + 1) * a + (u3 - 2 * u2 + u) * h * tan[i][k] + (-2 * u3 + 3 * u2) * vals[i + 1][k] + (u3 - u2) * h * tan[i + 1][k]);
  };
}
const hp = heroSlot.pos, wp = winSlot.pos;
const fz = -GROWS * DZ * 0.3;
const wideD = Math.max(40, (colX(GCOLS - 1)) * 1.25);
const deg = THREE.MathUtils.degToRad;
// [time, target x, y, z, log distance, elevation, azimuth]
// Framing points: each hero's head and screen.
const hh = flyHead(heroSlot), hs = new THREE.Vector3(hp.x, CAB.SCREEN_CENTER.y, hp.z + CAB.SCREEN_CENTER.z);
const wh = flyHead(winSlot), ws = new THREE.Vector3(wp.x, CAB.SCREEN_CENTER.y, wp.z + CAB.SCREEN_CENTER.z);
const at = (a, b, k) => a.clone().lerp(b, k).toArray();
// Where a hero fly is dt seconds after take-off: the take-off shader's offset, in world units.
function flyingAt(slot, t, seed = 0.37) {
  const dt = Math.max(t - 0.3, 0);
  const off = new THREE.Vector3(Math.sin(dt * 2.3 + seed * 6.283) * 0.9 * dt, 1.1 * dt + 2.6 * dt * dt,
    0.5 * dt + 0.35 * dt * dt + Math.cos(dt * 1.9 + seed * 4.1) * 0.5 * dt);
  return flyPoint(slot, new THREE.Vector3(...M.thorax).add(off).toArray()).toArray();
}
const KEYS = [
  [0, ...at(hh, hs, 0.0), Math.log(1.9), deg(4), deg(98)],
  [1.7, ...at(hh, hs, 0.3), Math.log(2.4), deg(8), deg(62)],
  [T.closeEnd, ...at(hh, hs, 0.55), Math.log(3.1), deg(12), deg(20)],
  [T.heroDie, ...at(hh, hs, 0.55), Math.log(3.3), deg(13), deg(16)],
  [5.45, ...at(hh, hs, 0.6), Math.log(3.35), deg(14), deg(14)],
  [6.0, ...flyingAt(heroSlot, 1.0), Math.log(4.0), deg(26), deg(12)],
  [6.6, ...flyingAt(heroSlot, 1.6), Math.log(7), deg(40), deg(8)],
  [7.2, hp.x * 0.6, 0.5, hp.z - 20, Math.log(wideD * 0.4), deg(42), deg(3)],
  [T.wide, 0, 0, fz, Math.log(wideD), deg(43), 0],
  [10.6, 0, 0, fz - 3, Math.log(wideD * 0.92), deg(46), deg(-3)],
  [11.9, (wp.x) * 0.5, 0, (fz + wp.z) / 2, Math.log(wideD * 0.66), deg(44), deg(-6)],
  [13.2, wp.x, 1.0, wp.z + 0.6, Math.log(11), deg(28), deg(12)],
  [T.arrive, ...at(wh, ws, 0.5), Math.log(3.0), deg(12), deg(30)],
  [T.end, ...at(wh, ws, 0.15), Math.log(2.4), deg(7), deg(78)],
];
const camAt = spline(KEYS.map(k => k[0]), KEYS.map(k => k.slice(1)));

const smooth = (a, b, x) => { const t = Math.min(Math.max((x - a) / (b - a), 0), 1); return t * t * (3 - 2 * t); };

function setTime(t) {
  U.uTime.value = t;
  const [tx, ty, tz, lnD, el, az] = camAt(t);
  const D = Math.exp(lnD);
  camera.position.set(tx + D * Math.sin(az) * Math.cos(el), ty + D * Math.sin(el), tz + D * Math.cos(az) * Math.cos(el));
  camera.lookAt(tx, ty, tz);
  scene.fog.density = 0.0012 + 0.028 / Math.max(1, D * 0.35);
  bloom.strength = 0.4 + 0.5 * smooth(3, 30, D);
  camera.far = Math.max(60, D * 6);
  camera.updateProjectionMatrix();
  const p = clock(t);
  updateBoards(p);
  if (D < 25) {
    for (const [slot, rig] of [[heroSlot, heroA], [winSlot, heroB]]) {
      if (!rig.group.visible) continue;
      const hand = handState(slot, p);
      rig.left.apply(leftAngles(rig.left, hand.tilt));
      rig.right.apply(rightAngles(rig.right, hand.u, hand.h));
      setJoystick(slot, hand.tilt);
    }
  }
  const alive = slots.reduce((a, s) => a + (s.death > t ? 1 : 0), 0);
  screenMat.uniforms.uLiveBoost.value = 1 + 1.2 * smooth(200, 5, alive) * smooth(T.massEnd - 3, T.massEnd, t) * smooth(4, 20, D);
  screenMat.uniforms.uWinGlow.value = smooth(T.runner, T.runner + 1.5, t);
  beamMat.uniforms.uBeam.value = smooth(400, 60, alive) * smooth(4, 14, D);
  beams.visible = beamMat.uniforms.uBeam.value > 0.001;
  winSpot.intensity = 20 * smooth(T.runner + 1, T.arrive, t);
  const heroVis = t < T.heroDie + 3;
  heroA.group.visible = heroVis;
  heroA.light.intensity = 9 * (t < heroSlot.death ? 1 : Math.max(0, 1 - (t - heroSlot.death) * 1.5));
  const c = smooth(T.crown, T.crown + 0.9, t);
  crown.visible = t > T.crown;
  const bounce = 1 - Math.pow(1 - c, 3) * Math.cos(c * 7);
  crown.position.copy(crownRest).add(new THREE.Vector3(0, (1 - bounce) * 1.2 + (1 - c) * 0.3, 0));
  crown.rotation.set(-M.pitch * 0.6, c * 1.2, 0);
  return alive;
}

function render(t) {
  const alive = setTime(t);
  composer.render();
  return alive;
}

window.flytris = {
  fps: FPS, duration: T.end, frames: Math.round(T.end * FPS), n: N,
  renderFrame: i => { render(i / FPS); return renderer.domElement.toDataURL('image/png'); },
  renderAt: t => render(t),
  renderView: (t, pos, look) => { setTime(t); camera.position.set(...pos); camera.lookAt(...look); scene.fog.density = 0.01; composer.render(); },
  heroPos: () => [heroSlot.pos.x, heroSlot.pos.z, winSlot.pos.x, winSlot.pos.z],
  timeline: () => {
    const presses = (slot, from, to) => {
      const out = [], g = slot.game;
      for (let k = 0; k < g.n; k++) {
        const plan = stepPlan(g.pieces[k], g.moves[k]);
        const events = plan.presses.map(pr => [pr.f, pr.button]);
        if (plan.x !== plan.x0) events.push([plan.slideStart, 3]);
        for (const [f, b] of events) {
          const t = timeAtPiece(k + f);
          if (t >= from && t < to) out.push([t, b]);
        }
      }
      return out;
    };
    const clears = slot => {
      const g = new Game(slot.game.pieces, slot.game.moves);
      g.advanceTo(g.n);
      return g.lineSteps.flatMap((l, k) => (l > (k ? g.lineSteps[k - 1] : 0) ? [timeAtPiece(k + LOCK)] : []))
        .filter(x => x < T.end);
    };
    return {
      T, fps: FPS, deaths: slots.map(s => s.death).filter(d => d < T.end), hero: heroSlot.death,
      runner: slots.find(s => s.g === runner).death, heroClears: clears(heroSlot), winnerClears: clears(winSlot),
      heroPresses: presses(heroSlot, 0, heroSlot.death), winnerPresses: presses(winSlot, T.runner - 0.3, T.end),
    };
  },
  info: () => ({ N, GCOLS, GROWS, opener: opener.n, openerLines: opener.lines, winner: [winner.seed, winner.lines, winner.n], runner: runner.n, winLast }),
  ready: true,
};

if (MODE === 'preview') {
  const hud = document.getElementById('hud');
  let t = +Q.get('t') || 0, playing = true, last = performance.now();
  addEventListener('keydown', e => {
    if (e.key === ' ') playing = !playing;
    if (e.key === 'ArrowRight') t += 1;
    if (e.key === 'ArrowLeft') t = Math.max(0, t - 1);
    if (e.key === 'Home') t = 0;
  });
  const loop = now => {
    if (playing) t += (now - last) / 1000;
    last = now;
    if (t > T.end) t = 0;
    const alive = render(t);
    if (hud) hud.textContent = `${t.toFixed(1)}s  alive ${alive}/${N}  [space] pause  [<-/->] seek`;
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
}
