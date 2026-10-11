// 3D battlefield: terrain, overlays, defenses, drones, interceptors, effects.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

export const G = 32;
const H_SCALE = 7;          // world units for height 1.0
const DRONE_ALT = 0.7;      // visual flight height above terrain
const OV = 24;              // overlay canvas pixels per cell
const FLOOR_Y = -1.6;       // bottom of the terrain plinth
const BG = 0x0d0e0f;

export const COLORS = {
  radar: '#3fd0ff', jammer: '#ffb547', pad: '#5cf0a0', enemy: '#ff4d5e',
  interceptor: '#7fe3ff', base: '#e8eef4', astar: '#ffd166', naive: '#9aa7b4',
};

const cellXY = (c) => [c % G, Math.floor(c / G)];

// Realistic terrain: a MeshStandardMaterial (lit, shadowed, fogged) whose albedo is generated in the
// shader from altitude, slope and noise: grass and dry scrub on low ground, dirt on slopes, rock on steep
// faces and snow above the flight ceiling. The gameplay overlay is painted on top and glows slightly so it
// stays readable in shadow.
const TERRAIN_PARS = /* glsl */`
uniform sampler2D uOverlay; uniform float uHScale; uniform float uGrid;
varying vec3 vWP; varying vec3 vWN; varying float vAo;
float tHash(vec2 p) { p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }
float tNoise(vec2 p) {
  vec2 i = floor(p), f = fract(p); vec2 u = f * f * (3.0 - 2.0 * f);
  return mix(mix(tHash(i), tHash(i + vec2(1.0, 0.0)), u.x), mix(tHash(i + vec2(0.0, 1.0)), tHash(i + vec2(1.0, 1.0)), u.x), u.y);
}
float tFbm3(vec2 p) { float v = 0.0, a = 0.5; for (int i = 0; i < 3; i++) { v += a * tNoise(p); p = p * 2.03 + vec2(17.1, 9.3); a *= 0.5; } return v / 0.875; }
float tFbm(vec2 p) { float v = 0.0, a = 0.5; for (int i = 0; i < 4; i++) { v += a * tNoise(p); p = p * 2.03 + vec2(17.1, 9.3); a *= 0.5; } return v / 0.9375; }
vec3 tLin(vec3 c) { return pow(c, vec3(2.2)); }
vec3 tPerturb(vec3 pos, vec3 n, vec2 dh) {
  vec3 sx = dFdx(pos), sy = dFdy(pos); vec3 r1 = cross(sy, n), r2 = cross(n, sx);
  float det = dot(sx, r1); vec3 g = sign(det) * (dh.x * r1 + dh.y * r2);
  return normalize(abs(det) * n - g);
}
`;

const TERRAIN_ALBEDO = /* glsl */`
  float hN = vWP.y / uHScale;
  vec3 wn = normalize(vWN);
  float slope = 1.0 - clamp(wn.y, 0.0, 1.0);
  vec2 q = vWP.xz;
  float n1 = tFbm3(q * 0.28);
  float n2 = tFbm(q * 1.7 + 3.1);
  float n3 = tFbm3(q * 6.0 + 7.7);
  // on steep faces sample the noise sideways (along the face, stretched vertically) so rock shows
  // horizontal layering instead of streaks smeared down the cliff
  float steep = smoothstep(0.35, 0.7, slope);
  if (steep > 0.0) {
    vec2 cq = vec2((q.x + q.y) * 0.9, vWP.y * 2.6);
    float nv = tFbm3(cq + 2.3);
    nv = mix(nv, 0.5, smoothstep(0.35, 0.9, length(fwidth(cq * 4.0))));   // fade sub-pixel detail (no shimmer)
    n3 = mix(n3, nv, steep);
  }
  n3 = mix(n3, 0.5, smoothstep(0.35, 0.9, length(fwidth(q * 24.0))));
  vec3 col = mix(tLin(vec3(0.25, 0.29, 0.16)), tLin(vec3(0.37, 0.37, 0.21)), smoothstep(0.3, 0.7, n2));
  float dry = smoothstep(0.42, 0.78, n1 + hN * 0.35 + slope * 0.7);
  col = mix(col, mix(tLin(vec3(0.42, 0.35, 0.25)), tLin(vec3(0.55, 0.47, 0.35)), n3), dry * 0.85);
  float rocky = smoothstep(0.24, 0.44, slope + (n2 - 0.5) * 0.25);
  rocky = max(rocky, smoothstep(0.62, 0.8, hN + (n1 - 0.5) * 0.12));
  col = mix(col, mix(tLin(vec3(0.33, 0.32, 0.31)), tLin(vec3(0.54, 0.52, 0.49)), smoothstep(0.25, 0.8, n3)), rocky);
  float snowLine = 0.79 + (n2 - 0.5) * 0.06;
  float snowAmt = smoothstep(snowLine, snowLine + 0.03, hN) * (1.0 - smoothstep(0.45, 0.8, slope));
  col = mix(col, tLin(vec3(0.93, 0.94, 0.96)), snowAmt);
  col *= 0.8 + 0.34 * n3;
  col *= vAo;
  vec2 gq = q + 16.0;
  vec2 gd = min(fract(gq), 1.0 - fract(gq)); vec2 gw = fwidth(gq) * 1.2;
  col *= 1.0 - uGrid * max(1.0 - smoothstep(0.0, gw.x, gd.x), 1.0 - smoothstep(0.0, gw.y, gd.y));
  vec4 ov = texture2D(uOverlay, (q + 16.0) / 32.0);
  vec3 ovc = tLin(ov.rgb);
  col = mix(col, ovc * 0.6, ov.a * 0.32);
  diffuseColor.rgb = col;
  float bumpH = (n3 * 0.6 + n2 * 0.4) * (0.05 + 0.2 * rocky) * (1.0 - 0.7 * snowAmt);
`;

// CPU-side noise, identical in spirit to the shader's, for the mesh's fine relief and tree placement
const hash2 = (x, y) => { const h = Math.sin(x * 127.1 + y * 311.7) * 43758.5453; return h - Math.floor(h); };
function vnoise(x, y) {
  const xi = Math.floor(x), yi = Math.floor(y), xf = x - xi, yf = y - yi;
  const u = xf * xf * (3 - 2 * xf), v = yf * yf * (3 - 2 * yf);
  const a = hash2(xi, yi), b = hash2(xi + 1, yi), c = hash2(xi, yi + 1), d = hash2(xi + 1, yi + 1);
  return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v;
}
function fbm2(x, y, oct = 4) { let s = 0, a = 0.5; for (let i = 0; i < oct; i++) { s += a * vnoise(x, y); x = x * 2.03 + 17.1; y = y * 2.03 + 9.3; a *= 0.5; } return s; }
const smooth = (a, b, x) => { const t = Math.min(Math.max((x - a) / (b - a), 0), 1); return t * t * (3 - 2 * t); };
const mulberry32 = (seed) => () => { seed |= 0; seed = (seed + 0x6D2B79F5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
const catmull = (p0, p1, p2, p3, t) => p1 + 0.5 * t * (p2 - p0 + t * (2 * p0 - 5 * p1 + 4 * p2 - p3 + t * (3 * (p1 - p2) + p3 - p0)));

const floorVert = /* glsl */`
varying vec3 vW;
void main() { vec4 w = modelMatrix * vec4(position, 1.0); vW = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }`;

const floorFrag = /* glsl */`
varying vec3 vW;
float line(vec2 p, float s) { vec2 q = p / s; vec2 d = abs(fract(q - 0.5) - 0.5) / fwidth(q); return 1.0 - clamp(min(d.x, d.y), 0.0, 1.0); }
void main() {
  float r = length(vW.xz);
  float fade = 1.0 - smoothstep(18.0, 60.0, r);
  float minor = line(vW.xz, 2.0) * 0.35;
  float major = line(vW.xz, 8.0);
  vec3 c = vec3(0.16, 0.165, 0.17) * max(minor, major) * fade;
  float glow = (1.0 - smoothstep(16.0, 34.0, r)) * 0.05;
  gl_FragColor = vec4(c + vec3(0.16, 0.16, 0.17) * glow, max(max(minor, major) * fade, glow * 4.0));
}`;

// small helpers for building models
const M = {
  std: (color, o = {}) => new THREE.MeshStandardMaterial({ color, roughness: 0.55, metalness: 0.35, ...o }),
  glow: (color, intensity = 1.2, o = {}) => new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: intensity, roughness: 0.4, ...o }),
};
const mesh = (geo, mat, x = 0, y = 0, z = 0) => { const m = new THREE.Mesh(geo, mat); m.position.set(x, y, z); return m; };

export class Scene3D {
  constructor(container) {
    this.container = container;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
    this.renderer.setClearColor(BG, 0);
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.0;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.Fog(BG, 55, 110);
    this.camera = new THREE.PerspectiveCamera(40, 1, 0.1, 400);
    this.camera.position.set(6, 30, 36);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(0, 1, 1.5);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.maxPolarAngle = Math.PI * 0.46;
    this.controls.minDistance = 10;
    this.controls.maxDistance = 90;
    this.controls.mouseButtons = { LEFT: null, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.ROTATE };
    this.controls.update();
    this.autoFrame = true;
    this.controls.addEventListener('start', () => { this.autoFrame = false; this.camAnim = null; });

    // late-afternoon sun from the west with soft shadows; sky and ground bounce from the hemisphere light
    this.scene.add(new THREE.HemisphereLight(0xc4d6e8, 0x4a3d2c, 1.2));
    const sun = new THREE.DirectionalLight(0xfff0d8, 3.1);
    sun.position.set(-22, 26, 12);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    Object.assign(sun.shadow.camera, { left: -24, right: 24, top: 24, bottom: -24, near: 1, far: 90 });
    sun.shadow.bias = -0.0004;
    sun.shadow.normalBias = 0.03;
    const fill = new THREE.DirectionalLight(0xd8e4f0, 0.35);
    fill.position.set(18, 10, -14);
    this.scene.add(sun, fill);
    this.tu = { uOverlay: { value: null }, uHScale: { value: H_SCALE }, uGrid: { value: 0.1 } };

    this.ovCanvas = document.createElement('canvas');
    this.ovCanvas.width = this.ovCanvas.height = G * OV;
    this.ovCtx = this.ovCanvas.getContext('2d');
    this.ovTex = new THREE.CanvasTexture(this.ovCanvas);
    this.ovTex.flipY = false;
    this.ovTex.anisotropy = 8;
    this.tu.uOverlay.value = this.ovTex;

    this.unitGroup = new THREE.Group();
    this.ghostGroup = new THREE.Group();
    this.actorGroup = new THREE.Group();
    this.fxGroup = new THREE.Group();
    this.pathGroup = new THREE.Group();
    this.scene.add(this.unitGroup, this.ghostGroup, this.actorGroup, this.fxGroup, this.pathGroup);
    this.effects = [];
    this.anims = [];        // per-frame unit animations: (t, dt) => void
    this.raycaster = new THREE.Raycaster();
    this.tags = [];
    this.onFrame = null;
    this.clock = new THREE.Clock();

    this._buildTextures();
    this._buildActorTemplates();
    this._buildFloor();
    new ResizeObserver(() => this.resize()).observe(container);
    this.resize();
    this.renderer.setAnimationLoop(() => this._loop());
  }

  resize() {
    const w = this.container.clientWidth, h = this.container.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h);
    // An overlay covering the right `r` pixels (the after-action report) shifts the projection centre
    // left, so the board is framed in the visible part while the canvas stays full size.
    const r = this.insetRight || 0;
    this.camera.aspect = (w + r) / h;
    if (r) this.camera.setViewOffset(w + r, h, r, 0, w, h); else this.camera.clearViewOffset();
    this.camera.updateProjectionMatrix();
    this.viewAspect = Math.max(0.3, (w - r) / h);
    if (this.autoFrame) this.frame(!!this._framed);   // animate refits after the first one
    this._framed = true;
  }

  setInset(px) {
    if ((this.insetRight || 0) === px) return;
    this.insetRight = px;
    this.resize();
  }

  // Fit the whole board in view from the default angle; animate unless told otherwise.
  frame(animate = true) {
    const dir = new THREE.Vector3(0.14, 0.66, 0.74).normalize();
    const vfov = THREE.MathUtils.degToRad(this.camera.fov);
    const aspect = this.viewAspect ?? this.camera.aspect;
    const hfov = 2 * Math.atan(Math.tan(vfov / 2) * aspect);
    const dist = THREE.MathUtils.clamp((aspect < 1.2 ? 21.5 : 19) / Math.sin(Math.min(vfov, hfov) / 2), 30, 90);
    const target = new THREE.Vector3(0, 0.6, 1.2);
    const pos = target.clone().addScaledVector(dir, dist);
    if (!animate) { this.camera.position.copy(pos); this.controls.target.copy(target); this.controls.update(); return; }
    this.camAnim = { t: 0, p0: this.camera.position.clone(), t0: this.controls.target.clone(), p1: pos, t1: target };
  }

  resetView() { this.autoFrame = true; this.frame(true); }

  setGrid(v) { this.tu.uGrid.value = v; }   // cell grid strength on the terrain (strong while placing units)

  // ------------------------------------------------------------- shared textures
  _buildTextures() {
    const radial = (stops, size = 64) => {
      const c = document.createElement('canvas');
      c.width = c.height = size;
      const g = c.getContext('2d');
      const grad = g.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
      for (const [o, col] of stops) grad.addColorStop(o, col);
      g.fillStyle = grad;
      g.fillRect(0, 0, size, size);
      const t = new THREE.CanvasTexture(c);
      t.colorSpace = THREE.SRGBColorSpace;
      return t;
    };
    this.glowTex = radial([[0, 'rgba(255,255,255,1)'], [0.22, 'rgba(255,255,255,0.5)'], [1, 'rgba(255,255,255,0)']]);
    this.shadowTex = radial([[0, 'rgba(0,0,0,0.75)'], [0.5, 'rgba(0,0,0,0.35)'], [1, 'rgba(0,0,0,0)']]);
    this.shadowGeo = new THREE.PlaneGeometry(1, 1).rotateX(-Math.PI / 2);
  }

  _glow(color, size, opacity = 1) {
    const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: this.glowTex, color, blending: THREE.AdditiveBlending, depthWrite: false, transparent: true, opacity }));
    s.scale.setScalar(size);
    return s;
  }

  _shadow(size, opacity = 0.8) {
    const m = new THREE.Mesh(this.shadowGeo, new THREE.MeshBasicMaterial({ map: this.shadowTex, transparent: true, depthWrite: false, opacity, toneMapped: false }));
    m.scale.setScalar(size);
    m.renderOrder = 1;
    return m;
  }

  // ------------------------------------------------------------- terrain
  _buildFloor() {
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(240, 240).rotateX(-Math.PI / 2),
      new THREE.ShaderMaterial({ vertexShader: floorVert, fragmentShader: floorFrag, transparent: true, depthWrite: false }));
    floor.position.y = FLOOR_Y - 0.02;
    floor.renderOrder = -1;
    this.scene.add(floor);
  }

  setMap(map) {
    this.map = map;
    this.heights = map.heights;
    this.baseXZ = (() => { const [x, y] = cellXY(map.base_cell); return [x - 15.5, y - 15.5]; })();
    if (this.terrain) { this.scene.remove(this.terrain); this.terrain.geometry.dispose(); }
    if (this.plinth) { this.scene.remove(this.plinth); this.plinth.traverse((o) => o.geometry?.dispose()); }
    const R = 160;
    const geo = new THREE.BufferGeometry();
    const pos = new Float32Array(R * R * 3), uv = new Float32Array(R * R * 2), hv = new Float32Array(R * R);
    for (let j = 0; j < R; j++) for (let i = 0; i < R; i++) {
      const wx = -16 + 32 * i / (R - 1), wz = -16 + 32 * j / (R - 1);
      const k = j * R + i;
      hv[k] = this.heightAt(wx, wz);
      pos[3 * k] = wx; pos[3 * k + 1] = hv[k]; pos[3 * k + 2] = wz;
      uv[2 * k] = (wx + 16) / 32; uv[2 * k + 1] = (wz + 16) / 32;
    }
    const idx = [];
    for (let j = 0; j < R - 1; j++) for (let i = 0; i < R - 1; i++) {
      const a = j * R + i, b = a + 1, c = a + R, d = c + 1;
      idx.push(a, c, b, b, c, d);
    }
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
    geo.setAttribute('ao', new THREE.BufferAttribute(this._bakeOcclusion(hv, R), 1));
    geo.setIndex(idx);
    geo.computeVertexNormals();
    const mat = new THREE.MeshStandardMaterial({ roughness: 0.94, metalness: 0 });
    mat.onBeforeCompile = (sh) => {
      Object.assign(sh.uniforms, this.tu);
      sh.vertexShader = 'attribute float ao;\nvarying vec3 vWP; varying vec3 vWN; varying float vAo;\n' + sh.vertexShader.replace(
        '#include <begin_vertex>',
        '#include <begin_vertex>\n  vWP = (modelMatrix * vec4(transformed, 1.0)).xyz; vWN = normalize(mat3(modelMatrix) * objectNormal); vAo = ao;');
      sh.fragmentShader = sh.fragmentShader
        .replace('#include <common>', '#include <common>\n' + TERRAIN_PARS)
        .replace('#include <map_fragment>', TERRAIN_ALBEDO)
        .replace('#include <normal_fragment_maps>', '#include <normal_fragment_maps>\n  normal = tPerturb(-vViewPosition, normal, vec2(dFdx(bumpH), dFdy(bumpH)));')
        .replace('#include <emissivemap_fragment>', '#include <emissivemap_fragment>\n  totalEmissiveRadiance += ovc * ov.a * ov.a * 0.75;');
    };
    mat.customProgramCacheKey = () => 'redcell-terrain';
    this.terrain = new THREE.Mesh(geo, mat);
    this.terrain.castShadow = this.terrain.receiveShadow = true;
    this.scene.add(this.terrain);
    this._buildPlinth(R);
    this._buildBase();
    this._buildTrees();
  }

  // Ambient occlusion baked per vertex: ground below its blurred surroundings (valleys, creases) gets
  // less sky light, ridges slightly more.
  _bakeOcclusion(hv, R) {
    const blur = (src) => {
      const r = 7, tmp = new Float32Array(R * R), out = new Float32Array(R * R);
      for (let j = 0; j < R; j++) for (let i = 0; i < R; i++) {
        let s = 0, n = 0;
        for (let d = -r; d <= r; d++) { const ii = Math.min(R - 1, Math.max(0, i + d)); s += src[j * R + ii]; n++; }
        tmp[j * R + i] = s / n;
      }
      for (let j = 0; j < R; j++) for (let i = 0; i < R; i++) {
        let s = 0, n = 0;
        for (let d = -r; d <= r; d++) { const jj = Math.min(R - 1, Math.max(0, j + d)); s += tmp[jj * R + i]; n++; }
        out[j * R + i] = s / n;
      }
      return out;
    };
    const b = blur(blur(hv));
    const ao = new Float32Array(R * R);
    for (let k = 0; k < R * R; k++) {
      const cavity = b[k] - hv[k];
      ao[k] = cavity > 0 ? Math.max(0.5, 1 - cavity * 0.55) : Math.min(1.1, 1 - cavity * 0.08);
    }
    return ao;
  }

  // Conifer stands in forest patches on gentle, low-to-mid slopes, kept off the base and out of unit cells.
  _buildTrees() {
    if (this.trees) { this.scene.remove(this.trees.canopy, this.trees.trunk); this.trees.canopy.dispose(); this.trees.trunk.dispose(); }
    const rng = mulberry32(1000 + (this.map.id ?? 0));
    const spots = [];
    const [bx, bz] = this.baseXZ;
    for (let t = 0; t < 12000 && spots.length < 650; t++) {
      const x = -15.7 + rng() * 31.4, z = -15.7 + rng() * 31.4;
      const y = this.heightAt(x, z), hN = y / H_SCALE;
      if (hN < 0.03 || hN > 0.6 || Math.hypot(x - bx, z - bz) < 3.0) continue;
      const slope = Math.hypot(this.heightAt(x + 0.3, z) - this.heightAt(x - 0.3, z), this.heightAt(x, z + 0.3) - this.heightAt(x, z - 0.3)) / 0.6;
      if (slope > 0.75) continue;
      if (rng() > smooth(0.46, 0.66, fbm2(x * 0.2 + 50, z * 0.2 + 20)) * 0.85) continue;
      spots.push({ x, y: y - 0.04, z, s: 0.7 + rng() * 0.6, r: rng() * Math.PI * 2, c: rng() });
    }
    const canopyGeo = new THREE.ConeGeometry(0.19, 0.62, 7).translate(0, 0.47, 0);
    const trunkGeo = new THREE.CylinderGeometry(0.028, 0.042, 0.22, 5).translate(0, 0.11, 0);
    const canopy = new THREE.InstancedMesh(canopyGeo, new THREE.MeshStandardMaterial({ roughness: 0.9, flatShading: true }), spots.length);
    const trunk = new THREE.InstancedMesh(trunkGeo, new THREE.MeshStandardMaterial({ color: 0x3b2a1c, roughness: 1 }), spots.length);
    const col = new THREE.Color();
    spots.forEach((p, i) => {
      canopy.setColorAt(i, col.setHSL(0.24 + p.c * 0.09, 0.32 + p.c * 0.18, 0.11 + p.c * 0.08));
    });
    canopy.castShadow = trunk.castShadow = true;
    canopy.receiveShadow = true;
    this.trees = { canopy, trunk, spots };
    this._placeTrees([]);
    this.scene.add(canopy, trunk);
  }

  _placeTrees(clear) {   // clear: [[x, z], ...] unit positions to keep free of trees
    if (!this.trees) return;
    const m = new THREE.Matrix4(), q = new THREE.Quaternion(), up = new THREE.Vector3(0, 1, 0), v = new THREE.Vector3(), sc = new THREE.Vector3();
    this.trees.spots.forEach((p, i) => {
      const hidden = clear.some(([x, z]) => Math.abs(x - p.x) < 0.75 && Math.abs(z - p.z) < 0.75);
      sc.setScalar(hidden ? 0 : p.s);
      m.compose(v.set(p.x, p.y, p.z), q.setFromAxisAngle(up, p.r), sc);
      this.trees.canopy.setMatrixAt(i, m);
      this.trees.trunk.setMatrixAt(i, m);
    });
    this.trees.canopy.instanceMatrix.needsUpdate = this.trees.trunk.instanceMatrix.needsUpdate = true;
  }

  // Solid sides under the terrain so the board reads as a physical table-top slab.
  _buildPlinth(R) {
    const g = new THREE.Group();
    const top = [];
    for (let i = 0; i < R; i++) top.push([-16 + 32 * i / (R - 1), -16]);
    for (let i = 1; i < R; i++) top.push([16, -16 + 32 * i / (R - 1)]);
    for (let i = R - 2; i >= 0; i--) top.push([-16 + 32 * i / (R - 1), 16]);
    for (let i = R - 2; i >= 1; i--) top.push([-16, -16 + 32 * i / (R - 1)]);
    top.push(top[0]);
    const n = top.length, K = 44;
    const pos = new Float32Array(n * K * 3), col = new Float32Array(n * K * 3);
    const rimPts = [];
    const c = new THREE.Color();
    const strata = [[0.50, 0.41, 0.29], [0.43, 0.30, 0.21], [0.36, 0.35, 0.33], [0.56, 0.48, 0.36], [0.40, 0.33, 0.25]];
    top.forEach(([x, z], i) => {
      const yt = this.heightAt(x, z);
      rimPts.push(new THREE.Vector3(x, yt + 0.01, z));
      for (let k = 0; k < K; k++) {
        const y = yt + (FLOOR_Y - yt) * (k / (K - 1));
        const o = (i * K + k) * 3;
        pos.set([x, y, z], o);
        const depth = yt - y;
        let rgb;
        if (depth < 0.18) rgb = [0.20, 0.15, 0.10];                 // topsoil
        else {
          const band = Math.floor((y + 2 + (vnoise(x * 0.35, z * 0.35) - 0.5) * 0.6) * 1.3);
          rgb = strata[((band % strata.length) + strata.length) % strata.length];
        }
        const shade = 0.85 + 0.15 * vnoise(x * 3, y * 6);
        c.setRGB(rgb[0] * shade, rgb[1] * shade, rgb[2] * shade, THREE.SRGBColorSpace);
        col.set([c.r, c.g, c.b], o);
      }
    });
    const idx = [];
    for (let i = 0; i < n - 1; i++) for (let k = 0; k < K - 1; k++) {
      const a = i * K + k, b = a + 1, cc = a + K, d = cc + 1;
      idx.push(a, b, cc, cc, b, d);
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
    geo.setIndex(idx);
    geo.computeVertexNormals();
    const sides = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ vertexColors: true, side: THREE.DoubleSide, roughness: 1 }));
    sides.receiveShadow = true;
    g.add(sides);
    g.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(rimPts), new THREE.LineBasicMaterial({ color: 0x6a6d72, transparent: true, opacity: 0.8 })));
    const bottom = [[-16, -16], [16, -16], [16, 16], [-16, 16], [-16, -16]].map(([x, z]) => new THREE.Vector3(x, FLOOR_Y, z));
    g.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(bottom), new THREE.LineBasicMaterial({ color: 0x6a6d72, transparent: true, opacity: 0.5 })));
    // corner markers
    for (const [x, z] of [[-16, -16], [16, -16], [16, 16], [-16, 16]]) {
      const post = mesh(new THREE.BoxGeometry(0.12, this.heightAt(x, z) - FLOOR_Y + 0.3, 0.12), new THREE.MeshBasicMaterial({ color: 0x3c3f43 }), x, (this.heightAt(x, z) + FLOOR_Y + 0.3) / 2, z);
      g.add(post);
    }
    this.plinth = g;
    this.scene.add(g);
  }

  heightAt(wx, wz) {
    // Catmull-Rom through the 32x32 game heights (smooth hills instead of bilinear facets) ...
    const fx = Math.min(Math.max(wx + 15.5, 0), G - 1), fz = Math.min(Math.max(wz + 15.5, 0), G - 1);
    const x0 = Math.floor(fx), z0 = Math.floor(fz), tx = fx - x0, tz = fz - z0, h = this.heights;
    const at = (x, z) => h[Math.min(G - 1, Math.max(0, z)) * G + Math.min(G - 1, Math.max(0, x))];
    const row = (z) => catmull(at(x0 - 1, z), at(x0, z), at(x0 + 1, z), at(x0 + 2, z), tx);
    const v = Math.max(0, catmull(row(z0 - 1), row(z0), row(z0 + 1), row(z0 + 2), tz));
    // ... plus fine relief that grows with altitude, faded out around the base so its pad sits flat.
    // Units, drones and labels all use this function, so everything stays seated on the visible ground.
    const [bx, bz] = this.baseXZ || [99, 99];
    const fade = smooth(2.6, 3.8, Math.hypot(wx - bx, wz - bz));
    const detail = (fbm2(wx * 0.55, wz * 0.55) - 0.47) * (0.12 + 0.45 * v) * fade;
    return v * H_SCALE + detail;
  }

  cellWorld(c, lift = 0) {
    const [x, y] = cellXY(c);
    const wx = x - 15.5, wz = y - 15.5;
    return new THREE.Vector3(wx, this.heightAt(wx, wz) + lift, wz);
  }

  pickCell(ev) {
    if (!this.terrain) return -1;
    const r = this.renderer.domElement.getBoundingClientRect();
    const p = new THREE.Vector2(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
    this.raycaster.setFromCamera(p, this.camera);
    const hit = this.raycaster.intersectObject(this.terrain)[0];
    if (!hit) return -1;
    const x = Math.floor(hit.point.x + 16), y = Math.floor(hit.point.z + 16);
    return x >= 0 && x < G && y >= 0 && y < G ? y * G + x : -1;
  }

  // ------------------------------------------------------------- overlay
  // layers: [{type:'mask', data, color, alpha, edge}, {type:'heat', data, color}, {type:'cells', cells, color, alpha}, ...]
  drawOverlay(layers) {
    const ctx = this.ovCtx, S = OV;
    ctx.clearRect(0, 0, G * S, G * S);
    for (const L of layers) {
      if (!L) continue;
      if (L.type === 'mask') this._mask(L.data, L.color, L.alpha ?? 0.3, L.edge ?? 0.9);
      else if (L.type === 'heat') this._heat(L.data, L.color, L.max ?? 0.8);
      else if (L.type === 'hatch') this._hatch(L.data);
      else if (L.type === 'cells') {
        ctx.fillStyle = L.color; ctx.globalAlpha = L.alpha ?? 0.5;
        for (const c of L.cells) { const [x, y] = cellXY(c); const m = L.inset ?? 0; ctx.fillRect(x * S + m, y * S + m, S - 2 * m, S - 2 * m); }
        ctx.globalAlpha = 1;
      } else if (L.type === 'outline') {
        const [x, y] = cellXY(L.cell);
        ctx.strokeStyle = L.color; ctx.lineWidth = 3; ctx.globalAlpha = 1;
        ctx.strokeRect(x * S + 2, y * S + 2, S - 4, S - 4);
        ctx.globalAlpha = 0.18; ctx.fillStyle = L.color;
        ctx.fillRect(x * S + 2, y * S + 2, S - 4, S - 4);
        ctx.globalAlpha = 1;
      }
    }
    this.ovTex.needsUpdate = true;
  }

  _mask(data, color, alpha, edgeAlpha) {
    const ctx = this.ovCtx, S = OV;
    ctx.fillStyle = color; ctx.globalAlpha = alpha;
    for (let c = 0; c < G * G; c++) if (data[c]) { const [x, y] = cellXY(c); ctx.fillRect(x * S, y * S, S, S); }
    ctx.globalAlpha = edgeAlpha; ctx.strokeStyle = color; ctx.lineWidth = 2.5;
    ctx.beginPath();
    for (let c = 0; c < G * G; c++) {
      if (!data[c]) continue;
      const [x, y] = cellXY(c);
      const off = (dx, dy) => { const nx = x + dx, ny = y + dy; return nx < 0 || ny < 0 || nx >= G || ny >= G || !data[ny * G + nx]; };
      if (off(0, -1)) { ctx.moveTo(x * S, y * S); ctx.lineTo((x + 1) * S, y * S); }
      if (off(0, 1)) { ctx.moveTo(x * S, (y + 1) * S); ctx.lineTo((x + 1) * S, (y + 1) * S); }
      if (off(-1, 0)) { ctx.moveTo(x * S, y * S); ctx.lineTo(x * S, (y + 1) * S); }
      if (off(1, 0)) { ctx.moveTo((x + 1) * S, y * S); ctx.lineTo((x + 1) * S, (y + 1) * S); }
    }
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  _heat(data, color, maxA) {
    const ctx = this.ovCtx, S = OV;
    ctx.fillStyle = color;
    for (let c = 0; c < G * G; c++) {
      const v = data[c];
      if (v <= 0.02) continue;
      const [x, y] = cellXY(c);
      ctx.globalAlpha = Math.min(maxA, 0.15 + Math.sqrt(v) * maxA);
      ctx.fillRect(x * S, y * S, S, S);
    }
    ctx.globalAlpha = 1;
  }

  _hatch(data) {
    const ctx = this.ovCtx, S = OV;
    ctx.strokeStyle = '#5d646c'; ctx.globalAlpha = 0.6; ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let c = 0; c < G * G; c++) {
      if (!data[c]) continue;
      const [x, y] = cellXY(c);
      for (let k = -S; k < S; k += 8) { ctx.moveTo(x * S + Math.max(0, k), y * S + Math.max(0, -k)); ctx.lineTo(x * S + Math.min(S, S + k), y * S + Math.min(S, S - k)); }
    }
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  // ------------------------------------------------------------- base
  _buildBase() {
    if (this.baseGroup) this.scene.remove(this.baseGroup);
    const g = new THREE.Group();
    g.position.copy(this.cellWorld(this.map.base_cell));
    const concrete = M.std(0x3c4651, { roughness: 0.9, metalness: 0.05 });
    const wall = M.std(0xc3ccd5, { roughness: 0.6, metalness: 0.15 });
    const roof = M.std(0x6d7a87, { roughness: 0.5, metalness: 0.5 });
    const window_ = M.glow(0xbfe9ff, 1.6);

    g.add(this._shadow(5.2, 0.55));
    const apron = mesh(new THREE.CylinderGeometry(1.75, 1.95, 0.3, 8), concrete, 0, 0.08, 0);
    apron.rotation.y = Math.PI / 8;
    g.add(apron);
    const stripe = mesh(new THREE.TorusGeometry(1.62, 0.025, 4, 8), M.glow(0xe8eef4, 0.6), 0, 0.24, 0);
    stripe.rotation.set(Math.PI / 2, 0, Math.PI / 8);
    g.add(stripe);

    // command building with a lit window band
    const hq = new THREE.Group();
    hq.add(mesh(new THREE.BoxGeometry(1.0, 0.55, 0.7), wall, 0, 0.5, 0));
    hq.add(mesh(new THREE.BoxGeometry(1.02, 0.08, 0.72), roof, 0, 0.8, 0));
    hq.add(mesh(new THREE.BoxGeometry(1.01, 0.09, 0.71), window_, 0, 0.58, 0));
    hq.add(mesh(new THREE.BoxGeometry(0.3, 0.12, 0.3), roof, 0.25, 0.9, 0.1));
    hq.position.set(-0.2, 0, 0.35);
    g.add(hq);

    // hangar (half-cylinder)
    const hangar = mesh(new THREE.CylinderGeometry(0.42, 0.42, 1.1, 16, 1, false, 0, Math.PI), roof, 0.75, 0.23, -0.55);
    hangar.rotation.set(0, 0, Math.PI / 2);
    hangar.rotation.order = 'ZYX';
    g.add(hangar);
    g.add(mesh(new THREE.BoxGeometry(0.06, 0.32, 0.6), M.std(0x1b2229), 1.31, 0.4, -0.55));

    // radome on a tower
    g.add(mesh(new THREE.CylinderGeometry(0.16, 0.2, 0.55, 8), wall, -0.9, 0.5, -0.55));
    g.add(mesh(new THREE.SphereGeometry(0.3, 24, 16, 0, Math.PI * 2, 0, Math.PI * 0.62), M.std(0xf0f4f8, { roughness: 0.35 }), -0.9, 0.82, -0.55));

    // helipad
    const heli = mesh(new THREE.CylinderGeometry(0.42, 0.42, 0.03, 24), M.std(0x2a323b, { roughness: 0.95 }), 0.8, 0.25, 0.75);
    g.add(heli);
    const hring = mesh(new THREE.TorusGeometry(0.34, 0.02, 4, 32), M.glow(0xffd166, 0.7), 0.8, 0.27, 0.75);
    hring.rotation.x = Math.PI / 2;
    g.add(hring);

    // comms mast with a blinking beacon
    g.add(mesh(new THREE.CylinderGeometry(0.03, 0.05, 1.6, 6), wall, 0.15, 1.6, 0.25));
    g.add(mesh(new THREE.BoxGeometry(0.28, 0.02, 0.02), wall, 0.15, 2.1, 0.25));
    this.baseBeacon = mesh(new THREE.SphereGeometry(0.07, 12, 8), new THREE.MeshBasicMaterial({ color: 0xff4d5e }), 0.15, 2.42, 0.25);
    this.baseBeaconGlow = this._glow(0xff4d5e, 0.9);
    this.baseBeacon.add(this.baseBeaconGlow);
    g.add(this.baseBeacon);

    // perimeter fence posts
    for (let i = 0; i < 16; i++) {
      const a = (i / 16) * Math.PI * 2;
      g.add(mesh(new THREE.BoxGeometry(0.04, 0.22, 0.04), roof, Math.cos(a) * 1.82, 0.32, Math.sin(a) * 1.82));
    }

    this.baseRing = new THREE.Mesh(new THREE.RingGeometry(2.0, 2.12, 64), new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.5, side: THREE.DoubleSide, depthWrite: false }));
    this.baseRing.rotation.x = -Math.PI / 2; this.baseRing.position.y = 0.3;
    g.add(this.baseRing);
    g.traverse((o) => { if (o.isMesh && !o.material.transparent) o.castShadow = o.receiveShadow = true; });
    this.baseGroup = g;
    this.scene.add(g);
  }

  // ------------------------------------------------------------- defensive units
  setUnits(layout, ghost = null) {
    this.unitGroup.clear();
    this.anims = [];
    for (const k of ['radar', 'jammer', 'pad']) for (const c of layout[k]) {
      const g = this._unitMesh(k, 1, this.anims);
      g.position.copy(this.cellWorld(c));
      g.rotation.y = ((c * 2654435761) % 360) * Math.PI / 180;   // stable per-cell variety
      this.unitGroup.add(g);
    }
    const clear = ['radar', 'jammer', 'pad'].flatMap((k) => layout[k].map((c) => { const [x, y] = cellXY(c); return [x - 15.5, y - 15.5]; }));
    this._placeTrees(clear);
    this.setGhost(ghost?.kind, ghost?.cell ?? -1);
  }

  // Translucent preview of the unit the player is about to place.
  setGhost(kind, cell) {
    if (this._ghostKey === `${kind}:${cell}`) return;
    this._ghostKey = `${kind}:${cell}`;
    this.ghostGroup.clear();
    this.ghostAnims = [];
    if (!kind || cell < 0) return;
    const g = this._unitMesh(kind, 0.45, this.ghostAnims);
    g.position.copy(this.cellWorld(cell));
    this.ghostGroup.add(g);
  }

  _unitMesh(kind, opacity, anims) {
    const g = new THREE.Group();
    const col = new THREE.Color(COLORS[kind]);
    const tr = opacity < 1;
    const o = tr ? { transparent: true, opacity, depthWrite: false } : {};
    const accent = M.glow(col, 1.3, o);
    const body = M.std(0x56626e, o);
    const dark = M.std(0x222a33, { roughness: 0.8, ...o });
    const light = M.std(0xd5dde4, { roughness: 0.35, metalness: 0.2, side: THREE.DoubleSide, ...o });

    if (!tr) g.add(this._shadow(1.7, 0.5));
    // shared hex pedestal with a team-colored light ring
    const ped = mesh(new THREE.CylinderGeometry(0.5, 0.56, 0.16, 6), dark, 0, 0.08, 0);
    g.add(ped);
    const ring = mesh(new THREE.TorusGeometry(0.5, 0.025, 4, 6), accent, 0, 0.165, 0);
    ring.rotation.x = Math.PI / 2;
    g.add(ring);
    if (!tr) {
      const halo = new THREE.Mesh(new THREE.RingGeometry(0.62, 0.78, 6).rotateX(-Math.PI / 2),
        new THREE.MeshBasicMaterial({ color: col, transparent: true, opacity: 0.35, blending: THREE.AdditiveBlending, depthWrite: false }));
      halo.position.y = 0.03;
      g.add(halo);
      anims.push((t) => { halo.material.opacity = 0.22 + 0.14 * Math.sin(t * 2.4 + g.id); });
    }

    if (kind === 'radar') {
      g.add(mesh(new THREE.CylinderGeometry(0.08, 0.14, 1.25, 8), body, 0, 0.78, 0));
      g.add(mesh(new THREE.CylinderGeometry(0.16, 0.16, 0.12, 12), dark, 0, 1.42, 0));
      const head = new THREE.Group();
      head.position.y = 1.5;
      head.add(mesh(new THREE.BoxGeometry(0.62, 0.06, 0.1), body, 0, 0.04, 0));
      head.add(mesh(new THREE.BoxGeometry(0.05, 0.36, 0.08), body, -0.29, 0.22, 0));
      head.add(mesh(new THREE.BoxGeometry(0.05, 0.36, 0.08), body, 0.29, 0.22, 0));
      // parabolic dish facing +z, tilted to look at the horizon
      const prof = [];
      for (let i = 0; i <= 10; i++) { const r = 0.5 * i / 10; prof.push(new THREE.Vector2(r, 0.32 * (r / 0.5) ** 2)); }
      const dish = new THREE.Group();
      const bowl = new THREE.Mesh(new THREE.LatheGeometry(prof, 28), light);
      dish.add(bowl);
      const rim = mesh(new THREE.TorusGeometry(0.5, 0.022, 6, 32), accent, 0, 0.32, 0);
      rim.rotation.x = Math.PI / 2;
      dish.add(rim);
      for (let i = 0; i < 3; i++) {
        const a = i * Math.PI * 2 / 3;
        const s = mesh(new THREE.CylinderGeometry(0.008, 0.008, 0.42), body, Math.cos(a) * 0.2, 0.35, Math.sin(a) * 0.2);
        s.lookAt(new THREE.Vector3(0, 0.55, 0));
        s.rotateX(Math.PI / 2);
        dish.add(s);
      }
      const feed = mesh(new THREE.SphereGeometry(0.05, 10, 8), accent, 0, 0.55, 0);
      dish.add(feed);
      dish.rotation.x = Math.PI / 2 - 0.35;
      dish.position.set(0, 0.28, 0);
      head.add(dish);
      g.add(head);
      // translucent sweep wedge on the ground
      if (!tr) {
        const sweep = new THREE.Mesh(new THREE.CircleGeometry(2.6, 24, -0.22, 0.44).rotateX(-Math.PI / 2),
          new THREE.MeshBasicMaterial({ color: col, transparent: true, opacity: 0.09, blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide }));
        sweep.position.y = 0.25;
        g.add(sweep);
        anims.push((t, dt) => { head.rotation.y += dt * 1.6; sweep.rotation.y = head.rotation.y + Math.PI / 2; });
      } else anims.push((t, dt) => { head.rotation.y += dt * 1.6; });
    } else if (kind === 'jammer') {
      g.add(mesh(new THREE.BoxGeometry(0.62, 0.36, 0.5), body, 0, 0.34, 0));
      g.add(mesh(new THREE.BoxGeometry(0.64, 0.05, 0.52), dark, 0, 0.54, 0));
      g.add(mesh(new THREE.BoxGeometry(0.5, 0.05, 0.01), accent, 0, 0.36, 0.255));
      g.add(mesh(new THREE.BoxGeometry(0.5, 0.05, 0.01), accent, 0, 0.36, -0.255));
      // phased-array panel and whip antennas
      const panel = mesh(new THREE.BoxGeometry(0.42, 0.36, 0.05), light, 0, 0.8, -0.1);
      panel.rotation.x = -0.3;
      g.add(panel);
      for (const [x, z, h] of [[-0.22, 0.16, 0.9], [0.22, 0.16, 0.75], [0.0, 0.18, 1.05]]) {
        g.add(mesh(new THREE.CylinderGeometry(0.012, 0.02, h, 5), body, x, 0.56 + h / 2, z));
        g.add(mesh(new THREE.SphereGeometry(0.04, 8, 6), accent, x, 0.56 + h, z));
      }
      // expanding interference rings
      if (!tr) {
        const waves = [0, 1, 2].map((i) => {
          const w = new THREE.Mesh(new THREE.TorusGeometry(1, 0.02, 4, 48),
            new THREE.MeshBasicMaterial({ color: col, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false }));
          w.rotation.x = Math.PI / 2;
          w.position.y = 1.2;
          g.add(w);
          return w;
        });
        anims.push((t) => waves.forEach((w, i) => {
          const k = (t * 0.6 + i / 3) % 1;
          w.scale.setScalar(0.2 + k * 1.6);
          w.position.y = 1.25 - k * 0.6;
          w.material.opacity = 0.75 * (1 - k);
        }));
      }
    } else {
      // interceptor launcher: turret ring, tilted 2x2 tube pod
      g.add(mesh(new THREE.CylinderGeometry(0.3, 0.36, 0.22, 12), body, 0, 0.27, 0));
      const turret = new THREE.Group();
      turret.position.y = 0.4;
      turret.add(mesh(new THREE.BoxGeometry(0.1, 0.32, 0.36), body, -0.32, 0.12, 0));
      turret.add(mesh(new THREE.BoxGeometry(0.1, 0.32, 0.36), body, 0.32, 0.12, 0));
      const pod = new THREE.Group();
      pod.position.y = 0.3;
      pod.rotation.x = -0.6;
      pod.add(mesh(new THREE.BoxGeometry(0.52, 0.42, 0.8), dark));
      for (const [x, y] of [[-0.12, -0.1], [0.12, -0.1], [-0.12, 0.1], [0.12, 0.1]]) {
        const tube = mesh(new THREE.CylinderGeometry(0.075, 0.075, 0.04, 12), body, x, y, 0.41);
        tube.rotation.x = Math.PI / 2;
        pod.add(tube);
        const tip = mesh(new THREE.ConeGeometry(0.05, 0.14, 10), accent, x, y, 0.47);
        tip.rotation.x = Math.PI / 2;
        pod.add(tip);
      }
      pod.add(mesh(new THREE.BoxGeometry(0.54, 0.03, 0.08), accent, 0, 0.215, 0.3));
      pod.add(mesh(new THREE.BoxGeometry(0.3, 0.06, 0.4), body, 0, 0.24, -0.1));
      turret.add(pod);
      g.add(turret);
      const base = g.rotation.y;
      anims.push((t) => { turret.rotation.y = base + Math.sin(t * 0.5 + g.id) * 0.9; });
    }
    g.scale.setScalar(1.15);
    if (!tr) g.traverse((o) => { if (o.isMesh && !o.material.transparent && o.material.blending !== THREE.AdditiveBlending) o.castShadow = o.receiveShadow = true; });
    return g;
  }

  // ------------------------------------------------------------- actors
  _buildActorTemplates() {
    this.droneGeo = {
      body: new THREE.BoxGeometry(0.3, 0.11, 0.42),
      canopy: new THREE.SphereGeometry(0.11, 12, 8, 0, Math.PI * 2, 0, Math.PI / 2),
      arm: new THREE.BoxGeometry(0.88, 0.035, 0.05),
      motor: new THREE.CylinderGeometry(0.05, 0.05, 0.08, 10),
      disc: new THREE.CylinderGeometry(0.19, 0.19, 0.005, 24),
      blade: new THREE.BoxGeometry(0.36, 0.008, 0.035),
      led: new THREE.SphereGeometry(0.035, 8, 6),
      lock: new THREE.RingGeometry(0.62, 0.7, 12, 1, -0.45, 0.9),
    };
    this.icptGeo = {
      body: new THREE.CylinderGeometry(0.045, 0.045, 0.42, 10).rotateX(Math.PI / 2),
      nose: new THREE.ConeGeometry(0.045, 0.14, 10).rotateX(Math.PI / 2).translate(0, 0, 0.28),
      fin: new THREE.BoxGeometry(0.16, 0.008, 0.08),
    };
    this.boomGeo = new THREE.SphereGeometry(1, 16, 12);
    this.debrisGeo = new THREE.BoxGeometry(0.07, 0.04, 0.07);
    this.shockGeo = new THREE.RingGeometry(0.85, 1, 48).rotateX(-Math.PI / 2);
  }

  resetActors(nDrones, nIcpt) {
    this.actorGroup.clear();
    this.fxGroup.clear();
    this.effects = [];
    this.drones = [];
    this.icpts = [];
    const D = this.droneGeo;
    for (let i = 0; i < nDrones; i++) {
      const g = new THREE.Group();
      g.rotation.order = 'YXZ';
      const frame = M.std(0x2b3036, { transparent: true, roughness: 0.5, metalness: 0.6 });
      const mat = new THREE.MeshStandardMaterial({ color: 0xff4d5e, emissive: 0xff2238, emissiveIntensity: 0.9, transparent: true });
      const disc = new THREE.MeshBasicMaterial({ color: 0xc8d2dc, transparent: true, opacity: 0.16, depthWrite: false });
      const blade = M.std(0x15191d, { transparent: true });
      const craft = new THREE.Group();
      craft.add(mesh(D.body, frame));
      craft.add(mesh(D.canopy, mat, 0, 0.05, 0.04));
      for (const s of [1, -1]) {
        const arm = mesh(D.arm, frame);
        arm.rotation.y = s * Math.PI / 4;
        craft.add(arm);
      }
      const rotors = [];
      for (const [x, z] of [[0.31, 0.31], [-0.31, 0.31], [0.31, -0.31], [-0.31, -0.31]]) {
        craft.add(mesh(D.motor, frame, x, 0.04, z));
        craft.add(mesh(D.disc, disc, x, 0.09, z));
        const b = mesh(D.blade, blade, x, 0.09, z);
        craft.add(b);
        rotors.push(b);
      }
      const led = mesh(D.led, mat, 0, 0, 0.23);
      craft.add(led);
      craft.add(mesh(D.led, mat, 0.31, -0.02, -0.31), mesh(D.led, mat, -0.31, -0.02, -0.31));
      craft.traverse((o) => { if (o.isMesh && o.material !== disc) o.castShadow = true; });
      g.add(craft);
      const glow = this._glow(0xff3048, 1.9);
      glow.position.y = 0.02;
      g.add(glow);
      // lock-on brackets shown while tracked by radar
      const lock = new THREE.Group();
      const lockMat = new THREE.MeshBasicMaterial({ color: 0x3fd0ff, transparent: true, opacity: 0.95, side: THREE.DoubleSide, depthWrite: false });
      for (let k = 0; k < 4; k++) { const r = new THREE.Mesh(D.lock, lockMat); r.rotation.z = k * Math.PI / 2; lock.add(r); }
      lock.rotation.x = -Math.PI / 2;
      lock.position.y = -0.2;
      g.add(lock);
      g.scale.setScalar(1.6);
      const shadow = this._shadow(1.5, 0.7);
      const trail = this._trail(0xff4d5e, 26);
      this.actorGroup.add(g, shadow, trail.line);
      this.drones.push({ g, craft, mat, frame, blade, disc, ring: lock, lockMat, trail, glow, rotors, shadow, last: null, yaw: 0, bank: 0 });
    }
    const I = this.icptGeo;
    for (let i = 0; i < nIcpt; i++) {
      const g = new THREE.Group();
      const hull = M.std(0xe4ebf1, { roughness: 0.3, metalness: 0.4 });
      const tip = M.glow(0x7fe3ff, 1.6);
      g.add(mesh(I.body, hull), mesh(I.nose, tip));
      for (let k = 0; k < 4; k++) {
        const f = mesh(I.fin, tip, 0, 0, -0.17);
        f.rotation.z = k * Math.PI / 2;
        f.position.set(Math.cos(k * Math.PI / 2) * 0.07, Math.sin(k * Math.PI / 2) * 0.07, -0.17);
        g.add(f);
      }
      const flame = this._glow(0x9fefff, 0.9);
      flame.position.z = -0.3;
      g.add(flame);
      g.add(this._glow(0x3fd0ff, 1.3, 0.6));
      g.scale.setScalar(1.4);
      const trail = this._trail(0x7fe3ff, 22);
      this.actorGroup.add(g, trail.line);
      this.icpts.push({ m: g, trail, flame });
    }
  }

  // Additive line whose vertex colors fade to black toward the tail.
  _trail(color, n = 14) {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(n * 3), 3));
    geo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(n * 3), 3));
    const line = new THREE.Line(geo, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, toneMapped: false }));
    line.frustumCulled = false;
    return { line, pts: [], n, color: new THREE.Color(color) };
  }

  _pushTrail(tr, p) {
    const last = tr.pts[tr.pts.length - 1];
    if (!last || last.distanceToSquared(p) > 0.02) { tr.pts.push(p.clone()); if (tr.pts.length > tr.n) tr.pts.shift(); }
    const pos = tr.line.geometry.attributes.position.array, col = tr.line.geometry.attributes.color.array;
    const L = tr.pts.length;
    for (let i = 0; i < tr.n; i++) {
      const q = i === L - 1 ? p : (tr.pts[Math.min(i, L - 1)] || p);
      pos[3 * i] = q.x; pos[3 * i + 1] = q.y; pos[3 * i + 2] = q.z;
      const k = L > 1 ? (i / (L - 1)) ** 1.6 : 1;
      col[3 * i] = tr.color.r * k; col[3 * i + 1] = tr.color.g * k; col[3 * i + 2] = tr.color.b * k;
    }
    tr.line.geometry.attributes.position.needsUpdate = true;
    tr.line.geometry.attributes.color.needsUpdate = true;
    tr.line.geometry.setDrawRange(0, L);
  }

  // drone state: {cellA, cellB, f, visible, detected, jammed}
  updateDrone(i, s) {
    const d = this.drones[i];
    d.g.visible = s.visible;
    d.trail.line.visible = s.visible;
    d.shadow.visible = s.visible;
    if (!s.visible) { d.trail.pts = []; d.last = null; return; }
    const now = performance.now();
    const a = this.cellWorld(s.cellA, DRONE_ALT), b = this.cellWorld(s.cellB, DRONE_ALT);
    const p = a.clone().lerp(b, s.f);
    p.y += Math.sin(now / 180 + i) * 0.05;
    d.g.position.copy(p);
    // heading and banking follow the direction of travel, smoothed
    const dx = b.x - a.x, dz = b.z - a.z;
    const moving = Math.abs(dx) + Math.abs(dz) > 0.01;
    if (moving) {
      const target = Math.atan2(dx, dz);
      let delta = target - d.yaw;
      delta = Math.atan2(Math.sin(delta), Math.cos(delta));
      d.yaw += delta * 0.18;
      d.bank = THREE.MathUtils.lerp(d.bank, THREE.MathUtils.clamp(-delta * 1.5, -0.5, 0.5), 0.15);
    } else d.bank *= 0.9;
    d.g.rotation.set(moving ? 0.22 : 0.05, d.yaw, d.bank);
    if (s.jammed) { d.g.rotation.x += (Math.random() - 0.5) * 0.25; d.g.rotation.z += (Math.random() - 0.5) * 0.25; }

    const alpha = s.detected ? 1 : 0.5;
    d.mat.opacity = d.frame.opacity = d.blade.opacity = alpha;
    d.disc.opacity = 0.16 * alpha;
    d.mat.emissiveIntensity = s.jammed ? 0.3 + Math.random() * 1.2 : (s.detected ? 1.3 : 0.45);
    d.mat.emissive.setHex(s.jammed ? 0xffa030 : 0xff2238);
    d.glow.material.opacity = s.detected ? 0.9 : 0.3;
    d.ring.visible = s.detected;
    d.ring.rotation.z = now / 600;
    d.ring.scale.setScalar(1 + 0.08 * Math.sin(now / 120));
    for (const r of d.rotors) r.rotation.y += 0.9;
    const gy = this.heightAt(p.x, p.z);
    d.shadow.position.set(p.x, gy + 0.03, p.z);
    d.shadow.material.opacity = 0.4 * alpha;
    this._pushTrail(d.trail, p);
  }

  updateIcpt(i, s) {
    const it = this.icpts[i];
    it.m.visible = s.visible;
    it.trail.line.visible = s.visible;
    if (!s.visible) { it.trail.pts = []; return; }
    const a = this.cellWorld(s.cellA, DRONE_ALT + 0.3), b = this.cellWorld(s.cellB, DRONE_ALT + 0.3);
    const p = a.clone().lerp(b, s.f);
    it.m.position.copy(p);
    if (a.distanceToSquared(b) > 0.001) it.m.lookAt(b);
    it.flame.scale.setScalar(0.7 + Math.random() * 0.5);
    this._pushTrail(it.trail, p);
  }

  explode(cell, color = 0xff9a3c, size = 1.4) {
    const p = this.cellWorld(cell, DRONE_ALT);
    const ball = new THREE.Mesh(this.boomGeo, new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.9, toneMapped: false }));
    ball.position.copy(p);
    ball.scale.setScalar(0.1);
    this.fxGroup.add(ball);
    this.effects.push({ m: ball, t: 0, dur: 0.7, run: (e, k) => { e.m.scale.setScalar(0.2 + size * Math.sqrt(k)); e.m.material.opacity = 0.9 * (1 - k) ** 1.5; } });
    const flash = this._glow(0xffe0b0, 0.1);
    flash.position.copy(p);
    this.fxGroup.add(flash);
    this.effects.push({ m: flash, t: 0, dur: 0.35, run: (e, k) => { e.m.scale.setScalar(5 * Math.sin(Math.PI * Math.min(1, k * 1.4))); e.m.material.opacity = 1 - k; } });
    const shock = new THREE.Mesh(this.shockGeo, new THREE.MeshBasicMaterial({ color: 0xffc27a, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide }));
    shock.position.copy(p);
    this.fxGroup.add(shock);
    this.effects.push({ m: shock, t: 0, dur: 0.6, run: (e, k) => { e.m.scale.setScalar(0.3 + 2.2 * k); e.m.material.opacity = 0.8 * (1 - k); } });
    const mat = new THREE.MeshStandardMaterial({ color: 0x2b3036, emissive: 0xff6a2a, emissiveIntensity: 1 });
    for (let n = 0; n < 10; n++) {
      const m = mesh(this.debrisGeo, mat, p.x, p.y, p.z);
      const v = new THREE.Vector3(Math.random() - 0.5, Math.random() * 0.8 + 0.3, Math.random() - 0.5).multiplyScalar(5);
      const spin = new THREE.Vector3(Math.random(), Math.random(), Math.random()).multiplyScalar(12);
      this.fxGroup.add(m);
      this.effects.push({ m, t: 0, dur: 1.1, run: (e, k, dt) => {
        v.y -= 9.8 * dt;
        e.m.position.addScaledVector(v, dt);
        e.m.rotation.x += spin.x * dt; e.m.rotation.y += spin.y * dt;
        const floor = this.heightAt(e.m.position.x, e.m.position.z) + 0.03;
        if (e.m.position.y < floor) { e.m.position.y = floor; v.multiplyScalar(0.3); v.y = Math.abs(v.y); }
        mat.emissiveIntensity = Math.max(0, 1 - k * 2);
      } });
    }
  }

  flashBase() {
    const base = this.cellWorld(this.map.base_cell);
    const col = new THREE.Mesh(new THREE.CylinderGeometry(0.35, 0.35, 16, 20, 1, true),
      new THREE.MeshBasicMaterial({ color: 0xff4d5e, transparent: true, opacity: 0.8, side: THREE.DoubleSide, blending: THREE.AdditiveBlending, depthWrite: false, toneMapped: false }));
    col.position.copy(base).y += 8;
    this.fxGroup.add(col);
    this.effects.push({ m: col, t: 0, dur: 1.2, run: (e, k) => { e.m.material.opacity = 0.8 * (1 - k); e.m.scale.set(1 + k * 1.5, 1, 1 + k * 1.5); } });
    const shock = new THREE.Mesh(this.shockGeo, new THREE.MeshBasicMaterial({ color: 0xff4d5e, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide }));
    shock.position.copy(base).y += 0.4;
    this.fxGroup.add(shock);
    this.effects.push({ m: shock, t: 0, dur: 1.0, run: (e, k) => { e.m.scale.setScalar(1 + 4 * k); e.m.material.opacity = 0.9 * (1 - k); } });
    this.baseAlarm = 1.6;
  }

  // ------------------------------------------------------------- paths
  clearPaths() { this.pathGroup.clear(); }

  drawPath(cells, color, { radius = 0.07, lift = 0.35, opacity = 0.95 } = {}) {
    if (cells.length < 2) return null;
    const pts = cells.map((c) => this.cellWorld(c, lift));
    const curve = new THREE.CatmullRomCurve3(pts, false, 'centripetal');
    const geo = new THREE.TubeGeometry(curve, Math.max(8, pts.length * 4), radius, 6, false);
    const m = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color, transparent: opacity < 1, opacity, depthWrite: opacity >= 1, toneMapped: false }));
    this.pathGroup.add(m);
    return m;
  }

  // ------------------------------------------------------------- html tags
  setTags(tags) { this.tags = tags; }   // [{el, cell, lift}]

  _updateTags() {
    const w = this.container.clientWidth, h = this.container.clientHeight;
    for (const t of this.tags) {
      const p = this.cellWorld(t.cell, t.lift ?? 2).project(this.camera);
      const vis = p.z < 1;
      t.el.style.display = vis ? '' : 'none';
      t.el.style.transform = `translate(-50%, -100%) translate(${(p.x * 0.5 + 0.5) * w}px, ${(-p.y * 0.5 + 0.5) * h}px)`;
    }
  }

  // If the first couple of seconds render slowly (integrated GPU, software rendering), drop to 1x pixel
  // ratio and cheaper, smaller shadows once. Gameplay is unaffected.
  _autoQuality() {
    const q = this._q || (this._q = { n: 0, t0: 0, done: false });
    if (q.done || !this.terrain) return;
    if (++q.n === 10) q.t0 = performance.now();   // skip shader-compile frames
    if (q.n <= 10) return;
    const elapsed = performance.now() - q.t0;
    if (q.n < 70 && elapsed < 2500) return;
    q.done = true;
    const fps = (q.n - 10) / (elapsed / 1000);
    if (fps < 30) {
      this.renderer.setPixelRatio(1);
      this.renderer.shadowMap.type = THREE.PCFShadowMap;
      this.scene.traverse((o) => {
        if (o.isDirectionalLight && o.castShadow) { o.shadow.mapSize.set(1024, 1024); o.shadow.map?.dispose(); o.shadow.map = null; }
        if (o.material) for (const m of [].concat(o.material)) m.needsUpdate = true;   // recompile for the new shadow type
      });
      this.resize();
      console.info(`RED CELL: ${Math.round(fps)} fps, using lower render quality`);
    }
  }

  // ------------------------------------------------------------- loop
  _loop() {
    const dt = Math.min(this.clock.getDelta(), 0.25);   // keeps replays near real time even at low frame rates
    this._autoQuality();
    const now = performance.now() / 1000;
    for (const a of this.anims) a(now, dt);
    for (const a of this.ghostAnims || []) a(now, dt);
    if (this.baseRing) {
      this.baseAlarm = Math.max(0, (this.baseAlarm || 0) - dt);
      const alarm = this.baseAlarm > 0;
      this.baseRing.material.color.setHex(alarm ? 0xff4d5e : 0xffffff);
      this.baseRing.material.opacity = alarm ? 0.5 + 0.5 * Math.sin(now * 20) : 0.3 + 0.2 * Math.sin(now * 2);
      const blink = (now % 1.4) < 0.18 || alarm;
      this.baseBeacon.visible = blink;
    }
    if (this.camAnim) {
      const c = this.camAnim;
      c.t = Math.min(1, c.t + dt / 0.9);
      const e = 1 - (1 - c.t) ** 3;
      this.camera.position.lerpVectors(c.p0, c.p1, e);
      this.controls.target.lerpVectors(c.t0, c.t1, e);
      if (c.t >= 1) this.camAnim = null;
    }
    this.effects = this.effects.filter((e) => {
      e.t += dt;
      const k = Math.min(1, e.t / e.dur);
      e.run(e, k, dt);
      if (k >= 1) { this.fxGroup.remove(e.m); return false; }
      return true;
    });
    if (this.onFrame) this.onFrame(dt);
    this.controls.update();
    this._updateTags();
    this.renderer.render(this.scene, this.camera);
  }
}
