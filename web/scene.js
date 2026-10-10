// 3D battlefield: terrain, overlays, defenses, drones, interceptors, effects.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

export const G = 32;
const H_SCALE = 7;          // world units for height 1.0
const DRONE_ALT = 0.7;      // visual flight height above terrain
const OV = 24;              // overlay canvas pixels per cell

export const COLORS = {
  radar: '#3fd0ff', jammer: '#ffb547', pad: '#5cf0a0', enemy: '#ff4d5e',
  interceptor: '#7fe3ff', base: '#e8eef4', astar: '#ffd166', naive: '#9aa7b4',
};

const cellXY = (c) => [c % G, Math.floor(c / G)];

const terrainVert = /* glsl */`
varying vec2 vUv; varying float vH; varying vec3 vN;
void main() {
  vUv = uv; vH = position.y;
  vN = normalize(mat3(modelMatrix) * normal);
  gl_Position = projectionMatrix * viewMatrix * modelMatrix * vec4(position, 1.0);
}`;

const terrainFrag = /* glsl */`
uniform sampler2D overlay; uniform float hScale; uniform vec3 lightDir;
varying vec2 vUv; varying float vH; varying vec3 vN;
void main() {
  float h = vH / hScale;
  vec3 c = mix(vec3(0.07, 0.10, 0.12), vec3(0.15, 0.20, 0.21), smoothstep(0.0, 0.45, h));
  c = mix(c, vec3(0.38, 0.43, 0.46), smoothstep(0.45, 0.80, h));
  c = mix(c, vec3(0.72, 0.75, 0.78), smoothstep(0.80, 0.86, h));
  float diff = clamp(dot(normalize(vN), normalize(lightDir)), 0.0, 1.0);
  c *= 0.45 + 0.8 * diff;
  float k = h * 14.0;
  float d = min(fract(k), 1.0 - fract(k));
  float contour = 1.0 - smoothstep(0.0, fwidth(k) * 1.1, d);
  c = mix(c, c * 1.6 + 0.03, contour * 0.35);
  vec2 g = vUv * 32.0;
  vec2 gd = min(fract(g), 1.0 - fract(g));
  vec2 gw = fwidth(g);
  float grid = max(1.0 - smoothstep(0.0, gw.x, gd.x), 1.0 - smoothstep(0.0, gw.y, gd.y));
  c = mix(c, vec3(0.35, 0.45, 0.5), grid * 0.10);
  vec4 o = texture2D(overlay, vUv);
  c = mix(c, o.rgb, o.a) + o.rgb * o.a * 0.25;
  gl_FragColor = vec4(c, 1.0);
}`;

export class Scene3D {
  constructor(container) {
    this.container = container;
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setClearColor(0x080b10);
    container.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.Fog(0x080b10, 45, 90);
    this.camera = new THREE.PerspectiveCamera(42, 1, 0.1, 300);
    this.camera.position.set(6, 30, 36);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(0, 1, 2);
    this.controls.enableDamping = true;
    this.controls.maxPolarAngle = Math.PI * 0.46;
    this.controls.minDistance = 12;
    this.controls.maxDistance = 70;
    this.controls.mouseButtons = { LEFT: null, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.ROTATE };
    this.controls.update();

    this.scene.add(new THREE.HemisphereLight(0x9fb8d0, 0x101418, 1.1));
    const sun = new THREE.DirectionalLight(0xffffff, 1.6);
    sun.position.set(-20, 30, 10);
    this.scene.add(sun);

    this.ovCanvas = document.createElement('canvas');
    this.ovCanvas.width = this.ovCanvas.height = G * OV;
    this.ovCtx = this.ovCanvas.getContext('2d');
    this.ovTex = new THREE.CanvasTexture(this.ovCanvas);
    this.ovTex.flipY = false;
    this.ovTex.anisotropy = 8;

    this.unitGroup = new THREE.Group();
    this.actorGroup = new THREE.Group();
    this.fxGroup = new THREE.Group();
    this.pathGroup = new THREE.Group();
    this.scene.add(this.unitGroup, this.actorGroup, this.fxGroup, this.pathGroup);
    this.effects = [];
    this.raycaster = new THREE.Raycaster();
    this.tags = [];
    this.onFrame = null;
    this.clock = new THREE.Clock();

    this._buildActorTemplates();
    new ResizeObserver(() => this.resize()).observe(container);
    this.resize();
    this.renderer.setAnimationLoop(() => this._loop());
  }

  resize() {
    const w = this.container.clientWidth, h = this.container.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  // ------------------------------------------------------------- terrain
  setMap(map) {
    this.map = map;
    this.heights = map.heights;
    if (this.terrain) { this.scene.remove(this.terrain); this.terrain.geometry.dispose(); }
    const R = 128;
    const geo = new THREE.BufferGeometry();
    const pos = new Float32Array(R * R * 3), uv = new Float32Array(R * R * 2);
    for (let j = 0; j < R; j++) for (let i = 0; i < R; i++) {
      const wx = -16 + 32 * i / (R - 1), wz = -16 + 32 * j / (R - 1);
      const k = j * R + i;
      pos[3 * k] = wx; pos[3 * k + 1] = this.heightAt(wx, wz); pos[3 * k + 2] = wz;
      uv[2 * k] = (wx + 16) / 32; uv[2 * k + 1] = (wz + 16) / 32;
    }
    const idx = [];
    for (let j = 0; j < R - 1; j++) for (let i = 0; i < R - 1; i++) {
      const a = j * R + i, b = a + 1, c = a + R, d = c + 1;
      idx.push(a, c, b, b, c, d);
    }
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
    geo.setIndex(idx);
    geo.computeVertexNormals();
    const mat = new THREE.ShaderMaterial({
      vertexShader: terrainVert, fragmentShader: terrainFrag,
      uniforms: { overlay: { value: this.ovTex }, hScale: { value: H_SCALE }, lightDir: { value: new THREE.Vector3(-0.6, 0.9, 0.3) } },
    });
    this.terrain = new THREE.Mesh(geo, mat);
    this.scene.add(this.terrain);
    this._buildBase();
  }

  heightAt(wx, wz) {
    const fx = Math.min(Math.max(wx + 15.5, 0), G - 1), fz = Math.min(Math.max(wz + 15.5, 0), G - 1);
    const x0 = Math.floor(fx), z0 = Math.floor(fz), x1 = Math.min(x0 + 1, G - 1), z1 = Math.min(z0 + 1, G - 1);
    const tx = fx - x0, tz = fz - z0, h = this.heights;
    const v = h[z0 * G + x0] * (1 - tx) * (1 - tz) + h[z0 * G + x1] * tx * (1 - tz) + h[z1 * G + x0] * (1 - tx) * tz + h[z1 * G + x1] * tx * tz;
    return v * H_SCALE;
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
    ctx.strokeStyle = '#c9d3dd'; ctx.globalAlpha = 0.35; ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let c = 0; c < G * G; c++) {
      if (!data[c]) continue;
      const [x, y] = cellXY(c);
      for (let k = -S; k < S; k += 8) { ctx.moveTo(x * S + Math.max(0, k), y * S + Math.max(0, -k)); ctx.lineTo(x * S + Math.min(S, S + k), y * S + Math.min(S, S - k)); }
    }
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  // ------------------------------------------------------------- units
  _buildBase() {
    if (this.baseGroup) this.scene.remove(this.baseGroup);
    const g = new THREE.Group();
    const p = this.cellWorld(this.map.base_cell);
    g.position.copy(p);
    const mat = new THREE.MeshStandardMaterial({ color: 0xc8d2dc, roughness: 0.6, metalness: 0.2 });
    const dark = new THREE.MeshStandardMaterial({ color: 0x4a5560, roughness: 0.8 });
    const pad = new THREE.Mesh(new THREE.CylinderGeometry(1.7, 1.9, 0.25, 6), dark);
    pad.position.y = 0.1; g.add(pad);
    [[0, 0, 0.9, 0.7, 0.9], [-0.8, 0.5, 0.5, 0.5, 0.6], [0.7, -0.6, 0.6, 0.4, 0.5]].forEach(([x, z, w, h, d]) => {
      const b = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat);
      b.position.set(x, 0.25 + h / 2, z); g.add(b);
    });
    const mast = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.04, 1.8), mat);
    mast.position.set(0, 1.1, 0); g.add(mast);
    this.baseBeacon = new THREE.Mesh(new THREE.SphereGeometry(0.12), new THREE.MeshBasicMaterial({ color: 0xffffff }));
    this.baseBeacon.position.set(0, 2.05, 0); g.add(this.baseBeacon);
    this.baseRing = new THREE.Mesh(new THREE.RingGeometry(1.9, 2.05, 48), new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.5, side: THREE.DoubleSide }));
    this.baseRing.rotation.x = -Math.PI / 2; this.baseRing.position.y = 0.3; g.add(this.baseRing);
    this.baseGroup = g;
    this.scene.add(g);
  }

  setUnits(layout, ghost = null) {
    this.unitGroup.clear();
    this.radarDishes = [];
    this.jammerCores = [];
    const add = (kind, c, opacity = 1) => {
      const g = this._unitMesh(kind, opacity);
      g.position.copy(this.cellWorld(c));
      this.unitGroup.add(g);
    };
    for (const k of ['radar', 'jammer', 'pad']) for (const c of layout[k]) add(k, c);
    if (ghost) add(ghost.kind, ghost.cell, 0.35);
  }

  _unitMesh(kind, opacity) {
    const g = new THREE.Group();
    const col = new THREE.Color(COLORS[kind]);
    const tr = opacity < 1;
    const emis = new THREE.MeshStandardMaterial({ color: col, emissive: col, emissiveIntensity: 0.8, transparent: tr, opacity });
    const body = new THREE.MeshStandardMaterial({ color: 0x5b6773, roughness: 0.7, transparent: tr, opacity });
    if (kind === 'radar') {
      const mast = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.12, 1.4), body);
      mast.position.y = 0.7; g.add(mast);
      const dish = new THREE.Group();
      const d = new THREE.Mesh(new THREE.BoxGeometry(0.9, 0.35, 0.06), emis);
      dish.add(d); dish.position.y = 1.45; g.add(dish);
      this.radarDishes.push(dish);
    } else if (kind === 'jammer') {
      const base = new THREE.Mesh(new THREE.CylinderGeometry(0.3, 0.35, 0.3, 8), body);
      base.position.y = 0.15; g.add(base);
      const core = new THREE.Mesh(new THREE.OctahedronGeometry(0.28), emis);
      core.position.y = 0.75; g.add(core);
      this.jammerCores.push(core);
    } else {
      const plat = new THREE.Mesh(new THREE.CylinderGeometry(0.62, 0.7, 0.18, 6), body);
      plat.position.y = 0.09; g.add(plat);
      for (let i = 0; i < 3; i++) {
        const m = new THREE.Mesh(new THREE.ConeGeometry(0.09, 0.42, 6), emis);
        const a = i * Math.PI * 2 / 3;
        m.position.set(Math.cos(a) * 0.3, 0.4, Math.sin(a) * 0.3);
        g.add(m);
      }
    }
    return g;
  }

  // ------------------------------------------------------------- actors
  _buildActorTemplates() {
    this.droneGeo = new THREE.OctahedronGeometry(0.3);
    this.rotorGeo = new THREE.CylinderGeometry(0.13, 0.13, 0.02, 12);
    this.ringGeo = new THREE.RingGeometry(0.45, 0.55, 32);
    this.icptGeo = new THREE.ConeGeometry(0.1, 0.5, 8);
    this.icptGeo.rotateX(Math.PI / 2);
    this.boomGeo = new THREE.SphereGeometry(1, 16, 12);
    const c = document.createElement('canvas');
    c.width = c.height = 64;
    const g = c.getContext('2d');
    const grad = g.createRadialGradient(32, 32, 0, 32, 32, 32);
    grad.addColorStop(0, 'rgba(255,255,255,1)');
    grad.addColorStop(0.25, 'rgba(255,255,255,0.45)');
    grad.addColorStop(1, 'rgba(255,255,255,0)');
    g.fillStyle = grad;
    g.fillRect(0, 0, 64, 64);
    this.glowTex = new THREE.CanvasTexture(c);
  }

  _glow(color, size) {
    const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: this.glowTex, color, blending: THREE.AdditiveBlending, depthWrite: false, transparent: true }));
    s.scale.setScalar(size);
    return s;
  }

  resetActors(nDrones, nIcpt) {
    this.actorGroup.clear();
    this.fxGroup.clear();
    this.effects = [];
    this.drones = [];
    this.icpts = [];
    for (let i = 0; i < nDrones; i++) {
      const g = new THREE.Group();
      const mat = new THREE.MeshStandardMaterial({ color: 0xff4d5e, emissive: 0xff2238, emissiveIntensity: 0.9, transparent: true });
      const body = new THREE.Mesh(this.droneGeo, mat);
      body.scale.set(1, 0.35, 1.3);
      g.add(body);
      const glow = this._glow(0xff3048, 2.2);
      g.add(glow);
      g.scale.setScalar(1.35);
      for (const [x, z] of [[0.3, 0.3], [-0.3, 0.3], [0.3, -0.3], [-0.3, -0.3]]) {
        const r = new THREE.Mesh(this.rotorGeo, mat);
        r.position.set(x, 0.05, z); g.add(r);
      }
      const ring = new THREE.Mesh(this.ringGeo, new THREE.MeshBasicMaterial({ color: 0xff4d5e, transparent: true, opacity: 0.9, side: THREE.DoubleSide }));
      ring.rotation.x = -Math.PI / 2; ring.position.y = -0.25;
      g.add(ring);
      const trail = this._trail(0xff4d5e);
      this.actorGroup.add(g, trail.line);
      this.drones.push({ g, mat, ring, trail, glow });
    }
    for (let i = 0; i < nIcpt; i++) {
      const mat = new THREE.MeshStandardMaterial({ color: 0x7fe3ff, emissive: 0x3fd0ff, emissiveIntensity: 1.2 });
      const m = new THREE.Mesh(this.icptGeo, mat);
      m.add(this._glow(0x3fd0ff, 1.4));
      m.scale.setScalar(1.3);
      const trail = this._trail(0x7fe3ff);
      this.actorGroup.add(m, trail.line);
      this.icpts.push({ m, trail });
    }
  }

  _trail(color, n = 14) {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(n * 3), 3));
    const line = new THREE.Line(geo, new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.55 }));
    line.frustumCulled = false;
    return { line, pts: [], n };
  }

  _pushTrail(tr, p) {
    const last = tr.pts[tr.pts.length - 1];
    if (!last || last.distanceToSquared(p) > 0.02) { tr.pts.push(p.clone()); if (tr.pts.length > tr.n) tr.pts.shift(); }
    const arr = tr.line.geometry.attributes.position.array;
    for (let i = 0; i < tr.n; i++) {
      const q = tr.pts[Math.min(i, tr.pts.length - 1)] || p;
      arr[3 * i] = q.x; arr[3 * i + 1] = q.y; arr[3 * i + 2] = q.z;
    }
    tr.line.geometry.attributes.position.needsUpdate = true;
    tr.line.geometry.setDrawRange(0, tr.pts.length);
  }

  // drone state: {cellA, cellB, f, visible, detected, jammed}
  updateDrone(i, s) {
    const d = this.drones[i];
    d.g.visible = s.visible;
    d.trail.line.visible = s.visible;
    if (!s.visible) { d.trail.pts = []; return; }
    const a = this.cellWorld(s.cellA, DRONE_ALT), b = this.cellWorld(s.cellB, DRONE_ALT);
    const p = a.lerp(b, s.f);
    p.y += Math.sin(performance.now() / 180 + i) * 0.05;
    d.g.position.copy(p);
    const dx = b.x - this.cellWorld(s.cellA).x, dz = b.z - this.cellWorld(s.cellA).z;
    if (Math.abs(dx) + Math.abs(dz) > 0.01) d.g.rotation.y = Math.atan2(dx, dz);
    d.mat.opacity = s.detected ? 1 : 0.38;
    d.mat.emissiveIntensity = s.jammed ? 0.3 + Math.random() * 0.9 : (s.detected ? 1.0 : 0.35);
    d.ring.visible = s.detected;
    d.glow.material.opacity = s.detected ? 1 : 0.35;
    d.ring.scale.setScalar(1 + 0.15 * Math.sin(performance.now() / 120));
    this._pushTrail(d.trail, p);
  }

  updateIcpt(i, s) {
    const it = this.icpts[i];
    it.m.visible = s.visible;
    it.trail.line.visible = s.visible;
    if (!s.visible) { it.trail.pts = []; return; }
    const a = this.cellWorld(s.cellA, DRONE_ALT + 0.3), b = this.cellWorld(s.cellB, DRONE_ALT + 0.3);
    const p = a.clone().lerp(b, s.f);
    if (a.distanceToSquared(b) > 0.001) it.m.lookAt(b);
    it.m.position.copy(p);
    this._pushTrail(it.trail, p);
  }

  explode(cell, color = 0xff9a3c, size = 1.4) {
    const m = new THREE.Mesh(this.boomGeo, new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.9 }));
    m.position.copy(this.cellWorld(cell, DRONE_ALT));
    m.scale.setScalar(0.1);
    this.fxGroup.add(m);
    this.effects.push({ m, t: 0, dur: 0.7, size });
  }

  flashBase() {
    const m = new THREE.Mesh(new THREE.CylinderGeometry(0.3, 0.3, 14, 16, 1, true),
      new THREE.MeshBasicMaterial({ color: 0xff4d5e, transparent: true, opacity: 0.8, side: THREE.DoubleSide }));
    m.position.copy(this.cellWorld(this.map.base_cell, 7));
    this.fxGroup.add(m);
    this.effects.push({ m, t: 0, dur: 1.2, size: 0, column: true });
  }

  // ------------------------------------------------------------- paths
  clearPaths() { this.pathGroup.clear(); }

  drawPath(cells, color, { radius = 0.07, lift = 0.35, opacity = 0.95 } = {}) {
    if (cells.length < 2) return null;
    const pts = cells.map((c) => this.cellWorld(c, lift));
    const curve = new THREE.CatmullRomCurve3(pts, false, 'centripetal');
    const geo = new THREE.TubeGeometry(curve, Math.max(8, pts.length * 4), radius, 6, false);
    const mesh = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color, transparent: opacity < 1, opacity, depthWrite: opacity >= 1 }));
    this.pathGroup.add(mesh);
    return mesh;
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

  // ------------------------------------------------------------- loop
  _loop() {
    const dt = Math.min(this.clock.getDelta(), 0.1);
    const now = performance.now() / 1000;
    for (const d of this.radarDishes || []) d.rotation.y += dt * 1.8;
    for (const c of this.jammerCores || []) { c.rotation.y += dt * 2; c.scale.setScalar(1 + 0.12 * Math.sin(now * 6)); }
    if (this.baseRing) { this.baseRing.material.opacity = 0.35 + 0.25 * Math.sin(now * 2); }
    this.effects = this.effects.filter((e) => {
      e.t += dt;
      const k = e.t / e.dur;
      if (e.column) e.m.material.opacity = 0.8 * (1 - k);
      else { e.m.scale.setScalar(0.2 + e.size * Math.sqrt(k)); e.m.material.opacity = 0.9 * (1 - k); }
      if (k >= 1) { this.fxGroup.remove(e.m); return false; }
      return true;
    });
    if (this.onFrame) this.onFrame(dt);
    this.controls.update();
    this._updateTags();
    this.renderer.render(this.scene, this.camera);
  }
}
