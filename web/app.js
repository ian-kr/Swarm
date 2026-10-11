import { Scene3D, COLORS } from '/static/scene.js';
import { sfx } from '/static/sound.js';

const $ = (s) => document.querySelector(s);
const STEPS_PER_SEC = 5;
const UNIT_INFO = {
  radar: { name: 'Radar', desc: 'Sees drones in range with clear line of sight.', key: '1' },
  jammer: { name: 'Jammer', desc: 'Scrambles navigation: jammed drones drift.', key: '2' },
  pad: { name: 'Interceptor pad', desc: 'Three interceptors that chase detected drones.', key: '3' },
};
// Unit symbols in the MIL-STD-2525 style: friendly units sit in rectangular frames.
const FRAME = '<rect x="1.5" y="1.5" width="41" height="27" fill="currentColor" fill-opacity=".14" stroke="currentColor" stroke-width="2"/>';
const SYM = {
  radar: `<svg viewBox="0 0 44 30" aria-hidden="true">${FRAME}<path d="M14.5 21.5a9 9 0 0 1 12.7-12.7z" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/><path d="M20.9 15.1l7-7M24.5 24h8M28.5 19v5" fill="none" stroke="currentColor" stroke-width="2"/></svg>`,
  jammer: `<svg viewBox="0 0 44 30" aria-hidden="true">${FRAME}<text x="22" y="20.5" text-anchor="middle" font-family="Barlow Condensed, Arial Narrow, sans-serif" font-weight="700" font-size="15" fill="currentColor">EW</text></svg>`,
  pad: `<svg viewBox="0 0 44 30" aria-hidden="true">${FRAME}<path d="M12 28.5a10 8.5 0 0 1 20 0" fill="none" stroke="currentColor" stroke-width="2"/><path d="M22 20V6M18.5 10.5L22 6l3.5 4.5" fill="none" stroke="currentColor" stroke-width="2"/></svg>`,
};
// Hostile air track: a diamond frame with the drone number.
const trackSym = (n) => `<svg viewBox="0 0 26 26" aria-hidden="true"><path class="fr" d="M13 1.5L24.5 13 13 24.5 1.5 13z"/><text x="13" y="17">${n}</text><path class="x" d="M5 21L21 5"/></svg>`;
const PHASES = ['defend', 'attack', 'debrief'];
const PHASE_STATUS = { defend: 'Placing defenses', attack: 'Attack in progress', debrief: 'After-action review' };
const MON = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'];
const pad2 = (n) => String(n).padStart(2, '0');
const dtg = (d = new Date()) => `${pad2(d.getUTCDate())}${pad2(d.getUTCHours())}${pad2(d.getUTCMinutes())}Z${MON[d.getUTCMonth()]}${String(d.getUTCFullYear()).slice(2)}`;
const fmtBig = (n) => (n >= 1e6 ? `${(n / 1e6).toFixed(n >= 1e8 ? 0 : 1)}M` : n >= 1e3 ? `${Math.round(n / 1e3)}K` : String(n));

const S = {
  cfg: null, training: null, mapId: 0, map: null, tool: 'radar', diff: 'scripted', phase: 'defend',
  layout: { radar: [], jammer: [], pad: [] }, coverage: null, preview: null, hover: -1,
  result: null, replay: null, astarAnim: null, ov: { breach: true, traffic: false, astar: false, sensors: true },
  learn: { stages: [], idx: -1, chain: false, busy: false },
  focus: -1, focusManual: false, focusEndT: null, mindKey: '',
  mode: 'north', tactic: -1,
  net: { open: false, key: '', data: null, cache: new Map(), drawn: '', att: true },
};
const TACTIC_COLORS = ['#ff8a3d', '#c77dff', '#f4f4f4', '#ff5fa2'];
const DIRS = ['north', 'north-east', 'east', 'south-east', 'south', 'south-west', 'west', 'north-west', 'hover'];
const fmtPct = (x) => `${Math.round(x * 100)}%`;

const scene = new Scene3D($('#stage'));
// every request carries the current theater ("north" or "360")
const api = async (path, body) => {
  if (body) body = { ...body, mode: S.mode };
  else path += `${path.includes('?') ? '&' : '?'}mode=${encodeURIComponent(S.mode)}`;
  const r = await fetch(path, body ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) } : {});
  const j = await r.json();
  if (!r.ok) throw new Error(j.detail || r.statusText);
  return j;
};
const store = {
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch { /* storage unavailable */ } },
};

// ------------------------------------------------------------------ setup
// Everything that depends on the theater: maps, enemies and their measured threat, the timeline.
async function applyConfig() {
  [S.cfg, S.training] = await Promise.all([api('/api/config'), api('/api/training').catch(() => ({}))]);
  $('#budget-total').textContent = S.cfg.budget;
  const sel = $('#sel-map');
  sel.innerHTML = '';
  for (const m of S.cfg.maps) sel.add(new Option(m.name, m.id));

  const ms = $('#seg-mode');
  ms.innerHTML = '';
  for (const m of S.cfg.modes) {
    const b = document.createElement('button');
    b.dataset.mode = m.key;
    b.setAttribute('role', 'radio');
    b.setAttribute('aria-checked', m.key === S.mode);
    b.classList.toggle('on', m.key === S.mode);
    b.innerHTML = `${m.label}${m.trained ? '' : '<span class="sub">A* only</span>'}`;
    b.title = m.key === '360' ? 'Drones enter from every edge; the base sits mid-map' : 'Drones enter along the northern edge';
    b.onclick = () => setMode(m.key);
    ms.appendChild(b);
  }
  document.body.dataset.mode = S.mode;
  $('#intro-threat').textContent = S.mode === '360' ? 'from every direction' : 'from the north';

  // opposing force: threat meter = held-out breach rate against strong defenses, relative to the strongest
  const seg = $('#seg-diff');
  seg.innerHTML = '';
  const top = Math.max(...S.cfg.difficulties.map((d) => d.eval_breach || 0), 1e-6);
  for (const d of S.cfg.difficulties) {
    const b = document.createElement('button');
    b.dataset.key = d.key;
    b.disabled = !d.available;
    b.setAttribute('role', 'radio');
    const lvl = d.eval_breach == null ? 0 : Math.max(1, Math.round((4 * d.eval_breach) / top));
    const sub = !d.available ? 'not trained' : d.key === 'scripted' ? 'A* pathfinding' : 'reinforcement learning';
    b.innerHTML = `<span class="nm">${d.label}</span><span class="lvl" title="Threat level">${[0, 1, 2, 3].map((i) => `<i class="${i < lvl ? 'f' : ''}"></i>`).join('')}</span><span class="sub">${sub}</span>`;
    b.title = d.blurb;
    b.onclick = () => setDiff(d.key);
    seg.appendChild(b);
  }
  const saved = store.get('rc-diff');
  const best = [...S.cfg.difficulties].reverse().find((d) => d.available);
  setDiff(S.cfg.difficulties.find((d) => d.key === saved && d.available) ? saved : best.key);
}

async function setMode(m) {
  if (m === S.mode || S.phase === 'attack') return;
  S.mode = m;
  store.set('rc-mode', m);
  sfx.click();
  await applyConfig();
  await loadMap(0);
}

async function init() {
  S.mode = store.get('rc-mode') === '360' ? '360' : 'north';
  await applyConfig();
  $('#sel-map').onchange = () => loadMap(+$('#sel-map').value);

  const units = $('#units');
  for (const [k, u] of Object.entries(S.cfg.units)) {
    const info = UNIT_INFO[k];
    const b = document.createElement('button');
    b.className = 'unit';
    b.dataset.kind = k;
    b.setAttribute('role', 'radio');
    b.style.setProperty('--c', COLORS[k]);
    b.innerHTML = `<span class="sym">${SYM[k]}</span><span class="u-name">${info.name}</span>
      <span class="u-meta"><b>${u.cost}</b> pts · <span data-count></span></span><span class="u-desc">${info.desc}</span><kbd>${info.key}</kbd>`;
    b.onclick = () => setTool(k);
    units.appendChild(b);
  }
  for (const el of document.querySelectorAll('[data-sym]')) el.innerHTML = SYM[el.dataset.sym];
  setTool('radar');

  const cv = scene.renderer.domElement;
  cv.addEventListener('click', onClick);
  cv.addEventListener('pointermove', onMove);
  cv.addEventListener('pointerleave', () => { S.hover = -1; S.preview = null; scene.setGhost(null, -1); redraw(); });
  cv.addEventListener('contextmenu', (e) => e.preventDefault());
  $('#btn-clear').onclick = () => { S.layout = { radar: [], jammer: [], pad: [] }; layoutChanged(); };
  $('#btn-launch').onclick = launch;
  $('#btn-ai').onclick = loadAiLayout;
  $('#btn-edit').onclick = () => setPhase('defend');
  $('#btn-replay').onclick = () => startReplay();
  $('#btn-skip').onclick = () => { S.learn.chain = false; showDebrief(); };
  $('#btn-suggest').onclick = placeSuggested;
  $('#btn-learn').onclick = runLearn;
  $('#btn-play').onclick = togglePlay;
  $('#btn-hood').onclick = openHood;
  $('#btn-help').onclick = () => { $('#intro-dtg').textContent = dtg(); $('#dlg-intro').showModal(); };
  $('#btn-view').onclick = () => scene.resetView();
  const sb = $('#btn-sound');
  const syncSound = () => { sb.textContent = sfx.enabled ? 'Sound on' : 'Sound off'; sb.setAttribute('aria-pressed', sfx.enabled); };
  sb.onclick = () => { sfx.setEnabled(!sfx.enabled); syncSound(); if (!sfx.enabled) sfx.humStop(); };
  syncSound();
  $('#mind-close').onclick = () => setFocus(-1, true);
  $('#mind-net').onclick = () => openNet(true);
  $('#net-close').onclick = () => openNet(false);
  $('#net-att').onchange = (e) => { S.net.att = e.target.checked; S.net.drawn = ''; redraw(); };
  window.addEventListener('resize', () => { S.net.drawn = ''; });
  $('#pips').onclick = (e) => { const t = e.target.closest('.trk'); if (t) setFocus(+t.dataset.i, true); };
  $('#scrub').oninput = (e) => { if (S.replay) { S.replay.t = +e.target.value; S.replay.lastIdx = Math.floor(S.replay.t); S.replay.done = false; } };
  for (const b of document.querySelectorAll('#seg-speed button')) b.onclick = () => setSpeed(+b.dataset.speed);
  for (const cb of document.querySelectorAll('#toggles input')) cb.onchange = () => {
    S.ov[cb.dataset.ov] = cb.checked;
    if (cb.dataset.ov === 'astar') startAstarAnim(cb.checked);
    redraw();
  };
  window.addEventListener('keydown', onKey);
  scene.onFrame = tick;
  window.addEventListener('resize', applyInset);

  await loadMap(0);
  if (!store.get('rc-intro')) { $('#intro-dtg').textContent = dtg(); $('#dlg-intro').showModal(); store.set('rc-intro', '1'); }
}

function setDiff(k) {
  S.diff = k;
  store.set('rc-diff', k);
  for (const b of document.querySelectorAll('#seg-diff button')) { b.classList.toggle('on', b.dataset.key === k); b.setAttribute('aria-checked', b.dataset.key === k); }
  $('#diff-blurb').textContent = S.cfg.difficulties.find((d) => d.key === k).blurb;
}

function setTool(k) {
  if (S.tool !== k) sfx.click();
  S.tool = k;
  for (const b of document.querySelectorAll('.unit')) { b.classList.toggle('on', b.dataset.kind === k); b.setAttribute('aria-checked', b.dataset.kind === k); }
  S.preview = null;
  if (S.hover >= 0 && S.phase === 'defend') scene.setGhost(!occupant(S.hover) && !canPlace(k, S.hover) ? k : null, S.hover);
  redraw();
}

async function loadMap(id) {
  S.mapId = id;
  $('#sel-map').value = id;
  S.map = await api(`/api/map/${id}`);
  $('#map-blurb').textContent = S.map.blurb;
  $('#btn-ai').hidden = !S.map.recommended;
  scene.setMap(S.map);
  scene.resetView();
  S.layout = { radar: [], jammer: [], pad: [] };
  S.result = null;
  setPhase('defend');
  await layoutChanged();
}

// ------------------------------------------------------------------ placement
const spend = () => Object.entries(S.layout).reduce((s, [k, v]) => s + v.length * S.cfg.units[k].cost, 0);
const occupant = (c) => Object.keys(S.layout).find((k) => S.layout[k].includes(c));

function canPlace(kind, c) {
  if (c < 0 || !S.map.place[c] || occupant(c)) return 'Not placeable there';
  if (S.layout[kind].length >= S.cfg.units[kind].max) return `Max ${S.cfg.units[kind].max} ${UNIT_INFO[kind].name.toLowerCase()}s`;
  if (spend() + S.cfg.units[kind].cost > S.cfg.budget) return 'Over budget';
  return null;
}

function onClick(ev) {
  if (S.phase === 'attack') { const i = scene.pickDrone(ev); if (i >= 0) setFocus(i, true); return; }
  if (S.phase !== 'defend') return;
  const c = scene.pickCell(ev);
  if (c < 0) return;
  const occ = occupant(c);
  if (occ) { S.layout[occ] = S.layout[occ].filter((x) => x !== c); sfx.remove(); layoutChanged(); return; }
  const err = canPlace(S.tool, c);
  if (err) { flashError(err); return; }
  sfx.place(S.tool);
  S.layout[S.tool].push(c);
  S.preview = null;
  layoutChanged();
}

let previewTimer = null, previewSeq = 0;
function onMove(ev) {
  if (S.phase !== 'defend') return;
  const c = scene.pickCell(ev);
  if (c === S.hover) return;
  S.hover = c;
  clearTimeout(previewTimer);
  S.preview = null;
  scene.setGhost(c >= 0 && !occupant(c) && !canPlace(S.tool, c) ? S.tool : null, c);
  if (S.tool === 'radar' && c >= 0 && !canPlace('radar', c)) {
    const seq = ++previewSeq;
    previewTimer = setTimeout(async () => {
      const lay = { radar: [c], jammer: [], pad: [] };
      try {
        const cov = await api('/api/coverage', { map: S.mapId, layout: lay });
        if (seq === previewSeq && S.hover === c) { S.preview = cov.radar; redraw(); }
      } catch { /* ignore */ }
    }, 90);
  }
  redraw();
}

async function layoutChanged() {
  $('#budget-left').textContent = S.cfg.budget - spend();
  renderBudget();
  for (const b of document.querySelectorAll('.unit')) {
    const k = b.dataset.kind;
    b.querySelector('[data-count]').textContent = `${S.layout[k].length}/${S.cfg.units[k].max}`;
  }
  scene.setUnits(S.layout);
  S.learn = { stages: [], idx: -1, chain: false, busy: false };
  try { S.coverage = await api('/api/coverage', { map: S.mapId, layout: S.layout }); } catch (e) { flashError(e.message); }
  redraw();
}

function renderBudget() {
  const cells = [];
  for (const k of ['radar', 'jammer', 'pad']) {
    S.layout[k].forEach(() => { for (let i = 0; i < S.cfg.units[k].cost; i++) cells.push({ k, first: i === 0 }); });
  }
  $('#budget-bar').innerHTML = Array.from({ length: S.cfg.budget }, (_, i) => {
    const c = cells[i];
    return c ? `<i class="on${c.first ? ' first' : ''}" style="--c:${COLORS[c.k]}"></i>` : '<i></i>';
  }).join('');
}

function flashError(msg) {
  sfx.error();
  const el = $('#err');
  el.textContent = msg;
  el.classList.remove('shake'); void el.offsetWidth; el.classList.add('shake');
  clearTimeout(flashError.t);
  flashError.t = setTimeout(() => (el.textContent = ''), 2600);
}

function loadAiLayout() {
  if (!S.map.recommended) return;
  S.layout = structuredClone(S.map.recommended.layout);
  for (const k of ['radar', 'jammer', 'pad']) S.layout[k] ??= [];
  setPhase('defend');
  layoutChanged();
}

function placeSuggested() {
  const c = S.result?.debrief?.blind_spot?.suggest_radar;
  if (c == null) return;
  setPhase('defend');
  const err = canPlace('radar', c);
  if (err) { flashError(err === 'Over budget' ? 'Free up 3 points first: click a unit to remove it.' : err); setTool('radar'); return; }
  S.layout.radar.push(c);
  layoutChanged();
}

// ------------------------------------------------------------------ phases
function setPhase(p) {
  S.phase = p;
  document.body.dataset.phase = p;
  const pi = PHASES.indexOf(p);
  for (const el of document.querySelectorAll('.phase')) {
    const i = PHASES.indexOf(el.dataset.phase);
    el.classList.toggle('on', i === pi);
    el.classList.toggle('done', i < pi);
  }
  $('#panel-left').hidden = p !== 'defend';
  $('#playback').hidden = p !== 'attack';
  $('#panel-right').hidden = p !== 'debrief';
  $('#sel-map').disabled = p === 'attack';
  for (const b of document.querySelectorAll('#seg-mode button')) b.disabled = p === 'attack';
  $('#hud-status').textContent = PHASE_STATUS[p];
  scene.setGrid(p === 'defend' ? 0.1 : 0.035);
  if (p !== 'defend') scene.setGhost(null, -1);
  if (p !== 'debrief') { scene.clearPaths(); S.astarAnim = null; }
  if (p === 'defend') { scene.drawTactic(null); S.tactic = -1; }
  if (p !== 'attack') { setCaption(null); delete document.body.dataset.learning; sfx.humStop(); setFocus(-1, false); $('#mind-hint').hidden = true; if (S.net.open) openNet(false, true); }
  if (p === 'defend') { scene.resetActors(0, 0); $('#feed').innerHTML = ''; S.replay = null; S.learn.chain = false; }
  if (p === 'attack') scene.resetActors(0, 0);
  applyInset();
  updateTags();
  redraw();
}

// Keep the board centered in the part of the map the report doesn't cover.
function applyInset() {
  const rep = $('#panel-right');
  const wide = window.innerWidth > 900;
  const net = $('#net');
  scene.setInset(S.phase === 'debrief' && wide ? rep.offsetWidth + 16 : 0, S.phase === 'attack' && S.net.open && wide ? net.offsetHeight + 14 : 0);
}

async function launch() {
  if (S.phase !== 'defend' || !$('#simulating').hidden) return;
  sfx.commit();
  $('#simulating').hidden = false;
  $('#sim-count').textContent = S.diff === 'scripted' ? 32 : 64;
  try {
    S.result = await api('/api/simulate', { map: S.mapId, layout: S.layout, difficulty: S.diff });
    S.result.mapName = S.cfg.maps[S.mapId].name;
    S.tactic = -1;
    S.learn = { stages: [], idx: -1, chain: false, busy: false };
    startReplay();
  } catch (e) {
    flashError(e.message);
  } finally {
    $('#simulating').hidden = true;
  }
}

// ------------------------------------------------------------------ replay
function setSpeed(v) {
  for (const x of document.querySelectorAll('#seg-speed button')) x.classList.toggle('on', +x.dataset.speed === v);
  if (S.replay) S.replay.speed = v;
}

function startReplay(res = S.result, opts = {}) {
  const frames = res.rep.frames;
  setPhase('attack');
  scene.resetActors(frames[0].d.length, frames[0].i.length);
  scene.setUnits(S.layout);
  $('#feed').innerHTML = '';
  const speed = opts.speed ?? +document.querySelector('#seg-speed .on').dataset.speed;
  S.replay = { res, t: 0, lastIdx: 0, playing: true, speed, seen: new Set(), done: false, onEnd: opts.onEnd };
  setSpeed(speed);
  $('#scrub').max = frames.length - 1;
  $('#btn-play').classList.remove('paused');
  $('#pips').innerHTML = frames[0].d.map((_, i) => `<span class="trk" data-i="${i}" data-s="alive" title="UAV-${i + 1}: click to see what it's thinking">${trackSym(i + 1)}</span>`).join('');
  S.focusManual = false;
  setFocus(res.rep.thinking ? 0 : -1, false);
  renderTicks(frames);
  feed('', `${S.cfg.drones} hostile UAVs inbound`, 'var(--enemy)');
}

// Event marks on the timeline: detections (small, blue), kills (green) and breaches (red).
function renderTicks(frames) {
  const F = frames.length, marks = [], seen = new Set();
  for (let t = 1; t < F; t++) frames[t].d.forEach((d, i) => {
    const p = frames[t - 1].d[i];
    const at = `${(100 * t) / (F - 1)}%`;
    if (p[1] === 0 && d[1] === 1) marks.push(`<i style="left:${at};--c:var(--pad)"></i>`);
    if (p[1] === 0 && d[1] === 2) marks.push(`<i style="left:${at};--c:var(--enemy)"></i>`);
    if (d[2] === 1 && !seen.has(i)) { seen.add(i); marks.push(`<i class="small" style="left:${at};--c:var(--radar)"></i>`); }
  });
  $('#ticks').innerHTML = marks.join('');
}

function togglePlay() {
  if (!S.replay) return;
  S.replay.playing = !S.replay.playing;
  $('#btn-play').classList.toggle('paused', !S.replay.playing);
  $('#btn-play').setAttribute('aria-label', S.replay.playing ? 'Pause' : 'Play');
}

function feed(who, what, color) {
  const el = document.createElement('div');
  el.style.setProperty('--c', color);
  const t = document.createElement('span');
  t.className = 't';
  t.textContent = `T+${pad2(Math.floor(S.replay?.t ?? 0))}`;
  const msg = document.createElement('span');
  if (who) { const b = document.createElement('b'); b.textContent = `${who} `; msg.append(b); }
  msg.append(what);
  el.append(t, msg);
  const f = $('#feed');
  f.prepend(el);
  while (f.children.length > 7) f.lastChild.remove();
}

function tick(dt) {
  if (S.astarAnim) {
    const a = S.astarAnim;
    a.shown = Math.min(a.total, a.shown + dt * a.total / 1.6);
    redraw();
    if (a.shown >= a.total && !a.pathsDrawn) { a.pathsDrawn = true; drawAstarPaths(); }
  }
  const R = S.replay;
  if (!R || S.phase !== 'attack') return;
  const frames = R.res.rep.frames, F = frames.length;
  if (R.playing) R.t = Math.min(F - 1, R.t + dt * STEPS_PER_SEC * R.speed);
  const idx = Math.min(Math.floor(R.t), F - 1), f = R.t - idx;
  const A = frames[idx], B = frames[Math.min(idx + 1, F - 1)];

  const quiet = idx - R.lastIdx > 3;   // scrubbing: update state without a burst of sounds
  while (R.lastIdx < idx) { R.lastIdx++; frameEvents(frames[R.lastIdx - 1], frames[R.lastIdx], quiet); }

  A.d.forEach((a, i) => {
    const b = B.d[i];
    scene.updateDrone(i, { visible: a[1] === 0, cellA: a[0], cellB: b[0], f, detected: (f < 0.5 ? a[2] : b[2]) === 1, jammed: b[3] === 1 });
  });
  A.i.forEach((a, k) => {
    const b = B.i[k];
    scene.updateIcpt(k, { visible: a[1] === 1 || (a[1] === 0 && b[1] === 1), cellA: a[0], cellB: b[0], f });
  });
  $('#scrub').value = R.t;
  $('#scrub').style.setProperty('--p', `${(100 * R.t) / Math.max(F - 1, 1)}%`);
  $('#clock').textContent = `T+${pad2(idx)}`;
  const pips = $('#pips').children;
  let down = 0, through = 0;
  A.d.forEach((a, i) => {
    const st = a[1] === 1 ? 'down' : a[1] === 2 ? 'through' : a[2] === 1 ? 'tracked' : 'alive';
    if (pips[i] && pips[i].dataset.s !== st) pips[i].dataset.s = st;
    if (st === 'down') down++; else if (st === 'through') through++;
  });
  if (!document.body.dataset.learning) $('#hud-status').textContent = `${A.d.length - down - through} airborne · ${down} down · ${through} through`;
  sfx.hum(R.playing && R.t < F - 1 ? A.d.length - down - through : 0);
  updateMind(R, idx, A);
  if (S.net.open && S.net.data) {
    const key = `${S.net.key}:${idx}`;
    if (key !== S.net.drawn) { S.net.drawn = key; drawNet(idx); redraw(); }
  }
  if (R.t >= F - 1 && !R.done) {
    R.done = true;
    const stalled = frames[F - 1].d.filter((d) => d[1] === 0).length;
    if (stalled) feed('', `Time limit: ${stalled} UAV${stalled > 1 ? 's' : ''} never committed to an attack`, 'var(--jammer)');
    setTimeout(() => { if (S.phase === 'attack' && S.replay === R) (R.onEnd ? R.onEnd() : showDebrief()); }, 1400);
  }
}

function frameEvents(prev, cur, quiet = false) {
  const R = S.replay;
  cur.d.forEach((d, i) => {
    const p = prev.d[i];
    const who = `UAV-${i + 1}`;
    if (p[1] === 0 && d[1] === 1) { scene.explode(d[0]); feed(who, 'destroyed', 'var(--pad)'); if (!quiet) sfx.boom(); }
    if (p[1] === 0 && d[1] === 2) {
      if (!quiet) sfx.breach();
      scene.flashBase();
      feed(who, 'reached the base', 'var(--enemy)');
      const st = $('#stage');
      st.classList.remove('hit'); void st.offsetWidth; st.classList.add('hit');
    }
    if (d[2] === 1 && !R.seen.has(i)) { R.seen.add(i); feed(who, 'detected by radar', 'var(--radar)'); if (!quiet) sfx.ping(); }
  });
  cur.i.forEach((it, k) => {
    if (prev.i[k][1] === 0 && it[1] === 1) { feed('', 'Interceptor launched', '#7fe3ff'); if (!quiet) sfx.launch(); }
  });
}

// ------------------------------------------------------------------ debrief
function showDebrief() {
  const r = S.result, db = r.debrief;
  setPhase('debrief');
  scene.resetActors(0, 0);
  $('#feed').innerHTML = '';
  const diff = S.cfg.difficulties.find((d) => d.key === r.difficulty);
  $('#db-title').textContent = `${r.mapName} vs. ${diff.label}`;
  $('#db-kind').textContent = S.mode === '360' ? 'After-action report · 360° threat' : 'After-action report';
  $('#db-dtg').textContent = dtg();
  const pct = Math.round(r.breach_rate * 100);
  const stall = r.stall_rate ?? 0;
  countUp($('#db-rate'), pct, (v) => `${v}%`);
  $('#db-rate').style.setProperty('--tone', pct <= 10 ? 'var(--ink-green)' : pct <= 30 ? 'var(--ink-amber)' : 'var(--ink-red)');
  const [stamp, ink] = stall >= 0.5 && r.breach_rate < 0.1 ? ['Deterred', 'var(--ink-blue)']
    : pct <= 10 ? ['Held', 'var(--ink-green)'] : pct <= 30 ? ['Penetrated', 'var(--ink-amber)'] : ['Overrun', 'var(--ink-red)'];
  const st = $('#db-grade');
  st.textContent = '';
  st.style.setProperty('--sc', ink);
  requestAnimationFrame(() => { st.textContent = stamp; setTimeout(() => sfx.stamp(), 300); });
  $('#db-n').textContent = r.battles;

  const hold = $('#db-holdback');
  hold.hidden = stall < 0.25;
  if (stall >= 0.25) hold.textContent = `Note: ${Math.round(stall * 100)}% of drones never committed. They loitered outside your sensors until time ran out. The enemy judged this defense too costly to attack, which deters but does not defeat it.`;

  const hist = $('#db-hist');
  hist.innerHTML = '';
  const mx = Math.max(...r.breach_hist, 1);
  r.breach_hist.forEach((v, i) => {
    const bar = document.createElement('div');
    bar.style.height = `${Math.max(2, (100 * v) / mx)}%`;
    bar.style.background = i === 0 ? 'var(--ink-green)' : `color-mix(in srgb, var(--ink-red) ${40 + i * 10}%, var(--paper-2))`;
    bar.style.animationDelay = `${i * 45}ms`;
    bar.innerHTML = `<span>${v}</span>`;
    bar.title = `${v} attacks with ${i} breaches`;
    hist.appendChild(bar);
  });
  $('#db-hist-axis').innerHTML = r.breach_hist.map((_, i) => `<span>${i}</span>`).join('');
  $('#db-det').textContent = `${Math.round(db.detection_rate * 100)}%`;
  $('#db-kills').textContent = r.mean_kills.toFixed(1);
  $('#db-stall-row').hidden = stall < 0.01;
  $('#db-stall').textContent = `${Math.round(stall * 100)}%`;
  $('#db-replay').textContent = `${r.rep.breaches} of ${S.cfg.drones}`;
  $('#db-follow-row').hidden = $('#db-hover-row').hidden = !r.policy;
  if (r.policy) { $('#db-follow').textContent = fmtPct(r.policy.follow); $('#db-hover').textContent = fmtPct(r.policy.hover); }

  renderTactics(r);

  const bs = db.blind_spot;
  $('#db-blind').hidden = !bs;
  if (bs) { $('#db-blind-text').textContent = bs.text; $('#db-blind-tip').textContent = bs.tip; }

  const rec = r.recommended;
  const recRate = rec?.breach?.[r.difficulty];
  $('#db-compare').hidden = recRate == null;
  if (recRate != null) {
    const diffPts = Math.round((r.breach_rate - recRate) * 100);
    $('#db-compare-text').innerHTML = `A defense found by genetic search for this map lets <b>${Math.round(recRate * 100)}%</b> of drones through against this enemy. ` +
      (diffPts <= 0 ? '<b>You matched or beat it.</b>' : `You are <b>${diffPts} points</b> behind it. Use <i>Load AI layout</i> to study it.`);
  }

  $('#db-learn').hidden = !(S.cfg.timeline?.length > 1);
  if (!S.learn.busy) $('#btn-learn').textContent = S.learn.stages.length && S.learn.stages.length === S.cfg.timeline?.length
    ? 'Play the time-lapse again' : 'Run the training time-lapse';
  renderStages();
  $('#astar-n').textContent = db.astar.explored.length;
  if (S.ov.astar) startAstarAnim(true);
  updateTags();
  redraw();
}

// ------------------------------------------------------------------ discovered tactics
function renderTactics(r) {
  const list = $('#db-tactic-list');
  const tacs = r.tactics || [];
  $('#db-tactics').hidden = !tacs.length;
  list.innerHTML = '';
  if (!tacs.length) { scene.drawTactic(null); return; }
  const learned = r.difficulty !== 'scripted';
  $('#db-tactics-title').textContent = learned ? 'Tactics the swarm found' : 'How the A* planner got through';
  $('#db-tactics-note').textContent = learned
    ? `Found automatically across all ${r.battles} attacks. Nobody programmed these: the swarm was only ever rewarded for getting through.`
    : `Patterns across all ${r.battles} attacks. The A* planner follows fixed rules; these are the routes and timings they produced.`;
  tacs.forEach((t, i) => {
    const li = document.createElement('li');
    li.style.setProperty('--tc', TACTIC_COLORS[i % TACTIC_COLORS.length]);
    li.classList.toggle('on', S.tactic === i);
    const share = t.kind === 'corridor' || t.kind === 'probing' ? `${fmtPct(t.share)} of breaches` : `${fmtPct(t.share)} of attacks`;
    li.innerHTML = `<div class="t-head"><b></b><span class="t-share">${share}</span></div><p></p>
      <div class="t-act"><button class="doc-btn small" data-show>${S.tactic === i ? 'Hide' : 'Show on map'}</button><button class="doc-btn small" data-watch>Watch an example</button></div>`;
    li.querySelector('b').textContent = t.name;
    li.querySelector('p').textContent = t.text;
    li.querySelector('[data-show]').onclick = () => { S.tactic = S.tactic === i ? -1 : i; showTactic(); renderTactics(r); };
    li.querySelector('[data-watch]').onclick = () => watchTactic(i);
    list.appendChild(li);
  });
  showTactic();
}

function showTactic() {
  const t = S.result?.tactics?.[S.tactic];
  scene.drawTactic(t ? t.paths : null, TACTIC_COLORS[S.tactic % TACTIC_COLORS.length]);
}

function watchTactic(i) {
  const t = S.result.tactics[i], ex = S.result.examples?.[String(t.battle)];
  if (!ex) return;
  S.tactic = i;
  startReplay({ rep: ex, seed: S.result.seed, difficulty: S.result.difficulty, battles: S.result.battles }, { onEnd: () => showDebrief() });
  setCaption({ k: `Tactic example · attack ${t.battle + 1} of ${S.result.battles}`, t: t.name, s: t.text, n: 0, i: 0 });
  showTactic();
}

function countUp(el, to, fmt, ms = 700) {
  const t0 = performance.now();
  const step = (t) => {
    const k = Math.min(1, (t - t0) / ms);
    el.textContent = fmt(Math.round(to * (1 - (1 - k) ** 3)));
    if (k < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

function startAstarAnim(on) {
  scene.clearPaths();
  $('#astar-note').hidden = !on;
  S.astarAnim = on && S.result ? { shown: 0, total: S.result.debrief.astar.explored.length, pathsDrawn: false } : null;
}

function drawAstarPaths() {
  const a = S.result.debrief.astar;
  scene.drawPath(a.naive_path, 0x9aa7b4, { radius: 0.05, lift: 0.3, opacity: 0.7 });
  scene.drawPath(a.path, 0xffd166, { radius: 0.11, lift: 0.45 });
  const frames = S.result.rep.frames;
  for (let d = 0; d < frames[0].d.length; d++) {
    const cells = [];
    for (const fr of frames) { if (fr.d[d][1] !== 0 && cells.length) { cells.push(fr.d[d][0]); break; } if (fr.d[d][1] === 0) cells.push(fr.d[d][0]); }
    scene.drawPath(cells, 0xff4d5e, { radius: 0.045, lift: 0.7, opacity: 0.6 });
  }
}

// ------------------------------------------------------------------ watch it learn
const stageLabel = (st) => (st.veteran ? `Veteran · update ${st.update}` : `Update ${st.update}`);

function renderStages() {
  const tl = S.cfg.timeline || [];
  const list = $('#learn-stages');
  list.innerHTML = '';
  if (!S.learn.stages.length && !S.learn.busy) return;
  tl.forEach((st, i) => {
    const res = S.learn.stages[i];
    const li = document.createElement('li');
    li.className = `${res ? '' : 'pending'}${i === S.learn.idx ? ' on' : ''}`;
    const pct = res ? Math.round(res.breach_rate * 100) : null;
    const bench = Math.round((st.strong ?? 0) * 100);
    const dec = st.steps ? `${fmtBig(st.steps * S.cfg.drones)} decisions` : '';
    const yours = pct == null ? 'yours ...' : `yours ${pct}%`;
    li.innerHTML = `<button ${res ? '' : 'disabled'}><span class="st-name">${stageLabel(st)}<span class="st-sub">${dec ? `${dec} · ` : ''}${yours}</span></span>
      <span class="st-bar"><i style="width:${bench}%"></i></span><span class="st-val">${bench}%</span></button>`;
    if (res) li.querySelector('button').onclick = () => { S.learn.chain = false; playStage(i); };
    list.appendChild(li);
  });
}

async function runLearn() {
  const tl = S.cfg.timeline || [];
  if (!S.result || tl.length < 2 || S.learn.busy) return;
  const btn = $('#btn-learn');
  if (S.learn.stages.length === tl.length) { S.learn.chain = true; playStage(0); return; }
  S.learn = { stages: [], idx: -1, chain: false, busy: true };
  btn.disabled = true;
  renderStages();
  try {
    for (let i = 0; i < tl.length; i++) {
      btn.textContent = `Simulating stage ${i + 1} of ${tl.length}...`;
      const reuse = tl[i].veteran && S.result.difficulty === 'veteran';
      const res = reuse ? S.result : await api('/api/simulate', { map: S.mapId, layout: S.layout, difficulty: `stage:${i}`, seed: 4242 });
      S.learn.stages.push(res);
      renderStages();
    }
  } catch (e) {
    flashError(e.message);
    S.learn.busy = false;
    btn.disabled = false;
    btn.textContent = 'Run the training time-lapse';
    return;
  }
  S.learn.busy = false;
  btn.disabled = false;
  btn.textContent = 'Play the time-lapse again';
  S.learn.chain = true;
  playStage(0);
}

function playStage(i) {
  const tl = S.cfg.timeline, st = tl[i], res = S.learn.stages[i];
  sfx.tick();
  S.learn.idx = i;
  startReplay(res, { speed: 2, onEnd: () => (S.learn.chain && i + 1 < tl.length ? playStage(i + 1) : endLearn()) });
  document.body.dataset.learning = '1';
  const dec = st.steps ? `${fmtBig(st.steps * S.cfg.drones)} drone decisions trained` : `update ${st.update}`;
  setCaption({
    k: `Training time-lapse · stage ${i + 1} of ${tl.length}`,
    t: stageLabel(st),
    s: `${dec} · ${Math.round((st.strong ?? 0) * 100)}% breach vs ${S.training?.strong_layouts ?? 'the'} strong benchmark defenses · ${Math.round(res.breach_rate * 100)}% on yours`,
    n: tl.length, i,
  });
  $('#hud-status').textContent = 'Training time-lapse';
}

function endLearn() {
  S.learn.chain = false;
  showDebrief();
  $('#db-learn').scrollIntoView({ block: 'nearest', behavior: 'smooth' });
}

function setCaption(c) {
  const el = $('#caption');
  el.hidden = !c;
  if (c) document.body.dataset.caption = '1'; else delete document.body.dataset.caption;
  if (!c) return;
  el.innerHTML = `<div class="cap-k"></div><div class="cap-t"></div><div class="cap-s"></div><div class="cap-steps">${Array.from({ length: c.n }, (_, j) => `<i class="${j < c.i ? 'done' : j === c.i ? 'on' : ''}"></i>`).join('')}</div>`;
  el.querySelector('.cap-k').textContent = c.k;
  el.querySelector('.cap-t').textContent = c.t;
  el.querySelector('.cap-s').textContent = c.s;
}

// ------------------------------------------------------------------ what the drone is thinking
function setFocus(i, manual) {
  S.focus = i;
  S.focusEndT = null;
  if (manual) S.focusManual = true;
  S.mindKey = '';
  scene.setFocus(i);
  scene.setIntent(null);
  for (const el of document.querySelectorAll('#pips .trk')) el.classList.toggle('focus', +el.dataset.i === i);
  if (S.net.open && i >= 0) { scene.follow(i); loadBrain(); }
  const attack = S.phase === 'attack';
  $('#mind').hidden = !(attack && i >= 0);
  $('#mind-hint').hidden = !(attack && i < 0);
}

function updateMind(R, idx, frame) {
  const i = S.focus;
  if (i < 0) return;
  const d = frame.d[i];
  // follow a live drone automatically until the viewer picks one themselves
  if (d[1] !== 0 && !S.focusManual) {
    if (S.focusEndT == null) S.focusEndT = R.t;
    else if (R.t - S.focusEndT > 6) {
      const next = frame.d.findIndex((x) => x[1] === 0);
      if (next >= 0) { setFocus(next, false); return; }
    }
  }
  const key = `${i}:${idx}:${R.res.rep.index}`;
  if (key === S.mindKey) return;
  S.mindKey = key;
  const th = R.res.rep.thinking;
  const mind = $('#mind');
  mind.classList.toggle('scripted', !th);
  $('#mind-who').textContent = `UAV-${i + 1}`;
  $('#mind-state').textContent = d[1] === 1 ? 'Destroyed' : d[1] === 2 ? 'Broke through'
    : [d[2] ? 'Tracked by radar' : 'Undetected', d[3] ? 'jammed' : ''].filter(Boolean).join(' · ');
  const say = $('#mind-say');
  if (!th) {
    say.textContent = 'Scripted A* enemy: no neural network. Every drone flies the cheapest threat-aware A* route around your known sensors.';
    scene.setIntent(null);
    return;
  }
  if (d[1] !== 0 || idx >= th.p.length) {
    say.textContent = d[1] === 1 ? `Shot down at T+${pad2(idx)}.` : d[1] === 2 ? `Reached the base at T+${pad2(idx)}.` : 'No decision this step.';
    scene.setIntent(null);
    return;
  }
  const p = th.p[idx][i].map((x) => x / 1000), a = th.a[idx][i], f = th.f[idx][i], v = th.v[idx][i];
  drawRose(p, a, f);
  $('#mind-follow').textContent = fmtPct(p[9]);
  $('#mind-follow-bar').style.width = fmtPct(p[9]);
  $('#mind-hover').textContent = fmtPct(p[8]);
  $('#mind-hover-bar').style.width = fmtPct(p[8]);
  // the critic's value estimate: the drone's own forecast of how this attack will turn out
  $('#mind-value').textContent = `${v >= 6 ? 'through' : v >= 2 ? 'unsure' : 'stopped'} ${v >= 0 ? '+' : ''}${v.toFixed(1)}`;
  $('#mind-value-bar').style.width = `${Math.max(3, Math.min(100, (v / 10) * 100))}%`;
  let msg = a === 9 ? `Following the pathfinder: heading ${DIRS[f]} (${fmtPct(p[9])} sure).`
    : a === 8 ? `Holding position, waiting for an opening (${fmtPct(p[8])}).`
    : a === f ? `Heading ${DIRS[a]}, the same way the pathfinder would go.`
    : `Improvising: heading ${DIRS[a]} while the pathfinder says ${DIRS[f]}.`;
  if (d[3]) msg += ' Jammed: its move may be scrambled.';
  say.textContent = msg;
  scene.setIntent({ chosen: a === 9 ? f : a, follow: f, followed: a === 9 });
}

// Compass rose: one wedge per direction (north up). Red = probability of choosing that move directly,
// gold = probability of reaching it by "follow the pathfinder", white outline = the move it took.
function drawRose(p, a, f) {
  const R0 = 10, R1 = 54, span = R1 - R0, h = 0.34;
  const P = (r, t) => `${(r * Math.sin(t)).toFixed(1)},${(-r * Math.cos(t)).toFixed(1)}`;
  const wedge = (r0, r1, t, cls) => `<path class="${cls}" d="M${P(r0, t - h)} L${P(r1, t - h)} A${r1} ${r1} 0 0 1 ${P(r1, t + h)} L${P(r0, t + h)} A${r0} ${r0} 0 0 0 ${P(r0, t - h)} Z"/>`;
  let svg = `<circle class="ring" r="${R1}"/><circle class="ring" r="${R0 + span / 2}"/>`;
  [['N', 0], ['E', 2], ['S', 4], ['W', 6]].forEach(([l, k]) => { svg += `<text class="lbl" x="${P(R1 + 8, (k * Math.PI) / 4).split(',')[0]}" y="${P(R1 + 8, (k * Math.PI) / 4).split(',')[1]}">${l}</text>`; });
  const rad = (x) => R0 + span * Math.min(1, Math.sqrt(x) * 1.3);   // area-like scaling: 10% is still visible
  for (let k = 0; k < 8; k++) {
    const t = (k * Math.PI) / 4;
    const own = rad(p[k]);
    const via = f === k ? p[9] : 0;
    const top = rad(p[k] + via);
    if (p[k] > 0.004) svg += wedge(R0, own, t, 'own');
    if (via > 0.004) svg += wedge(own, top, t, 'via');
    if (a === k || (a === 9 && f === k)) svg += wedge(R0, Math.max(top, R0 + 6), t, 'chosen');
  }
  svg += `<circle class="hov" r="${(3 + 14 * p[8]).toFixed(1)}"/>`;
  if (a === 8) svg += `<circle class="chosen" r="${(3 + 14 * p[8]).toFixed(1)}"/>`;
  $('#mind-rose').innerHTML = svg;
}

// ------------------------------------------------------------------ the neural network view
const unb64 = (b) => Uint8Array.from(atob(b), (c) => c.charCodeAt(0));
const NET_OUT = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'Hover', 'Follow pathfinder'];
const HEAT = (() => {   // 256-step color ramp for neuron activity: dark -> orange -> pale
  const stops = [[0, [30, 33, 37]], [0.45, [194, 65, 12]], [0.8, [255, 166, 77]], [1, [255, 244, 214]]];
  return Array.from({ length: 256 }, (_, n) => {
    const v = n / 255;
    for (let i = 1; i < stops.length; i++) if (v <= stops[i][0]) {
      const [a, ca] = stops[i - 1], [b, cb] = stops[i], k = (v - a) / (b - a);
      return `rgb(${ca.map((x, j) => Math.round(x + (cb[j] - x) * k)).join(',')})`;
    }
    return 'rgb(255,244,214)';
  });
})();

function openNet(on, silent = false) {
  S.net.open = on;
  $('#net').hidden = !on;
  if (on) document.body.dataset.net = '1'; else delete document.body.dataset.net;
  S.net.drawn = '';
  applyInset();
  if (!silent) redraw();
  if (on) {
    if (S.focus < 0) setFocus(0, true);
    scene.follow(S.focus);
    loadBrain();
  } else {
    scene.resetView();
  }
}

async function loadBrain() {
  const R = S.replay;
  if (!R || S.focus < 0) return;
  const res = R.res;
  $('#net-who').textContent = `UAV-${S.focus + 1}`;
  if (!res.rep.thinking) {
    S.net.data = null; S.net.key = '';
    $('#net-meta').textContent = 'Scripted A* enemy: no neural network to show. Pick a learned enemy.';
    $('#net-why').textContent = '';
    const cv = $('#net-canvas'); cv.getContext('2d').clearRect(0, 0, cv.width, cv.height);
    return;
  }
  const key = `${S.mode}|${res.difficulty}|${res.seed}|${res.battles}|${res.rep.index}|${S.focus}`;
  if (S.net.key === key && (S.net.data || S.net.pending === key)) return;
  S.net.key = key;
  S.net.data = null;
  let d = S.net.cache.get(key);
  if (!d) {
    $('#net-loading').hidden = false;
    S.net.pending = key;
    try {
      d = await api('/api/brain', { map: S.mapId, layout: S.layout, difficulty: res.difficulty, seed: res.seed, n: res.battles, battle: res.rep.index, drone: S.focus });
    } catch (e) {
      if (S.net.key === key) { $('#net-loading').hidden = true; flashError(e.message); }
      return;
    } finally {
      if (S.net.pending === key) S.net.pending = '';
    }
    d.localU8 = unb64(d.local); d.globU8 = unb64(d.glob); d.attU8 = unb64(d.att_map); d.gattU8 = unb64(d.att_gmap);
    d.layersU8 = d.layers.map(unb64);
    d.stepAt = new Map(d.t.map((t, i) => [t, i]));
    // the server re-ran the battle from its seed: it must be the battle on screen, step for step
    const frames = res.rep.frames;
    d.matches = d.t.every((t, i) => !frames[t] || frames[t].d[S.focus][0] === d.cell[i]);
    if (!d.matches) console.warn('RED CELL: network re-run did not reproduce this replay exactly');
    S.net.cache.set(key, d);
    if (S.net.cache.size > 8) S.net.cache.delete(S.net.cache.keys().next().value);
  }
  if (S.net.key !== key) return;   // the viewer moved on while this was loading
  $('#net-loading').hidden = true;
  S.net.data = d;
  S.net.drawn = '';
}

function drawNet(idx) {
  const d = S.net.data, cv = $('#net-canvas');
  const dpr = Math.min(window.devicePixelRatio || 1, 2), W = cv.clientWidth, H = cv.clientHeight;
  if (!W || !H) return;
  if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); }
  const ctx = cv.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  // which recorded decision belongs to this frame (after the drone's last decision, keep showing it, dimmed)
  let i = d.stepAt.get(idx);
  const live = i !== undefined;
  if (!live) i = idx > d.t[d.t.length - 1] ? d.t.length - 1 : 0;
  S.net.step = live ? i : -1;
  ctx.globalAlpha = live ? 1 : 0.45;
  const label = (t, x, y, color = '#a3a097', size = 11) => { ctx.fillStyle = color; ctx.font = `600 ${size}px "Barlow Condensed", sans-serif`; ctx.fillText(t.toUpperCase(), x, y); };
  const mono = (t, x, y, color = '#ebe9e4', align = 'left') => { ctx.fillStyle = color; ctx.font = '10.5px "IBM Plex Mono", monospace'; ctx.textAlign = align; ctx.fillText(t, x, y); ctx.textAlign = 'left'; };

  // ---------------- inputs
  const x0 = 14, y0 = 24, LV = Math.min(132, H - 150), cL = LV / 11;
  label('Input · local view', x0, y0 - 8);
  const lb = i * 9 * 121, ch = (c, r, k) => d.localU8[lb + c * 121 + r * 11 + k] / 255;
  for (let r = 0; r < 11; r++) for (let k = 0; k < 11; k++) {
    const h = ch(0, r, k), blocked = ch(1, r, k) > 0.5;
    let rgb = blocked ? [52, 56, 61] : [40 + 120 * h, 48 + 112 * h, 36 + 92 * h];
    const mix = (c2, a) => { rgb = rgb.map((v, j) => v + (c2[j] - v) * a); };
    mix([63, 208, 255], 0.45 * ch(2, r, k));
    mix([255, 181, 71], 0.4 * ch(3, r, k));
    mix([255, 255, 255], 0.7 * ch(7, r, k));
    ctx.fillStyle = `rgb(${rgb.map(Math.round).join(',')})`;
    ctx.fillRect(x0 + k * cL, y0 + r * cL, cL - 0.5, cL - 0.5);
    const cxp = x0 + (k + 0.5) * cL, cyp = y0 + (r + 0.5) * cL;
    if (ch(4, r, k) > 0.2) { ctx.fillStyle = '#5cf0a0'; ctx.fillRect(cxp - cL * 0.25, cyp - cL * 0.25, cL * 0.5, cL * 0.5); }
    if (ch(5, r, k) > 0.1) { ctx.fillStyle = '#7fe3ff'; ctx.beginPath(); ctx.arc(cxp, cyp, cL * 0.22, 0, 7); ctx.fill(); }
    if (ch(6, r, k) > 0.1 && !(r === 5 && k === 5)) { ctx.fillStyle = '#ff4d5e'; ctx.beginPath(); ctx.arc(cxp, cyp, cL * 0.28, 0, 7); ctx.fill(); }
    if (S.net.att) { ctx.fillStyle = `rgba(255,79,216,${(0.7 * (d.attU8[i * 121 + r * 11 + k] / 255) ** 1.8).toFixed(3)})`; ctx.fillRect(x0 + k * cL, y0 + r * cL, cL - 0.5, cL - 0.5); }
  }
  ctx.strokeStyle = '#ff8080'; ctx.lineWidth = 2; ctx.strokeRect(x0 + 5 * cL, y0 + 5 * cL, cL, cL);   // the drone itself
  // the nine channels the network actually receives
  const TS = Math.floor((LV - 36) / 3), tx = x0 + LV + 14;
  const SHORT = ['terrain', 'blocked', 'radar', 'jammers', 'pads', 'intercept', 'drones', 'base', 'threat'];
  label('Its 9 input channels', tx, y0 - 8);
  for (let c = 0; c < 9; c++) {
    const gx = tx + (c % 3) * (TS + 10), gy = y0 + Math.floor(c / 3) * (TS + 12), cs = TS / 11;
    for (let r = 0; r < 11; r++) for (let k = 0; k < 11; k++) { ctx.fillStyle = HEAT[d.localU8[lb + c * 121 + r * 11 + k]]; ctx.fillRect(gx + k * cs, gy + r * cs, cs, cs); }
    ctx.fillStyle = '#6e6b65'; ctx.font = '9px "IBM Plex Mono", monospace';
    ctx.fillText(SHORT[c], gx, gy + TS + 9);
  }
  // whole-map summary and status
  const y1 = y0 + LV + 24, MV = Math.min(96, H - y1 - 8), cG = MV / 16, gb = i * 8 * 256;
  label('Input · whole map', x0, y1 - 8);
  for (let r = 0; r < 16; r++) for (let k = 0; k < 16; k++) {
    const g = (c) => d.globU8[gb + c * 256 + r * 16 + k] / 255;
    let rgb = [40 + 120 * g(0), 48 + 112 * g(0), 36 + 92 * g(0)];
    const mix = (c2, a) => { rgb = rgb.map((v, j) => v + (c2[j] - v) * Math.min(1, a)); };
    mix([52, 56, 61], g(1)); mix([63, 208, 255], 0.5 * g(2)); mix([255, 181, 71], 0.45 * g(3)); mix([255, 255, 255], 0.8 * g(5)); mix([255, 77, 94], 1.6 * g(6));
    ctx.fillStyle = `rgb(${rgb.map(Math.round).join(',')})`;
    ctx.fillRect(x0 + k * cG, y1 + r * cG, cG - 0.3, cG - 0.3);
    if (S.net.att) { ctx.fillStyle = `rgba(255,79,216,${(0.65 * (d.gattU8[i * 256 + r * 16 + k] / 255) ** 1.8).toFixed(3)})`; ctx.fillRect(x0 + k * cG, y1 + r * cG, cG - 0.3, cG - 0.3); }
  }
  const sx = x0 + MV + 14, sw = LV + 14 + 3 * TS + 20 - MV - 14 - 92;
  label('Input · status', sx, y1 - 8);
  const vec = d.vec[i], signed = [true, true, false, false, false, false, false, false, false, false], scale = [1, 1, 1, 1, 1, 1, 1, 1, 1, 2];
  const short = ['base dx', 'base dy', 'distance', 'detected', 'jammed', 'time', 'pos x', 'pos y', 'swarm', 'threat'];
  vec.forEach((v, k) => {
    const yy = y1 + k * Math.max(9.6, (MV - 2) / 10);
    ctx.fillStyle = '#6e6b65'; ctx.font = '8.5px "IBM Plex Mono", monospace'; ctx.fillText(short[k], sx, yy + 7);
    const bx = sx + 58, bw = Math.max(30, sw);
    ctx.fillStyle = '#26282b'; ctx.fillRect(bx, yy + 1, bw, 6);
    ctx.fillStyle = HEAT[220];
    const f = Math.max(-1, Math.min(1, v / scale[k]));
    if (signed[k]) ctx.fillRect(f < 0 ? bx + bw / 2 + f * bw / 2 : bx + bw / 2, yy + 1, Math.abs(f) * bw / 2, 6);
    else ctx.fillRect(bx, yy + 1, Math.max(0, f) * bw, 6);
  });

  // ---------------- hidden layers
  const ox = W - 200;                                  // outputs column
  const ax = x0 + LV + 14 + 3 * TS + 20 + 34;          // first hidden column
  const avail = H - y0 - 10;
  const cs = Math.max(2.4, Math.min(5.2, (avail - 3 * 16) / 40));
  const ct = Math.min(cs * 1.45, (ox - 60 - ax - 48 * cs - 10 - 70) / 32);
  const layerAt = (li) => { const sz = d.layer_info[li].size; return { cols: sz === 512 ? 32 : sz === 64 ? 8 : 16, rows: sz === 512 ? 16 : sz === 64 ? 8 : 16 }; };
  const grid = (li, gx, gy, c, title) => {
    const { cols, rows } = layerAt(li), u = d.layersU8[li], n = cols * rows, off = i * n;
    let sum = 0;
    for (let j = 0; j < n; j++) { const v = u[off + j]; sum += v; ctx.fillStyle = HEAT[v]; ctx.fillRect(gx + (j % cols) * c, gy + Math.floor(j / cols) * c, c - 0.6, c - 0.6); }
    if (title) label(title, gx, gy - 5, '#a3a097', 10.5);
    return { x: gx, y: gy, w: cols * c, h: rows * c, mean: sum / n / 255, cols, c };
  };
  const bundle = (a, b, strength) => {   // a schematic sheaf of connections from block a to block b
    for (let k = 0; k < 9; k++) {
      const ya = a.y + a.h * (k + 0.5) / 9, yb = b.y + b.h * (k + 0.5) / 9;
      ctx.strokeStyle = `rgba(255,170,90,${(0.05 + 0.3 * strength).toFixed(3)})`;
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(a.x + a.w + 2, ya);
      ctx.bezierCurveTo(a.x + a.w + 30, ya, b.x - 30, yb, b.x - 2, yb); ctx.stroke();
    }
  };
  const rowH = 16 * cs + 16;
  const enc = [['Local view encoder', 0, 1], ['Whole-map encoder', 2, 3], ['Status encoder', 4, -1]];
  const blocks = enc.map(([name, l1, l2], r) => {
    const gy = y0 + 4 + r * rowH;
    const g1 = grid(l1, ax, gy, cs, `${name} · ${d.layer_info[l1].size}${l2 >= 0 ? ' → ' + d.layer_info[l2].size : ''}`);
    if (l2 < 0) return g1;
    const g2 = grid(l2, ax + 32 * cs + 22, gy, cs, '');
    bundle(g1, g2, g1.mean);
    return g2;
  });
  const ty = y0 + 4 + (3 * rowH - 16 * ct - 16) / 2 + 8;
  const tA = grid(5, ax + 48 * cs + 22 + 60, ty, ct, 'Trunk · 256');
  const tB = grid(6, tA.x + tA.w + 46, ty, ct, 'Trunk · 256');
  blocks.forEach((b) => bundle(b, tA, b.mean));
  bundle(tA, tB, tA.mean);

  // ---------------- outputs
  const probs = d.p[i], a = d.a[i], f = d.f[i];
  label('Outputs · 10 actions', ox, y0 - 8);
  const oy = (k) => y0 + 8 + k * Math.min(22, (H - y0 - 52) / 10);
  const outX = ox + 12;
  // the last-layer neurons that pushed hardest for (orange) or against (blue) the chosen action
  const cmax = Math.max(...d.contrib[i].map(Math.abs), 1e-9);
  d.contrib_idx[i].forEach((j, k) => {
    const c = d.contrib[i][k], w = Math.abs(c) / cmax;
    const nx = tB.x + (j % 16 + 0.5) * ct, ny = tB.y + (Math.floor(j / 16) + 0.5) * ct;
    ctx.strokeStyle = c > 0 ? `rgba(255,138,61,${(0.25 + 0.65 * w).toFixed(3)})` : `rgba(92,200,255,${(0.25 + 0.65 * w).toFixed(3)})`;
    ctx.lineWidth = 0.6 + 2.6 * w;
    ctx.beginPath(); ctx.moveTo(nx, ny); ctx.bezierCurveTo(nx + 60, ny, outX - 60, oy(a), outX - 8, oy(a)); ctx.stroke();
    ctx.strokeStyle = c > 0 ? '#ff8a3d' : '#5cc8ff'; ctx.lineWidth = 1.2; ctx.strokeRect(nx - ct / 2, ny - ct / 2, ct, ct);
  });
  NET_OUT.forEach((name, k) => {
    const p = probs[k], y = oy(k);
    ctx.fillStyle = HEAT[Math.round(Math.min(1, p * 1.6) * 255)];
    ctx.beginPath(); ctx.arc(outX, y, 3 + 6 * Math.sqrt(p), 0, 7); ctx.fill();
    if (k === a) { ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(outX, y, 3 + 6 * Math.sqrt(p) + 3, 0, 7); ctx.stroke(); }
    ctx.fillStyle = k === a ? '#ffffff' : '#a3a097'; ctx.font = `${k === a ? 700 : 500} 12px "Barlow Condensed", sans-serif`;
    ctx.fillText(k === 9 ? `Follow pathfinder (${NET_OUT[f]})` : name, outX + 14, y + 4);
    mono(`${Math.round(p * 100)}%`, W - 14, y + 4, k === a ? '#fff' : '#a3a097', 'right');
  });
  const v = d.v[i], vy = oy(10) + 6;
  label('Forecast', ox, vy, '#a3a097', 11);
  mono(`${v >= 0 ? '+' : ''}${v.toFixed(1)}  ${v >= 6 ? 'likely through' : v >= 2 ? 'unsure' : 'likely stopped'}`, W - 14, vy, '#ebe9e4', 'right');
  ctx.globalAlpha = 1;

  // ---------------- header and footer text
  const total = d.params >= 1e6 ? `${(d.params / 1e6).toFixed(1)}M` : `${Math.round(d.params / 1e3)}K`;
  $('#net-who').textContent = `UAV-${d.drone + 1}`;
  $('#net-meta').textContent = `${total} parameters · 7 hidden layers · T+${pad2(idx)}${live ? '' : ' · no longer flying'}${d.matches ? '' : ' · re-run differs'}`;
  const names = [...d.names.local.map((n) => `${n} nearby`), ...d.names.glob.map((n) => `map: ${n.toLowerCase()}`), ...d.names.vec.map((n) => n.toLowerCase())];
  const top = d.att_share[i].map((v2, k) => [v2, names[k]]).sort((p1, p2) => p2[0] - p1[0]).slice(0, 3);
  const why = $('#net-why');
  why.textContent = '';
  why.append('What drove this decision: ');
  top.forEach(([v2, n], k) => { const b = document.createElement('b'); b.textContent = `${n} ${Math.round(v2 * 100)}%`; why.append(b); if (k < 2) why.append(' · '); });
}

// ------------------------------------------------------------------ overlay + tags
function redraw() {
  if (!S.map) return;
  const L = [
    { type: 'hatch', data: S.map.passable.map((p) => !p) },
    { type: 'mask', data: S.map.spawn, color: COLORS.enemy, alpha: 0.16, edge: 0.6 },
  ];
  const showSensors = S.phase !== 'debrief' || S.ov.sensors;
  if (S.coverage && showSensors) {
    L.push({ type: 'mask', data: S.coverage.pad, color: COLORS.pad, alpha: 0.14, edge: 0.6 });
    L.push({ type: 'mask', data: S.coverage.jammer, color: COLORS.jammer, alpha: 0.2, edge: 0.75 });
    L.push({ type: 'mask', data: S.coverage.radar, color: COLORS.radar, alpha: 0.22, edge: 0.9 });
  }
  if (S.phase === 'defend') {
    if (S.preview) L.push({ type: 'mask', data: S.preview, color: '#bff0ff', alpha: 0.18, edge: 0.7 });
    if (S.hover >= 0) {
      const occ = occupant(S.hover);
      const ok = !canPlace(S.tool, S.hover);
      L.push({ type: 'outline', cell: S.hover, color: occ ? '#ff4d5e' : ok ? COLORS[S.tool] : '#556270' });
    }
  }
  if (S.phase === 'attack' && S.net.open && S.net.att && S.net.data && S.net.step >= 0) {
    // project the drone's attention (over its 11x11 local view) onto the map cells around it
    const d = S.net.data, i = S.net.step, c = d.cell[i], cx = c % 32, cy = Math.floor(c / 32);
    const data = new Array(1024).fill(0);
    for (let r = 0; r < 11; r++) for (let k = 0; k < 11; k++) {
      const x = cx - 5 + k, y = cy - 5 + r;
      if (x >= 0 && y >= 0 && x < 32 && y < 32) data[y * 32 + x] = (d.attU8[i * 121 + r * 11 + k] / 255) ** 2.2;
    }
    L.push({ type: 'heat', data, color: '#ff4fd8', max: 0.62 });
  }
  if (S.phase === 'debrief' && S.result) {
    const db = S.result.debrief;
    if (S.ov.traffic) L.push({ type: 'heat', data: db.heat_traffic, color: '#b07cff', max: 0.6 });
    if (S.ov.breach) L.push({ type: 'heat', data: db.heat_breach, color: '#ff5a4d', max: 0.8 });
    if (S.ov.astar && S.astarAnim) L.push({ type: 'cells', cells: db.astar.explored.slice(0, Math.floor(S.astarAnim.shown)), color: COLORS.radar, alpha: 0.35, inset: 7 });
    if (db.blind_spot) {
      L.push({ type: 'outline', cell: db.blind_spot.cell, color: '#ff4d5e' });
      L.push({ type: 'outline', cell: db.blind_spot.suggest_radar, color: COLORS.radar });
    }
  }
  L.push({ type: 'mask', data: S.map.base, color: '#ffffff', alpha: 0.12, edge: 0.5 });
  scene.drawOverlay(L);
}

function updateTags() {
  const root = $('#tags');
  root.innerHTML = '';
  const tags = [];
  const add = (text, cell, color, lift = 2.4, pulse = false) => {
    const el = document.createElement('div');
    el.className = `tag${pulse ? ' pulse' : ''}`;
    el.textContent = text;
    el.style.setProperty('--c', color);
    root.appendChild(el);
    tags.push({ el, cell, lift });
  };
  if (!S.map) return;
  add('Base', S.map.base_cell, '#e8eef4', 3);
  const spawnCells = S.map.spawn.map((v, i) => (v ? i : -1)).filter((i) => i >= 0);
  if (S.mode === '360') {
    // one label per side that has an entry zone, at the entry cell nearest the middle of that edge
    const sides = { N: (x, y) => y <= 1, S: (x, y) => y >= 30, W: (x, y) => x <= 1, E: (x, y) => x >= 30 };
    for (const [name, on] of Object.entries(sides)) {
      const cells = spawnCells.filter((c) => on(c % 32, Math.floor(c / 32)));
      if (cells.length < 3) continue;
      const mid = cells.reduce((b, c) => (Math.hypot(c % 32 - 15.5, Math.floor(c / 32) - 15.5) < Math.hypot(b % 32 - 15.5, Math.floor(b / 32) - 15.5) ? c : b));
      add(`Entry ${name}`, mid, COLORS.enemy, 1.2);
    }
  } else if (spawnCells.length) add('Drone entry · north', spawnCells[Math.floor(spawnCells.length / 2)], COLORS.enemy, 1.2);
  const bs = S.phase === 'debrief' && S.result?.debrief?.blind_spot;
  if (bs) {
    add('Blind spot', bs.cell, COLORS.enemy, 1.6, true);
    add('Suggested radar', bs.suggest_radar, COLORS.radar, 1.6);
  }
  scene.setTags(tags);
}

// ------------------------------------------------------------------ keyboard
function onKey(e) {
  if (e.target.matches('input, select, textarea') || document.querySelector('dialog[open]')) return;
  const tool = Object.entries(UNIT_INFO).find(([, u]) => u.key === e.key);
  if (tool && S.phase === 'defend') setTool(tool[0]);
  if (e.key === ' ' && S.phase === 'attack') { e.preventDefault(); togglePlay(); }
  if ((e.key === 'n' || e.key === 'N') && S.phase === 'attack') openNet(!S.net.open);
  if (e.key === 'Enter' && S.phase === 'defend' && !e.target.matches('button')) { e.preventDefault(); launch(); }
}

// ------------------------------------------------------------------ technical annex
async function openHood() {
  $('#dlg-hood').showModal();
  const chart = $('#chart');
  let data;
  try { data = await api('/api/training'); } catch { data = { metrics: [] }; }
  const stepsAt = new Map(data.metrics.map((m) => [m.update, m.steps]));
  const strong = data.curve?.length > 1;
  const pts = strong
    ? data.curve.filter(([u]) => stepsAt.has(u)).map(([u, s, r]) => ({ update: u, steps: stepsAt.get(u), main: s, other: r }))
    : data.metrics.filter((m) => m.eval_breach != null).map((m) => ({ update: m.update, steps: m.steps, main: m.eval_breach }));
  if (pts.length < 2) {
    chart.innerHTML = '<div class="empty"><div>No training run found yet. Run <code>python train.py</code>, then <code>python scripts/select_difficulties.py</code>.</div></div>';
    $('#hood-stats').innerHTML = '';
    return;
  }
  $('#chart-label').textContent = strong
    ? `Training curve${S.mode === '360' ? ', 360° theater' : ''}: breach rate vs. ${data.strong_layouts} strong defenses`
    : 'Training curve: breach rate on a fixed benchmark of defenses';
  const W = 600, H = 300, P = { l: 46, r: 16, t: 16, b: 34 };
  const D = S.cfg.drones;
  const maxX = pts[pts.length - 1].steps * D;
  const x = (v) => P.l + (v / maxX) * (W - P.l - P.r);
  const y = (v) => H - P.b - v * (H - P.t - P.b);
  const line = (key) => pts.map((m, i) => `${i ? 'L' : 'M'}${x(m.steps * D).toFixed(1)},${y(m[key] ?? 0).toFixed(1)}`).join('');
  let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Breach rate over training">`;
  svg += `<path class="ch-area" d="${line('main')}L${x(pts[pts.length - 1].steps * D)},${y(0)}L${x(pts[0].steps * D)},${y(0)}Z"/>`;
  for (const v of [0, 0.25, 0.5, 0.75, 1]) svg += `<line class="ch-grid" x1="${P.l}" x2="${W - P.r}" y1="${y(v)}" y2="${y(v)}"/><text class="ch-label" x="${P.l - 8}" y="${y(v) + 4}" text-anchor="end">${v * 100}%</text>`;
  for (let i = 0; i <= 4; i++) { const v = (maxX * i) / 4; svg += `<text class="ch-label" x="${x(v)}" y="${H - 14}" text-anchor="middle">${fmtBig(v)}</text>`; }
  svg += `<text class="ch-note" x="${W - P.r}" y="${H - 2}" text-anchor="end">drone decisions trained</text>`;
  for (const st of data.timeline || []) {
    if (!stepsAt.has(st.update)) continue;
    const px = x(stepsAt.get(st.update) * D);
    svg += `<line class="ch-stage" x1="${px}" x2="${px}" y1="${H - P.b}" y2="${H - P.b - 6}"/>`;
  }
  if (data.baseline != null) svg += `<line class="ch-base" x1="${P.l}" x2="${W - P.r}" y1="${y(data.baseline)}" y2="${y(data.baseline)}"/><text class="ch-base-label" x="${W - P.r - 4}" y="${y(data.baseline) + 15}" text-anchor="end">Scripted A* attacker</text>`;
  if (strong) svg += `<path class="ch-other" d="${line('other')}"/><text class="ch-note" x="${W - P.r - 4}" y="${y(pts[pts.length - 1].other) - 7}" text-anchor="end">vs. random defenses</text>`;
  svg += `<path class="ch-main" d="${line('main')}"/>`;
  const vet = data.levels?.veteran;
  if (vet && stepsAt.has(vet.update)) {
    const vp = pts.find((p) => p.update === vet.update);
    const px = x(stepsAt.get(vet.update) * D), py = y(vp ? vp.main : vet.eval_breach);
    svg += `<circle class="ch-vet" cx="${px}" cy="${py}" r="5"/><text class="ch-vet-label" x="${px + 8}" y="${py - 9}">Veteran</text>`;
  }
  chart.innerHTML = `${svg}</svg>`;

  // training time and steps; a resumed run continues the counters (select_difficulties.py offsets them)
  const last = data.metrics[data.metrics.length - 1];
  const rows = [
    ['Drone decisions trained', fmtBig(last.steps * D)],
    ['Training time', `${Math.round(last.time / 60)} min`],
    ['Throughput', `${fmtBig(last.sps * D)} decisions/s`],
    ['Hardware', data.device || 'GPU'],
  ];
  if (data.heldout) {
    rows.push([`Veteran breach rate, ${data.heldout.layouts} held-out strong defenses`, `${Math.round(data.heldout.veteran * 100)}%`]);
    rows.push(['Scripted A* breach rate, same defenses', `${Math.round(data.heldout.scripted * 100)}%`]);
    if (data.heldout.veteran_stall != null) rows.push(['Veteran drones that never committed', `${Math.round(data.heldout.veteran_stall * 100)}%`]);
  }
  const tb = $('#hood-stats');
  tb.innerHTML = '';
  for (const [k, v] of rows) {
    const tr = tb.insertRow();
    const th = document.createElement('th');
    th.textContent = k;
    tr.appendChild(th);
    tr.insertCell().textContent = v;
  }
}

window.__rc = { S, scene };   // handy for debugging from the console
init().catch((e) => { console.error(e); flashError(`Failed to start: ${e.message}`); });
