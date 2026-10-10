import { Scene3D, G, COLORS } from '/static/scene.js';

const $ = (s) => document.querySelector(s);
const STEPS_PER_SEC = 5;
const UNIT_INFO = {
  radar: { glyph: 'R', name: 'Radar', desc: 'Detects drones in range with clear line of sight.', key: '1' },
  jammer: { glyph: 'J', name: 'Jammer', desc: 'Scrambles navigation: jammed drones drift.', key: '2' },
  pad: { glyph: 'P', name: 'Interceptor pad', desc: '3 interceptors that chase detected drones.', key: '3' },
};

const S = {
  cfg: null, mapId: 0, map: null, tool: 'radar', diff: 'scripted', phase: 'defend',
  layout: { radar: [], jammer: [], pad: [] }, coverage: null, preview: null, hover: -1,
  result: null, replay: null, astarAnim: null, ov: { breach: true, traffic: false, astar: false, sensors: true },
};

const scene = new Scene3D($('#stage'));
const api = async (path, body) => {
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
async function init() {
  S.cfg = await api('/api/config');
  $('#budget-total').textContent = S.cfg.budget;
  const sel = $('#sel-map');
  for (const m of S.cfg.maps) sel.add(new Option(m.name, m.id));
  sel.onchange = () => loadMap(+sel.value);

  const seg = $('#seg-diff');
  for (const d of S.cfg.difficulties) {
    const b = document.createElement('button');
    b.dataset.key = d.key;
    b.disabled = !d.available;
    b.setAttribute('role', 'radio');
    b.innerHTML = `${d.label}<span class="sub">${d.available ? (d.key === 'scripted' ? 'pathfinding' : 'learned') : 'not trained'}</span>`;
    b.title = d.blurb;
    b.onclick = () => setDiff(d.key);
    seg.appendChild(b);
  }
  const best = [...S.cfg.difficulties].reverse().find((d) => d.available);
  setDiff(store.get('rc-diff') && S.cfg.difficulties.find((d) => d.key === store.get('rc-diff') && d.available) ? store.get('rc-diff') : best.key);

  const units = $('#units');
  for (const [k, u] of Object.entries(S.cfg.units)) {
    const info = UNIT_INFO[k];
    const b = document.createElement('button');
    b.className = 'unit';
    b.dataset.kind = k;
    b.style.setProperty('--c', COLORS[k]);
    b.innerHTML = `<span class="glyph">${info.glyph}</span>
      <span><div class="name">${info.name} <kbd>${info.key}</kbd></div><div class="desc">${info.desc}</div></span>
      <span class="meta"><b>${u.cost}</b> pts<br><span data-count></span></span>`;
    b.onclick = () => setTool(k);
    units.appendChild(b);
  }
  setTool('radar');

  const cv = scene.renderer.domElement;
  cv.addEventListener('click', onClick);
  cv.addEventListener('pointermove', onMove);
  cv.addEventListener('pointerleave', () => { S.hover = -1; S.preview = null; redraw(); });
  cv.addEventListener('contextmenu', (e) => e.preventDefault());
  $('#btn-clear').onclick = () => { S.layout = { radar: [], jammer: [], pad: [] }; layoutChanged(); };
  $('#btn-launch').onclick = launch;
  $('#btn-ai').onclick = loadAiLayout;
  $('#btn-edit').onclick = () => setPhase('defend');
  $('#btn-replay').onclick = () => startReplay();
  $('#btn-skip').onclick = () => showDebrief();
  $('#btn-suggest').onclick = placeSuggested;
  $('#btn-play').onclick = togglePlay;
  $('#btn-hood').onclick = openHood;
  $('#btn-help').onclick = () => $('#dlg-intro').showModal();
  $('#scrub').oninput = (e) => { if (S.replay) { S.replay.t = +e.target.value; S.replay.lastIdx = Math.floor(S.replay.t); S.replay.done = false; } };
  for (const b of document.querySelectorAll('#seg-speed button')) b.onclick = () => {
    document.querySelectorAll('#seg-speed button').forEach((x) => x.classList.toggle('on', x === b));
    if (S.replay) S.replay.speed = +b.dataset.speed;
  };
  for (const cb of document.querySelectorAll('#toggles input')) cb.onchange = () => {
    S.ov[cb.dataset.ov] = cb.checked;
    if (cb.dataset.ov === 'astar') startAstarAnim(cb.checked);
    redraw();
  };
  window.addEventListener('keydown', onKey);
  scene.onFrame = tick;

  await loadMap(0);
  if (!store.get('rc-intro')) { $('#dlg-intro').showModal(); store.set('rc-intro', '1'); }
}

function setDiff(k) {
  S.diff = k;
  store.set('rc-diff', k);
  for (const b of document.querySelectorAll('#seg-diff button')) { b.classList.toggle('on', b.dataset.key === k); b.setAttribute('aria-checked', b.dataset.key === k); }
  $('#diff-blurb').textContent = S.cfg.difficulties.find((d) => d.key === k).blurb;
}

function setTool(k) {
  S.tool = k;
  for (const b of document.querySelectorAll('.unit')) b.classList.toggle('on', b.dataset.kind === k);
  S.preview = null;
  redraw();
}

async function loadMap(id) {
  S.mapId = id;
  $('#sel-map').value = id;
  S.map = await api(`/api/map/${id}`);
  $('#map-blurb').textContent = S.map.blurb;
  $('#btn-ai').hidden = !S.map.recommended;
  scene.setMap(S.map);
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
  if (S.phase !== 'defend') return;
  const c = scene.pickCell(ev);
  if (c < 0) return;
  const occ = occupant(c);
  if (occ) { S.layout[occ] = S.layout[occ].filter((x) => x !== c); layoutChanged(); return; }
  const err = canPlace(S.tool, c);
  if (err) { flashError(err); return; }
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
  if (S.tool === 'radar' && c >= 0 && !canPlace('radar', c)) {
    const seq = ++previewSeq;
    previewTimer = setTimeout(async () => {
      const lay = { ...S.layout, radar: [c] , jammer: [], pad: [] };
      try {
        const cov = await api('/api/coverage', { map: S.mapId, layout: lay });
        if (seq === previewSeq && S.hover === c) { S.preview = cov.radar; redraw(); }
      } catch { /* ignore */ }
    }, 90);
  }
  redraw();
}

async function layoutChanged() {
  const left = S.cfg.budget - spend();
  $('#budget-left').textContent = left;
  $('#budget-fill').style.width = `${(100 * left) / S.cfg.budget}%`;
  for (const b of document.querySelectorAll('.unit')) {
    const k = b.dataset.kind;
    b.querySelector('[data-count]').textContent = `${S.layout[k].length}/${S.cfg.units[k].max}`;
  }
  scene.setUnits(S.layout);
  try { S.coverage = await api('/api/coverage', { map: S.mapId, layout: S.layout }); } catch (e) { flashError(e.message); }
  redraw();
}

function flashError(msg) {
  const el = $('#err');
  el.textContent = msg;
  clearTimeout(flashError.t);
  flashError.t = setTimeout(() => (el.textContent = ''), 2200);
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
  if (err) { flashError(err === 'Over budget' ? 'Free up 3 budget points first: click a unit to remove it.' : err); setTool('radar'); return; }
  S.layout.radar.push(c);
  layoutChanged();
}

// ------------------------------------------------------------------ phases
function setPhase(p) {
  S.phase = p;
  for (const el of document.querySelectorAll('.phase')) el.classList.toggle('on', el.dataset.phase === p);
  $('#panel-right').hidden = p !== 'debrief';
  $('#playback').hidden = p !== 'attack';
  $('#legend').hidden = p === 'attack';
  for (const el of document.querySelectorAll('#panel-left button, #panel-left select')) el.disabled = p === 'attack';
  if (p !== 'debrief') { scene.clearPaths(); S.astarAnim = null; }
  if (p === 'defend') { scene.resetActors(0, 0); $('#feed').innerHTML = ''; S.replay = null; }
  if (p === 'attack') scene.resetActors(0, 0);
  scene.resize();
  updateTags();
  redraw();
}

async function launch() {
  $('#simulating').hidden = false;
  $('#sim-count').textContent = S.diff === 'scripted' ? 32 : 64;
  try {
    S.result = await api('/api/simulate', { map: S.mapId, layout: S.layout, difficulty: S.diff });
    startReplay();
  } catch (e) {
    flashError(e.message);
  } finally {
    $('#simulating').hidden = true;
  }
}

// ------------------------------------------------------------------ replay
function startReplay() {
  const frames = S.result.rep.frames;
  setPhase('attack');
  scene.resetActors(frames[0].d.length, frames[0].i.length);
  scene.setUnits(S.layout);
  $('#feed').innerHTML = '';
  S.replay = { t: 0, lastIdx: 0, playing: true, speed: +document.querySelector('#seg-speed .on').dataset.speed, seen: new Set(), done: false };
  $('#scrub').max = frames.length - 1;
  $('#btn-play').textContent = '❚❚';
  feed(`${S.cfg.drones} hostile drones inbound`, 'var(--enemy)');
}

function togglePlay() {
  if (!S.replay) return;
  S.replay.playing = !S.replay.playing;
  $('#btn-play').textContent = S.replay.playing ? '❚❚' : '▶';
  $('#btn-play').setAttribute('aria-label', S.replay.playing ? 'Pause' : 'Play');
}

function feed(msg, color) {
  const el = document.createElement('div');
  el.style.setProperty('--c', color);
  el.textContent = `T+${String(Math.floor(S.replay?.t ?? 0)).padStart(2, '0')}  ${msg}`;
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
  const frames = S.result.rep.frames, F = frames.length;
  if (R.playing) R.t = Math.min(F - 1, R.t + dt * STEPS_PER_SEC * R.speed);
  const idx = Math.min(Math.floor(R.t), F - 1), f = R.t - idx;
  const A = frames[idx], B = frames[Math.min(idx + 1, F - 1)];

  // events for every frame boundary we crossed
  while (R.lastIdx < idx) { R.lastIdx++; frameEvents(frames[R.lastIdx - 1], frames[R.lastIdx]); }

  A.d.forEach((a, i) => {
    const b = B.d[i];
    scene.updateDrone(i, { visible: a[1] === 0, cellA: a[0], cellB: b[0], f, detected: (f < 0.5 ? a[2] : b[2]) === 1, jammed: b[3] === 1 });
  });
  A.i.forEach((a, k) => {
    const b = B.i[k];
    const visible = a[1] === 1 || (a[1] === 0 && b[1] === 1);
    scene.updateIcpt(k, { visible, cellA: a[0], cellB: b[0], f });
  });
  $('#scrub').value = R.t;
  $('#clock').textContent = `T+${String(idx).padStart(2, '0')}`;
  if (R.t >= F - 1 && !R.done) { R.done = true; setTimeout(() => { if (S.phase === 'attack' && S.replay === R) showDebrief(); }, 1400); }
}

function frameEvents(prev, cur) {
  const R = S.replay;
  cur.d.forEach((d, i) => {
    const p = prev.d[i];
    if (p[1] === 0 && d[1] === 1) { scene.explode(d[0]); feed(`Drone ${i + 1} destroyed`, 'var(--pad)'); }
    if (p[1] === 0 && d[1] === 2) { scene.flashBase(); feed(`BREACH: drone ${i + 1} reached the base`, 'var(--enemy)'); }
    if (d[2] === 1 && !R.seen.has(i)) { R.seen.add(i); feed(`Drone ${i + 1} detected`, 'var(--radar)'); }
  });
  cur.i.forEach((it, k) => {
    const p = prev.i[k];
    if (p[1] === 0 && it[1] === 1) feed('Interceptor launched', 'var(--interceptor, #7fe3ff)');
  });
}

// ------------------------------------------------------------------ debrief
function showDebrief() {
  const r = S.result, db = r.debrief;
  setPhase('debrief');
  scene.resetActors(0, 0);
  $('#feed').innerHTML = '';
  const diff = S.cfg.difficulties.find((d) => d.key === r.difficulty);
  $('#db-diff').textContent = diff.label.toUpperCase();
  const pct = Math.round(r.breach_rate * 100);
  $('#db-rate').textContent = `${pct}%`;
  $('#db-rate').style.color = pct <= 15 ? 'var(--good)' : pct <= 40 ? 'var(--jammer)' : 'var(--enemy)';
  $('#db-n').textContent = r.battles;
  const hist = $('#db-hist');
  hist.innerHTML = '';
  const mx = Math.max(...r.breach_hist);
  r.breach_hist.forEach((v, i) => {
    const bar = document.createElement('div');
    bar.style.height = `${Math.max(3, (100 * v) / mx)}%`;
    bar.style.background = i === 0 ? 'var(--good)' : `color-mix(in srgb, var(--enemy) ${30 + i * 12}%, var(--line-2))`;
    bar.innerHTML = `<span>${v}</span>`;
    bar.title = `${v} attacks with ${i} breaches`;
    hist.appendChild(bar);
  });
  $('#db-det').textContent = `${Math.round(db.detection_rate * 100)}%`;
  $('#db-kills').textContent = r.mean_kills.toFixed(1);
  $('#db-replay').textContent = `${r.rep.breaches}/${S.cfg.drones}`;

  const bs = db.blind_spot;
  $('#db-blind').hidden = !bs;
  if (bs) { $('#db-blind-text').textContent = bs.text; $('#db-blind-tip').textContent = bs.tip; }

  const rec = r.recommended;
  const cmp = $('#db-compare');
  const recRate = rec?.breach?.[r.difficulty];
  cmp.hidden = recRate == null;
  if (recRate != null) {
    const diffPts = Math.round((r.breach_rate - recRate) * 100);
    $('#db-compare-text').innerHTML = `An AI-searched layout lets <b>${Math.round(recRate * 100)}%</b> through against this enemy. ` +
      (diffPts <= 0 ? '<b style="color:var(--good)">You matched or beat it.</b>' : `You're <b>${diffPts} points</b> behind. Try <i>Load AI layout</i> to study it.`);
  }
  $('#astar-n').textContent = db.astar.explored.length;
  if (S.ov.astar) startAstarAnim(true);
  updateTags();
  redraw();
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
  // the learned / scripted drones' actual routes in the replay, for comparison
  const frames = S.result.rep.frames;
  for (let d = 0; d < frames[0].d.length; d++) {
    const cells = [];
    for (const fr of frames) { if (fr.d[d][1] !== 0 && cells.length) { cells.push(fr.d[d][0]); break; } if (fr.d[d][1] === 0) cells.push(fr.d[d][0]); }
    scene.drawPath(cells, 0xff4d5e, { radius: 0.045, lift: 0.7, opacity: 0.6 });
  }
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
  add('BASE', S.map.base_cell, '#e8eef4', 3);
  const spawnCells = S.map.spawn.map((v, i) => (v ? i : -1)).filter((i) => i >= 0);
  if (spawnCells.length) add('DRONE ENTRY', spawnCells[Math.floor(spawnCells.length / 2)], COLORS.enemy, 1.2);
  const bs = S.phase === 'debrief' && S.result?.debrief?.blind_spot;
  if (bs) {
    add('BLIND SPOT', bs.cell, COLORS.enemy, 1.6, true);
    add('SUGGESTED RADAR', bs.suggest_radar, COLORS.radar, 1.6);
  }
  scene.setTags(tags);
}

// ------------------------------------------------------------------ keyboard
function onKey(e) {
  if (e.target.matches('input, select, textarea') || document.querySelector('dialog[open]')) return;
  const tool = Object.entries(UNIT_INFO).find(([, u]) => u.key === e.key);
  if (tool && S.phase === 'defend') setTool(tool[0]);
  if (e.key === ' ' && S.phase === 'attack') { e.preventDefault(); togglePlay(); }
}

// ------------------------------------------------------------------ under the hood
async function openHood() {
  $('#dlg-hood').showModal();
  const chart = $('#chart');
  let data;
  try { data = await api('/api/training'); } catch { data = { metrics: [] }; }
  // Prefer the per-checkpoint curve on "strong" defenses (from select_difficulties.py); fall back to the training log.
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
    ? `Training curve: breach rate vs. ${data.strong_layouts} strong defenses (ones that stop A* most of the time)`
    : 'Training curve: breach rate on a fixed benchmark of 512 defenses';
  const W = 600, H = 300, P = { l: 46, r: 16, t: 16, b: 34 };
  const maxX = pts[pts.length - 1].steps * S.cfg.drones;
  const x = (v) => P.l + (v / maxX) * (W - P.l - P.r);
  const y = (v) => H - P.b - v * (H - P.t - P.b);
  const line = (key) => pts.map((m, i) => `${i ? 'L' : 'M'}${x(m.steps * S.cfg.drones).toFixed(1)},${y(m[key] ?? 0).toFixed(1)}`).join('');
  let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Breach rate rising over training">`;
  for (const v of [0, 0.25, 0.5, 0.75, 1]) svg += `<line x1="${P.l}" x2="${W - P.r}" y1="${y(v)}" y2="${y(v)}" stroke="#1f2a36"/><text x="${P.l - 8}" y="${y(v) + 4}" fill="#7f8c99" font-size="11" text-anchor="end" font-family="monospace">${v * 100}%</text>`;
  for (let i = 0; i <= 4; i++) { const v = (maxX * i) / 4; svg += `<text x="${x(v)}" y="${H - 12}" fill="#7f8c99" font-size="11" text-anchor="middle" font-family="monospace">${(v / 1e6).toFixed(v < 1e7 ? 1 : 0)}M</text>`; }
  svg += `<text x="${W - P.r}" y="${H - 1}" fill="#7f8c99" font-size="10" text-anchor="end">drone decisions trained</text>`;
  if (data.baseline != null) svg += `<line x1="${P.l}" x2="${W - P.r}" y1="${y(data.baseline)}" y2="${y(data.baseline)}" stroke="#ffd166" stroke-dasharray="5 4"/><text x="${W - P.r - 4}" y="${y(data.baseline) + 14}" fill="#ffd166" font-size="11" text-anchor="end">Scripted A* attacker</text>`;
  if (strong) svg += `<path d="${line('other')}" fill="none" stroke="#7f8c99" stroke-width="1.4" stroke-dasharray="2 3"/><text x="${W - P.r - 4}" y="${y(pts[pts.length - 1].other) - 6}" fill="#7f8c99" font-size="10" text-anchor="end">vs. random defenses</text>`;
  for (const [key, lv] of Object.entries(data.levels || {})) {
    if (key !== 'veteran' || !stepsAt.has(lv.update)) continue;   // easier levels are the same checkpoint with noise
    const px = x(stepsAt.get(lv.update) * S.cfg.drones), py = y(lv.eval_breach);
    svg += `<line x1="${px}" x2="${px}" y1="${P.t}" y2="${H - P.b}" stroke="#2a3846" stroke-dasharray="3 3"/><circle cx="${px}" cy="${py}" r="5" fill="#ff4d5e" stroke="#080b10" stroke-width="2"/><text x="${px + 7}" y="${py - 8}" fill="#dbe4ec" font-size="11" font-weight="600">${key[0].toUpperCase() + key.slice(1)}</text>`;
  }
  svg += `<path d="${line('main')}" fill="none" stroke="#ff4d5e" stroke-width="2.2"/></svg>`;
  chart.innerHTML = svg;
  const last = data.metrics[data.metrics.length - 1];
  const fmt = (n) => (n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : n >= 1e3 ? `${(n / 1e3).toFixed(0)}K` : n);
  $('#hood-stats').innerHTML = `
    <div><b>${fmt(last.steps * S.cfg.drones)}</b><span class="muted small">drone decisions</span></div>
    <div><b>${(last.time / 60).toFixed(0)} min</b><span class="muted small">training time</span></div>
    <div><b>${fmt(last.sps * S.cfg.drones)}/s</b><span class="muted small">${data.device || 'GPU'}</span></div>` +
    (data.heldout ? `<div class="wide-stat"><b>${Math.round(data.heldout.veteran * 100)}% vs ${Math.round(data.heldout.scripted * 100)}%</b><span class="muted small">Veteran vs scripted A* breach rate on ${data.heldout.layouts} held-out strong defenses (never used for training or selection)</span></div>` : '');
}

window.__rc = { S, scene };   // handy for debugging from the console
init().catch((e) => { console.error(e); flashError(`Failed to start: ${e.message}`); });
