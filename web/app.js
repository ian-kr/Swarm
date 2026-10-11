import { Scene3D, COLORS } from '/static/scene.js';

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
  [S.cfg, S.training] = await Promise.all([api('/api/config'), api('/api/training').catch(() => ({}))]);
  $('#budget-total').textContent = S.cfg.budget;
  const sel = $('#sel-map');
  for (const m of S.cfg.maps) sel.add(new Option(m.name, m.id));
  sel.onchange = () => loadMap(+sel.value);

  // opposing force: threat meter = held-out breach rate against strong defenses, relative to the strongest
  const seg = $('#seg-diff');
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
  $('#hud-status').textContent = PHASE_STATUS[p];
  scene.setGrid(p === 'defend' ? 0.1 : 0.035);
  if (p !== 'defend') scene.setGhost(null, -1);
  if (p !== 'debrief') { scene.clearPaths(); S.astarAnim = null; }
  if (p !== 'attack') { setCaption(null); delete document.body.dataset.learning; }
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
  scene.setInset(S.phase === 'debrief' && wide ? rep.offsetWidth + 16 : 0);
}

async function launch() {
  if (S.phase !== 'defend' || !$('#simulating').hidden) return;
  $('#simulating').hidden = false;
  $('#sim-count').textContent = S.diff === 'scripted' ? 32 : 64;
  try {
    S.result = await api('/api/simulate', { map: S.mapId, layout: S.layout, difficulty: S.diff });
    S.result.mapName = S.cfg.maps[S.mapId].name;
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
  $('#pips').innerHTML = frames[0].d.map((_, i) => `<span class="trk" data-s="alive" title="Drone ${i + 1}">${trackSym(i + 1)}</span>`).join('');
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

  while (R.lastIdx < idx) { R.lastIdx++; frameEvents(frames[R.lastIdx - 1], frames[R.lastIdx]); }

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
  if (R.t >= F - 1 && !R.done) {
    R.done = true;
    const stalled = frames[F - 1].d.filter((d) => d[1] === 0).length;
    if (stalled) feed('', `Time limit: ${stalled} UAV${stalled > 1 ? 's' : ''} never committed to an attack`, 'var(--jammer)');
    setTimeout(() => { if (S.phase === 'attack' && S.replay === R) (R.onEnd ? R.onEnd() : showDebrief()); }, 1400);
  }
}

function frameEvents(prev, cur) {
  const R = S.replay;
  cur.d.forEach((d, i) => {
    const p = prev.d[i];
    const who = `UAV-${i + 1}`;
    if (p[1] === 0 && d[1] === 1) { scene.explode(d[0]); feed(who, 'destroyed', 'var(--pad)'); }
    if (p[1] === 0 && d[1] === 2) {
      scene.flashBase();
      feed(who, 'reached the base', 'var(--enemy)');
      const st = $('#stage');
      st.classList.remove('hit'); void st.offsetWidth; st.classList.add('hit');
    }
    if (d[2] === 1 && !R.seen.has(i)) { R.seen.add(i); feed(who, 'detected by radar', 'var(--radar)'); }
  });
  cur.i.forEach((it, k) => {
    if (prev.i[k][1] === 0 && it[1] === 1) feed('', 'Interceptor launched', '#7fe3ff');
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
  requestAnimationFrame(() => { st.textContent = stamp; });
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
  if (!c) return;
  el.innerHTML = `<div class="cap-k"></div><div class="cap-t"></div><div class="cap-s"></div><div class="cap-steps">${Array.from({ length: c.n }, (_, j) => `<i class="${j < c.i ? 'done' : j === c.i ? 'on' : ''}"></i>`).join('')}</div>`;
  el.querySelector('.cap-k').textContent = c.k;
  el.querySelector('.cap-t').textContent = c.t;
  el.querySelector('.cap-s').textContent = c.s;
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
  add('Base', S.map.base_cell, '#e8eef4', 3);
  const spawnCells = S.map.spawn.map((v, i) => (v ? i : -1)).filter((i) => i >= 0);
  if (spawnCells.length) add('Drone entry · north', spawnCells[Math.floor(spawnCells.length / 2)], COLORS.enemy, 1.2);
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
    ? `Training curve: breach rate vs. ${data.strong_layouts} strong defenses`
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
