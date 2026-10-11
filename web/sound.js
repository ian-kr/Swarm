// Procedural sound effects with the Web Audio API: no audio files, so the game stays small and offline.
// Browsers only allow audio after a user gesture, so the context is created on the first click or key.

let ctx = null, master = null, noiseBuf = null, hum = null;
let enabled = true;
const lastAt = {};

try { enabled = localStorage.getItem('rc-sound') !== 'off'; } catch { /* storage unavailable */ }

function unlock() {
  if (!ctx) {
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) return;
    ctx = new AC();
    const comp = ctx.createDynamicsCompressor();
    comp.threshold.value = -16;
    comp.ratio.value = 4;
    master = ctx.createGain();
    master.gain.value = enabled ? 0.7 : 0;
    master.connect(comp).connect(ctx.destination);
    noiseBuf = ctx.createBuffer(1, ctx.sampleRate * 2, ctx.sampleRate);
    const d = noiseBuf.getChannelData(0);
    for (let i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1;
  }
  if (ctx.state === 'suspended') ctx.resume();
}
window.addEventListener('pointerdown', unlock, { capture: true });
window.addEventListener('keydown', unlock, { capture: true });

const ready = () => ctx && ctx.state === 'running' && enabled;
// rate limit per effect, so scrubbing or simultaneous events don't pile up into noise
const limited = (key, ms) => { const t = performance.now(); if (lastAt[key] && t - lastAt[key] < ms) return true; lastAt[key] = t; return false; };

function tone({ type = 'sine', f0, f1, dur, gain = 0.2, attack = 0.005, delay = 0 }) {
  const t = ctx.currentTime + delay;
  const o = ctx.createOscillator(), g = ctx.createGain();
  o.type = type;
  o.frequency.setValueAtTime(f0, t);
  if (f1) o.frequency.exponentialRampToValueAtTime(f1, t + dur);
  g.gain.setValueAtTime(0.0001, t);
  g.gain.exponentialRampToValueAtTime(gain, t + attack);
  g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
  o.connect(g).connect(master);
  o.start(t);
  o.stop(t + dur + 0.05);
}

function noise({ dur, gain = 0.2, type = 'lowpass', f0 = 2000, f1 = 200, q = 0.8, delay = 0 }) {
  const t = ctx.currentTime + delay;
  const s = ctx.createBufferSource(), f = ctx.createBiquadFilter(), g = ctx.createGain();
  s.buffer = noiseBuf;
  f.type = type;
  f.Q.value = q;
  f.frequency.setValueAtTime(f0, t);
  f.frequency.exponentialRampToValueAtTime(Math.max(f1, 20), t + dur);
  g.gain.setValueAtTime(gain, t);
  g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
  s.connect(f).connect(g).connect(master);
  s.start(t, Math.random());
  s.stop(t + dur + 0.05);
}

export const sfx = {
  get enabled() { return enabled; },
  setEnabled(on) {
    enabled = on;
    try { localStorage.setItem('rc-sound', on ? 'on' : 'off'); } catch { /* storage unavailable */ }
    if (master) master.gain.setTargetAtTime(on ? 0.7 : 0, ctx.currentTime, 0.05);
  },

  // ---- planning
  click() { if (ready() && !limited('click', 40)) tone({ f0: 1100, dur: 0.05, gain: 0.05 }); },
  place(kind) {
    if (!ready()) return;
    const f = { radar: 330, jammer: 262, pad: 196 }[kind] || 262;
    tone({ type: 'triangle', f0: f, f1: f * 0.75, dur: 0.14, gain: 0.22 });
    noise({ dur: 0.05, gain: 0.12, type: 'highpass', f0: 3000, f1: 1500 });
  },
  remove() { if (ready()) tone({ type: 'triangle', f0: 180, f1: 110, dur: 0.14, gain: 0.16 }); },
  error() {
    if (!ready() || limited('error', 200)) return;
    tone({ type: 'square', f0: 150, dur: 0.11, gain: 0.06 });
    tone({ type: 'square', f0: 118, dur: 0.14, gain: 0.06, delay: 0.09 });
  },
  commit() {
    if (!ready()) return;
    tone({ f0: 220, f1: 880, dur: 0.45, gain: 0.14 });
    noise({ dur: 0.5, gain: 0.08, type: 'bandpass', f0: 400, f1: 3000, q: 2 });
  },

  // ---- battle
  ping() {   // radar detection: a sonar-like ping with a faint echo
    if (!ready() || limited('ping', 120)) return;
    tone({ f0: 1320, f1: 1290, dur: 0.4, gain: 0.14 });
    tone({ f0: 2640, dur: 0.18, gain: 0.03 });
    tone({ f0: 1320, f1: 1290, dur: 0.3, gain: 0.04, delay: 0.2 });
  },
  launch() {   // interceptor: low thump and a rising whoosh
    if (!ready() || limited('launch', 90)) return;
    tone({ type: 'sawtooth', f0: 95, f1: 40, dur: 0.22, gain: 0.1 });
    noise({ dur: 0.5, gain: 0.18, type: 'bandpass', f0: 500, f1: 3200, q: 1.2 });
  },
  boom() {
    if (!ready() || limited('boom', 80)) return;
    noise({ dur: 0.9, gain: 0.45, f0: 1800, f1: 70 });
    tone({ f0: 72, f1: 34, dur: 0.6, gain: 0.35 });
  },
  breach() {   // two-tone alarm
    if (!ready() || limited('breach', 400)) return;
    for (let i = 0; i < 4; i++) tone({ type: 'triangle', f0: i % 2 ? 660 : 880, dur: 0.19, gain: 0.16, delay: i * 0.2 });
  },
  // continuous rotor hum, louder with more drones airborne
  hum(n) {
    if (!ctx) return;
    if (!hum && n > 0 && enabled) {
      const o1 = ctx.createOscillator(), o2 = ctx.createOscillator(), lfo = ctx.createOscillator();
      const f = ctx.createBiquadFilter(), g = ctx.createGain(), am = ctx.createGain(), lg = ctx.createGain();
      o1.type = o2.type = 'sawtooth';
      o1.frequency.value = 148;
      o2.frequency.value = 153;   // slight detune = the beating of several rotors
      lfo.frequency.value = 38;
      lg.gain.value = 0.35;
      f.type = 'lowpass';
      f.frequency.value = 700;
      g.gain.value = 0;
      am.gain.value = 0.65;
      lfo.connect(lg).connect(am.gain);
      o1.connect(f); o2.connect(f);
      f.connect(am).connect(g).connect(master);
      for (const o of [o1, o2, lfo]) o.start();
      hum = { g, nodes: [o1, o2, lfo] };
    }
    if (hum) hum.g.gain.setTargetAtTime(enabled ? Math.min(0.05, 0.009 * n) : 0, ctx.currentTime, 0.25);
  },
  humStop() {
    if (!hum) return;
    const h = hum;
    hum = null;
    h.g.gain.setTargetAtTime(0, ctx.currentTime, 0.15);
    setTimeout(() => h.nodes.forEach((o) => o.stop()), 800);
  },

  // ---- documents
  stamp() {
    if (!ready()) return;
    noise({ dur: 0.18, gain: 0.35, f0: 900, f1: 120 });
    tone({ f0: 95, f1: 60, dur: 0.16, gain: 0.28 });
  },
  tick() { if (ready()) tone({ f0: 1500, dur: 0.05, gain: 0.05 }); },
};
