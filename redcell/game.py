"""Game-side logic: run a player's defense against an opponent and build the debrief.

Kept separate from the HTTP layer so it can be tested and scripted directly.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from . import config as C
from .astar import ScriptedAttacker, astar, threat_cost
from .env import ALIVE, BREACHED, FLYING, KILLED, SPENT, DefenseEnv, Layout, MapTensors
from .policy import load_policy
from .terrain import load_maps

ROOT = Path(__file__).resolve().parent.parent
G, N = C.GRID, C.N_CELLS
COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
COMPASS_LONG = {"N": "northern", "NE": "north-eastern", "E": "eastern", "SE": "south-eastern",
                "S": "southern", "SW": "south-western", "W": "western", "NW": "north-western"}

DIFFICULTIES = [
    ("scripted", "Scripted (A*)", "Classic AI: every drone flies the cheapest A* route around your known sensors."),
    ("recruit", "Recruit", "The trained swarm with very noisy decisions: it often wanders into your sensors."),
    ("regular", "Regular", "The trained swarm with some decision noise: about as dangerous as the A* planner."),
    ("veteran", "Veteran", "The fully trained swarm: about 40% more breaches than A* against strong defenses."),
]


class LayoutError(ValueError):
    pass


class Game:
    def __init__(self, ckpt_dir: Path | None = None, device: str = "cpu"):
        self.ms = load_maps()
        self.dev = torch.device(device)
        self.mt = MapTensors(self.ms, self.dev)
        self.euclid = self.mt.euclid.cpu().numpy()
        self.policies = {}
        self.difficulty_info = {}
        self._load_difficulties(ckpt_dir)
        rec = ROOT / "data" / "recommended.json"
        self.recommended = json.loads(rec.read_text()) if rec.exists() else {}

    # ----------------------------------------------------------- loading
    def _load_difficulties(self, ckpt_dir):
        path = ROOT / "data" / "difficulties.json"
        if not path.exists():
            return
        spec = json.loads(path.read_text())
        for key, entry in spec.get("levels", {}).items():
            p = ROOT / entry["checkpoint"]
            if not p.exists():
                continue
            try:
                net = load_policy(p, self.dev)
                net.temperature = float(entry.get("temperature", 1.0))
                self.policies[key] = net
                self.difficulty_info[key] = entry
            except (RuntimeError, KeyError) as e:    # e.g. a model from an older network version
                print(f"skipping difficulty '{key}': {p} does not match the current network ({type(e).__name__})")

    def difficulties(self):
        out = []
        for key, label, blurb in DIFFICULTIES:
            available = key == "scripted" or key in self.policies
            info = self.difficulty_info.get(key, {})
            out.append(dict(key=key, label=label, blurb=blurb, available=available,
                            update=info.get("update"), eval_breach=info.get("eval_breach")))
        return out

    # ----------------------------------------------------------- helpers
    def map_payload(self, i):
        ms = self.ms
        return dict(id=i, name=ms.names[i], blurb=ms.blurbs[i],
                    heights=np.round(ms.heights[i].reshape(-1), 3).tolist(),
                    passable=ms.passable[i].astype(int).tolist(),
                    place=ms.place_mask[i].astype(int).tolist(),
                    spawn=ms.spawn_mask[i].astype(int).tolist(),
                    base_cell=int(ms.base_cell[i]),
                    base=ms.base_mask[i].astype(int).tolist(),
                    recommended=self.recommended.get(str(i)))

    def validate(self, map_id: int, layout: dict) -> dict:
        if not 0 <= map_id < self.ms.n:
            raise LayoutError("unknown map")
        clean, used, spend = {}, set(), 0
        for kind, spec in C.UNITS.items():
            cells = [int(c) for c in layout.get(kind, [])]
            if len(cells) > spec["max"]:
                raise LayoutError(f"at most {spec['max']} {kind}s")
            for c in cells:
                if not (0 <= c < N) or not self.ms.place_mask[map_id, c]:
                    raise LayoutError(f"{kind} at cell {c} is not on placeable ground")
                if c in used:
                    raise LayoutError("two units on one cell")
                used.add(c)
            spend += spec["cost"] * len(cells)
            clean[kind] = cells
        if spend > C.BUDGET:
            raise LayoutError(f"over budget ({spend}/{C.BUDGET})")
        return clean

    def coverage(self, map_id: int, layout: dict) -> dict:
        layout = self.validate(map_id, layout)
        env = DefenseEnv(self.mt, 1, self.dev, auto_reset=False, seed=0)
        env.reset(map_id=map_id, layout=Layout.from_lists([layout], self.dev))
        return dict(radar=env.coverage[0].int().tolist(), jammer=env.jam_zone[0].int().tolist(),
                    pad=env.pad_zone[0].int().tolist())

    # ----------------------------------------------------------- simulate
    def simulate(self, map_id: int, layout: dict, difficulty: str, seed: int = 0, n: int | None = None) -> dict:
        layout = self.validate(map_id, layout)
        if difficulty != "scripted" and difficulty not in self.policies:
            raise LayoutError(f"difficulty '{difficulty}' is not available (no trained checkpoint)")
        n = n or (32 if difficulty == "scripted" else 64)
        env = DefenseEnv(self.mt, n, self.dev, auto_reset=False, seed=seed)
        env.reset(map_id=map_id, layout=Layout.from_lists([layout] * n, self.dev))
        env.record_events = True
        atk = ScriptedAttacker(self.ms) if difficulty == "scripted" else None
        net = self.policies.get(difficulty)

        cells = [env.drone_cell.cpu().numpy()]
        status = [env.drone_status.cpu().numpy()]
        det = [env.detected.cpu().numpy()]
        jam = [np.zeros_like(det[0])]
        icell = [env.int_cell.cpu().numpy()]
        istate = [env.int_state.cpu().numpy()]
        with torch.no_grad():
            for _ in range(C.MAX_STEPS):
                a = atk.act(env) if atk else net.act_env(env.observe())[0]
                _, _, _, info = env.step(a)
                cells.append(env.drone_cell.cpu().numpy())
                status.append(env.drone_status.cpu().numpy())
                det.append(env.detected.cpu().numpy())
                jam.append(env.jammed.cpu().numpy())
                icell.append(env.int_cell.cpu().numpy())
                istate.append(env.int_state.cpu().numpy())
                if info["env_done"].all():
                    break
        cells, status, det, jam = map(np.stack, (cells, status, det, jam))     # (T, n, D)
        icell, istate = np.stack(icell), np.stack(istate)                     # (T, n, KI)
        breaches = (status[-1] == BREACHED).sum(1)
        kills = (status[-1] == KILLED).sum(1)

        # representative battle: breach count closest to the median
        med = np.median(breaches)
        rep = int(np.argmin(np.abs(breaches - med) + 1e-3 * np.arange(n)))
        T = cells.shape[0]
        last = T
        for t in range(T):
            if not (status[t, rep] == ALIVE).any():
                last = t + 1
                break
        frames = [dict(d=[[int(cells[t, rep, d]), int(status[t, rep, d]), int(det[t, rep, d]), int(jam[t, rep, d])]
                          for d in range(C.N_DRONES)],
                       i=[[int(icell[t, rep, k]), int(istate[t, rep, k])] for k in range(icell.shape[2])])
                  for t in range(last)]
        events = [dict(kind=k, drone=d, cell=c) for (k, e, d, c) in env.events if e == rep]

        debrief = self._debrief(map_id, layout, cells, status, det, env)
        result = dict(
            difficulty=difficulty, battles=n, seed=seed,
            breach_rate=float(breaches.mean() / C.N_DRONES),
            breach_hist=np.bincount(breaches, minlength=C.N_DRONES + 1).tolist(),
            mean_kills=float(kills.mean()),
            rep=dict(index=rep, breaches=int(breaches[rep]), kills=int(kills[rep]), frames=frames, events=events),
            debrief=debrief,
        )
        rec = self.recommended.get(str(map_id))
        if rec:
            result["recommended"] = rec
        return result

    # ----------------------------------------------------------- debrief
    def _debrief(self, map_id, layout, cells, status, det, env):
        ms = self.ms
        T, n, D = cells.shape
        final = status[-1]
        breached = final == BREACHED
        h = ms.heights[map_id].reshape(-1)
        bc = int(ms.base_cell[map_id])
        bx, by = bc % G, bc // G

        traffic = np.zeros(N)
        sneak = np.zeros(N)            # cells breaching drones flew through UNDETECTED
        ever_det = det.any(0)
        for t in range(T):
            alive_t = status[t] == ALIVE
            np.add.at(traffic, cells[t][alive_t], 1)
            m = alive_t & breached & ~det[t]
            np.add.at(sneak, cells[t][m], 1)
        y = np.arange(N) // G
        far = np.hypot(np.arange(N) % G - bx, y - by) > C.BASE_RADIUS + 1
        sneak *= (y >= C.SPAWN_ROWS + 2) & far

        out = dict(
            heat_breach=_norm(sneak), heat_traffic=_norm(traffic),
            detection_rate=float(ever_det.mean()),
            undetected_breaches=int((breached & ~ever_det).sum()),
            total_breaches=int(breached.sum()),
        )

        # blind spot: densest corridor of undetected breaching traffic
        if sneak.sum() > 0:
            sm = _box(sneak.reshape(G, G)).reshape(-1)
            c = int(np.argmax(sm))
            cx, cy = c % G, c // G
            ang = math.degrees(math.atan2(cx - bx, -(cy - by))) % 360
            sector = COMPASS[int((ang + 22.5) // 45) % 8]
            radars = layout.get("radar", [])
            in_range = [r for r in radars if math.hypot(r % G - cx, r // G - cy) <= C.UNITS["radar"]["range"]]
            if not radars:
                why = "with no radar coverage at all"
            elif not in_range:
                why = "outside the range of every radar"
            elif not any(ms.vis[map_id, r, c] for r in in_range):
                why = "in a radar shadow: terrain blocks the line of sight from your radar"
            else:
                why = "through gaps between sensor footprints"
            local = h.reshape(G, G)[max(0, cy - 2):cy + 3, max(0, cx - 2):cx + 3]
            terrain = "low ground" if h[c] < local.mean() - 0.03 else ("high ground" if h[c] > local.mean() + 0.03 else "open ground")
            ghosts = int((breached & ~ever_det).sum())
            ghost_txt = (f" {ghosts} of {int(breached.sum())} drones that got through were never detected at all."
                         if ghosts else " They were only picked up when it was too late to intercept.")

            # suggested fix: the placeable cell whose radar would see most of that corridor
            R = C.UNITS["radar"]["range"]
            seen = ms.vis[map_id] & (self.euclid <= R)
            score = seen.astype(np.float32) @ sm
            occupied = set(sum(layout.values(), []))
            score[~ms.place_mask[map_id]] = -1
            score[list(occupied)] = -1
            fix = int(np.argmax(score))
            out["blind_spot"] = dict(
                cell=c, sector=sector, terrain=terrain, reason=why, suggest_radar=fix,
                text=(f"Your blind spot was the {COMPASS_LONG[sector]} approach: {terrain} about "
                      f"{round(math.hypot(cx - bx, cy - by))} cells from the base. Drones that broke through "
                      f"flew this stretch undetected, {why}.{ghost_txt}"),
                tip=f"A radar at the marked spot would cover {round(100 * score[fix] / max(sm.sum(), 1e-9))}% of that corridor.",
            )
        else:
            out["blind_spot"] = None

        # pathfinding view: what the textbook A* planner does against this layout
        spawn = np.nonzero(ms.spawn_mask[map_id])[0]
        start = int(spawn[len(spawn) // 2])
        extra = threat_cost(env.coverage[0].cpu().numpy(), env.jam_zone[0].cpu().numpy(), env.pad_zone[0].cpu().numpy())
        path, order = astar(ms.passable[map_id], ms.cell_cost[map_id], start, ms.base_mask[map_id], (bx, by), extra, trace=True)
        plain = astar(ms.passable[map_id], ms.cell_cost[map_id], start, ms.base_mask[map_id], (bx, by))
        out["astar"] = dict(start=start, explored=order, path=path, naive_path=plain)
        return out


def _norm(a):
    m = a.max()
    return np.round(a / m, 3).tolist() if m > 0 else a.tolist()


def _box(a):
    p = np.pad(a, 1)
    return sum(p[i:i + G, j:j + G] for i in range(3) for j in range(3))
