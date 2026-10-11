"""Game-side logic: run a player's defense against an opponent and build the debrief.

Kept separate from the HTTP layer so it can be tested and scripted directly.
"""

from __future__ import annotations

import base64
import json
import math
from pathlib import Path

import numpy as np
import torch

from . import config as C
from .astar import ScriptedAttacker, astar, threat_cost
from .env import ALIVE, BREACHED, FLYING, KILLED, SPENT, DefenseEnv, Layout, MapTensors
from .policy import load_policy
from .terrain import THEATERS, artifacts, load_maps

ROOT = Path(__file__).resolve().parent.parent
G, N = C.GRID, C.N_CELLS
COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
COMPASS_NOUN = {"N": "north", "NE": "north-east", "E": "east", "SE": "south-east",
                "S": "south", "SW": "south-west", "W": "west", "NW": "north-west"}
COMPASS_LONG = {"N": "northern", "NE": "north-eastern", "E": "eastern", "SE": "south-eastern",
                "S": "southern", "SW": "south-western", "W": "western", "NW": "north-western"}

DIFFICULTIES = [
    ("scripted", "Scripted (A*)", "Classic AI: every drone flies the cheapest A* route around your known sensors."),
    ("recruit", "Recruit", "The trained swarm with very noisy decisions: it often wanders into your sensors."),
    ("regular", "Regular", "The trained swarm with some decision noise."),
    ("veteran", "Veteran", "The fully trained swarm."),
]


class LayoutError(ValueError):
    pass


class Game:
    def __init__(self, ckpt_dir: Path | None = None, device: str = "cpu", mode: str = "north"):
        self.mode = mode
        self.art = artifacts(mode)
        self.ms = load_maps(mode=mode)
        self.dev = torch.device(device)
        self.mt = MapTensors(self.ms, self.dev)
        self.euclid = self.mt.euclid.cpu().numpy()
        self.policies = {}
        self.difficulty_info = {}
        self.timeline = []        # [(info, net)]: snapshots of the Veteran's own training run, oldest first
        self.spec = {}
        self._load_difficulties(ckpt_dir)
        rec = ROOT / self.art["recommended"]
        self.recommended = json.loads(rec.read_text()) if rec.exists() else {}

    # ----------------------------------------------------------- loading
    def _load_difficulties(self, ckpt_dir):
        path = ROOT / self.art["difficulties"]
        if not path.exists():
            return
        spec = self.spec = json.loads(path.read_text())
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
        for entry in spec.get("timeline", []):
            p = ROOT / entry["checkpoint"]
            if not p.exists():
                continue
            try:
                self.timeline.append((entry, load_policy(p, self.dev)))
            except (RuntimeError, KeyError) as e:
                print(f"skipping timeline stage {p} ({type(e).__name__})")

    def timeline_info(self):
        return [dict(stage=i, update=e["update"], steps=e.get("steps"), strong=e.get("strong"),
                     random=e.get("random"), veteran=bool(e.get("veteran")))
                for i, (e, _) in enumerate(self.timeline)]

    def _policy(self, difficulty: str):
        """'scripted' -> None; 'recruit'/'regular'/'veteran'; or 'stage:<i>' for a training-timeline snapshot."""
        if difficulty == "scripted":
            return None
        if difficulty.startswith("stage:"):
            try:
                return self.timeline[int(difficulty[6:])][1]
            except (ValueError, IndexError):
                raise LayoutError(f"unknown training stage '{difficulty}'")
        if difficulty not in self.policies:
            raise LayoutError(f"difficulty '{difficulty}' is not available (no trained checkpoint)")
        return self.policies[difficulty]

    def difficulties(self):
        out = []
        for key, label, blurb in DIFFICULTIES:
            available = key == "scripted" or key in self.policies
            info = self.difficulty_info.get(key, {})
            ev = self.spec.get("heldout", {}).get("scripted") if key == "scripted" else info.get("eval_breach")
            astar = self.spec.get("heldout", {}).get("scripted")
            if key != "scripted" and ev is not None and astar:
                # measured on the held-out strong defenses, so the claim always matches the shipped model
                rel = round(100 * (ev / astar - 1))
                cmp = (f"{ev / astar:.1f}x as many drones through as A*" if rel >= 100 else
                       f"{rel}% more drones through than A*" if rel >= 5 else
                       f"{-rel}% fewer drones through than A*" if rel <= -5 else "about as many drones through as A*")
                blurb = f"{blurb} Against strong defenses it gets {cmp} ({round(100 * ev)}% vs {round(100 * astar)}%)."
            out.append(dict(key=key, label=label, blurb=blurb, available=available,
                            update=info.get("update"), eval_breach=ev))
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
    def _rollout(self, map_id, layout, net, seed, n, on_step=None):
        """Run n battles of one layout and record everything the replay and debrief need.

        Fully deterministic for a given (layout, enemy, seed, n): the environment and the drones' action
        sampling both draw from the battle's own seeded generator. That is what lets brain() re-run a
        battle later and look inside the network at exactly the decisions the replay showed.
        """
        env = DefenseEnv(self.mt, n, self.dev, auto_reset=False, seed=seed)
        env.reset(map_id=map_id, layout=Layout.from_lists([layout] * n, self.dev))
        env.record_events = True
        atk = ScriptedAttacker(self.ms) if net is None else None
        rec = {k: [] for k in ("cells", "status", "det", "jam", "icell", "istate")}

        def snap(jam):
            rec["cells"].append(env.drone_cell.cpu().numpy())
            rec["status"].append(env.drone_status.cpu().numpy())
            rec["det"].append(env.detected.cpu().numpy())
            rec["jam"].append(jam)
            rec["icell"].append(env.int_cell.cpu().numpy())
            rec["istate"].append(env.int_state.cpu().numpy())

        snap(np.zeros((n, C.N_DRONES), bool))
        # For a learned enemy, record what the network was "thinking" at every step: its action
        # probabilities, its value estimate (its own forecast of how the attack will go), the action it
        # sampled, and the move the pathfinder would have made.
        think = {"p": [], "v": [], "a": [], "f": [], "alive": []}
        with torch.no_grad():
            for t in range(C.MAX_STEPS):
                if atk:
                    a = atk.act(env)
                else:
                    obs = env.observe()
                    logits, value = net(obs["local"], obs["glob"], obs["cell"], obs["vec"])
                    probs = torch.softmax(logits / net.temperature, -1)
                    a = torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1, generator=env.gen).view(probs.shape[:-1])
                    fm = env.follow_moves()
                    think["p"].append(probs.cpu().numpy())
                    think["v"].append(value.cpu().numpy())
                    think["a"].append(a.cpu().numpy())
                    think["f"].append(fm.cpu().numpy())
                    think["alive"].append((env.drone_status == ALIVE).cpu().numpy())
                    if on_step:
                        on_step(t, env, obs, probs, value, a, fm)
                _, _, _, info = env.step(a)
                snap(env.jammed.cpu().numpy())
                if info["env_done"].all():
                    break
        return env, rec, think

    def simulate(self, map_id: int, layout: dict, difficulty: str, seed: int = 0, n: int | None = None) -> dict:
        layout = self.validate(map_id, layout)
        net = self._policy(difficulty)
        n = n or (32 if difficulty == "scripted" else 64)
        env, rec, think = self._rollout(map_id, layout, net, seed, n)
        cells, status, det, jam, icell, istate = (rec[k] for k in ("cells", "status", "det", "jam", "icell", "istate"))
        cells, status, det, jam = map(np.stack, (cells, status, det, jam))     # (T, n, D)
        icell, istate = np.stack(icell), np.stack(istate)                     # (T, n, KI)
        breaches = (status[-1] == BREACHED).sum(1)
        kills = (status[-1] == KILLED).sum(1)
        stalled = (status[-1] == ALIVE).sum(1)       # still airborne at the time limit: never committed

        T = cells.shape[0]
        P = Vv = A = Fm = Al = None
        if think["p"]:
            P, Vv, A, Fm, Al = (np.stack(think[k]) for k in ("p", "v", "a", "f", "alive"))   # (steps, n, D, ...)

        def replay(b):
            """Everything the client needs to replay battle b (and show the network's thinking in it)."""
            last = T
            for t in range(T):
                if not (status[t, b] == ALIVE).any():
                    last = t + 1
                    break
            frames = [dict(d=[[int(cells[t, b, d]), int(status[t, b, d]), int(det[t, b, d]), int(jam[t, b, d])]
                              for d in range(C.N_DRONES)],
                           i=[[int(icell[t, b, k]), int(istate[t, b, k])] for k in range(icell.shape[2])])
                      for t in range(last)]
            thinking = None
            if P is not None:
                steps = min(len(P), last - 1)
                thinking = dict(p=(np.round(P[:steps, b] * 1000)).astype(int).tolist(),   # per mille, (steps, D, 10)
                                v=np.round(Vv[:steps, b], 2).tolist(),
                                a=A[:steps, b].astype(int).tolist(),
                                f=Fm[:steps, b].astype(int).tolist())
            return dict(index=int(b), breaches=int(breaches[b]), kills=int(kills[b]), stalled=int(stalled[b]), frames=frames,
                        events=[dict(kind=k, drone=d, cell=c) for (k, e, d, c) in env.events if e == b], thinking=thinking)

        # representative battle: breach count closest to the median
        med = np.median(breaches)
        rep = int(np.argmin(np.abs(breaches - med) + 1e-3 * np.arange(n)))
        policy = None
        if A is not None:
            n_alive = max(int(Al.sum()), 1)
            policy = dict(follow=float(((A == C.FOLLOW) & Al).sum() / n_alive),
                          hover=float(((A == C.HOVER) & Al).sum() / n_alive))
        tactics = self._tactics(map_id, cells, status, det, jam, env)
        examples = {}
        for tac in tactics:
            if tac["battle"] not in examples:
                examples[tac["battle"]] = replay(tac["battle"])

        debrief = self._debrief(map_id, layout, cells, status, det, env)
        result = dict(
            difficulty=difficulty, battles=n, seed=seed,
            breach_rate=float(breaches.mean() / C.N_DRONES),
            breach_hist=np.bincount(breaches, minlength=C.N_DRONES + 1).tolist(),
            mean_kills=float(kills.mean()),
            stall_rate=float(stalled.mean() / C.N_DRONES),
            rep=replay(rep),
            policy=policy,
            tactics=tactics,
            examples={str(k): v for k, v in examples.items()},
            debrief=debrief,
        )
        rec = self.recommended.get(str(map_id))
        if rec:
            result["recommended"] = rec
        return result

    # ----------------------------------------------------------- the network view
    LOCAL_CH = ["Terrain height", "Blocked", "Radar coverage", "Jammers", "Interceptor pads", "Interceptors in flight",
                "Other drones", "Base", "Threat field"]
    GLOBAL_CH = ["Terrain height", "Blocked", "Radar coverage", "Jammers", "Interceptor pads", "Base", "Drones", "Threat field"]
    VEC = ["Base direction x", "Base direction y", "Distance to base", "Detected", "Jammed", "Time", "Position x",
           "Position y", "Swarm left", "Threat field here"]
    LOCAL_SCALE = [1, 1, 1, 1, 1, 2, 2, 1, 2]          # value ranges of the input channels, for 8-bit transport
    GLOBAL_SCALE = [1, 1, 1, 1, 1, 1, 1.5, 2]

    def brain(self, map_id: int, layout: dict, difficulty: str, seed: int, n: int, battle: int, drone: int) -> dict:
        """Re-run a simulated battle exactly and record what one drone's network did at every step:
        its inputs, every hidden layer's activations, its outputs, which last-layer neurons pushed for the
        move it took, and attention (input x gradient) over its inputs."""
        layout = self.validate(map_id, layout)
        net = self._policy(difficulty)
        if net is None:
            raise LayoutError("the scripted A* enemy has no neural network")
        if not (0 <= battle < n and 0 <= drone < C.N_DRONES):
            raise LayoutError("no such battle or drone")
        steps = []

        def capture(t, env, obs, probs, value, a, fm):
            if env.drone_status[battle, drone] != ALIVE:
                return
            act = int(a[battle, drone])
            ex = net.explain(obs["local"][battle, drone], obs["glob"][battle], obs["cell"][battle, drone],
                             obs["vec"][battle, drone], act)
            steps.append(dict(t=t, cell=int(obs["cell"][battle, drone]), a=act, f=int(fm[battle, drone]),
                              p=probs[battle, drone].numpy().copy(), v=float(value[battle, drone]),
                              local=obs["local"][battle, drone].numpy().copy(), glob=obs["glob"][battle].numpy().copy(),
                              vec=obs["vec"][battle, drone].numpy().copy(), ex=ex))

        self._rollout(map_id, layout, net, seed, n, on_step=capture)
        if not steps:
            raise LayoutError("that drone made no decisions")
        S = len(steps)
        b64 = lambda x: base64.b64encode(np.ascontiguousarray(x, dtype=np.uint8).tobytes()).decode()
        q = lambda x, scale: np.clip(np.round(x / scale * 255), 0, 255)      # value -> 0..255 given its range

        local = np.stack([st["local"] for st in steps]) / np.array(self.LOCAL_SCALE)[None, :, None, None]
        glob = np.stack([st["glob"] for st in steps]) / np.array(self.GLOBAL_SCALE)[None, :, None, None]
        layers, layer_scale = [], []
        for i in range(len(net.LAYERS)):
            a = np.stack([st["ex"]["acts"][i].numpy() for st in steps])     # (S, size), ReLU >= 0
            sc = float(np.percentile(a, 99.5)) or 1.0                         # one scale per layer over the whole flight
            layers.append(b64(q(a, sc)))
            layer_scale.append(sc)
        att_l = np.stack([st["ex"]["att_local"].numpy() for st in steps])   # (S, 9, 11, 11)
        att_g = np.stack([st["ex"]["att_glob"].numpy() for st in steps])    # (S, 8, 16, 16)
        att_v = np.stack([st["ex"]["att_vec"].numpy() for st in steps])     # (S, 10)
        total = att_l.sum((1, 2, 3)) + att_g.sum((1, 2, 3)) + att_v.sum(1) + 1e-12
        amap = att_l.sum(1)                                                   # (S, 11, 11): where in the local view
        gmap = att_g.sum(1)                                                   # (S, 16, 16)
        contrib = np.stack([st["ex"]["contrib"].numpy() for st in steps])   # (S, 256)
        top = np.argsort(-np.abs(contrib), 1)[:, :14]
        return dict(
            difficulty=difficulty, seed=seed, n=n, battle=battle, drone=drone, steps=S,
            t=[st["t"] for st in steps], cell=[st["cell"] for st in steps], a=[st["a"] for st in steps],
            f=[st["f"] for st in steps], v=[round(st["v"], 2) for st in steps],
            p=[np.round(st["p"], 3).tolist() for st in steps],
            local=b64(q(local, 1.0)), glob=b64(q(glob, 1.0)),
            vec=np.round(np.stack([st["vec"] for st in steps]), 3).tolist(),
            layers=layers, layer_scale=layer_scale,
            layer_info=[dict(name=nm, size=sz) for nm, sz in net.LAYERS],
            att_map=b64(q(amap / (amap.reshape(S, -1).max(1)[:, None, None] + 1e-12), 1.0)),
            att_gmap=b64(q(gmap / (gmap.reshape(S, -1).max(1)[:, None, None] + 1e-12), 1.0)),
            # share of all attention per input channel / feature: local 9, map 8, status 10
            att_share=np.round(np.concatenate([att_l.sum((2, 3)), att_g.sum((2, 3)), att_v], 1) / total[:, None], 4).tolist(),
            contrib_idx=top.tolist(),
            contrib=np.round(np.take_along_axis(contrib, top, 1), 4).tolist(),
            names=dict(local=self.LOCAL_CH, glob=self.GLOBAL_CH, vec=self.VEC),
            params=int(sum(p.numel() for p in net.parameters())),
        )

    # ----------------------------------------------------------- tactic discovery
    def _tactics(self, map_id, cells, status, det, jam, env, max_corridors=2):
        """Find recurring patterns in how drones broke through, across all simulated battles.

        Corridors: every breaching drone's route is resampled to a few points and grouped with routes that
        stay close to it all the way (greedy clustering on mean point distance). Each corridor is named by
        the direction it reaches the base from and by its character: flown undetected, along low ground,
        or with a long wait before the final dash.
        Swarm patterns, checked per battle: saturation (3+ breaches within 4 steps), pincer (breaches from
        directions at least 90 degrees apart within 8 steps), decoy (a drone shot down on one side shortly
        before another breaks through from a different side).
        These describe what happened, not intent: the swarm was only ever rewarded for getting through.
        """
        ms = self.ms
        T, n, D = cells.shape
        bc = int(ms.base_cell[map_id])
        bx, by = bc % G, bc // G
        h = ms.heights[map_id]
        local = sum(np.pad(h, 2, mode="edge")[i:i + G, j:j + G] for i in range(5) for j in range(5)) / 25
        low = (h < local - 0.02).reshape(-1)
        jam_zone = env.jam_zone[0].cpu().numpy()
        first = lambda m: np.where(m.any(0), m.argmax(0), -1)               # first step where m is true
        bt, kt = first(status == BREACHED), first(status == KILLED)         # (n, D)
        ang = lambda c: math.degrees(math.atan2(c % G - bx, -(c // G - by))) % 360
        sector = lambda a: COMPASS[int((a + 22.5) // 45) % 8]
        diff = lambda a, b: abs((a - b + 180) % 360 - 180)

        routes = []
        for b, d in zip(*np.nonzero(bt >= 0)):
            path = cells[:bt[b, d] + 1, b, d]
            keep = np.r_[True, path[1:] != path[:-1]]
            pts = np.stack([path[keep] % G, path[keep] // G], 1).astype(float)
            seg = np.r_[0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
            u = np.linspace(0, seg[-1], 10)
            res = np.stack([np.interp(u, seg, pts[:, 0]), np.interp(u, seg, pts[:, 1])], 1)
            dist_b = np.hypot(pts[:, 0] - bx, pts[:, 1] - by)
            entry = path[keep][np.argmax(dist_b <= 6)] if (dist_b <= 6).any() else path[-1]
            flight = slice(0, bt[b, d])
            routes.append(dict(b=int(b), d=int(d), cells=path[keep].tolist(), pts=res, t=int(bt[b, d]),
                               angle=ang(int(entry)), undetected=float(1 - det[flight, b, d].mean()),
                               low=float(low[path].mean()), wait=int((path[1:] == path[:-1]).sum()),
                               jammed=float(jam[flight, b, d].mean()), through_jam=bool(jam_zone[path].any())))
        total = len(routes)
        out = []
        if total >= 2:
            clusters = []
            for r in sorted(routes, key=lambda r: r["t"]):
                best = min(clusters, key=lambda c: np.hypot(*(c[0]["pts"] - r["pts"]).T).mean(), default=None)
                if best is not None and np.hypot(*(best[0]["pts"] - r["pts"]).T).mean() < 3.2:
                    best.append(r)
                else:
                    clusters.append([r])
            named = {}
            for cl in sorted(clusters, key=len, reverse=True):
                if len(cl) < 2:
                    continue
                med = min(cl, key=lambda r: sum(np.hypot(*(r["pts"] - o["pts"]).T).mean() for o in cl))
                und = np.mean([r["undetected"] for r in cl])
                lowg = np.mean([r["low"] for r in cl])
                wait = np.median([r["wait"] for r in cl])
                where = COMPASS_NOUN[sector(med["angle"])]
                # character, most distinctive first (nearly every breaching drone flew mostly undetected)
                if wait >= 5:
                    kind, how = "wait and dash", f"holding position for about {int(wait)} steps before the final dash"
                elif lowg >= 0.45:
                    kind, how = "valley run", f"keeping to low ground for {round(100 * lowg)}% of the route"
                elif und >= 0.75:
                    kind, how = "radar-shadow run", "staying in radar shadow"
                else:
                    kind, how = "direct run", f"fast and direct, about {int(np.median([r['t'] for r in cl]))} steps to the base"
                g = named.setdefault(f"{where.capitalize()} {kind}", dict(where=where, how=how, routes=[], meds=[]))
                g["routes"] += cl
                g["meds"].append(med)
            for name, g in sorted(named.items(), key=lambda kv: -len(kv[1]["routes"]))[:max_corridors]:
                cl = g["routes"]
                if len(cl) / total < 0.12:
                    continue
                und = np.mean([r["undetected"] for r in cl])
                extra = []
                if "radar" not in name:
                    extra.append(f"undetected for {round(100 * und)}% of the flight")
                if np.mean([r["through_jam"] for r in cl]) >= 0.5:
                    extra.append("straight through your jammer coverage")
                text = (f"{len(cl)} of {total} breaches reached the base from the {g['where']}, {g['how']}"
                        + "".join(f", {e}" for e in extra) + ".")
                out.append(dict(kind="corridor", name=name, count=len(cl), share=len(cl) / total, text=text,
                                paths=[m["cells"] for m in g["meds"][:2]], battle=g["meds"][0]["b"]))
            # no dominant corridor but breaches from many directions: the swarm is probing every side
            sectors = {}
            for r in routes:
                sectors.setdefault(sector(r["angle"]), []).append(r)
            top = max((len(v) for v in sectors.values()), default=0)
            if total >= 6 and len(sectors) >= 4 and top / total < 0.4:
                picks = [min(v, key=lambda r: r["t"]) for v in sorted(sectors.values(), key=len, reverse=True)[:4]]
                ex = max(set(r["b"] for r in routes), key=lambda b: len({sector(r["angle"]) for r in routes if r["b"] == b}))
                out.insert(0, dict(kind="probing", name="Multi-axis probing", count=total, share=1.0,
                                   text=(f"Breaches came from {len(sectors)} of 8 directions and no single approach carried more "
                                         f"than {round(100 * top / total)}% of them: the swarm spreads out and finds whichever side is thin."),
                                   paths=[r["cells"] for r in picks], battle=int(ex)))

        # swarm-level patterns, per battle
        by_battle = {}
        for r in routes:
            by_battle.setdefault(r["b"], []).append(r)
        found = {"saturation": [], "pincer": [], "decoy": []}
        for b, rs in by_battle.items():
            ts = sorted(r["t"] for r in rs)
            if any(ts[i + 2] - ts[i] <= 4 for i in range(len(ts) - 2)):
                found["saturation"].append((b, [r for r in rs]))
            pair = next(((r1, r2) for r1 in rs for r2 in rs if r1 is not r2 and diff(r1["angle"], r2["angle"]) >= 90
                         and abs(r1["t"] - r2["t"]) <= 8), None)
            if pair:
                found["pincer"].append((b, list(pair)))
            for d in np.nonzero(kt[b] >= 0)[0]:
                kc = int(cells[kt[b, d], b, d])
                hit = next((r for r in rs if 0 < r["t"] - kt[b, d] <= 10 and diff(ang(kc), r["angle"]) >= 90), None)
                if hit:
                    found["decoy"].append((b, [dict(cells=cells[:kt[b, d] + 1, b, d].tolist()), hit]))
                    break
        names = {
            "saturation": ("Saturation", "battles had three or more drones reach the base within 4 steps of each other, "
                                         "more than the interceptors could engage at once"),
            "pincer": ("Pincer", "battles had drones break through from directions at least 90 degrees apart, within "
                                 "8 steps of each other, splitting your defenses"),
            "decoy": ("Decoy pattern", "battles had a drone shot down on one side shortly before another broke through "
                                       "from a different side, while your interceptors were committed"),
        }
        need = max(3, int(0.08 * n))
        for k, hits in found.items():
            if len(hits) >= need:
                b, rs = max(hits, key=lambda x: len(by_battle[x[0]]))
                paths = []
                for r in rs[:3]:
                    c = np.asarray(r["cells"])
                    paths.append(c[np.r_[True, c[1:] != c[:-1]]].tolist())
                out.append(dict(kind=k, name=names[k][0], count=len(hits), share=len(hits) / n,
                                text=f"{len(hits)} of {n} {names[k][1]}.", paths=paths, battle=int(b)))
        return out[:4]

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
        # interceptors only engage detected drones, so a kill implies detection, even when the drone was
        # spotted and shot down within the same step (its per-step flag is cleared once it is down)
        ever_det = det.any(0) | (final == KILLED)
        for t in range(T):
            alive_t = status[t] == ALIVE
            np.add.at(traffic, cells[t][alive_t], 1)
            m = alive_t & breached & ~det[t]
            np.add.at(sneak, cells[t][m], 1)
        y = np.arange(N) // G
        far = np.hypot(np.arange(N) % G - bx, y - by) > C.BASE_RADIUS + 1
        spawn_cells = np.nonzero(ms.spawn_mask[map_id])[0]
        off_spawn = self.euclid[:, spawn_cells].min(1) >= 2   # ignore the entry zone itself
        sneak *= off_spawn & far

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
        # start where the drones that broke through actually entered (the busiest entry cell), so the
        # planner view is about the approach that hurt; with no breaches, the middle of the entry zone
        entries = cells[0][breached]
        start = int(np.bincount(entries).argmax()) if entries.size else int(spawn_cells[len(spawn_cells) // 2])
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
