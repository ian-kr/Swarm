"""Batched counter-drone simulation in pure PyTorch.

One `DefenseEnv` holds B independent battles. Every rule is a tensor op over
the batch, so the same code runs 4096 battles on a GPU for training and a single
battle on a CPU for the playable game.

Per battle:
  * attackers: N_DRONES drones controlled by the policy (or the scripted A* attacker)
  * defenders: radars (line-of-sight detection), jammers (GPS drift), and
    interceptor pads whose interceptors fly shortest paths (Dijkstra next-hop
    tables) to chase tracked drones.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from . import config as C
from .astar import W_JAM, W_PAD, W_RADAR
from .terrain import MapSet, cell_xy

G, N, D = C.GRID, C.N_CELLS, C.N_DRONES
KR, KJ, KP = C.UNITS["radar"]["max"], C.UNITS["jammer"]["max"], C.UNITS["pad"]["max"]
KI = KP * C.INTERCEPTORS_PER_PAD

ALIVE, KILLED, BREACHED = 0, 1, 2
READY, FLYING, SPENT = 0, 1, 2

N_LOCAL_CH = 9      # height, blocked, radar, jammer, pad, interceptors, drones, base, threat field
N_GLOBAL_CH = 8     # height, blocked, radar, jammer, pad, base, drones, threat field
N_VEC = 10
FIELD_SCALE = 60.0  # threat-field distances are divided by this for the network


class MapTensors:
    """The precomputed map tables, moved to a device."""

    def __init__(self, ms: MapSet, device):
        t = lambda a, dt=None: torch.as_tensor(np.asarray(a), device=device, dtype=dt)
        self.n = ms.n
        self.heights = t(ms.heights.reshape(ms.n, N), torch.float32)
        self.passable = t(ms.passable, torch.bool)
        self.base_cell = t(ms.base_cell, torch.long)
        self.base_mask = t(ms.base_mask, torch.bool)
        self.spawn_mask = t(ms.spawn_mask, torch.bool)
        self.place_mask = t(ms.place_mask, torch.bool)
        self.vis = t(ms.vis, torch.bool)
        self.dist = t(ms.dist, torch.float32)
        self.next_hop = t(ms.next_hop.astype(np.int64), torch.long)
        xy = cell_xy()
        self.xy = t(xy, torch.long)
        self.euclid = torch.cdist(self.xy.float(), self.xy.float())          # (N, N)
        # move table: cell, action -> target cell (or -1 if off-map)
        mv = np.full((N, len(C.MOVES)), -1, np.int64)
        for a, (dx, dy) in enumerate(C.MOVES):
            nx, ny = xy[:, 0] + dx, xy[:, 1] + dy
            ok = (nx >= 0) & (nx < G) & (ny >= 0) & (ny < G)
            mv[ok, a] = ny[ok] * G + nx[ok]
        self.move = t(mv, torch.long)
        # distance from every cell to its map's base (shortest path)
        self.dist_to_base = self.dist[torch.arange(ms.n, device=device), self.base_cell]   # (M, N)
        bxy = self.xy[self.base_cell].float()
        self.base_xy = bxy


@dataclass
class Layout:
    """Defense placement for a batch. Cells are flat indices, `on` marks used slots."""
    radar_cell: torch.Tensor
    radar_on: torch.Tensor
    jam_cell: torch.Tensor
    jam_on: torch.Tensor
    pad_cell: torch.Tensor
    pad_on: torch.Tensor

    FIELDS = ("radar_cell", "radar_on", "jam_cell", "jam_on", "pad_cell", "pad_on")

    def index(self, i) -> "Layout":
        return Layout(*(getattr(self, f)[i] for f in self.FIELDS))

    def select_merge(self, mask, other: "Layout") -> "Layout":
        """Copy of self with rows where `mask` is set replaced (in order) by rows of `other`."""
        out = []
        for f in self.FIELDS:
            t = getattr(self, f).clone()
            t[mask] = getattr(other, f)
            out.append(t)
        return Layout(*out)

    @staticmethod
    def cat(layouts) -> "Layout":
        return Layout(*(torch.cat([getattr(l, f) for l in layouts]) for f in Layout.FIELDS))

    def repeat(self, k) -> "Layout":
        return Layout(*(getattr(self, f).repeat_interleave(k, 0) for f in self.FIELDS))

    def to_lists(self) -> list[dict]:
        out = []
        for b in range(self.radar_cell.shape[0]):
            out.append({k: getattr(self, c)[b][getattr(self, o)[b]].tolist()
                        for k, c, o in (("radar", "radar_cell", "radar_on"), ("jammer", "jam_cell", "jam_on"), ("pad", "pad_cell", "pad_on"))})
        return out

    @staticmethod
    def from_lists(layouts: list[dict], device) -> "Layout":
        """Build from [{'radar': [cells], 'jammer': [...], 'pad': [...]}, ...]."""
        def pack(key, k):
            cell = torch.zeros(len(layouts), k, dtype=torch.long)
            on = torch.zeros(len(layouts), k, dtype=torch.bool)
            for b, lay in enumerate(layouts):
                cs = list(lay.get(key, []))[:k]
                if cs:
                    cell[b, :len(cs)] = torch.tensor(cs, dtype=torch.long)
                    on[b, :len(cs)] = True
            return cell.to(device), on.to(device)
        r = pack("radar", KR)
        j = pack("jammer", KJ)
        p = pack("pad", KP)
        return Layout(*r, *j, *p)


def _compositions():
    out = []
    cr, cj, cp = (C.UNITS[k]["cost"] for k in ("radar", "jammer", "pad"))
    for r in range(1, KR + 1):
        for j in range(KJ + 1):
            for p in range(KP + 1):
                spend = r * cr + j * cj + p * cp
                if spend <= C.BUDGET and spend >= C.BUDGET - 3:
                    out.append((r, j, p))
    return out


COMPOSITIONS = _compositions()


class DefenseEnv:
    def __init__(self, maps: MapTensors, batch: int, device, auto_reset: bool = True, seed: int | None = None):
        self.m = maps
        self.B = batch
        self.dev = device
        self.auto_reset = auto_reset
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(seed if seed is not None else int(np.random.randint(1 << 31)))
        B = batch
        z = lambda *s, dt=torch.long: torch.zeros(*s, dtype=dt, device=device)
        self.map_id = z(B)
        self.t = z(B)
        self.drone_cell = z(B, D)
        self.prev_cell = z(B, D)
        self.drone_status = z(B, D)
        self.detected = z(B, D, dt=torch.bool)
        self.jammed = z(B, D, dt=torch.bool)
        self.int_cell = z(B, KI)
        self.int_state = z(B, KI)
        self.int_target = z(B, KI)
        self.int_goal = z(B, KI)
        self.int_fuel = z(B, KI)
        self.coverage = z(B, N, dt=torch.bool)
        self.jam_zone = z(B, N, dt=torch.bool)
        self.pad_zone = z(B, N, dt=torch.bool)
        self.tfield = torch.zeros(B, N, device=device)      # threat-aware distance to base
        self.threat = torch.zeros(B, N, device=device)      # per-cell threat penalty
        self.layout = Layout(z(B, KR), z(B, KR, dt=torch.bool), z(B, KJ), z(B, KJ, dt=torch.bool),
                             z(B, KP), z(B, KP, dt=torch.bool))
        self._static = torch.zeros(B, 7, N, device=device)   # height, blocked, radar, jammer, pad, base, field
        self.events: list = []                               # (kind, env, drone, cell) for replays
        self.record_events = False
        cmp = torch.tensor(COMPOSITIONS, device=device)
        self._comp = cmp
        # placement preference weights per map (M, N) for the random layout sampler
        place = self.m.place_mask.float()
        dbase = self.m.euclid[self.m.base_cell]                            # (M, N)
        h = self.m.heights
        self._w_radar = place * torch.exp(-dbase / 10) * (0.3 + h) ** 2
        self._w_jam = place * torch.exp(-dbase / 5)
        self._w_pad = place * torch.exp(-dbase / 6)
        self._w_uniform = place
        self._step_len = torch.tensor([1.41421356 if dx and dy else 1.0 for dx, dy in C.MOVES], device=device)
        self.team_reward = 0.0    # bonus to every still-flying drone when a teammate breaches
        self.pool = None          # optional pool of hard layouts mined by the blue-team search
        self.pool_frac = 0.0

    # ------------------------------------------------------------------ reset
    def random_layout(self, idx: torch.Tensor, map_id: torch.Tensor) -> Layout:
        n = idx.numel()
        comp = self._comp[torch.randint(len(self._comp), (n,), device=self.dev, generator=self.gen)]
        uniform = torch.rand(n, device=self.dev, generator=self.gen) < 0.35

        def pick(w, k, count):
            ww = torch.where(uniform[:, None], self._w_uniform[map_id], w[map_id]) + 1e-9
            cells = torch.multinomial(ww, k, replacement=False, generator=self.gen)
            on = torch.arange(k, device=self.dev)[None] < count[:, None]
            return cells, on

        return Layout(*pick(self._w_radar, KR, comp[:, 0]), *pick(self._w_jam, KJ, comp[:, 1]),
                      *pick(self._w_pad, KP, comp[:, 2]))

    def mutate_layout(self, layout: Layout, map_id: torch.Tensor, radius: float = 4.0) -> Layout:
        """Move one active unit of each layout to a random placeable cell nearby (blue-team search step)."""
        n = map_id.numel()
        dev = self.dev
        out = [getattr(layout, f).clone() for f in Layout.FIELDS]
        kind = torch.randint(3, (n,), device=dev, generator=self.gen)
        for ki, (ci, oi) in enumerate(((0, 1), (2, 3), (4, 5))):
            cells, on = out[ci], out[oi]
            rows = (kind == ki) & on.any(1)
            if not rows.any():
                continue
            r = rows.nonzero().squeeze(1)
            slot = torch.multinomial(on[r].float() + 1e-12, 1, generator=self.gen).squeeze(1)
            cur = cells[r, slot]
            near = (self.m.euclid[cur] <= radius) & self.m.place_mask[map_id[r]]
            new = torch.multinomial(near.float() + 1e-12, 1, generator=self.gen).squeeze(1)
            cells[r, slot] = new
        return Layout(*out)

    def reset(self, idx: torch.Tensor | None = None, map_id=None, layout: Layout | None = None):
        """Reset envs `idx` (default all). map_id/layout default to random."""
        if idx is None:
            idx = torch.arange(self.B, device=self.dev)
        n = idx.numel()
        if n == 0:
            return
        if map_id is None:
            map_id = torch.randint(self.m.n, (n,), device=self.dev, generator=self.gen)
        elif not torch.is_tensor(map_id):
            map_id = torch.full((n,), int(map_id), device=self.dev, dtype=torch.long)
        if layout is None:
            layout = self.random_layout(idx, map_id)
            if self.pool is not None and self.pool_frac > 0:
                use = torch.rand(n, device=self.dev, generator=self.gen) < self.pool_frac
                if use.any():
                    pmap, play = self.pool
                    pi = torch.randint(pmap.numel(), (int(use.sum()),), device=self.dev, generator=self.gen)
                    map_id = map_id.clone()
                    map_id[use] = pmap[pi]
                    layout = layout.select_merge(use, play.index(pi))
        self.map_id[idx] = map_id
        self.t[idx] = 0
        L = self.layout
        L.radar_cell[idx], L.radar_on[idx] = layout.radar_cell, layout.radar_on
        L.jam_cell[idx], L.jam_on[idx] = layout.jam_cell, layout.jam_on
        L.pad_cell[idx], L.pad_on[idx] = layout.pad_cell, layout.pad_on

        # sensor / jammer footprints
        U = C.UNITS
        vis = self.m.vis[map_id[:, None], layout.radar_cell]                  # (n, KR, N)
        rng = self.m.euclid[layout.radar_cell] <= U["radar"]["range"]
        self.coverage[idx] = (vis & rng & layout.radar_on[..., None]).any(1)
        self.jam_zone[idx] = ((self.m.euclid[layout.jam_cell] <= U["jammer"]["range"]) & layout.jam_on[..., None]).any(1)
        self.pad_zone[idx] = ((self.m.euclid[layout.pad_cell] <= U["pad"]["range"]) & layout.pad_on[..., None]).any(1)

        # drones enter along the northern edge
        sw = self.m.spawn_mask[map_id].float()
        self.drone_cell[idx] = torch.multinomial(sw, D, replacement=False, generator=self.gen)
        self.prev_cell[idx] = self.drone_cell[idx]
        self.drone_status[idx] = ALIVE
        self.detected[idx] = False
        self.jammed[idx] = False

        # interceptors start on their pads
        pad_of = torch.arange(KI, device=self.dev) // C.INTERCEPTORS_PER_PAD
        self.int_cell[idx] = layout.pad_cell[:, pad_of]
        self.int_state[idx] = torch.where(layout.pad_on[:, pad_of], READY, SPENT)
        self.int_target[idx] = 0
        self.int_goal[idx] = self.int_cell[idx]
        self.int_fuel[idx] = C.INTERCEPTOR_FUEL

        self.threat[idx] = W_RADAR * self.coverage[idx].float() + W_JAM * self.jam_zone[idx].float() + W_PAD * self.pad_zone[idx].float()
        self.tfield[idx] = self.threat_field(map_id, self.coverage[idx], self.jam_zone[idx], self.pad_zone[idx])
        pad_mark = torch.zeros(n, N, device=self.dev)
        pad_mark.scatter_(1, layout.pad_cell, layout.pad_on.float())
        self._static[idx] = torch.stack([
            self.m.heights[map_id],
            (~self.m.passable[map_id]).float(),
            self.coverage[idx].float(),
            self.jam_zone[idx].float(),
            pad_mark,
            self.m.base_mask[map_id].float(),
            (self.tfield[idx] / FIELD_SCALE).clamp(max=2.0),
        ], 1)
        self._update_detection(idx)

    def threat_field(self, map_id, cov, jam, pad, iters: int = 96) -> torch.Tensor:
        """Threat-aware distance to the base for every cell, for a batch of battles at once.

        Same cost model as the scripted A* attacker (step length + penalties for entering
        radar / jammer / interceptor-sensor cells), solved for the whole grid by parallel
        Bellman-Ford relaxation on the GPU. Each drone gets this field as part of its view,
        so the learned swarm starts from A*-quality route knowledge.
        """
        n = map_id.numel()
        INF = 1e4
        extra = (W_RADAR * cov.float() + W_JAM * jam.float() + W_PAD * pad.float()).view(n, G, G)
        blocked = (~self.m.passable[map_id]).view(n, G, G)
        d = torch.where(self.m.base_mask[map_id].view(n, G, G), 0.0, INF)
        for it in range(iters):
            p = F.pad(d + extra, (1, 1, 1, 1), value=INF)
            best = d
            for dx, dy in C.MOVES[:-1]:
                L = 1.41421356 if dx and dy else 1.0
                best = torch.minimum(best, p[:, 1 + dy:1 + dy + G, 1 + dx:1 + dx + G] + L)
            best = torch.where(blocked, INF, best)
            if it % 16 == 15 and torch.equal(best, d):
                break
            d = best
        return d.view(n, N)

    # ------------------------------------------------------------------- step
    def _update_detection(self, idx=None):
        b = torch.arange(self.B, device=self.dev)[:, None]
        det = self.coverage[b, self.drone_cell] | self.pad_zone[b, self.drone_cell]
        det &= self.drone_status == ALIVE
        if idx is None:
            self.detected = det
        else:
            self.detected[idx] = det[idx]

    def follow_moves(self) -> torch.Tensor:
        """(B, D) move index each drone would take following the threat-aware pathfinding field."""
        B, m = self.B, self.m
        b = torch.arange(B, device=self.dev)[:, None, None]
        nb = m.move[self.drone_cell]                                          # (B, D, 9)
        ok = (nb >= 0) & m.passable[self.map_id[:, None, None], nb.clamp(min=0)]
        nbc = nb.clamp(min=0)
        cost = self.tfield[b, nbc] + self.threat[b, nbc] + self._step_len
        cost = torch.where(ok, cost, torch.full_like(cost, 1e9))
        cost[..., C.HOVER] = 1e9
        return cost.argmin(-1)

    def step(self, actions: torch.Tensor):
        """actions: (B, D) long in [0, N_ACTIONS). Returns reward, done, valid, info."""
        B, dev, m = self.B, self.dev, self.m
        b = torch.arange(B, device=dev)[:, None]
        mid = self.map_id[:, None]
        active = self.drone_status == ALIVE                                   # valid this step
        d_prev = m.dist_to_base[mid, self.drone_cell]

        # 0. FOLLOW: step to the neighbour that minimises (threat-aware distance + threat + step length),
        #    i.e. the next move of a threat-aware A* route from here
        actions = torch.where(actions == C.FOLLOW, self.follow_moves(), actions)

        # 1. jammers scramble navigation: a jammed drone's move may be replaced by a random one
        self.jammed = self.jam_zone[b, self.drone_cell] & active
        drift = self.jammed & (torch.rand(B, D, device=dev, generator=self.gen) < C.JAM_DRIFT_PROB)
        rnd = torch.randint(len(C.MOVES), (B, D), device=dev, generator=self.gen)
        actions = torch.where(drift, rnd, actions)

        # 2. move
        self.prev_cell = self.drone_cell.clone()      # for the interceptors' lead pursuit
        tgt = m.move[self.drone_cell, actions]
        ok = (tgt >= 0) & m.passable[mid, tgt.clamp(min=0)] & active
        self.drone_cell = torch.where(ok, tgt, self.drone_cell)
        d_now = m.dist_to_base[mid, self.drone_cell]

        reward = torch.full((B, D), C.R_STEP, device=dev)
        reward += C.R_PROGRESS * (d_prev - d_now).clamp(-2, 2)

        # 3. breach
        breach = active & m.base_mask[mid, self.drone_cell]
        self.drone_status = torch.where(breach, BREACHED, self.drone_status)
        reward += C.R_BREACH * breach
        if self.team_reward:
            others = breach.sum(1, keepdim=True) - breach.long()
            reward += self.team_reward * others * (active & ~breach)

        # 4. sensors
        self._update_detection()
        reward += C.R_DETECTED * self.detected

        # 5. interceptors
        killed = self._interceptors()
        reward += C.R_KILLED * killed

        if self.record_events:
            for kind, mask in (("breach", breach), ("kill", killed)):
                for e, d in mask.nonzero().tolist():
                    self.events.append((kind, e, d, int(self.drone_cell[e, d])))

        # 6. termination
        self.t += 1
        timeout = self.t >= C.MAX_STEPS
        still = self.drone_status == ALIVE
        done = active & (~still | timeout[:, None])
        reward += C.R_TIMEOUT * (still & active & timeout[:, None])
        env_done = timeout | ~still.any(1)
        info = {
            "env_done": env_done,
            "breaches": (self.drone_status == BREACHED).sum(1),
            "kills": (self.drone_status == KILLED).sum(1),
        }
        if self.auto_reset and env_done.any():
            self.reset(env_done.nonzero().squeeze(1))
        return reward, done, active, info

    def _interceptors(self) -> torch.Tensor:
        B, dev, m = self.B, self.dev, self.m
        b1 = torch.arange(B, device=dev)
        mid = self.map_id
        alive = self.drone_status == ALIVE
        killed = torch.zeros(B, D, dtype=torch.bool, device=dev)
        tracked = alive & self.detected
        INF = 1e6

        # assignment (threat evaluation and weapon assignment): each free interceptor takes the tracked drone
        # that is most urgent, i.e. closest to the base along its flight path, weighted by how far the
        # interceptor has to fly to reach it; drones it cannot reach on its fuel are skipped, and drones
        # nobody is already chasing are preferred
        flying = self.int_state == FLYING
        tgt_alive = alive[b1[:, None], self.int_target]
        chasing = torch.zeros(B, D, device=dev)
        chasing.scatter_add_(1, self.int_target, (flying & tgt_alive).float())
        for k in range(KI):
            st = self.int_state[:, k]
            need = (st == READY) | ((st == FLYING) & ~alive[b1, self.int_target[:, k]])
            if not need.any():
                continue
            dd = m.dist[mid[:, None], self.drone_cell, self.int_cell[:, k:k + 1]]   # (B, D)
            urgency = m.dist_to_base[mid[:, None], self.drone_cell]                    # flight distance to the base
            reach = dd <= C.INTERCEPTOR_SPEED * self.int_fuel[:, k:k + 1]
            score = torch.where(tracked & reach, C.INTERCEPT_TEWA * urgency + (1 - 0.5 * min(C.INTERCEPT_TEWA, 1)) * dd + 100.0 * chasing,
                                torch.full_like(dd, INF))
            best = score.argmin(1)
            has = need & (score.min(1).values < INF)
            self.int_target[:, k] = torch.where(has, best, self.int_target[:, k])
            self.int_state[:, k] = torch.where(has, FLYING, st)
            chasing[b1[has], best[has]] += 1

        # update goal (lead pursuit): aim where the tracked drone will be by the time the interceptor
        # gets there, extrapolating its last move; fall back to its current cell if that point is not
        # flyable. A drone that drops off radar is hunted at its last known cell.
        tcell = self.drone_cell[b1[:, None], self.int_target]                 # (B, KI)
        ttracked = tracked[b1[:, None], self.int_target]
        txy = m.xy[tcell]
        vel = txy - m.xy[self.prev_cell[b1[:, None], self.int_target]]
        gap = m.dist[mid[:, None], tcell, self.int_cell]
        lead = (gap / (C.INTERCEPTOR_SPEED + 1)).round().clamp(0, C.INTERCEPT_LEAD_MAX)
        lead = torch.where(gap <= C.INTERCEPT_TERMINAL, torch.zeros_like(lead), lead)
        pxy = (txy + vel * lead[..., None].long()).clamp(0, G - 1)
        pcell = pxy[..., 1] * G + pxy[..., 0]
        pcell = torch.where(m.passable[mid[:, None], pcell], pcell, tcell)
        self.int_goal = torch.where(ttracked, pcell, self.int_goal)

        flying = self.int_state == FLYING
        for _ in range(C.INTERCEPTOR_SPEED):
            nxt = m.next_hop[mid[:, None], self.int_goal, self.int_cell]
            self.int_cell = torch.where(flying, nxt, self.int_cell)
            # proximity kill
            tcell = self.drone_cell[b1[:, None], self.int_target]
            txy, ixy = m.xy[tcell], m.xy[self.int_cell]
            close = (txy - ixy).abs().max(-1).values <= 1
            talive = (self.drone_status[b1[:, None], self.int_target] == ALIVE)
            hit = flying & close & talive
            if hit.any():
                roll = torch.rand(B, KI, device=dev, generator=self.gen) < C.KILL_PROB
                for k in range(KI):              # resolve slot by slot so two interceptors can't double-count
                    hk = hit[:, k] & (self.drone_status[b1, self.int_target[:, k]] == ALIVE)
                    succ = hk & roll[:, k]
                    tk = self.int_target[:, k]
                    self.drone_status[b1[succ], tk[succ]] = KILLED
                    killed[b1[succ], tk[succ]] = True
                    self.int_state[:, k] = torch.where(hk, SPENT, self.int_state[:, k])
                flying = self.int_state == FLYING

        self.int_fuel = torch.where(flying, self.int_fuel - 1, self.int_fuel)
        self.int_state = torch.where(flying & (self.int_fuel <= 0), SPENT, self.int_state)
        self.detected &= self.drone_status == ALIVE
        return killed

    # -------------------------------------------------------------- observe
    def observe(self):
        """Returns dict(local (B,D,C,L,L), glob (B,Cg,Gs,Gs), vec (B,D,F), cell (B,D))."""
        B, dev, m = self.B, self.dev, self.m
        b = torch.arange(B, device=dev)[:, None]
        alive = self.drone_status == ALIVE
        icept = torch.zeros(B, N, device=dev)
        icept.scatter_add_(1, self.int_cell, (self.int_state == FLYING).float())
        drones = torch.zeros(B, N, device=dev)
        drones.scatter_add_(1, self.drone_cell, alive.float())

        s = self._static
        planes = torch.stack([s[:, 0], s[:, 1], s[:, 2], s[:, 3], s[:, 4], icept, drones, s[:, 5], s[:, 6]], 1)
        planes = planes.view(B, N_LOCAL_CH, G, G)
        P = C.LOCAL // 2
        padded = F.pad(planes, (P, P, P, P))
        padded[:, 1] = F.pad(planes[:, 1], (P, P, P, P), value=1.0)            # off-map is blocked
        padded[:, 8] = F.pad(planes[:, 8], (P, P, P, P), value=2.0)            # off-map is "far"
        Wp = G + 2 * P
        xy = m.xy[self.drone_cell]                                            # (B, D, 2)
        off = torch.arange(C.LOCAL, device=dev)
        rows = (xy[..., 1, None] + off)[..., :, None]                         # (B, D, L, 1)
        cols = (xy[..., 0, None] + off)[..., None, :]                         # (B, D, 1, L)
        flat = (rows * Wp + cols).view(B, 1, -1).expand(B, N_LOCAL_CH, -1)
        local = padded.view(B, N_LOCAL_CH, -1).gather(2, flat)
        local = local.view(B, N_LOCAL_CH, D, C.LOCAL, C.LOCAL).permute(0, 2, 1, 3, 4)

        gplanes = torch.stack([s[:, 0], s[:, 1], s[:, 2], s[:, 3], s[:, 4], s[:, 5], drones, s[:, 6]], 1).view(B, N_GLOBAL_CH, G, G)
        glob = F.avg_pool2d(gplanes, G // C.GLOBAL)

        bxy = m.base_xy[self.map_id][:, None]                                 # (B, 1, 2)
        dxy = (bxy - xy.float()) / G
        dbase = m.dist_to_base[self.map_id[:, None], self.drone_cell] / 40.0
        vec = torch.stack([
            dxy[..., 0], dxy[..., 1], dbase,
            self.detected.float(),
            self.jam_zone[b, self.drone_cell].float(),
            (self.t.float() / C.MAX_STEPS)[:, None].expand(B, D),
            xy[..., 0].float() / G, xy[..., 1].float() / G,
            alive.float().mean(1, keepdim=True).expand(B, D),
            s[:, 6].gather(1, self.drone_cell),
        ], -1)
        return {"local": local, "glob": glob, "vec": vec, "cell": self.drone_cell.clone(), "alive": alive}
