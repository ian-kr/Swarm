"""Blue-team search: find defense layouts that (almost) stop a given attacker.

A simple genetic search on the GPU. Candidates are fresh random layouts plus
mutations (one unit nudged to a nearby cell) of the current best. Each is scored
by running the attacker against it over several seeds, and the lowest breach
rates survive.

Used during training (the swarm trains against the hardest defenses found so
far: an arms race) and offline to produce the "AI-optimized" layout for each map.
"""

from __future__ import annotations

import torch

from . import config as C
from .env import DefenseEnv, Layout, MapTensors


@torch.no_grad()
def evaluate_layouts(net, maps: MapTensors, layout: Layout, map_id: torch.Tensor, seeds: int, dev,
                     gen_seed: int = 0, chunk: int = 4096, return_stall: bool = False):
    """Mean breach rate of `net` against each layout (averaged over `seeds` battles).

    With return_stall, also returns the stall rate: the fraction of drones still airborne when
    time ran out (they neither broke through nor were shot down, i.e. they never committed).
    Battles are run `chunk` at a time to bound GPU memory.
    """
    n = map_id.numel()
    all_map = map_id.repeat_interleave(seeds)
    all_lay = layout.repeat(seeds)
    out, stall = [], []
    for s in range(0, n * seeds, chunk):
        idx = torch.arange(s, min(s + chunk, n * seeds), device=dev)
        env = DefenseEnv(maps, idx.numel(), dev, auto_reset=False, seed=gen_seed + s)
        env.reset(map_id=all_map[idx], layout=all_lay.index(idx))
        for _ in range(C.MAX_STEPS):
            a, _, _ = net.act_env(env.observe())
            _, _, _, info = env.step(a)
            if info["env_done"].all():
                break
        out.append(info["breaches"].float() / C.N_DRONES)
        stall.append((C.N_DRONES - info["breaches"] - info["kills"]).float() / C.N_DRONES)
    breach = torch.cat(out).view(n, seeds).mean(1)
    if return_stall:
        return breach, torch.cat(stall).view(n, seeds).mean(1)
    return breach


class BluePool:
    """The current set of hard-but-beatable defense layouts (a curriculum for the swarm)."""

    def __init__(self, maps: MapTensors, dev, size: int = 1024, chunk: int = 4096, target: float = 0.2,
                 stall_weight: float = 0.0):
        self.maps, self.dev, self.size, self.chunk, self.target = maps, dev, size, chunk, target
        self.stall_weight = stall_weight
        self.map_id = None
        self.layout = None
        self.score = None
        self._env = DefenseEnv(maps, 1, dev, auto_reset=False)   # used for its samplers only

    def __len__(self):
        return 0 if self.map_id is None else self.map_id.numel()

    def as_env_pool(self):
        return (self.map_id, self.layout) if len(self) else None

    @torch.no_grad()
    def step(self, net, n_fresh: int, n_mutate: int, seeds: int = 2, seed: int = 0):
        """One generation of search against `net`. Returns stats."""
        env = self._env
        env.gen.manual_seed(seed)
        fresh_map = torch.randint(self.maps.n, (n_fresh,), device=self.dev, generator=env.gen)
        cands_m = [fresh_map]
        cands_l = [env.random_layout(torch.arange(n_fresh, device=self.dev), fresh_map)]
        if len(self):
            pick = torch.randint(len(self), (n_mutate,), device=self.dev, generator=env.gen)
            pm = self.map_id[pick]
            cands_m += [self.map_id, pm]
            cands_l += [self.layout, env.mutate_layout(self.layout.index(pick), pm)]
        map_id = torch.cat(cands_m)
        layout = Layout.cat(cands_l)
        score, stall = evaluate_layouts(net, self.maps, layout, map_id, seeds, self.dev, gen_seed=seed + 1,
                                        chunk=self.chunk, return_stall=True)
        # Keep the frontier: defenses the swarm beats about `target` of the time. Easy layouts teach
        # nothing new. Layouts it can't beat are normally dropped (no learning signal), except the ones
        # where it stalls (loiters until timeout): those expose a habit the timeout penalty must train out,
        # so stall_weight moves them to the front of the queue.
        key = (score - self.target).abs() - self.stall_weight * stall
        keep = key.argsort()[: self.size]
        self.map_id, self.layout, self.score = map_id[keep], layout.index(keep), score[keep]
        return {"pool_breach": round(self.score.mean().item(), 4), "pool_best": round(self.score.min().item(), 4),
                "pool_stall": round(stall[keep].mean().item(), 4),
                "fresh_breach": round(score[:n_fresh].mean().item(), 4), "fresh_stall": round(stall[:n_fresh].mean().item(), 4)}

    def state_dict(self):
        if not len(self):
            return None
        return {"map_id": self.map_id.cpu(), "score": self.score.cpu(),
                **{f: getattr(self.layout, f).cpu() for f in Layout.FIELDS}}

    def load_state_dict(self, sd):
        if sd is None:
            return
        self.map_id = sd["map_id"].to(self.dev)
        self.score = sd["score"].to(self.dev)
        self.layout = Layout(*(sd[f].to(self.dev) for f in Layout.FIELDS))
