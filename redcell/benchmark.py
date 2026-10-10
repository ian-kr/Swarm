"""Benchmarks shared by training and difficulty selection.

'strong' = random defense layouts the scripted A* attacker mostly fails against (breach <= 1/3):
the kind of defense a good player builds. A* is re-scored on fresh seeds so its number is unbiased.
"""

import torch

from . import config as C
from .astar import ScriptedAttacker
from .env import DefenseEnv


def scripted_rates(ms, maps_cpu, map_id, layout, seed):
    """Scripted A* breach rate per layout (one battle each)."""
    n = map_id.numel()
    env = DefenseEnv(maps_cpu, n, "cpu", auto_reset=False, seed=seed)
    env.reset(map_id=map_id, layout=layout)
    atk = ScriptedAttacker(ms)
    for _ in range(C.MAX_STEPS):
        _, _, _, info = env.step(atk.act(env))
        if info["env_done"].all():
            break
    return info["breaches"].float() / C.N_DRONES


def build_strong_set(ms, maps_cpu, n=1536, seed=4321):
    env = DefenseEnv(maps_cpu, 1, "cpu", auto_reset=False, seed=seed)
    map_id = torch.arange(n) % ms.n
    layout = env.random_layout(torch.arange(n), map_id)
    first = scripted_rates(ms, maps_cpu, map_id, layout, seed + 1)
    keep = (first <= 1 / 3).nonzero().squeeze(1)
    map_id, layout = map_id[keep], layout.index(keep)
    # unbiased A* score: average over fresh seeds
    fresh = torch.stack([scripted_rates(ms, maps_cpu, map_id, layout, seed + 10 + k) for k in range(4)]).mean(0)
    return map_id, layout, fresh.mean().item()


