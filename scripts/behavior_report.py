"""Compare how the trained swarm and the scripted A* attacker fly against the same defenses.

    python scripts/behavior_report.py [--level veteran] [--n 240]

Prints, per attacker: breach rate, share of drone-steps spent detected, share of drones
never detected, steps to breach, route spread (distinct cells used per battle), and how
often drones hover (waiting for an opening).
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from redcell import config as C  # noqa: E402
from redcell.astar import ScriptedAttacker  # noqa: E402
from redcell.env import ALIVE, BREACHED, DefenseEnv, MapTensors  # noqa: E402
from redcell.policy import load_policy  # noqa: E402
from redcell.terrain import load_maps  # noqa: E402
from redcell.benchmark import build_strong_set  # noqa: E402


def run(ms, maps, map_id, layout, policy):
    n = map_id.numel()
    env = DefenseEnv(maps, n, "cpu", auto_reset=False, seed=99)
    env.reset(map_id=map_id, layout=layout)
    atk = ScriptedAttacker(ms) if policy is None else None
    det_steps = alive_steps = hover = follow = 0
    ever = torch.zeros(n, C.N_DRONES, dtype=torch.bool)
    visited = torch.zeros(n, C.N_CELLS, dtype=torch.bool)
    breach_t = torch.full((n, C.N_DRONES), -1)
    with torch.no_grad():
        for t in range(C.MAX_STEPS):
            alive = env.drone_status == ALIVE
            a = atk.act(env) if atk else policy.act_env(env.observe())[0]
            hover += ((a == C.HOVER) & alive).sum().item()
            follow += ((a == C.FOLLOW) & alive).sum().item()
            prev = env.drone_status.clone()
            _, _, _, info = env.step(a)
            alive_steps += alive.sum().item()
            det_steps += (env.detected & alive).sum().item()
            ever |= env.detected
            visited.scatter_(1, env.drone_cell, alive)
            newly = (prev == ALIVE) & (env.drone_status == BREACHED)
            breach_t[newly] = t + 1
            if info["env_done"].all():
                break
    br = env.drone_status == BREACHED
    return {
        "breach_rate": round(br.float().mean().item(), 3),
        "detected_share_of_flight": round(det_steps / max(alive_steps, 1), 3),
        "never_detected": round((~ever).float().mean().item(), 3),
        "steps_to_breach": round(breach_t[br].float().mean().item(), 1) if br.any() else None,
        "cells_used_per_battle": round(visited.sum(1).float().mean().item(), 1),
        "hover_share": round(hover / max(alive_steps, 1), 3),
        "follow_pathfinder_share": round(follow / max(alive_steps, 1), 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="veteran")
    args = ap.parse_args()
    ms = load_maps()
    maps = MapTensors(ms, "cpu")
    spec = json.loads((ROOT / "data" / "difficulties.json").read_text())
    net = load_policy(ROOT / spec["levels"][args.level]["checkpoint"])
    map_id, layout, _ = build_strong_set(ms, maps)
    print(f"{map_id.numel()} strong defense layouts (A* breach <= 1/3 on first try)")
    for name, pol in (("scripted A*", None), (args.level, net)):
        print(f"{name:12s}", run(ms, maps, map_id, layout, pol))


if __name__ == "__main__":
    main()
