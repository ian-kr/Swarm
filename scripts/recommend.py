"""Search for the best defense on each map against the Veteran swarm.

Genetic search (see redcell/search.py), then re-scores the winner against every
opponent with more battles. Writes data/recommended.json for the game.

    python scripts/recommend.py [--generations 12]
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
from redcell.env import DefenseEnv, Layout, MapTensors  # noqa: E402
from redcell.policy import load_policy  # noqa: E402
from redcell.search import evaluate_layouts  # noqa: E402
from redcell.terrain import load_maps  # noqa: E402


def has_duplicates(lay: Layout) -> torch.Tensor:
    cells = torch.cat([lay.radar_cell, lay.jam_cell, lay.pad_cell], 1)
    on = torch.cat([lay.radar_on, lay.jam_on, lay.pad_on], 1)
    k = torch.arange(cells.shape[1], device=cells.device)
    cells = torch.where(on, cells, -1 - k)              # inactive slots never collide
    s = cells.sort(1).values
    return (s[:, 1:] == s[:, :-1]).any(1)


STALL_PENALTY = 0.5


def scripted_rate(ms, maps_cpu, layout: dict, map_id: int, n=32, seed=7):
    env = DefenseEnv(maps_cpu, n, "cpu", auto_reset=False, seed=seed)
    env.reset(map_id=map_id, layout=Layout.from_lists([layout] * n, "cpu"))
    atk = ScriptedAttacker(ms)
    for _ in range(C.MAX_STEPS):
        _, _, _, info = env.step(atk.act(env))
        if info["env_done"].all():
            break
    return (info["breaches"].float() / C.N_DRONES).mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=12)
    ap.add_argument("--population", type=int, default=512)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = torch.device(args.device)

    spec = json.loads((ROOT / "data" / "difficulties.json").read_text())
    nets = {k: load_policy(ROOT / v["checkpoint"], dev) for k, v in spec["levels"].items()}
    for k, net in nets.items():
        net.amp = dev.type != "cpu"          # bf16 matmuls: far faster on AMD GPUs
        net.temperature = float(spec["levels"][k].get("temperature", 1.0))
    target = nets["veteran"]
    ms = load_maps()
    maps = MapTensors(ms, dev)
    maps_cpu = MapTensors(ms, "cpu")
    helper = DefenseEnv(maps, 1, dev, auto_reset=False, seed=0)
    out = {}
    for mi in range(ms.n):
        P = args.population
        mid = torch.full((P,), mi, device=dev)
        lay = helper.random_layout(torch.arange(P, device=dev), mid)
        best_lay, best_score = None, None
        for g in range(args.generations):
            if best_lay is not None:
                parents = best_lay.index(torch.randint(best_lay.radar_cell.shape[0], (P,), device=dev))
                kids = helper.mutate_layout(parents, mid)
                fresh = helper.random_layout(torch.arange(P // 4, device=dev), mid[: P // 4])
                lay = Layout.cat([best_lay, kids, fresh])
            m = torch.full((lay.radar_cell.shape[0],), mi, device=dev)
            breach, stall = evaluate_layouts(target, maps, lay, m, args.seeds, dev, gen_seed=1000 * mi + g, return_stall=True)
            # A defense the swarm refuses to attack is not one a player can learn from, so prefer
            # defenses that win by detecting and shooting drones down.
            score = breach + STALL_PENALTY * stall + has_duplicates(lay).float()
            keep = score.argsort()[: P // 4]
            best_lay, best_score = lay.index(keep), score[keep]
            print(f"{ms.names[mi]:13s} gen {g:2d}: best {best_score[0]:.3f}  top-quarter mean {best_score.mean():.3f}")

        # re-score the top 16 with many more battles and keep the true best
        top = best_lay.index(torch.arange(16, device=dev))
        m16 = torch.full((16,), mi, device=dev)
        fb, fs = evaluate_layouts(target, maps, top, m16, 64, dev, gen_seed=99, return_stall=True)
        fine = fb + STALL_PENALTY * fs + has_duplicates(top).float()
        i = int(fine.argmin())
        layout = top.index(torch.tensor([i], device=dev)).to_lists()[0]
        breach = {k: round(evaluate_layouts(net, maps, top.index(torch.tensor([i], device=dev)),
                                            m16[:1], 128, dev, gen_seed=5).item(), 4) for k, net in nets.items()}
        breach["scripted"] = round(scripted_rate(ms, maps_cpu, layout, mi), 4)
        breach["veteran_stall"] = round(fs[i].item(), 4)
        out[str(mi)] = {"layout": layout, "breach": breach}
        print(f"  -> {layout}  {breach}")

    path = ROOT / "data" / "recommended.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
