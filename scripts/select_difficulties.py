"""Pick which training checkpoints become the Recruit / Regular / Veteran opponents.

Two benchmarks:
  random  a fixed set of random defense layouts (the curve logged during training)
  strong  layouts the scripted A* attacker mostly fails against (breach <= 1/3). This is
          what a good player builds, so it is the benchmark that decides the difficulty levels.
          A* is re-scored on fresh battle seeds so its number is not biased by the selection.

Picks veteran = best on 'strong', regular ≈ 60% of that, recruit ≈ 25%. Exports the chosen
weights to models/ and writes data/difficulties.json for the game.

    python scripts/select_difficulties.py checkpoints/amd
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from redcell import config as C  # noqa: E402
from redcell.env import DefenseEnv, Layout, MapTensors  # noqa: E402
from redcell.policy import load_policy  # noqa: E402
from redcell.benchmark import build_strong_set, scripted_rates  # noqa: E402
from redcell.search import evaluate_layouts  # noqa: E402
from redcell.terrain import load_maps  # noqa: E402
from train import evaluate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_dir")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n", type=int, default=1024, help="random-benchmark battles per checkpoint")
    ap.add_argument("--seeds", type=int, default=8, help="battles per strong layout per checkpoint")
    args = ap.parse_args()

    d = Path(args.ckpt_dir)
    ckpts = sorted(d.glob("ckpt_*.pt"))
    if not ckpts:
        sys.exit(f"no checkpoints in {d}")
    ms = load_maps()
    maps = MapTensors(ms, args.device)
    maps_cpu = MapTensors(ms, "cpu")

    print("building strong-defense benchmark with the scripted A* attacker...")
    s_map, s_lay, s_scripted = build_strong_set(ms, maps_cpu)
    print(f"  {s_map.numel()} strong layouts, A* breach on fresh seeds: {s_scripted:.3f}")
    s_map_d = s_map.to(args.device)
    s_lay_d = Layout(*(getattr(s_lay, f).to(args.device) for f in Layout.FIELDS))

    scores = []
    for p in ckpts:
        net = load_policy(p, args.device)
        net.amp = args.device != "cpu"
        rate, _ = evaluate(net, maps, args.device, n_envs=args.n)
        strong = evaluate_layouts(net, maps, s_lay_d, s_map_d, args.seeds, args.device, gen_seed=77).mean().item()
        upd = torch.load(p, map_location="cpu", weights_only=False)["update"]
        scores.append(dict(update=upd, random=rate, strong=strong, path=p))
        print(f"  {p.name}: random {rate:.3f}  strong {strong:.3f}")

    best = max(scores, key=lambda s: s["strong"])
    pick = lambda frac: min((s for s in scores if s["update"] <= best["update"]),
                            key=lambda s: abs(s["strong"] - frac * best["strong"]))
    levels = {"recruit": pick(0.25), "regular": pick(0.6), "veteran": best}

    print("scripted A* on the random benchmark...")
    nr = 256
    rmap = torch.arange(nr) % ms.n
    rlay = DefenseEnv(maps_cpu, 1, "cpu", seed=1234).random_layout(torch.arange(nr), rmap)
    rand_scripted = scripted_rates(ms, maps_cpu, rmap, rlay, 1235).mean().item()

    # Held-out check: picking the best of many noisy checkpoints inflates its score, so re-measure the
    # Veteran and A* on a separate strong set (different layouts and seeds) that selection never saw.
    print("held-out strong benchmark...")
    h_map, h_lay, h_scripted = build_strong_set(ms, maps_cpu, seed=98765)
    h_lay_d = Layout(*(getattr(h_lay, f).to(args.device) for f in Layout.FIELDS))
    vet = load_policy(levels["veteran"]["path"], args.device)
    vet.amp = args.device != "cpu"
    h_vet = evaluate_layouts(vet, maps, h_lay_d, h_map.to(args.device), args.seeds, args.device, gen_seed=555).mean().item()
    print(f"  held-out ({h_map.numel()} layouts): veteran {h_vet:.3f}  scripted A* {h_scripted:.3f}")

    models = ROOT / "models"
    models.mkdir(exist_ok=True)
    out_levels = {}
    for k, s in levels.items():
        c = torch.load(s["path"], map_location="cpu", weights_only=False)
        dst = models / f"{k}.pt"
        torch.save({"model": c["model"], "hidden": c.get("hidden", 256), "update": s["update"], "device": c.get("device")}, dst)
        out_levels[k] = {"checkpoint": str(dst.relative_to(ROOT)), "update": s["update"],
                         "eval_breach": round(s["strong"], 4), "random_breach": round(s["random"], 4)}

    shutil.copy(d / "metrics.jsonl", ROOT / "data" / "training_metrics.jsonl")
    ck = torch.load(best["path"], map_location="cpu", weights_only=False)
    spec = {
        "metrics": "data/training_metrics.jsonl",
        "device": ck.get("device"),
        "benchmark": "strong",
        "strong_layouts": int(s_map.numel()),
        "scripted_eval_breach": round(s_scripted, 4),
        "scripted_random_breach": round(rand_scripted, 4),
        "heldout": {"layouts": int(h_map.numel()), "veteran": round(h_vet, 4), "scripted": round(h_scripted, 4)},
        "curve": [[s["update"], round(s["strong"], 4), round(s["random"], 4)] for s in scores],
        "levels": out_levels,
    }
    out = ROOT / "data" / "difficulties.json"
    out.write_text(json.dumps(spec, indent=2))
    print(json.dumps({k: v for k, v in spec.items() if k != "curve"}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
