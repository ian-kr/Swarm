"""Pick the Veteran checkpoint, define the difficulty levels and export the training timeline.

Benchmarks:
  random  a fixed set of random defense layouts (the curve logged during training)
  strong  layouts the scripted A* attacker mostly fails against (breach <= 1/3). This is what a good
          player builds, so it decides which checkpoint becomes the Veteran.
  stall   on the strong set, the fraction of drones still airborne at the time limit. A checkpoint
          that "wins" by loitering outside radar range is no use as an opponent, so the Veteran must
          stall on at most --max-stall of its drones.

Levels: Veteran is the best checkpoint; Regular and Recruit fly the same network with noisier decisions
(sampling temperature 2 and 3). All three are re-measured on a held-out strong set that selection never
saw. The timeline is a handful of snapshots from the Veteran's own training run, from the first update up
to the Veteran, for the in-game "watch it learn" replay.

    python scripts/select_difficulties.py checkpoints/amd_v3
    python scripts/select_difficulties.py checkpoints/amd_ft --prior checkpoints/amd   # a resumed run
"""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from redcell.benchmark import build_strong_set, scripted_rates  # noqa: E402
from redcell.env import DefenseEnv, Layout, MapTensors  # noqa: E402
from redcell.policy import load_policy  # noqa: E402
from redcell.search import evaluate_layouts  # noqa: E402
from redcell.terrain import load_maps  # noqa: E402
from train import evaluate  # noqa: E402

TEMPERATURES = {"veteran": 1.0, "regular": 2.0, "recruit": 3.0}


def read_metrics(d: Path):
    p = d / "metrics.jsonl"
    return [json.loads(line) for line in p.read_text().splitlines()] if p.exists() else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_dir")
    ap.add_argument("--prior", default=None, help="run this one was resumed from (its earlier checkpoints and metrics)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n", type=int, default=1024, help="random-benchmark battles per checkpoint")
    ap.add_argument("--seeds", type=int, default=8, help="battles per strong layout per checkpoint")
    ap.add_argument("--max-stall", type=float, default=0.1)
    ap.add_argument("--stages", type=int, default=5, help="training-timeline snapshots to export")
    args = ap.parse_args()
    dev = args.device

    d = Path(args.ckpt_dir)
    own = sorted(d.glob("ckpt_*.pt"))
    if not own:
        sys.exit(f"no checkpoints in {d}")
    upd = lambda p: int(p.stem.split("_")[1])
    first_own = upd(own[0])
    prior = sorted(p for p in Path(args.prior).glob("ckpt_*.pt") if upd(p) < first_own) if args.prior else []
    before = [m for m in (read_metrics(Path(args.prior)) if args.prior else []) if m["update"] < first_own]
    # a resumed run restarts its step and wall-clock counters: continue them from the resume point
    off_steps, off_time = (before[-1]["steps"], before[-1]["time"]) if before else (0, 0)
    metrics = before + [{**m, "steps": m["steps"] + off_steps, "time": round(m["time"] + off_time, 1)} for m in read_metrics(d)]
    steps_at = {m["update"]: m["steps"] for m in metrics}

    ms = load_maps()
    maps = MapTensors(ms, dev)
    maps_cpu = MapTensors(ms, "cpu")
    print("building strong-defense benchmark with the scripted A* attacker...")
    s_map, s_lay, s_scripted = build_strong_set(ms, maps_cpu)
    print(f"  {s_map.numel()} strong layouts, A* breach on fresh seeds: {s_scripted:.3f}")
    s_map_d = s_map.to(dev)
    s_lay_d = Layout(*(getattr(s_lay, f).to(dev) for f in Layout.FIELDS))

    def load(p, temperature=1.0):
        net = load_policy(p, dev)
        net.amp = dev != "cpu"
        net.temperature = temperature
        return net

    scores = []
    for p in prior + own:
        net = load(p)
        rate, _ = evaluate(net, maps, dev, n_envs=args.n)
        strong, stall = evaluate_layouts(net, maps, s_lay_d, s_map_d, args.seeds, dev, gen_seed=77, return_stall=True)
        scores.append(dict(update=upd(p), random=rate, strong=strong.mean().item(), stall=stall.mean().item(),
                           path=p, own=p in own))
        print(f"  {p.parent.name}/{p.name}: random {rate:.3f}  strong {scores[-1]['strong']:.3f}  stall {scores[-1]['stall']:.3f}")

    cands = [s for s in scores if s["own"] and s["stall"] <= args.max_stall] or [s for s in scores if s["own"]]
    best = max(cands, key=lambda s: s["strong"])
    print(f"veteran: update {best['update']} (strong {best['strong']:.3f}, stall {best['stall']:.3f})")

    print("scripted A* on the random benchmark...")
    nr = 256
    rmap = torch.arange(nr) % ms.n
    rlay = DefenseEnv(maps_cpu, 1, "cpu", seed=1234).random_layout(torch.arange(nr), rmap)
    rand_scripted = scripted_rates(ms, maps_cpu, rmap, rlay, 1235).mean().item()

    # Held-out check: picking the best of many noisy checkpoints inflates its score, so re-measure every
    # level and A* on a separate strong set (different layouts and seeds) that selection never saw.
    print("held-out strong benchmark...")
    h_map, h_lay, h_scripted = build_strong_set(ms, maps_cpu, seed=98765)
    h_map_d = h_map.to(dev)
    h_lay_d = Layout(*(getattr(h_lay, f).to(dev) for f in Layout.FIELDS))
    models = ROOT / "models"
    models.mkdir(exist_ok=True)
    c = torch.load(best["path"], map_location="cpu", weights_only=False)
    torch.save({"model": c["model"], "hidden": c.get("hidden", 256), "update": best["update"], "device": c.get("device")},
               models / "veteran.pt")
    levels = {}
    for key, temp in TEMPERATURES.items():
        net = load(best["path"], temp)
        hb, hs = evaluate_layouts(net, maps, h_lay_d, h_map_d, args.seeds, dev, gen_seed=555, return_stall=True)
        rate, _ = evaluate(net, maps, dev, n_envs=args.n)
        levels[key] = {"checkpoint": "models/veteran.pt", "update": best["update"], "temperature": temp,
                       "eval_breach": round(hb.mean().item(), 4), "stall": round(hs.mean().item(), 4),
                       "random_breach": round(rate, 4)}
        print(f"  {key:8s} (T={temp}): held-out breach {hb.mean():.3f}  stall {hs.mean():.3f}  random {rate:.3f}")

    # Training timeline: geometric spacing from the first snapshot to the Veteran, so the early
    # (fast-learning) part of the run gets more stages.
    lineage = [s for s in scores if s["update"] <= best["update"]]
    u0, u1 = lineage[0]["update"], best["update"]
    picks = []
    for k in range(args.stages):
        target = u0 * (u1 / u0) ** (k / max(args.stages - 1, 1))
        s = min(lineage, key=lambda s: abs(s["update"] - target))
        if s not in picks:
            picks.append(s)
    tdir = models / "timeline"
    tdir.mkdir(exist_ok=True)
    for old in tdir.glob("*.pt"):
        old.unlink()
    timeline = []
    for s in picks:
        c = torch.load(s["path"], map_location="cpu", weights_only=False)
        dst = tdir / f"update_{s['update']:05d}.pt"
        torch.save({"model": c["model"], "hidden": c.get("hidden", 256), "update": s["update"]}, dst)
        timeline.append({"update": s["update"], "checkpoint": str(dst.relative_to(ROOT)), "steps": steps_at.get(s["update"]),
                         "strong": round(s["strong"], 4), "random": round(s["random"], 4), "stall": round(s["stall"], 4),
                         "veteran": s is best})

    with open(ROOT / "data" / "training_metrics.jsonl", "w") as f:
        for m in metrics:
            f.write(json.dumps(m) + "\n")
    spec = {
        "metrics": "data/training_metrics.jsonl",
        "device": c.get("device"),
        "benchmark": "strong",
        "strong_layouts": int(s_map.numel()),
        "scripted_eval_breach": round(s_scripted, 4),
        "scripted_random_breach": round(rand_scripted, 4),
        "heldout": {"layouts": int(h_map.numel()), "veteran": levels["veteran"]["eval_breach"],
                    "veteran_stall": levels["veteran"]["stall"], "scripted": round(h_scripted, 4)},
        "curve": [[s["update"], round(s["strong"], 4), round(s["random"], 4), round(s["stall"], 4)] for s in scores],
        "levels": levels,
        "levels_note": f"Measured on the held-out strong set ({int(h_map.numel())} layouts, {args.seeds} battles each). "
                       "Recruit/Regular = Veteran with noisier decisions.",
        "timeline": timeline,
        "runs": ([Path(args.prior).name] if prior else []) + [d.name],
    }
    out = ROOT / "data" / "difficulties.json"
    out.write_text(json.dumps(spec, indent=2))
    print(json.dumps({k: v for k, v in spec.items() if k != "curve"}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
