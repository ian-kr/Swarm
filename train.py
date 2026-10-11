"""PPO training for the SWARM attacker swarm.

Runs thousands of battles in parallel on one GPU (AMD via ROCm, or NVIDIA via
CUDA: PyTorch exposes both through `torch.cuda`). Every drone is an agent;
all drones share one policy network.

    python train.py --envs 4096 --updates 600          # AMD cloud
    python train.py --envs 512 --updates 50            # laptop smoke test

Writes checkpoints/ckpt_XXXXX.pt and checkpoints/metrics.jsonl.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from redcell import config as C
from redcell.env import DefenseEnv, MapTensors
from redcell.policy import AttackerPolicy
from redcell.benchmark import build_strong_set
from redcell.env import Layout
from redcell.search import BluePool, evaluate_layouts
from redcell.terrain import load_maps

ROOT = Path(__file__).resolve().parent


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def describe_device(dev: torch.device) -> str:
    if dev.type != "cuda":
        return "CPU"
    backend = "ROCm/HIP " + torch.version.hip if getattr(torch.version, "hip", None) else "CUDA " + str(torch.version.cuda)
    return f"{torch.cuda.get_device_name(dev)} ({backend})"


@torch.no_grad()
def evaluate(net, maps, dev, n_envs=512, seed=1234):
    """Breach rate against a fixed benchmark set of defense layouts (same every call)."""
    env = DefenseEnv(maps, n_envs, dev, auto_reset=False, seed=seed)
    env.reset(map_id=torch.arange(n_envs, device=dev) % maps.n)
    env.gen.manual_seed(seed + 1)
    for _ in range(C.MAX_STEPS):
        a, _, _ = net.act_env(env.observe())
        _, _, _, info = env.step(a)
        if info["env_done"].all():
            break
    br = info["breaches"].float() / C.N_DRONES
    per_map = [br[i::maps.n].mean().item() for i in range(maps.n)]
    return br.mean().item(), per_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=1024)
    ap.add_argument("--rollout", type=int, default=64)
    ap.add_argument("--updates", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=2048, help="battles (x6 drones) per minibatch")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", type=float, default=0.005)
    ap.add_argument("--vf", type=float, default=0.5)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--amp", action="store_true", help="bf16 matmuls (much faster on AMD Radeon / MI300 and recent NVIDIA)")
    ap.add_argument("--out", default=str(ROOT / "checkpoints"))
    ap.add_argument("--resume", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", default="north", choices=["north", "360"], help="theater: north-axis or 360-degree threat")
    ap.add_argument("--follow-bias", type=float, default=2.0,
                    help="initial logit bonus for the follow-the-pathfinder action (0 = start from random flight)")
    ap.add_argument("--stall-weight", type=float, default=0.5,
                    help="blue-team pool priority for defenses where the swarm loiters until timeout")
    ap.add_argument("--team-reward", type=float, default=0.0, help="bonus to each flying drone when a teammate breaches")
    ap.add_argument("--no-strong-eval", action="store_true", help="skip the strong-defense benchmark (needs ~2 min of CPU A* at start)")
    ap.add_argument("--mine-every", type=int, default=10, help="run a blue-team search generation every N updates (0 = off)")
    ap.add_argument("--mine-start", type=int, default=10)
    ap.add_argument("--mine-fresh", type=int, default=1024)
    ap.add_argument("--mine-mutate", type=int, default=1024)
    ap.add_argument("--mine-seeds", type=int, default=2)
    ap.add_argument("--pool-size", type=int, default=1024)
    ap.add_argument("--mine-chunk", type=int, default=2048, help="battles evaluated at once during the search (GPU memory)")
    ap.add_argument("--pool-frac", type=float, default=0.5, help="fraction of training battles drawn from the blue-team pool")
    ap.add_argument("--pool-target", type=float, default=0.2,
                    help="keep defenses the swarm beats about this often: hard but still learnable")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = pick_device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"device: {describe_device(dev)}")

    maps = MapTensors(load_maps(mode=args.mode), dev)
    env = DefenseEnv(maps, args.envs, dev, auto_reset=True, seed=args.seed)
    env.team_reward = args.team_reward
    env.reset()
    strong = None
    if not args.no_strong_eval:
        from redcell.terrain import load_maps as _lm
        ms_cpu = _lm(mode=args.mode)
        s_map, s_lay, s_astar = build_strong_set(ms_cpu, MapTensors(ms_cpu, "cpu"))
        strong = (s_map.to(dev), Layout(*(getattr(s_lay, f).to(dev) for f in Layout.FIELDS)))
        print(f"strong benchmark: {s_map.numel()} layouts, scripted A* breach {s_astar:.3f}")
    pool = BluePool(maps, dev, size=args.pool_size, chunk=args.mine_chunk, target=args.pool_target,
                    stall_weight=args.stall_weight)
    net = AttackerPolicy(args.hidden, amp=args.amp, follow_bias=args.follow_bias).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)
    start = 0
    metrics_path = out / "metrics.jsonl"
    if args.resume:
        ck = torch.load(args.resume, map_location=dev, weights_only=False)
        net.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start = ck["update"]
        pool.load_state_dict(ck.get("pool"))
        if len(pool):
            env.pool, env.pool_frac = pool.as_env_pool(), args.pool_frac
    elif metrics_path.exists():
        metrics_path.unlink()

    T, B, Dn = args.rollout, args.envs, C.N_DRONES
    obs0 = env.observe()
    buf = {
        "local": torch.zeros(T, B, Dn, *obs0["local"].shape[2:], device=dev, dtype=torch.float16),
        "glob": torch.zeros(T, B, *obs0["glob"].shape[1:], device=dev, dtype=torch.float16),
        "vec": torch.zeros(T, B, Dn, obs0["vec"].shape[-1], device=dev),
        "cell": torch.zeros(T, B, Dn, device=dev, dtype=torch.long),
        "act": torch.zeros(T, B, Dn, device=dev, dtype=torch.long),
        "logp": torch.zeros(T, B, Dn, device=dev),
        "val": torch.zeros(T, B, Dn, device=dev),
        "rew": torch.zeros(T, B, Dn, device=dev),
        "done": torch.zeros(T, B, Dn, device=dev),
        "valid": torch.zeros(T, B, Dn, device=dev, dtype=torch.bool),
    }

    ep_breach, ep_kill, ep_count = 0.0, 0.0, 0
    ep_stall = 0.0
    t_start = time.time()
    total_steps = 0
    obs = obs0
    for update in range(start + 1, start + args.updates + 1):
        frac = (update - start - 1) / args.updates
        for g in opt.param_groups:
            g["lr"] = args.lr * (1 - 0.9 * frac)
        ent_coef = args.ent * (1 - 0.7 * frac)

        # ---------------- rollout
        net.eval()
        for t in range(T):
            a, logp, v = net.act_env(obs)
            buf["local"][t] = obs["local"]
            buf["glob"][t] = obs["glob"]
            buf["vec"][t] = obs["vec"]
            buf["cell"][t] = obs["cell"]
            buf["act"][t], buf["logp"][t], buf["val"][t] = a, logp, v
            rew, done, valid, info = env.step(a)
            buf["rew"][t], buf["done"][t], buf["valid"][t] = rew, done.float(), valid
            ed = info["env_done"]
            if ed.any():
                ep_breach += (info["breaches"][ed].float() / Dn).sum().item()
                ep_kill += (info["kills"][ed].float() / Dn).sum().item()
                ep_stall += ((Dn - info["breaches"][ed] - info["kills"][ed]).float() / Dn).sum().item()
                ep_count += int(ed.sum())
            obs = env.observe()
        total_steps += T * B

        # ---------------- GAE (per drone; a drone's samples end when it is done)
        with torch.no_grad():
            _, _, last_v = net.act_env(obs)
            adv = torch.zeros_like(buf["rew"])
            gae = torch.zeros(B, Dn, device=dev)
            next_v = last_v
            for t in reversed(range(T)):
                nonterm = 1.0 - buf["done"][t]
                delta = buf["rew"][t] + args.gamma * next_v * nonterm - buf["val"][t]
                gae = delta + args.gamma * args.lam * nonterm * gae
                gae = gae * buf["valid"][t]
                adv[t] = gae
                next_v = buf["val"][t]
            ret = adv + buf["val"]

        # ---------------- PPO update. Minibatches are whole battles at one timestep,
        # so each battle's global map is encoded once; dead drones are masked out.
        net.train()
        groups = buf["valid"].any(-1).nonzero()                               # (n, 2): t, b
        n = groups.shape[0]
        vmask = buf["valid"]
        a_valid = adv[vmask]
        adv_mean, adv_std = a_valid.mean(), a_valid.std() + 1e-8
        stats = {"pg": 0.0, "vf": 0.0, "ent": 0.0, "kl": 0.0}
        nb = 0
        for _ in range(args.epochs):
            perm = groups[torch.randperm(n, device=dev)]
            for s in range(0, n, args.minibatch):
                mb = perm[s:s + args.minibatch]
                ti, bi = mb[:, 0], mb[:, 1]
                m = vmask[ti, bi].float()
                msum = m.sum()
                logits, v = net(buf["local"][ti, bi].float(), buf["glob"][ti, bi].float(),
                                buf["cell"][ti, bi], buf["vec"][ti, bi])
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(buf["act"][ti, bi])
                old = buf["logp"][ti, bi]
                ratio = (logp - old).exp()
                A = (adv[ti, bi] - adv_mean) / adv_std
                pg = -(torch.min(ratio * A, ratio.clamp(1 - args.clip, 1 + args.clip) * A) * m).sum() / msum
                vf = 0.5 * ((v - ret[ti, bi]).pow(2) * m).sum() / msum
                ent = (dist.entropy() * m).sum() / msum
                loss = pg + args.vf * vf - ent_coef * ent
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()
                with torch.no_grad():
                    stats["kl"] += (((old - logp) * m).sum() / msum).item()
                stats["pg"] += pg.item()
                stats["vf"] += vf.item()
                stats["ent"] += ent.item()
                nb += 1

        # ---------------- blue-team search: mine the defenses that best stop the current swarm
        mine = {}
        if args.mine_every and update >= args.mine_start and update % args.mine_every == 0:
            net.eval()
            mine = pool.step(net, args.mine_fresh, args.mine_mutate, args.mine_seeds, seed=args.seed * 100003 + update)
            env.pool, env.pool_frac = pool.as_env_pool(), args.pool_frac

        # ---------------- logging
        elapsed = time.time() - t_start
        rec = {
            "update": update,
            "steps": total_steps,
            "time": round(elapsed, 1),
            "sps": round(total_steps / elapsed),
            "train_breach": round(ep_breach / max(ep_count, 1), 4),
            "train_kill": round(ep_kill / max(ep_count, 1), 4),
            "train_stall": round(ep_stall / max(ep_count, 1), 4),
            "episodes": ep_count,
            "reward_per_step": round(buf["rew"][buf["valid"]].mean().item(), 4),
            **{k: round(v / nb, 4) for k, v in stats.items()},
            **mine,
        }
        ep_breach, ep_kill, ep_count = 0.0, 0.0, 0
        ep_stall = 0.0
        if update % args.eval_every == 0 or update == start + 1:
            net.eval()
            rec["eval_breach"], per_map = evaluate(net, maps, dev)
            rec["eval_breach_per_map"] = [round(x, 3) for x in per_map]
            if strong is not None:
                sb, ss = evaluate_layouts(net, maps, strong[1], strong[0], 4, dev, gen_seed=77, return_stall=True)
                rec["strong_breach"], rec["strong_stall"] = round(sb.mean().item(), 4), round(ss.mean().item(), 4)
        with open(metrics_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{update:4d}] breach {rec['train_breach']:.3f} stall {rec['train_stall']:.3f} eval {rec.get('eval_breach', float('nan')):.3f} "
              f"strong {rec.get('strong_breach', float('nan')):.3f}/{rec.get('strong_stall', float('nan')):.3f} ent {rec['ent']:.2f} kl {rec['kl']:.4f} vf {rec['vf']:.3f} {rec['sps']} steps/s"
              + (f" | blue pool breach {mine['pool_breach']:.3f} stall {mine['pool_stall']:.3f} (fresh {mine['fresh_breach']:.3f})" if mine else ""))

        early = update <= 30 and update % 2 == 0          # learning is fastest early: finer snapshots for Recruit
        if early or update % args.save_every == 0 or update == start + 1 or update == start + args.updates:
            torch.save({"model": net.state_dict(), "opt": opt.state_dict(), "update": update,
                        "hidden": args.hidden, "follow_bias": args.follow_bias, "mode": args.mode, "metrics": rec, "pool": pool.state_dict(), "device": describe_device(dev)},
                       out / f"ckpt_{update:05d}.pt")


if __name__ == "__main__":
    main()
