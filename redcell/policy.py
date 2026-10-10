"""Shared attacker policy: every drone runs the same network on its own view.

Inputs per battle
  glob   (8, 16, 16)     downsampled whole map: terrain, blocked, radar coverage, jammers,
                         pads, base, drones, threat-aware distance field
Inputs per drone
  local  (9, 11, 11)     egocentric patch: terrain, blocked, radar coverage, jammers, pads,
                         flying interceptors, other drones, base, threat-aware distance field
  cell   ()              position, fed as a one-hot on an 8x8 grid next to the global map
  vec    (10,)           direction/distance to base, detected?, jammed?, time, position,
                         swarm size, threat-field value here
Outputs: logits over 9 moves, and a value estimate (for PPO).

All layers are plain MLPs: on the Radeon W7900 (ROCm) they run in bf16 at ~60 TFLOPS,
about 30-60x faster than the equivalent small convolutions.
"""

import torch
import torch.nn as nn

from . import config as C
from .env import N_GLOBAL_CH, N_LOCAL_CH, N_VEC

POS = 8   # position one-hot grid


def mlp(*sizes):
    layers = []
    for a, b in zip(sizes, sizes[1:]):
        layers += [nn.Linear(a, b), nn.ReLU()]
    return nn.Sequential(*layers)


class AttackerPolicy(nn.Module):
    def __init__(self, hidden: int = 256, amp: bool = False):
        super().__init__()
        self.amp = amp
        self.temperature = 1.0   # >1 = noisier decisions (used for the easier difficulty levels)
        self.local = mlp(N_LOCAL_CH * C.LOCAL * C.LOCAL, 512, hidden)
        self.glob = mlp(N_GLOBAL_CH * C.GLOBAL * C.GLOBAL + POS * POS, 512, hidden)
        self.vec = mlp(N_VEC, 64)
        self.trunk = mlp(2 * hidden + 64, hidden, hidden)
        self.actor = nn.Linear(hidden, C.N_ACTIONS)
        self.critic = nn.Linear(hidden, 1)
        nn.init.orthogonal_(self.actor.weight, 0.01)
        nn.init.zeros_(self.actor.bias)
        with torch.no_grad():                  # start near the pathfinder: FOLLOW gets ~45% initial probability
            self.actor.bias[C.FOLLOW] = 2.0

    def forward(self, local, glob, cell, vec):
        """local (B,D,9,11,11), glob (B,8,16,16), cell (B,D), vec (B,D,10) -> logits (B,D,9), value (B,D)."""
        B, D = cell.shape
        s = C.GRID // POS
        pi = (cell // C.GRID) // s * POS + (cell % C.GRID) // s
        here = torch.zeros(B, D, POS * POS, device=cell.device, dtype=glob.dtype).scatter_(2, pi[..., None], 1.0)
        g = torch.cat([glob.flatten(1)[:, None].expand(B, D, -1), here], -1)
        with torch.autocast(cell.device.type, dtype=torch.bfloat16, enabled=self.amp):
            z = torch.cat([self.local(local.flatten(2)), self.glob(g), self.vec(vec)], -1)
            z = self.trunk(z)
            logits, value = self.actor(z), self.critic(z).squeeze(-1)
        return logits.float(), value.float()

    @torch.no_grad()
    def act_env(self, obs, greedy: bool = False):
        """Run on a full env observation; returns actions (B,D), logp, value."""
        logits, value = self(obs["local"], obs["glob"], obs["cell"], obs["vec"])
        dist = torch.distributions.Categorical(logits=logits)
        if greedy:
            a = logits.argmax(-1)
        elif self.temperature != 1.0:
            a = torch.distributions.Categorical(logits=logits / self.temperature).sample()
        else:
            a = dist.sample()
        return a, dist.log_prob(a), value


def load_policy(path, device="cpu") -> AttackerPolicy:
    ck = torch.load(path, map_location=device, weights_only=False)
    net = AttackerPolicy(ck.get("hidden", 256)).to(device)
    net.load_state_dict(ck["model"])
    net.eval()
    return net
