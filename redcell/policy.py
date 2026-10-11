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
    def __init__(self, hidden: int = 256, amp: bool = False, follow_bias: float = 2.0):
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
        with torch.no_grad():                  # 2.0 = start near the pathfinder (FOLLOW ~45% initial probability);
            self.actor.bias[C.FOLLOW] = follow_bias   # 0 = start from uniformly random flight

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

    # names and sizes of the hidden layers, in the order explain() returns them
    LAYERS = [("Local encoder", 512), ("Local encoder", 256), ("Map encoder", 512), ("Map encoder", 256),
              ("Status encoder", 64), ("Trunk", 256), ("Trunk", 256)]

    def explain(self, local, glob, cell, vec, action: int) -> dict:
        """Look inside one drone's decision (for the in-game network view).

        local (9, 11, 11), glob (8, 16, 16), cell (), vec (10,) for ONE drone; action = the action it took.
        Returns the ReLU activations of every hidden layer, the logits and value, each last-layer neuron's
        contribution to the chosen action's logit (activation x output weight), and attention:
        |input x d logit[action] / d input| for every input element, i.e. how much each input moved
        the score of the move it picked (gradient x input attribution).
        """
        with torch.enable_grad():
            loc = local.detach().float().reshape(1, 1, -1).requires_grad_(True)
            glo = glob.detach().float().reshape(1, -1).requires_grad_(True)
            vin = vec.detach().float().reshape(1, 1, -1).requires_grad_(True)
            s, c = C.GRID // POS, int(cell)
            here = torch.zeros(1, 1, POS * POS)
            here[0, 0, ((c // C.GRID) // s) * POS + (c % C.GRID) // s] = 1.0
            g = torch.cat([glo[:, None], here], -1)
            acts = []

            def run(seq, x):
                for layer in seq:
                    x = layer(x)
                    if isinstance(layer, nn.ReLU):
                        acts.append(x[0, 0])
                return x

            z = run(self.trunk, torch.cat([run(self.local, loc), run(self.glob, g), run(self.vec, vin)], -1))
            logits, value = self.actor(z)[0, 0], self.critic(z)[0, 0, 0]
            logits[action].backward()
        return dict(
            acts=[a.detach() for a in acts], logits=logits.detach(), value=float(value.detach()),
            contrib=(z[0, 0] * self.actor.weight[action]).detach(),
            att_local=(loc.grad * loc).detach().abs().view(N_LOCAL_CH, C.LOCAL, C.LOCAL),
            att_glob=(glo.grad * glo).detach().abs().view(N_GLOBAL_CH, C.GLOBAL, C.GLOBAL),
            att_vec=(vin.grad * vin).detach().abs().view(-1),
        )

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
