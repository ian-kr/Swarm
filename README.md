# RED CELL

**A counter-drone defense trainer where the enemy swarm learned to attack using reinforcement learning on AMD GPUs.**

Cheap attack drones are now one of the biggest threats to bases and critical infrastructure. Operators and planners
usually practice against scripted enemies that act the same way every time, which teaches the wrong lessons.
In RED CELL you build an air defense on real-feeling terrain, then face a drone swarm that has played millions of
battles and learned to find the holes: radar shadows behind ridges, gaps between sensors, jammer-free lanes.
The debrief shows *where* and *why* your defense failed, and how to fix it.

Built for the **Full Sail "Pathways & Perception" serious games & simulation challenge** (pathfinding) at KnightHacks.

## Results (AMD Radeon Pro W7900D, ROCm)

Trained for 41 minutes: 630M drone decisions at ≈260K/s. Measured on **197 held-out strong defenses** (layouts the
scripted A\* attacker mostly fails against, never used for training or checkpoint selection), 8 battles each:

| Attacker | Breach rate vs. strong defenses | vs. random defenses |
|---|---|---|
| Scripted A\* | 22.6% | 78.5% |
| **Veteran (learned)** | **31.7%** (+40%) | 78.0% |
| Regular (Veteran, noisier decisions) | 23.4% | 72.1% |
| Recruit (Veteran, very noisy decisions) | 12.4% | 61.2% |

How the Veteran differs from A\* against strong defenses (`scripts/behavior_report.py`): it spends half as much of
its flight detected (5% vs 10%), follows the pathfinder on 60% of its moves and improvises on the rest, hovers 5% of
the time waiting for openings, and spreads its attacks over 104 cells per battle vs 65 (harder to predict).
Most of the gain came in the first ~25 updates; the best checkpoint is kept as the Veteran.

## How to play (60 seconds)

1. **Defend.** Spend a budget of 15 on radars (line-of-sight detection), jammers (scramble drone navigation) and
   interceptor pads (3 interceptors each, which chase *detected* drones). Hover with the radar tool to preview exactly
   what a radar would see from that spot.
2. **Attack.** Pick an enemy: *Scripted (A\*)*, *Recruit*, *Regular* or *Veteran*, then launch. RED CELL simulates
   64 attacks and replays a representative one.
3. **Debrief.** See your breach rate across all attacks, the corridors the drones used, an automatic blind-spot
   diagnosis with a suggested radar position, the A\* planner view, and how you compare to an AI-optimized defense.

## Where pathfinding is used

| Where | Algorithm | Why it matters |
|---|---|---|
| Interceptors chasing drones | All-pairs **Dijkstra** next-hop tables (precomputed per map) | Lets 4,096 battles route interceptors in parallel on the GPU with one tensor lookup |
| Scripted enemy | **Threat-aware A\***: cost = distance + penalties for radar coverage, jammer and interceptor zones; replans when jamming knocks a drone off course | The classic game-AI opponent, and the baseline the learned swarm is compared against |
| Debrief "A\* planner view" | A\* with its explored-node trace | Shows how a textbook planner would route around *your* defense vs. the plain shortest path |
| Learned swarm's senses | **Threat-aware distance field**: the same cost model as A\*, solved for every cell of every battle at once by parallel Bellman-Ford relaxation on the GPU | Each drone *sees* a pathfinding field, so the swarm starts from A\*-quality routes and learns tactics on top: timing, splitting up, dodging interceptors, riding jamming drift |
| Learned swarm's reward | Progress along the plain **shortest-path distance** to the base (Dijkstra tables) | Rewards getting closer without dictating the route |

A unit test checks that A\* and the Dijkstra tables produce *identical* routes. A tiny per-cell cost jitter makes every
shortest path unique.

## How the swarm was trained

```
terrain tables ──► batched GPU simulation ──► PPO (shared policy, 1 agent per drone) ──► checkpoints = difficulty levels
 (LOS + Dijkstra)     4096 battles at once            ▲                                     Recruit / Regular / Veteran
                      + threat-aware distance          │
                        field per battle   blue-team genetic search: every 10 updates, find the
                                           defenses at the edge of what the *current* swarm can
                                           beat and train against them (an arms race)
```

* **Simulation** (`redcell/env.py`): every rule (movement, line-of-sight detection, jamming drift, interceptor
  assignment and pursuit, kill rolls) is a PyTorch tensor operation over a batch of battles. The same code runs
  4,096 battles on the GPU for training and one battle on a CPU for the game, so training and gameplay can never disagree.
* **Observation** per drone: an 11×11 egocentric patch (terrain, blocked cells, radar coverage, jammers, pads, flying
  interceptors, other drones, base, threat-aware distance field), a 16×16 whole-map summary plus a position one-hot,
  and a few scalars (direction and distance to base, detected?, jammed?, time, threat-field value).
* **Policy** (`redcell/policy.py`): an MLP shared by all drones, run in bf16. We profiled on the Radeon Pro W7900D:
  small convolutions took 60–140 ms per minibatch under ROCm, while the equivalent MLP takes 2 ms. Switching made
  training 13× faster (≈60K battle-steps/s).
* **Actions**: 8 compass moves, hover, and **follow the pathfinder** (step along the threat-aware field, i.e. what
  A\* would do). The swarm starts out matching A\* and learns *when to deviate*.
* **Reward**: +progress along the shortest path to base, +10 breach, −1 shot down, −1 still airborne at timeout,
  and a small time penalty. (An earlier version also penalised every step spent detected. The swarm learned to hide
  and time out instead of attacking, so we removed it.)
* **Blue-team search** (`redcell/search.py`): a genetic search over defense layouts, scored against the current
  policy. It keeps the *frontier*: layouts the swarm beats only ~20% of the time (unbeatable ones give no learning
  signal; our first version filled up with them and stalled). Half of all training battles use them. The same search produces the
  "AI-optimized" layout for each map (`scripts/recommend.py`).
* **Difficulty levels** (`scripts/select_difficulties.py`): Veteran is the checkpoint with the highest breach rate on
  strong defenses. Regular and Recruit fly the same network with a higher sampling temperature (2.0 and 3.0).

## Running it

```bash
pip install -r requirements.txt        # install the right torch build for your GPU first
python -m redcell.terrain              # build map tables (≈10 s, cached in data/maps.npz)
python server.py                       # http://localhost:8000
```

The *Scripted (A\*)* enemy works with no training. To enable the learned difficulties:

```bash
# AMD cloud GPU (ROCm). Run inside tmux:
bash scripts/cloud_train.sh            # check_rocm → train → select difficulties → recommend layouts
# or manually / on any GPU:
python train.py --envs 4096 --updates 600 --amp --out checkpoints/amd
python scripts/select_difficulties.py checkpoints/amd
python scripts/recommend.py
```

Then copy `models/`, `data/difficulties.json`, `data/recommended.json` and `data/training_metrics.jsonl` to the demo laptop and restart the server.

```bash
python -m pytest tests/test_core.py    # simulation + A* / Dijkstra agreement tests
python tests/e2e_ui.py OUT_DIR         # drives the UI in headless Chromium (needs playwright), saves screenshots
```

## Project layout

```
redcell/config.py     every game rule and constant (shared by training and the game)
redcell/terrain.py    procedural maps, line-of-sight tables, Dijkstra next-hop tables
redcell/env.py        batched PyTorch simulation
redcell/astar.py      A* search, scripted attacker
redcell/policy.py     attacker policy network
redcell/search.py     blue-team genetic search
redcell/game.py       running a player's layout, debrief analysis
train.py              PPO training loop
server.py             FastAPI server for the web client
web/                  Three.js client (vendored, works offline)
scripts/              ROCm check, cloud training, difficulty selection, layout recommendation
```
