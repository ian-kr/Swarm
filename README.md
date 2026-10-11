# RED CELL

**A counter-drone defense trainer where the enemy swarm learned to attack using reinforcement learning on AMD GPUs.**

Cheap attack drones are now one of the biggest threats to bases and critical infrastructure. Operators and planners
usually practice against scripted enemies that act the same way every time, which teaches the wrong lessons.
In RED CELL you build an air defense on real-feeling terrain, then face a drone swarm that has played millions of
battles and learned to find the holes: radar shadows behind ridges, gaps between sensors, jammer-free lanes.
The debrief shows *where* and *why* your defense failed, and how to fix it.

Built for the **Full Sail "Pathways & Perception" serious games & simulation challenge** (pathfinding) at KnightHacks.

## Results (AMD Radeon Pro W7900D, ROCm)

Trained from scratch (random initial flight) for 17 minutes: 252M drone decisions at ≈251K/s. Measured on **205
held-out strong defenses** (layouts the scripted A\* attacker mostly fails against, never used for training or
checkpoint selection), 24 battles each:

| Attacker | Breach rate vs. strong defenses | Drones that never committed | vs. random defenses |
|---|---|---|---|
| Scripted A\* | 21.4% | | 78.0% |
| **Veteran (learned)** | **29.2%** (+36%) | 5.5% | 81.0% |
| Regular (Veteran, noisier decisions) | 25.8% | 12.8% | 76.7% |
| Recruit (Veteran, very noisy decisions) | 15.0% | 34.8% | 67.7% |

The swarm went from random flight (2% vs. strong defenses) to A\*-level by update 6 (9M decisions), then kept
improving slowly; update 130 is the Veteran. How it differs from A\* against strong defenses
(`scripts/behavior_report.py`): it spends 4% of its flight detected vs 11%, takes 70 steps to break through vs 30
because it waits for openings (10% of its moves are hovers), spreads its attacks over 116 cells per battle vs 66
(harder to predict), and follows the pathfinder on 43% of its moves, improvising the rest.

**360° theater** (separate swarm, same pipeline and settings, 16 minutes from random flight), on 252 held-out strong
360° defenses, 24 battles each:

| Attacker | Breach rate vs. strong defenses | Drones that never committed | vs. random defenses |
|---|---|---|---|
| Scripted A\* | 25.0% | | 76.5% |
| **Veteran (learned)** | **61.7%** (2.5×) | 8.3% | 83.0% |
| Regular | 42.8% | 35.1% | 74.2% |
| Recruit | 32.2% | 45.6% | 61.2% |

With every edge open, choosing *where* to attack matters more, and the learned swarm's lead over A\* is about
two and a half times. It went from 16% to 63% on the strong benchmark in 18 updates.

All numbers above are against the upgraded interceptors (threat-based targeting and lead pursuit, see the table
below). Both swarms were retrained against them. Compared with the previous run (each measured on its own held-out set,
which is rebuilt from the current interceptors), A\* lost ground (22.6% → 21.4% north, 26.6% → 25.0% 360°) while
the retrained swarms gained (26.1% → 29.2%, 56.7% → 61.7%): the learned enemy adapts to a smarter defense, the
fixed pathfinder cannot.

Our first Veteran scored higher on paper (31.5% on its held-out set, against the earlier interceptors) but 12% of its drones never attacked, and
against the strongest defenses none did. The fix (below) traded some headline breach rate for an enemy that always
commits.

## How to play (60 seconds)

1. **Defend.** Spend a budget of 15 on radars (line-of-sight detection), jammers (scramble drone navigation) and
   interceptor pads (3 interceptors each, which chase *detected* drones). Hover with the radar tool to preview exactly
   what a radar would see from that spot.
2. **Attack.** Pick an enemy: *Scripted (A\*)*, *Recruit*, *Regular* or *Veteran*, then launch. RED CELL simulates
   64 attacks and replays a representative one.
3. **Review.** The after-action report shows your breach rate across all attacks, the corridors the drones used, an
   automatic blind-spot diagnosis with a suggested radar position, the A\* planner view, and how you compare to an
   AI-optimized defense.
   During any replay against a learned enemy, the **What it's thinking** panel shows the selected drone's live
   network output: a compass of move probabilities (its own choice vs. "follow the pathfinder"), and the critic's
   value estimate as the drone's own forecast. Click any drone to switch.
   The report's **Tactics the swarm found** section mines all 64 attacks for recurring patterns: approach
   corridors (named by direction and character: wait-and-dash, valley run, radar-shadow run), saturation, pincers,
   decoy patterns and multi-axis probing. Each one can be drawn on the map or watched in an example attack. Nobody
   programmed these: the swarm was only rewarded for getting through. Against the same defenses, the scripted A\*
   planner's breaches come out as plain radar-shadow or direct runs; the learned swarm's as wait-and-dash.
   Press **N** (or **Open network** in that panel) for the full **neural network view**: the drone's actual
   inputs (its 11×11 local view and 9 input channels, the 16×16 map summary, 10 status values), every hidden layer
   lighting up as it fires, the 10 action outputs and the value head, the last-layer neurons that pushed hardest for
   or against the move it chose, and attention (input × gradient) showing which inputs changed the decision most,
   also projected onto the terrain around the drone while the camera follows it. The server re-runs the exact battle
   from its seed to record this (`/api/brain`), so every number shown is what the network computed in that replay.
4. **Watch it learn.** Replay *your* defense against snapshots of the swarm from its own training run, from its first
   update (random flight) to the Veteran, and watch the breach rate climb as it learns.

## Two theaters

Switch with **Threat axis** above the enemy list. Each theater has its own maps and its own trained swarm.

* **North axis** (the original): six maps built around an attack from the north, with the base near the southern edge.
* **360°**: six maps with the base mid-map (Crossroads, Citadel, Hill Country, Escarpment, Steppe, Saddle). Drones
  enter along every edge, so the defense has to cover all sides, and the swarm can probe for the weakest one.
  Defenses stay at least 6 cells in from every edge, the same buffer the north mode keeps from its entry line.
  Same budget: the scripted A\* attacker gets through random defenses about equally often in both theaters (78%).

## Where pathfinding is used

| Where | Algorithm | Why it matters |
|---|---|---|
| Interceptors chasing drones | All-pairs **Dijkstra** next-hop tables (precomputed per map), aimed by **lead pursuit** | Lets 4,096 battles route interceptors in parallel on the GPU with one tensor lookup. Each interceptor aims where its target is heading (up to 2 moves ahead, straight homing within 3 cells) and picks targets by **threat evaluation**: the tracked drone closest to the base, weighed against its own flight distance, skipping drones beyond its fuel |
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
  A\* would do). The current Veteran started from uniformly random flight (`--follow-bias 0`), learned to use the
  pathfinder within ~10 updates, and then learned *when to deviate*.
* **Reward**: +progress along the shortest path to base, +10 breach, −1 shot down, −3 still airborne at timeout,
  and a small time penalty. (An earlier version also penalised every step spent detected. The swarm learned to hide
  and time out instead of attacking, so we removed it.)
* **The swarm that refused to attack.** Our first Veteran never attacked the strongest defenses: it loitered outside
  radar range until time ran out (0% breach, 0% detected, every drone still airborne). Two things caused it. Timing
  out cost the same as being shot down, and the blue-team search threw away every defense the swarm couldn't beat,
  including the ones where it hid, so it never trained against them. The AI-layout search then *found* that habit and
  exploited it. The fix: timing out now costs 3× more than being shot down, the blue-team pool keeps the defenses
  where the swarm stalls at the front of the queue, and difficulty selection rejects any checkpoint whose drones stall
  more than 10% of the time. Every simulation now reports a **stall rate** (drones that neither broke through nor
  were shot down), and the debrief calls it out when the enemy held back.
* **Blue-team search** (`redcell/search.py`): a genetic search over defense layouts, scored against the current
  policy. It keeps the *frontier*: layouts the swarm beats only ~20% of the time (unbeatable ones give no learning
  signal; our first version filled up with them and stalled), plus the layouts where the swarm stalls. Half of all
  training battles use them. The same search produces the "AI-optimized" layout for each map (`scripts/recommend.py`),
  penalising layouts that only win because the swarm refuses to attack.
* **Difficulty levels** (`scripts/select_difficulties.py`): Veteran is the checkpoint with the highest breach rate on
  strong defenses among those that stall on at most 10% of drones. Regular and Recruit fly the same network with a
  higher sampling temperature (2.0 and 3.0). All three are re-measured on a held-out set of strong defenses.
* **Training timeline**: the same script exports five snapshots of the Veteran's own training run, spaced
  geometrically from the first update to the Veteran, for the in-game "watch it learn" time-lapse.

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
python scripts/select_difficulties.py checkpoints/amd     # add --prior DIR for a run resumed from DIR
python scripts/recommend.py
# the 360-degree theater: same pipeline with --mode 360 (models/360/, data/*_360.*)
python -m redcell.terrain 360
python train.py --mode 360 --follow-bias 0 --amp --out checkpoints/amd_360
python scripts/select_difficulties.py checkpoints/amd_360 --mode 360 && python scripts/recommend.py --mode 360
```

Then copy `models/` (including `models/timeline/`), `data/difficulties.json`, `data/recommended.json` and
`data/training_metrics.jsonl` to the demo laptop and restart the server.

```bash
python -m pytest tests/test_core.py    # simulation + A* / Dijkstra agreement tests
python tests/e2e_ui.py OUT_DIR         # drives the UI in headless Chromium (needs playwright), saves screenshots,
                                       # including the training time-lapse when a timeline is installed
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
web/                  Three.js client (Three.js and fonts vendored, procedural Web Audio sound; works offline)
scripts/              ROCm check, cloud training, difficulty selection, layout recommendation
```
