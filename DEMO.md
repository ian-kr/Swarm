# RED CELL: demo & pitch guide

## Before judging
- [ ] `bash scripts/serve.sh`, open http://localhost:8000 in a full-screen browser window
- [ ] Hard-refresh once. The mission briefing shows only on first visit (click **?** to reopen it)
- [ ] Check that all four enemies are enabled (the learned ones need `models/` and `data/difficulties.json`)
- [ ] Record a backup screen capture of one full run in case the laptop or venue misbehaves
- [ ] Everything runs offline (Three.js is vendored), so wifi isn't needed

## The 3-minute pitch

**1. Hook (20 s)**
> "Cheap attack drones are now one of the biggest threats to bases and infrastructure. Operators practice against
> scripted enemies that do the same thing every time. We built a trainer where the enemy *learned* to attack."

**2. The judge plays (60 s)**: hand them the mouse.
- "You have 15 points. Radars see, but hills block them. Hover here and you'll see exactly what this radar can see."
- Let them place 4–5 units, pick **Veteran**, press **Launch attack**.
- While it replays: "Faded drones are ones your sensors haven't detected. Watch them use the valley."

**3. Debrief (40 s)**
- Read the breach rate: "That's averaged over 64 simulated attacks, not one lucky run."
- Point at the blind-spot callout and the red corridor. Click **Place suggested radar** and relaunch if time allows.
- Turn on **A\* planner view**: "Gold is what a classic A\* planner does against your defense, gray is the plain
  shortest path, red is what the trained swarm actually flew."

**4. Under the hood (40 s)**: click the button.
- "Every rule is a tensor op, so we run 4,096 battles at once on an AMD GPU with ROCm."
- "One shared neural network flies every drone, trained with PPO."
- "Every 10 updates a genetic search finds the defenses that best stop the *current* swarm, and the swarm trains
  against those. It's an arms race."
- The numbers: "Against strong defenses that stop the classic A\* planner, our trained swarm gets **40% more drones
  through**: 32% vs 23%, measured on 197 defenses it never trained on."
- How: "It spends half as long detected, follows the pathfinder 60% of the time and improvises the rest, waits
  for openings, and spreads its approach so it's harder to predict."

**5. Close (20 s)**
> "Who it helps: counter-drone operators and base planners. It lets them practice against an adversary that adapts,
> and see exactly why their defense failed."

## How it hits the Full Sail brief
| Brief asks for | RED CELL |
|---|---|
| Serious game / training simulation | Defend → attack → debrief training loop with measurable score (breach rate over 64 runs) |
| Pathfinding that's meaningful | Interceptors route by Dijkstra tables. The scripted enemy uses threat-aware A\* with replanning. The learned swarm is rewarded along the shortest-path field. The debrief visualizes A\*'s search |
| Explain who it helps / what problem | Counter-UAS operators and planners. Scripted enemies teach predictable habits |
| Demonstrate it works | Live play, 64-run statistics, A\* vs. learned routes overlay, training curve |

## Likely judge questions (know these cold)
- **Why is the training curve flat after the Veteran point?** The swarm starts at A\* level because it can choose
  "follow the pathfinder". Most of the learning (when to deviate) happened in the first ~25 updates. Later updates
  were noisier, so we kept the best checkpoint and verified it on held-out defenses.
- **How are Recruit/Regular easier?** Same network, but it samples its moves with more randomness (temperature 3 and 2).
- **How does line-of-sight work?** For every pair of cells we march along the ray from the radar mast to a drone
  flying 0.03 above the terrain and check whether terrain pokes above the line (bilinear height sampling). Precomputed
  once per map into a 1024×1024 table.
- **Why Dijkstra tables instead of A\* for interceptors?** Thousands of battles run in parallel on the GPU. Running A\*
  per interceptor per step would serialize everything. All-pairs Dijkstra gives a next-hop table, so routing is one
  gather. A test proves A\* returns identical routes (per-cell cost jitter makes shortest paths unique).
- **What does A\*'s heuristic look like?** Straight-line distance to the base edge. Admissible because every step
  costs at least its length.
- **What does the network see?** An 11×11 local patch (9 channels, including the threat-aware distance field), a
  16×16 whole-map summary with a position one-hot, and 10 scalars. It outputs 9 moves (8 directions + hover).
- **What's the threat field?** The A\* cost model (distance + penalties for radar/jammer/interceptor zones) solved
  for *every* cell at once by Bellman-Ford relaxation: each cell repeatedly takes the min over its 8 neighbours of
  (neighbour's distance + step + threat). That's ~60 tensor ops on the GPU for thousands of battles. The swarm starts
  from A\*-level route knowledge and learns what A\* can't: timing, spreading out, dodging interceptors.
- **Why an MLP, not a CNN?** We profiled on the Radeon: small convolutions were 30–60× slower than the equivalent
  matrix multiplies under ROCm, and fp32 GEMMs were 40× slower than bf16. Switching to a bf16 MLP made training 13× faster.
- **Reward?** Progress along shortest-path distance, +10 breach, −1 shot down, −1 if still flying at timeout. We removed a
  per-step detection penalty because it made the swarm hide and time out instead of attacking.
- **Why PPO?** Stable on-policy algorithm that handles many parallel agents well. Shared parameters across drones.
- **What did you learn / what was hard?** (Answer honestly.) E.g. keeping the training simulation and the game
  identical (solved by making them the *same code*), GPU memory limits, reward shaping, game balance (the first
  version let 90% of drones through any defense), an arms race that stalled because the blue team only kept
  unbeatable defenses (fixed by keeping the ~20%-beatable frontier), and ROCm performance profiling, a swarm that learned to be *too* cautious (reward fix), and a comparison bug where
  CPU and GPU random generators gave the two attackers different defenses (fixed: identical layouts, held-out set).
- **What used AI assistance?** The rubric rewards honesty here. Say what you built and what tools helped.

## If something breaks
- Learned enemies greyed out → `data/difficulties.json` or `models/` missing. The Scripted (A\*) enemy always works.
- Server won't start → `python -m redcell.terrain` to rebuild `data/maps.npz`, then `bash scripts/serve.sh`.
- Slow simulation → the server runs on CPU by design. Scripted runs 32 battles (≈3 s), learned runs 64 (≈1–2 s).
