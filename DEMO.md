# RED CELL: demo & pitch guide

## Before judging
- [ ] `bash scripts/serve.sh`, open http://localhost:8000 in a full-screen browser window
- [ ] Hard-refresh once. The mission briefing shows only on first visit (click **Briefing** to reopen it)
- [ ] Check that all four enemies are enabled (the learned ones need `models/` and `data/difficulties.json`)
- [ ] Record a backup screen capture of one full run in case the laptop or venue misbehaves
- [ ] Sound is on by default (radar pings, interceptor launches, explosions, breach alarm). Toggle it with **Sound on** at the top right if the room needs quiet
- [ ] Everything runs offline (Three.js and the fonts are vendored), so wifi isn't needed

## The 3-minute pitch

**1. Hook (20 s)**
> "Cheap attack drones are now one of the biggest threats to bases and infrastructure. Operators practice against
> scripted enemies that do the same thing every time. We built a trainer where the enemy *learned* to attack."

**2. The judge plays (60 s)**: hand them the mouse.
- "You have 15 points. Radars see, but hills block them. Hover here and you'll see exactly what this radar can see."
- Let them place 4–5 units, pick **Veteran**, press **Launch attack**.
- While it replays: "Faded drones are ones your sensors haven't detected. Watch them use the valley." The six
  diamonds in the bottom bar are the drones (military symbology for hostile air tracks): they fill in when radar
  tracks one, cross out when it's shot down, and go solid red when it breaks through.

- **Point at the "What it's thinking" panel** (bottom right; click any drone or its diamond to switch). "This is the
  neural network's actual output for this drone, right now. The compass shows where it wants to go: red is its own
  choice, gold is 'follow the pathfinder', white is what it picked. The forecast is its value estimate: the network's
  own guess at whether this drone gets through." In the time-lapse, the first update's compass is an even blur
  (it has no idea yet), and the Veteran's is sharp.

- **Press N for the network view** (15 s, the strongest Technical Understanding moment): "This is the actual
  network: 2.1 million parameters. Left, what this drone sees. Middle, every layer firing. Right, its 10 possible
  moves. Orange lines are the neurons that voted for the move it took. Pink is attention: which inputs changed its
  decision most, here mostly the threat field around it, and you can see it on the map too." Press N again to close.

**3. After-action report (40 s)**
- Lead with **Tactics the swarm found**: "We never programmed tactics. The swarm was only rewarded for getting
  through. RED CELL mines all 64 attacks and names what emerged." Click **Show on map**, then **Watch an example**.
  Typical finds: the learned swarm *waits* (holds position 9 to 13 steps, then dashes), while A\* against the same
  defense just takes the radar-shadow route; in 360°, pincers in about a fifth of attacks.
- Read the breach rate: "That's averaged over 64 simulated attacks, not one lucky run."
- Point at the blind-spot callout and the red corridor. Click **Place suggested radar** and relaunch if time allows.
- Turn on **A\* planner view**: "Gold is what a classic A\* planner does against your defense, gray is the plain
  shortest path, red is what the trained swarm actually flew."

**4. Watch it learn (30 s)**: in the report, click **Run the training time-lapse**.
- "This is *your* defense against the same network at five points in its own training run."
- "At the first update it flies almost at random and barely gets through. Watch the breach rate climb as it learns
  to follow the pathfinder, then terrain masking and timing."

**5. How it was trained (40 s)**: click the link at the top right.
- "Every rule is a tensor op, so we run 4,096 battles at once on an AMD GPU with ROCm."
- "One shared neural network flies every drone, trained with PPO."
- "Every 10 updates a genetic search finds the defenses that best stop the *current* swarm, and the swarm trains
  against those. It's an arms race."
- The numbers: "It learned from random flight in 16 minutes on one AMD GPU. Against strong defenses that stop the
  classic A\* planner, it gets **15% more drones through**: 26% vs 23%, measured on 197 defenses it never trained on."
- How: "It spends less than half as long detected (4% vs 10%), waits for openings instead of rushing, spreads its
  approach over twice as many cells, and follows the pathfinder about a third of the time, improvising the rest."

**Optional, if there's time: 360° (20 s).** Flip **Threat axis** to **360°**. "Same pipeline, new theater: the base is
mid-map and drones come from every edge. We retrained a separate swarm for it on the AMD GPU in 16 minutes, and
there it gets 2.1 times as many drones through as A\* (57% vs 27% on held-out defenses)." Launch Veteran on
Crossroads and point at drones converging from several sides. The 360° time-lapse is the most dramatic one.

**6. Close (20 s)**
> "Who it helps: counter-drone operators and base planners. It lets them practice against an adversary that adapts,
> and see exactly why their defense failed."

## How it hits the Full Sail brief
| Brief asks for | RED CELL |
|---|---|
| Serious game / training simulation | Defend → attack → debrief training loop with measurable score (breach rate over 64 runs) |
| Pathfinding that's meaningful | Interceptors route by Dijkstra tables. The scripted enemy uses threat-aware A\* with replanning. The learned swarm is rewarded along the shortest-path field. The debrief visualizes A\*'s search |
| Explain who it helps / what problem | Counter-UAS operators and planners. Scripted enemies teach predictable habits |
| Demonstrate it works | Live play, 64-run statistics, A\* vs. learned routes overlay, training curve, training time-lapse on the player's own defense |

## Design talking points (the rubric asks if the design is intentional)
- **Two materials.** The live console is dark: the map, the controls, the battle. Anything you'd hand to someone is
  paper: the operation order (briefing), the after-action report, the technical annex. Paper means "a record you can
  act on", the console means "live".
- **Real conventions.** Unit symbols follow MIL-STD-2525 / NATO APP-6: friendly units in rectangular frames (the
  jammer is "EW", electronic warfare; the interceptor pad has the air-defense dome), hostile air tracks are diamonds.
  Report timestamps are military date-time groups (e.g. 101932ZOCT26 = 10th, 19:32 Zulu, Oct 2026). The green
  banners mimic the classification markings on real ops software. It's styling: nothing here is classified.
- **One accent per job.** Map colors identify units (cyan radar, amber jammer, green interceptors, red enemy). The
  interface itself stays neutral so the map carries the color. The report's stamp (Held, Penetrated, Overrun,
  Deterred) gives the verdict at a glance.
- **Sand-table terrain.** The board is a realistic scale model: grass and forest on low ground, bare rock on steep
  faces, and snow only on peaks above the drones' flight ceiling, so "white = drones can't fly here" reads at a
  glance. Sun shadows and darker valleys make the relief, and therefore the radar shadows, easy to judge.
- **RTS command bar.** Units, budget and the launch button live in one bottom bar like a strategy game, so the map
  gets the whole screen.

## Likely judge questions (know these cold)
- **Is the network view real or an illustration?** Real. The server re-runs the exact battle from its random seed
  (everything, including the drones' action sampling, is seeded) and records the network's activations at every
  step for the drone you picked. A unit test checks the re-run makes exactly the moves the replay showed.
- **How is "attention" computed?** Gradient times input: for the move the drone chose, how much each input value
  changed that move's score. It's a standard attribution method; it shows what the decision was sensitive to, not
  a human-style reason.
- **Why does it look at the threat field so much?** That input already encodes A\*-style cost-to-go around every
  sensor, so it's the most useful single signal. The network learned to lean on it and to deviate when timing or
  other drones matter.
- **How are tactics detected? Isn't that you labeling things?** Every breaching drone's route is resampled to ten
  points and grouped with routes that stay within about 3 cells of it the whole way. Each group gets a name from
  the direction it reached the base from and simple measured features (stationary steps, share of route on low
  ground, share undetected). Swarm patterns are timing rules across drones in the same attack (3+ breaches within 4
  steps, breaches 90 degrees apart within 8 steps, a kill on one side followed by a breach on another). The labels
  describe what happened, not intent, and the same analysis runs on the scripted A\* enemy for comparison.
- **Why does the learned swarm beat A\* by more in 360° than north?** With every edge open, picking *where* to
  attack matters more than *how* to fly a fixed corridor. A\* sends each drone down its cheapest route from wherever
  it entered; the swarm learned to probe and shift toward the weakest side. In the north theater the terrain
  funnels everyone, so there is less to choose.
- **Did you just rotate the old maps?** No. The old maps were designed around a northern threat (ridges shielding a
  base near the south edge). The 360° theater has six new maps with central bases, its own precomputed
  line-of-sight and Dijkstra tables, and its own trained network.
- **What is the "forecast" in the thinking panel?** PPO trains two heads on the same network: the policy (the
  action probabilities in the compass) and a critic that predicts the future reward from this state. We show the
  critic's number: high means the network expects this drone to break through (+10 reward), low means it expects
  to be shot down or time out. It's the network's own confidence, not something we compute separately.
- **Why does the training curve flatten?** It starts from random flight (2%), learns to use the pathfinder and
  reaches A\* level by update 10, then improves slowly (when to wait, when to deviate). Later checkpoints are noisy,
  so we keep the best one on the selection benchmark and verify it on held-out defenses.
- **Why did the breach rate go down from your first model (32% to 26%)?** The first model loitered: 12% of its drones
  never attacked, and against the strongest defenses none did, which made the game look broken. Fixing that cost a
  few points of breach rate, but the enemy now always commits. We'd rather report the honest number.
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
- **Reward?** Progress along shortest-path distance, +10 breach, −1 shot down, −3 if still flying at timeout. We removed a
  per-step detection penalty because it made the swarm hide and time out instead of attacking.
- **What's "never committed" / the stall rate?** Drones still airborne at the time limit: they neither broke through
  nor got shot down. Our first Veteran did this against the strongest defenses: it loitered outside radar range
  until time ran out. Timing out cost the same as being shot down, and the blue-team search discarded every defense
  the swarm couldn't beat, including the ones where it hid, so it never practiced against them. The AI-layout
  search then *found* that habit and exploited it (0% breach, 0% detected, all six drones still flying). We made
  timing out 3× as costly as being shot down, made the blue-team search keep stall-inducing defenses, retrained,
  and now reject any checkpoint that stalls on more than 10% of drones. A good example of an RL agent exploiting
  its reward, and of an arms race finding the exploit before we did.
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
