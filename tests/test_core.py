import numpy as np
import torch

from redcell import config as C
from redcell.astar import ScriptedAttacker, astar
from redcell.env import N_GLOBAL_CH, N_LOCAL_CH, DefenseEnv, Layout, MapTensors
from redcell.terrain import load_maps

MS = load_maps()
MT = MapTensors(MS, "cpu")


def test_astar_matches_dijkstra_tables():
    """A* (no threat cost) must reproduce the exact routes the next-hop tables encode."""
    rng = np.random.default_rng(0)
    for mi in range(MS.n):
        cells = np.nonzero(MS.passable[mi] & np.isfinite(MS.dist[mi, MS.base_cell[mi]]) & (MS.dist[mi, MS.base_cell[mi]] < 900))[0]
        for _ in range(25):
            s, t = rng.choice(cells, 2, replace=False)
            goal = np.zeros(C.N_CELLS, bool)
            goal[t] = True
            path = astar(MS.passable[mi], MS.cell_cost[mi], int(s), goal, (t % C.GRID, t // C.GRID))
            table = [int(s)]
            while table[-1] != t:
                table.append(int(MS.next_hop[mi, t, table[-1]]))
                assert len(table) < 2000
            assert path == table, (MS.names[mi], s, t)


def test_env_runs_and_terminates():
    env = DefenseEnv(MT, 64, "cpu", auto_reset=False, seed=1)
    env.reset()
    for _ in range(C.MAX_STEPS):
        r, done, valid, info = env.step(torch.randint(C.N_ACTIONS, (64, C.N_DRONES)))
        assert torch.isfinite(r).all()
        assert not (done & ~valid).any()
    assert info["env_done"].all()
    obs = env.observe()
    assert obs["local"].shape == (64, C.N_DRONES, N_LOCAL_CH, C.LOCAL, C.LOCAL)
    assert obs["glob"].shape == (64, N_GLOBAL_CH, C.GLOBAL, C.GLOBAL)


def test_drones_stay_on_passable_cells():
    env = DefenseEnv(MT, 32, "cpu", auto_reset=True, seed=2)
    env.reset(map_id=4)   # canyon: lots of blocked terrain
    for _ in range(200):
        env.step(torch.randint(C.N_ACTIONS, (32, C.N_DRONES)))
        assert MT.passable[env.map_id[:, None], env.drone_cell].all()


def test_no_defense_means_scripted_attack_breaches():
    env = DefenseEnv(MT, 4, "cpu", auto_reset=False, seed=3)
    empty = Layout.from_lists([{} for _ in range(4)], "cpu")
    env.reset(map_id=0, layout=empty)
    atk = ScriptedAttacker(MS)
    for _ in range(C.MAX_STEPS):
        _, _, _, info = env.step(atk.act(env))
        if info["env_done"].all():
            break
    assert (info["breaches"] == C.N_DRONES).all()


def test_strong_defense_kills_scripted_attackers():
    mi = 0
    bc = int(MS.base_cell[mi])
    bx, by = bc % C.GRID, bc // C.GRID
    near = lambda dx, dy: (by + dy) * C.GRID + bx + dx
    lay = {"radar": [near(0, -4), near(-6, -3)], "pad": [near(2, -2), near(-2, -2)]}
    env = DefenseEnv(MT, 16, "cpu", auto_reset=False, seed=4)
    env.reset(map_id=mi, layout=Layout.from_lists([lay] * 16, "cpu"))
    atk = ScriptedAttacker(MS)
    for _ in range(C.MAX_STEPS):
        _, _, _, info = env.step(atk.act(env))
    assert info["kills"].float().mean() >= 1.0


def test_threat_field_matches_astar_cost_with_no_defenses():
    """With no sensors the threat field is plain shortest flight distance to the base edge."""
    env = DefenseEnv(MT, 6, "cpu", auto_reset=False, seed=5)
    env.reset(map_id=torch.arange(6), layout=Layout.from_lists([{}] * 6, "cpu"))
    z = torch.zeros(6, C.N_CELLS, dtype=torch.bool)
    field = env.threat_field(torch.arange(6), z, z, z)
    for mi in range(6):
        spawn = MS.spawn_mask[mi].nonzero()[0]
        base_cells = MS.base_mask[mi].nonzero()[0]
        # table distance to the nearest base cell (tables include a tiny cost jitter, hence the tolerance)
        ref = MS.dist[mi][base_cells][:, spawn].min(0)
        got = field[mi, torch.as_tensor(spawn)].numpy()
        assert np.allclose(got, ref, rtol=0.03), MS.names[mi]


def test_follow_action_reaches_base_without_defenses():
    env = DefenseEnv(MT, 12, "cpu", auto_reset=False, seed=8)
    env.reset(map_id=torch.arange(12) % 6, layout=Layout.from_lists([{}] * 12, "cpu"))
    for _ in range(C.MAX_STEPS):
        _, _, _, info = env.step(torch.full((12, C.N_DRONES), C.FOLLOW))
        if info["env_done"].all():
            break
    assert (info["breaches"] == C.N_DRONES).all()


def test_threat_field_routes_around_radar():
    env = DefenseEnv(MT, 1, "cpu", auto_reset=False, seed=6)
    bc = int(MS.base_cell[5])
    lay = {"radar": [(bc // C.GRID - 8) * C.GRID + bc % C.GRID]}
    env.reset(map_id=5, layout=Layout.from_lists([lay], "cpu"))
    z = torch.zeros(1, C.N_CELLS, dtype=torch.bool)
    plain = env.threat_field(torch.tensor([5]), z, z, z)
    threat = env.threat_field(torch.tensor([5]), env.coverage, env.jam_zone, env.pad_zone)
    assert (threat >= plain - 1e-4).all()
    assert (threat - plain).max() > 4.0       # detours (or penalties) appear behind the radar


def test_stall_rate_counts_drones_that_never_commit():
    """With no defenses, a policy that only hovers can neither break through nor be shot down: every drone stalls."""
    from redcell.search import evaluate_layouts

    class Hover:
        def act_env(self, obs):
            B, D = obs["cell"].shape
            return torch.full((B, D), C.MOVES.index((0, 0))), None, None

    m = torch.arange(8) % MS.n
    lay = Layout.from_lists([{"radar": [], "jammer": [], "pad": []}] * 8, "cpu")
    breach, stall = evaluate_layouts(Hover(), MT, lay, m, 2, "cpu", return_stall=True)
    assert (breach == 0).all()
    assert torch.allclose(stall, torch.ones_like(stall))


def test_timeout_costs_more_than_being_shot_down():
    # otherwise loitering until the time limit is as good as a failed attack, and the swarm learns to hide
    assert C.R_TIMEOUT < C.R_KILLED < 0


def test_game_reports_stall_and_rejects_unknown_stage():
    import pytest
    from redcell.game import Game, LayoutError

    g = Game()
    r = g.simulate(0, {"radar": [], "jammer": [], "pad": []}, "scripted", seed=1, n=4)
    assert 0.0 <= r["stall_rate"] <= 1.0 and r["rep"]["stalled"] >= 0
    with pytest.raises(LayoutError):
        g.simulate(0, {}, f"stage:{len(g.timeline)}", n=2)


def test_360_theater_spawns_on_every_side_with_a_central_base():
    ms360 = load_maps(mode="360")
    G = C.GRID
    x, y = np.arange(C.N_CELLS) % G, np.arange(C.N_CELLS) // G
    edge = np.minimum.reduce([x, y, G - 1 - x, G - 1 - y])
    for i in range(ms360.n):
        bx, by = ms360.base_cell[i] % G, ms360.base_cell[i] // G
        assert abs(bx - 15.5) <= 2 and abs(by - 15.5) <= 2, ms360.names[i]
        spawn = ms360.spawn_mask[i]
        assert (edge[spawn] < C.SPAWN_ROWS).all()
        sides = [(y < C.SPAWN_ROWS), (y >= G - C.SPAWN_ROWS), (x < C.SPAWN_ROWS), (x >= G - C.SPAWN_ROWS)]
        assert sum((spawn & s).sum() >= 8 for s in sides) >= 3, ms360.names[i]   # Escarpment: cliff on one side
        assert (edge[ms360.place_mask[i]] >= C.PLACE_MIN_ROW).all()             # same buffer as the north mode
        assert (ms360.dist[i, ms360.base_cell[i]][spawn] < 900).all()           # every entry cell can reach the base


def test_tactic_discovery_reports_patterns_with_examples():
    from redcell.game import Game
    g = Game()
    weak = {"radar": [688, 713], "jammer": [661], "pad": [786, 750]}   # lets plenty of scripted drones through
    r = g.simulate(0, weak, "scripted", seed=5)
    assert r["tactics"], "expected at least one pattern from a leaky defense"
    for t in r["tactics"]:
        assert t["kind"] in {"corridor", "probing", "saturation", "pincer", "decoy"}
        assert t["name"] and t["text"] and 0 < t["share"] <= 1 and t["paths"]
        assert str(t["battle"]) in r["examples"] and r["examples"][str(t["battle"])]["frames"]
    tight = g.simulate(0, {"radar": [], "jammer": [], "pad": []}, "scripted", seed=5, n=4)
    assert isinstance(tight["tactics"], list)


def test_network_view_reproduces_the_replayed_battle():
    """brain() re-runs a battle from its seed: the inspected drone must make exactly the moves the replay showed."""
    from redcell.game import Game
    g = Game()
    lay = {"radar": [688, 713], "jammer": [661], "pad": [786, 750]}
    r = g.simulate(0, lay, "veteran", seed=99, n=16)
    b = r["rep"]["index"]
    br = g.brain(0, lay, "veteran", 99, 16, b, 2)
    frames = r["rep"]["frames"]
    assert br["steps"] > 0 and len(br["layers"]) == 7
    assert all(frames[t]["d"][2][0] == c for t, c in zip(br["t"], br["cell"]))
    assert all(abs(sum(p) - 1) < 0.01 for p in br["p"])
    assert all(abs(sum(s) - 1) < 0.01 for s in br["att_share"])


def test_north_theater_unchanged():
    assert MS.mode == "north" and MS.names[0] == "Ridgeback"
    assert (np.arange(C.N_CELLS)[MS.spawn_mask[0]] // C.GRID < C.SPAWN_ROWS).all()


def test_interceptor_prefers_the_drone_nearest_the_base_and_leads_its_target():
    """Threat evaluation: of two tracked drones, a free interceptor goes for the one closer to the base,
    even if the other is nearer to the interceptor. Lead pursuit: it aims ahead of a moving target."""
    from redcell.env import FLYING
    mi = 5                                              # Flatlands: open ground
    G = C.GRID
    base = int(MS.base_cell[mi])
    bx, by = base % G, base // G
    pad = (by - 6) * G + bx
    env = DefenseEnv(MT, 1, "cpu", auto_reset=False, seed=0)
    env.reset(map_id=torch.tensor([mi]), layout=Layout.from_lists([{"radar": [], "jammer": [], "pad": [pad]}], "cpu"))
    near_pad = (by - 12) * G + bx                       # 6 cells from the pad, 12 from the base
    near_base = (by - 3) * G + bx + 6                   # farther from the pad, much closer to the base
    cells = torch.tensor([[near_pad, near_base] + [0] * (C.N_DRONES - 2)])
    env.drone_cell[:] = cells
    env.prev_cell[:] = cells
    env.prev_cell[0, 1] = near_base + 1                 # drone 1 is moving west, one cell per step
    env.drone_status[:] = 0
    env.drone_status[0, 2:] = 1                         # only drones 0 and 1 are in play
    env.detected[:] = False
    env.detected[0, :2] = True
    env._interceptors()
    flying = (env.int_state[0] == FLYING).nonzero().squeeze(1)
    assert flying.numel() >= 1
    first = int(flying[0])
    assert int(env.int_target[0, first]) == 1, "should engage the drone closer to the base first"
    goal = int(env.int_goal[0, first])
    assert goal % G < (near_base % G), "should aim west of the drone, where it is heading"
