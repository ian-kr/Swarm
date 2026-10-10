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
