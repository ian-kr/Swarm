"""Terrain generation and the precomputed tables everything else relies on.

For every map we precompute, once:
  * heights / passable cells / base / spawn zone
  * vis[radar_cell, drone_cell]   line-of-sight from a radar mast to a low-flying drone
  * dist[target, cell]            shortest flight distance (all-pairs Dijkstra)
  * next_hop[target, cell]        first step on that shortest path

The tables turn pathfinding and line-of-sight into O(1) tensor lookups, which is
what lets thousands of simulations run in parallel on the GPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
try:                                   # scipy is optional: a pure-Python Dijkstra is used if it is missing
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra
except ImportError:                    # pragma: no cover
    csr_matrix = dijkstra = None

from . import config as C

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE = DATA_DIR / "maps.npz"
CACHE_VERSION = 3

G = C.GRID
N = C.N_CELLS

# neighbour offsets for the flight graph (8-connected)
NEIGHBOURS = [(dx, dy) for dx, dy in C.MOVES if (dx, dy) != (0, 0)]


def _grid():
    y, x = np.mgrid[0:G, 0:G].astype(np.float32)
    return x, y


def _bump(x, y, cx, cy, sx, sy=None, h=1.0):
    sy = sx if sy is None else sy
    return h * np.exp(-(((x - cx) / sx) ** 2 + ((y - cy) / sy) ** 2))


def _ridge(x, y, pts, width, h):
    """A ridge following a polyline of (x, y) points."""
    out = np.zeros_like(x)
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        dx, dy = x1 - x0, y1 - y0
        L2 = dx * dx + dy * dy
        t = np.clip(((x - x0) * dx + (y - y0) * dy) / L2, 0, 1)
        d = np.hypot(x - (x0 + t * dx), y - (y0 + t * dy))
        out = np.maximum(out, h * np.exp(-((d / width) ** 2)))
    return out


def _noise(rng, scale=0.08):
    """Cheap smooth noise: sum of random small bumps."""
    x, y = _grid()
    out = np.zeros((G, G), np.float32)
    for _ in range(40):
        out += _bump(x, y, rng.uniform(0, G), rng.uniform(0, G), rng.uniform(1.5, 4), h=rng.uniform(-1, 1))
    return scale * out / (np.abs(out).max() + 1e-6)


# Each recipe returns (heights, base_xy). Coordinates: x east, y south.
def _ridgeback(rng):
    x, y = _grid()
    h = 0.15 + _ridge(x, y, [(-2, 15), (9, 13), (20, 16), (34, 14)], 2.6, 0.85)
    h -= _bump(x, y, 9, 13.5, 1.6, 3, 0.55) + _bump(x, y, 24, 15.5, 1.6, 3, 0.6)   # two passes
    h += _bump(x, y, 4, 24, 3, h=0.35) + _bump(x, y, 28, 25, 3, h=0.3)
    return h, (16, 26)


def _twin_valleys(rng):
    x, y = _grid()
    h = 0.2 + _ridge(x, y, [(10, -2), (11, 12), (9, 24)], 2.2, 0.75)
    h += _ridge(x, y, [(22, -2), (21, 10), (23, 22)], 2.2, 0.75)
    h += _bump(x, y, 16, 6, 3, h=0.25)
    return h, (16, 27)


def _broken_hills(rng):
    x, y = _grid()
    h = np.full((G, G), 0.15, np.float32)
    r = np.random.default_rng(7)
    for _ in range(14):
        h += _bump(x, y, r.uniform(2, 30), r.uniform(5, 25), r.uniform(1.8, 3.2), h=r.uniform(0.35, 0.7))
    return h, (15, 27)


def _crater(rng):
    x, y = _grid()
    d = np.hypot(x - 16, y - 21)
    h = 0.12 + 0.75 * np.exp(-(((d - 7) / 2.0) ** 2))
    for ang in (-1.9, -0.6, 2.4):   # gaps in the rim
        h -= _bump(x, y, 16 + 7 * math.cos(ang), 21 + 7 * math.sin(ang), 2.2, h=0.6)
    h += _bump(x, y, 6, 6, 3, h=0.3) + _bump(x, y, 27, 5, 2.5, h=0.35)
    return h, (16, 21)


def _canyon_run(rng):
    x, y = _grid()
    h = np.full((G, G), 0.62, np.float32)
    h -= _ridge(x, y, [(5, -2), (8, 8), (20, 12), (14, 20), (17, 30)], 2.0, 0.5)
    h -= _ridge(x, y, [(28, -2), (24, 9), (20, 12)], 1.6, 0.42)
    h -= _bump(x, y, 16, 27, 5, 3, 0.45)       # open ground around the base
    h -= _bump(x, y, 16, 0, 12, 2.5, 0.4)      # open staging area in the north
    return h, (16, 27)


def _flatlands(rng):
    x, y = _grid()
    h = 0.12 + _bump(x, y, 8, 18, 2.5, h=0.45) + _bump(x, y, 24, 12, 2.2, h=0.4)
    h += _ridge(x, y, [(2, 8), (12, 6)], 1.4, 0.3) + _ridge(x, y, [(20, 22), (30, 20)], 1.4, 0.3)
    return h, (16, 27)


# ---- 360-degree theater: the base sits mid-map and drones enter along every edge
def _crossroads(rng):
    x, y = _grid()
    h = 0.15 + sum(_bump(x, y, cx, cy, 4.6, h=0.8) for cx, cy in [(6, 6), (26, 7), (7, 26), (25, 25)])
    h += _bump(x, y, 16, 3, 2.5, h=0.3) + _bump(x, y, 29, 16, 2.2, h=0.3)
    return h, (16, 16)


def _citadel(rng):
    x, y = _grid()
    d = np.hypot(x - 16, y - 16)
    h = 0.12 + 0.8 * np.exp(-(((d - 9) / 1.8) ** 2))
    for ang in (0.5, 2.0, 3.4, 5.0):   # four gaps, not evenly spaced
        h -= _bump(x, y, 16 + 9 * math.cos(ang), 16 + 9 * math.sin(ang), 2.0, h=0.65)
    h += _bump(x, y, 3, 28, 2.5, h=0.3) + _bump(x, y, 29, 4, 2.5, h=0.3)
    return h, (16, 16)


def _hill_country(rng):
    x, y = _grid()
    h = np.full((G, G), 0.15, np.float32)
    r = np.random.default_rng(11)
    placed = 0
    while placed < 16:
        cx, cy = r.uniform(2, 30), r.uniform(2, 30)
        if math.hypot(cx - 16, cy - 16) < 5:
            continue
        h += _bump(x, y, cx, cy, r.uniform(1.8, 3.0), h=r.uniform(0.35, 0.7))
        placed += 1
    return h, (16, 16)


def _escarpment(rng):
    x, y = _grid()
    h = 0.15 + 0.9 / (1 + np.exp((x - 7.5) * 1.6))          # high plateau along the west edge
    h += _bump(x, y, 22, 7, 2.6, h=0.4) + _bump(x, y, 24, 24, 2.8, h=0.45) + _bump(x, y, 14, 27, 2.0, h=0.3)
    return h, (17, 16)


def _steppe(rng):
    x, y = _grid()
    h = 0.12 + _ridge(x, y, [(4, 9), (12, 6)], 1.4, 0.3) + _ridge(x, y, [(20, 25), (28, 22)], 1.4, 0.3)
    h += _bump(x, y, 25, 9, 2.0, h=0.4) + _bump(x, y, 7, 22, 2.0, h=0.35)
    return h, (16, 16)


def _saddle(rng):
    x, y = _grid()
    h = 0.15 + _bump(x, y, 9, 9, 5.0, h=0.95) + _bump(x, y, 23, 23, 5.0, h=0.95)
    h += _bump(x, y, 27, 6, 2.2, h=0.3) + _bump(x, y, 5, 26, 2.2, h=0.3)
    return h, (16, 16)


MAPS = [
    ("Ridgeback", "An east-west ridge with two passes shields the base.", _ridgeback),
    ("Twin Valleys", "Two north-south valleys funnel attackers toward the base.", _twin_valleys),
    ("Broken Hills", "Scattered hills create a maze of radar shadows.", _broken_hills),
    ("Crater", "The base sits inside a crater rim broken by three gaps.", _crater),
    ("Canyon Run", "A winding canyon hides anything flying low inside it.", _canyon_run),
    ("Flatlands", "Open ground: radar sees far, but there is nowhere to hide sensors.", _flatlands),
]

MAPS_360 = [
    ("Crossroads", "Four massifs leave two crossing valleys, with the base at the junction.", _crossroads),
    ("Citadel", "A ring ridge with four gaps surrounds the base.", _citadel),
    ("Hill Country", "Hills on every side cast radar shadows in all directions.", _hill_country),
    ("Escarpment", "A cliff wall guards the west, so the threat comes from three sides.", _escarpment),
    ("Steppe", "Open ground in every direction: radar sees far, and nothing hides.", _steppe),
    ("Saddle", "The base sits on a saddle between two peaks.", _saddle),
]

# A theater = a map set plus where drones enter. "north": the original game, drones enter along the
# northern edge. "360": drones enter along every edge, the base is mid-map, and defenses must stay
# PLACE_MIN_ROW cells in from every edge (the same buffer the north mode keeps from the spawn line).
THEATERS = {
    "north": dict(maps=MAPS, cache="maps.npz", spawn="north", label="North axis"),
    "360": dict(maps=MAPS_360, cache="maps_360.npz", spawn="ring", label="360°"),
}


def artifacts(mode: str = "north") -> dict:
    """Where a theater's trained models and data live, relative to the repo root."""
    sfx = "" if mode == "north" else f"_{mode}"
    return dict(difficulties=f"data/difficulties{sfx}.json", recommended=f"data/recommended{sfx}.json",
                metrics=f"data/training_metrics{sfx}.jsonl", models="models" if mode == "north" else f"models/{mode}")


# ---------------------------------------------------------------------------
# Precompute
# ---------------------------------------------------------------------------

def cell_xy():
    idx = np.arange(N)
    return np.stack([idx % G, idx // G], 1)


def _line_of_sight(heights: np.ndarray, src_off: float, dst_off: float, max_range: float) -> np.ndarray:
    """vis[s, t] = True if a point src_off above cell s can see a point dst_off above cell t."""
    xy = cell_xy().astype(np.float32)
    hflat = heights.reshape(-1)
    vis = np.zeros((N, N), bool)
    S = 24
    k = (np.arange(1, S) / S).astype(np.float32)            # sample fractions along the ray
    for s in range(N):
        d = np.hypot(*(xy - xy[s]).T)
        tgt = np.nonzero(d <= max_range)[0]
        p = xy[s][None, None] + (xy[tgt] - xy[s])[:, None] * k[None, :, None]   # (T, S-1, 2)
        # bilinear terrain height along the ray
        px, py = p[..., 0], p[..., 1]
        x0, y0 = np.floor(px).astype(int), np.floor(py).astype(int)
        x1, y1 = np.minimum(x0 + 1, G - 1), np.minimum(y0 + 1, G - 1)
        fx, fy = px - x0, py - y0
        th = (heights[y0, x0] * (1 - fx) * (1 - fy) + heights[y0, x1] * fx * (1 - fy)
              + heights[y1, x0] * (1 - fx) * fy + heights[y1, x1] * fx * fy)
        hs = hflat[s] + src_off
        ht = hflat[tgt] + dst_off
        line = hs + (ht - hs)[:, None] * k[None, :]
        vis[s, tgt] = ~(th > line).any(1)
    return vis


def _dijkstra_all_pairs(rows, cols, w):
    """All-pairs shortest paths -> (dist, pred) with scipy's conventions (pred=-9999 for none)."""
    import heapq
    adj = [[] for _ in range(N)]
    for i, j, c in zip(rows, cols, w):
        adj[i].append((j, c))
    dist = np.full((N, N), np.inf)
    pred = np.full((N, N), -9999, np.int64)
    for s in range(N):
        d, p = dist[s], pred[s]
        d[s] = 0.0
        q = [(0.0, s)]
        while q:
            du, u = heapq.heappop(q)
            if du > d[u]:
                continue
            for v, c in adj[u]:
                nd = du + c
                if nd < d[v]:
                    d[v] = nd
                    p[v] = u
                    heapq.heappush(q, (nd, v))
    return dist, pred


def _flight_graph(passable: np.ndarray, rng):
    # Tiny per-cell cost jitter makes every shortest path unique, so A* and the
    # all-pairs Dijkstra tables always agree on the exact route.
    cost = 1.0 + 0.02 * rng.random(N)
    rows, cols, w = [], [], []
    for i in range(N):
        if not passable[i]:
            continue
        x, y = i % G, i // G
        for dx, dy in NEIGHBOURS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < G and 0 <= ny < G:
                j = ny * G + nx
                if passable[j]:
                    rows.append(i)
                    cols.append(j)
                    w.append(math.hypot(dx, dy) * 0.5 * (cost[i] + cost[j]))
    return (rows, cols, w), cost


def build_map(recipe, seed: int, spawn: str = "north"):
    rng = np.random.default_rng(seed)
    h, (bx, by) = recipe(rng)
    h = h + _noise(rng)
    h = (h - h.min()) / (h.max() - h.min())
    h = h.astype(np.float32)

    xy = cell_xy()
    hflat = h.reshape(-1)
    passable = hflat < C.IMPASSABLE_HEIGHT
    dbase = np.hypot(xy[:, 0] - bx, xy[:, 1] - by)
    base_mask = dbase <= C.BASE_RADIUS
    passable |= base_mask                       # base is always reachable
    h_flat_base = np.where(base_mask, np.minimum(hflat, 0.3), hflat)
    h = h_flat_base.reshape(G, G).astype(np.float32)

    (rows, cols, w), cell_cost = _flight_graph(passable, rng)
    if dijkstra is not None:
        dist, pred = dijkstra(csr_matrix((w, (rows, cols)), shape=(N, N)), directed=True, return_predecessors=True)
    else:
        dist, pred = _dijkstra_all_pairs(rows, cols, w)
    # graph is symmetric, so pred[t, c] (predecessor of c on the path from t)
    # is the next hop from c toward t.
    next_hop = np.where(pred < 0, np.arange(N)[None, :], pred)

    base_cell = int(by * G + bx)
    reach = np.isfinite(dist[base_cell])
    if spawn == "ring":
        edge = np.minimum.reduce([xy[:, 0], xy[:, 1], G - 1 - xy[:, 0], G - 1 - xy[:, 1]])
        spawn_mask = (edge < C.SPAWN_ROWS) & passable & reach
        place_mask = passable & reach & (edge >= C.PLACE_MIN_ROW) & ~base_mask
    else:
        spawn_mask = (xy[:, 1] < C.SPAWN_ROWS) & passable & reach
        place_mask = passable & reach & (xy[:, 1] >= C.PLACE_MIN_ROW) & ~base_mask
    assert spawn_mask.sum() >= 8, "map has too few spawn cells"

    vis = _line_of_sight(h, C.RADAR_MAST, C.DRONE_ALT, C.UNITS["radar"]["range"] + 0.5)

    dist = np.where(np.isfinite(dist), dist, 999.0).astype(np.float32)
    return dict(heights=h, passable=passable, base_cell=base_cell, base_mask=base_mask,
                spawn_mask=spawn_mask, place_mask=place_mask, vis=vis,
                dist=dist, next_hop=next_hop.astype(np.int16), cell_cost=cell_cost.astype(np.float32))


@dataclass
class MapSet:
    names: list
    blurbs: list
    heights: np.ndarray      # (M, G, G)
    passable: np.ndarray     # (M, N)
    base_cell: np.ndarray    # (M,)
    base_mask: np.ndarray    # (M, N)
    spawn_mask: np.ndarray   # (M, N)
    place_mask: np.ndarray   # (M, N)
    vis: np.ndarray          # (M, N, N) bool
    dist: np.ndarray         # (M, N, N) float32  dist[target, cell]
    next_hop: np.ndarray     # (M, N, N) int16    next_hop[target, cell]
    cell_cost: np.ndarray    # (M, N)
    mode: str = "north"      # theater key in THEATERS

    @property
    def n(self):
        return len(self.names)


def load_maps(rebuild: bool = False, mode: str = "north") -> MapSet:
    th = THEATERS[mode]
    maps, cache = th["maps"], DATA_DIR / th["cache"]
    if cache.exists() and not rebuild:
        z = np.load(cache, allow_pickle=False)
        if int(z["version"]) == CACHE_VERSION:
            return MapSet(names=[m[0] for m in maps], blurbs=[m[1] for m in maps], mode=mode,
                          **{k: z[k] for k in ("heights", "passable", "base_cell", "base_mask", "spawn_mask",
                                               "place_mask", "vis", "dist", "next_hop", "cell_cost")})
    built = [build_map(fn, seed=100 + i, spawn=th["spawn"]) for i, (_, _, fn) in enumerate(maps)]
    arrays = {k: np.stack([b[k] for b in built]) for k in built[0]}
    DATA_DIR.mkdir(exist_ok=True)
    np.savez_compressed(cache, version=CACHE_VERSION, **arrays)
    return load_maps(mode=mode)


if __name__ == "__main__":
    import sys
    import time
    mode = sys.argv[1] if len(sys.argv) > 1 else "north"
    t0 = time.time()
    ms = load_maps(rebuild=True, mode=mode)
    print(f"built {ms.n} maps in {time.time() - t0:.1f}s")
    for i, name in enumerate(ms.names):
        print(f"  {name:13s} passable={ms.passable[i].mean():.2f} spawn={ms.spawn_mask[i].sum():2d} "
              f"base->spawn dist={ms.dist[i, ms.base_cell[i]][ms.spawn_mask[i]].mean():.1f} "
              f"radar vis frac={ms.vis[i].mean():.3f}")
