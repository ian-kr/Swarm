"""A* search over the flight grid.

Used two ways:
  * `ScriptedAttacker`: the classic baseline enemy. Each drone plans a
    threat-aware route to the base (distance + penalties for flying through
    radar coverage, jammer zones and interceptor sensor range) and replans when
    jamming knocks it off course.
  * `astar(...)` with `trace=True`: returns the explored-node order so the game
    can animate the search for the pathfinding view.
"""

from __future__ import annotations

import heapq
import math

import numpy as np
import torch

from . import config as C

G, N = C.GRID, C.N_CELLS
NEIGHBOURS = [(dx, dy, math.hypot(dx, dy)) for dx, dy in C.MOVES if (dx, dy) != (0, 0)]

# threat penalties for the scripted attacker, per cell entered
W_RADAR, W_JAM, W_PAD = 4.0, 1.5, 4.0


def astar(passable, cell_cost, start: int, goal_mask, goal_xy, extra=None, trace=False):
    """Shortest path from `start` to any cell in `goal_mask`.

    Edge cost = step length * mean(cell_cost) + extra[next_cell]. Using the same
    cell_cost as the Dijkstra tables means that with extra=None, A* returns the
    exact route the next-hop tables encode. The heuristic is straight-line
    distance to the goal center minus the goal radius, which never overestimates.
    """
    gx, gy = goal_xy
    def h(c):
        return max(0.0, math.hypot(c % G - gx, c // G - gy) - C.BASE_RADIUS)

    g = {start: 0.0}
    parent = {start: -1}
    openq = [(h(start), 0.0, start)]
    closed = set()
    order = []
    while openq:
        f, gc, c = heapq.heappop(openq)
        if c in closed:
            continue
        closed.add(c)
        if trace:
            order.append(c)
        if goal_mask[c]:
            path = [c]
            while parent[path[-1]] != -1:
                path.append(parent[path[-1]])
            path.reverse()
            return (path, order) if trace else path
        x, y = c % G, c // G
        for dx, dy, L in NEIGHBOURS:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < G and 0 <= ny < G):
                continue
            n = ny * G + nx
            if not passable[n] or n in closed:
                continue
            ng = gc + L * 0.5 * (cell_cost[c] + cell_cost[n]) + (extra[n] if extra is not None else 0.0)
            if ng < g.get(n, math.inf):
                g[n] = ng
                parent[n] = c
                heapq.heappush(openq, (ng + h(n), ng, n))
    return ([start], order) if trace else [start]


def threat_cost(coverage, jam_zone, pad_zone):
    return W_RADAR * coverage + W_JAM * jam_zone + W_PAD * pad_zone


class ScriptedAttacker:
    """Drones that each fly an A* route minimizing distance + exposure."""

    def __init__(self, maps_np):
        self.ms = maps_np
        self.paths = {}

    def reset(self):
        self.paths = {}

    def act(self, env) -> torch.Tensor:
        B, D = env.drone_cell.shape
        cells = env.drone_cell.cpu().numpy()
        status = env.drone_status.cpu().numpy()
        mids = env.map_id.cpu().numpy()
        cov = env.coverage.cpu().numpy()
        jam = env.jam_zone.cpu().numpy()
        pad = env.pad_zone.cpu().numpy()
        acts = np.full((B, D), C.HOVER, np.int64)
        for b in range(B):
            mi = mids[b]
            extra = None
            for d in range(D):
                if status[b, d] != 0:
                    continue
                c = int(cells[b, d])
                path = self.paths.get((b, d))
                if path is None or c not in path:          # first plan, or knocked off course
                    if extra is None:
                        extra = threat_cost(cov[b], jam[b], pad[b])
                    bc = int(self.ms.base_cell[mi])
                    path = astar(self.ms.passable[mi], self.ms.cell_cost[mi], c, self.ms.base_mask[mi],
                                 (bc % G, bc // G), extra)
                    self.paths[(b, d)] = path
                i = path.index(c)
                if i + 1 < len(path):
                    n = path[i + 1]
                    dx, dy = n % G - c % G, n // G - c // G
                    acts[b, d] = C.MOVES.index((dx, dy))
        return torch.as_tensor(acts, device=env.dev)
