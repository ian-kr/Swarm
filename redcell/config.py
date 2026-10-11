"""Shared constants for the RED CELL simulation.

Everything that defines the rules of the game lives here so the GPU training
environment and the playable game server can never disagree.
"""

GRID = 32                 # map is GRID x GRID cells
N_CELLS = GRID * GRID

# --- attackers -------------------------------------------------------------
N_DRONES = 6
MAX_STEPS = 90
DRONE_ALT = 0.03          # flight height above terrain (nap-of-the-earth)

# --- defenses --------------------------------------------------------------
BUDGET = 15
UNITS = {
    #            cost  max  range
    "radar":    dict(cost=3, max=4, range=9.0),
    "jammer":   dict(cost=2, max=3, range=4.0),
    "pad":      dict(cost=3, max=3, range=3.0),   # range = short-range sensor
}
RADAR_MAST = 0.06         # radar antenna height above terrain
INTERCEPTORS_PER_PAD = 3
INTERCEPTOR_SPEED = 2     # cells per step (drones move 1)
INTERCEPTOR_FUEL = 30     # steps of flight before it is spent
KILL_PROB = 0.8
INTERCEPT_TEWA = 1.0      # weight of a drone's closeness to the base when picking targets (0 = nearest drone first)
INTERCEPT_LEAD_MAX = 2    # lead pursuit: aim up to this many of the target's moves ahead (0 = pure pursuit)
INTERCEPT_TERMINAL = 3    # within this many cells, home straight onto the drone (terminal guidance)
JAM_DRIFT_PROB = 0.5      # chance a jammed drone's move is replaced by a random one

# --- terrain ---------------------------------------------------------------
IMPASSABLE_HEIGHT = 0.82  # peaks above this are above the flight ceiling
BASE_RADIUS = 1.6
SPAWN_ROWS = 2            # attackers enter from the northern edge
PLACE_MIN_ROW = 6         # defenses can't be placed right on the spawn line

# --- rewards (attacker's perspective) -------------------------------------
R_PROGRESS = 0.05         # per cell of shortest-path distance closed to base
R_BREACH = 10.0
R_KILLED = -1.0
R_DETECTED = 0.0          # no per-step penalty: an earlier version learned to hide and time out instead of attacking
R_TIMEOUT = -3.0          # still airborne when time runs out = mission failed. Worse than being shot down
                          # (-1): with equal penalties the swarm learned to loiter outside radar range
                          # against strong defenses instead of committing to an attack.
R_STEP = -0.01

# --- observation -----------------------------------------------------------
LOCAL = 11                # local egocentric patch size
GLOBAL = 16               # downsampled global map size

# 8 compass moves + hover, plus FOLLOW below. (dx, dy) with y growing southward.
MOVES = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, 0)]
HOVER = MOVES.index((0, 0))
FOLLOW = len(MOVES)        # extra action: take the next step along the threat-aware pathfinding field
N_ACTIONS = len(MOVES) + 1
