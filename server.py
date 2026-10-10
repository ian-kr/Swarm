"""RED CELL game server: serves the web client and runs simulations.

    python server.py            # http://localhost:8000
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from redcell import config as C
from redcell.game import Game, LayoutError

ROOT = Path(__file__).resolve().parent
app = FastAPI(title="RED CELL")
game = Game(device=os.environ.get("REDCELL_DEVICE", "cpu"))


class LayoutIn(BaseModel):
    radar: list[int] = Field(default_factory=list)
    jammer: list[int] = Field(default_factory=list)
    pad: list[int] = Field(default_factory=list)


class CoverageReq(BaseModel):
    map: int
    layout: LayoutIn


class SimReq(BaseModel):
    map: int
    layout: LayoutIn
    difficulty: str = "scripted"
    seed: int | None = None


@app.get("/api/config")
def get_config():
    return dict(grid=C.GRID, budget=C.BUDGET, units=C.UNITS, drones=C.N_DRONES, max_steps=C.MAX_STEPS,
                interceptors_per_pad=C.INTERCEPTORS_PER_PAD, kill_prob=C.KILL_PROB,
                difficulties=game.difficulties(),
                maps=[dict(id=i, name=n, blurb=b) for i, (n, b) in enumerate(zip(game.ms.names, game.ms.blurbs))])


@app.get("/api/map/{map_id}")
def get_map(map_id: int):
    if not 0 <= map_id < game.ms.n:
        raise HTTPException(404, "unknown map")
    return game.map_payload(map_id)


@app.post("/api/coverage")
def post_coverage(req: CoverageReq):
    try:
        return game.coverage(req.map, req.layout.model_dump())
    except LayoutError as e:
        raise HTTPException(400, str(e))


@app.post("/api/simulate")
def post_simulate(req: SimReq):
    seed = req.seed if req.seed is not None else random.randrange(1 << 30)
    try:
        return game.simulate(req.map, req.layout.model_dump(), req.difficulty, seed)
    except LayoutError as e:
        raise HTTPException(400, str(e))


@app.get("/api/training")
def get_training():
    """Training curve + which checkpoints back each difficulty, for the 'Under the hood' panel."""
    spec_path = ROOT / "data" / "difficulties.json"
    spec = json.loads(spec_path.read_text()) if spec_path.exists() else {}
    metrics = []
    mpath = ROOT / spec.get("metrics", "checkpoints/metrics.jsonl")
    if mpath.exists():
        for line in mpath.read_text().splitlines():
            r = json.loads(line)
            metrics.append({k: r.get(k) for k in ("update", "steps", "time", "train_breach", "eval_breach", "ent", "sps")})
    return dict(metrics=metrics, levels=spec.get("levels", {}), device=spec.get("device"),
                baseline=spec.get("scripted_eval_breach"), baseline_random=spec.get("scripted_random_breach"),
                curve=spec.get("curve"), strong_layouts=spec.get("strong_layouts"), heldout=spec.get("heldout"))


app.mount("/static", StaticFiles(directory=ROOT / "web"), name="static")


@app.get("/")
def index():
    return FileResponse(ROOT / "web" / "index.html")


if __name__ == "__main__":
    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 8000)))
