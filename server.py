"""
server.py - FastAPI server for HuggingFace Space deployment

Wraps AutoScalingEnvironment as HTTP endpoints so the OpenEnv
automated validator can ping /reset and judges can test via browser.

Endpoints:
    POST /reset          → start fresh episode, returns initial obs
    POST /step           → apply action, returns (obs, reward, done, info)
    GET  /state          → current observation snapshot
    GET  /health         → liveness check (returns 200)
    GET  /tasks          → list all task configs
    GET  /               → basic info page

Run locally:
    pip install fastapi uvicorn
    uvicorn server:app --host 0.0.0.0 --port 7860

HF Space:
    The Dockerfile exposes port 7860 and runs this server.
    Automated validator hits POST /reset — must return 200.
"""

from __future__ import annotations

from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from environment import (
    AutoScalingEnvironment,
    ACTION_SCALE_UP,
    ACTION_SCALE_DOWN,
    ACTION_HOLD,
)
from tasks import ALL_TASKS

# ─────────────────────────────────────────────
# App
# ─────────────────────────────────────────────

app = FastAPI(
    title="Auto-Scaling Infrastructure Agent — OpenEnv",
    description="OpenEnv environment for AI-driven server auto-scaling",
    version="1.0.0",
)

# One environment instance per server process (stateful)
env = AutoScalingEnvironment()


# ─────────────────────────────────────────────
# Request / Response models
# ─────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: int = 1


class StepRequest(BaseModel):
    action: int  # 0=scale_up, 1=scale_down, 2=hold


class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.get("/health")
def health():
    """Liveness check — automated validator uses this."""
    return {"status": "ok"}


@app.post("/reset")
def reset(req: ResetRequest | None = None):
    """
    Start a fresh episode.
    Automated validator pings this — must return 200.
    """
    task_id = req.task_id if req else 1
    if task_id not in (1, 2, 3):
        raise HTTPException(status_code=400, detail="task_id must be 1, 2, or 3")
    obs = env.reset(task_id=task_id)
    return {"observation": obs, "task_id": task_id}


@app.post("/step")
def step(req: StepRequest):
    """Apply one action and advance the simulation."""
    if req.action not in (ACTION_SCALE_UP, ACTION_SCALE_DOWN, ACTION_HOLD):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid action {req.action}. Must be 0 (scale_up), 1 (scale_down), or 2 (hold)."
        )
    if env.task is None:
        raise HTTPException(status_code=400, detail="Call /reset before /step")

    obs, reward, done, info = env.step(req.action)
    return StepResponse(observation=obs, reward=reward, done=done, info=info)


@app.get("/state")
def state():
    """Return current observation without advancing the step."""
    if env.task is None:
        raise HTTPException(status_code=400, detail="Call /reset first")
    return {"observation": env.state()}


@app.get("/tasks")
def tasks():
    """List all task configurations."""
    return {
        "tasks": [
            {
                "task_id": t.task_id,
                "name": t.name,
                "difficulty": t.difficulty,
                "max_steps": t.max_steps,
                "max_instances": t.max_instances,
                "budget": t.budget,
                "description": t.description,
            }
            for t in ALL_TASKS
        ]
    }


@app.get("/", response_class=HTMLResponse)
def index():
    """Basic info page for HF Space."""
    return """
    <html>
    <head><title>Auto-Scaling OpenEnv</title></head>
    <body style="font-family:monospace;background:#111;color:#eee;padding:2rem">
    <h1>Auto-Scaling Infrastructure Agent</h1>
    <p>OpenEnv environment for AI-driven server auto-scaling.</p>
    <h2>Endpoints</h2>
    <ul>
      <li><b>POST /reset</b>  — Start episode: <code>{"task_id": 1}</code></li>
      <li><b>POST /step</b>   — Take action: <code>{"action": 0}</code> (0=scale_up, 1=scale_down, 2=hold)</li>
      <li><b>GET  /state</b>  — Current observation</li>
      <li><b>GET  /tasks</b>  — List all tasks</li>
      <li><b>GET  /health</b> — Liveness check</li>
      <li><b>GET  /docs</b>   — Swagger UI</li>
    </ul>
    <h2>Tasks</h2>
    <ul>
      <li>Task 1 (Easy)   — Single Spike Recovery (30 steps)</li>
      <li>Task 2 (Medium) — Traffic Wave Management (50 steps)</li>
      <li>Task 3 (Hard)   — Adaptive Scaling Under Uncertainty (60 steps)</li>
    </ul>
    </body>
    </html>
    """