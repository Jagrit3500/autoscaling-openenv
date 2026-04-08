from __future__ import annotations

from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

from environment import (
    AutoScalingEnvironment,
    ACTION_SCALE_UP,
    ACTION_SCALE_DOWN,
    ACTION_HOLD,
)
from tasks import ALL_TASKS

app = FastAPI(
    title="Auto-Scaling Infrastructure Agent — OpenEnv",
    description="OpenEnv environment for AI-driven server auto-scaling",
    version="1.0.0",
)

env = AutoScalingEnvironment()


class ResetRequest(BaseModel):
    task_id: int = 1


class StepRequest(BaseModel):
    action: int


class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset")
def reset(req: ResetRequest | None = None):
    task_id = req.task_id if req else 1
    if task_id not in (1, 2, 3):
        raise HTTPException(status_code=400, detail="task_id must be 1, 2, or 3")
    obs = env.reset(task_id=task_id)
    return {"observation": obs, "task_id": task_id}


@app.post("/step")
def step(req: StepRequest):
    if req.action not in (ACTION_SCALE_UP, ACTION_SCALE_DOWN, ACTION_HOLD):
        raise HTTPException(status_code=400, detail="Invalid action")
    if env.task is None:
        raise HTTPException(status_code=400, detail="Call /reset before /step")

    obs, reward, done, info = env.step(req.action)
    return StepResponse(observation=obs, reward=reward, done=done, info=info)


@app.get("/state")
def state():
    if env.task is None:
        raise HTTPException(status_code=400, detail="Call /reset first")
    return {"observation": env.state()}


@app.get("/tasks")
def tasks():
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
    return """
    <html>
    <body style="font-family: monospace; padding: 2rem;">
      <h1>Auto-Scaling Infrastructure Agent</h1>
      <p>OpenEnv environment for AI-driven server auto-scaling.</p>
    </body>
    </html>
    """


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()