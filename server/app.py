from __future__ import annotations

from typing import Any, Dict

import json
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

try:
    from environment import (
        AutoScalingEnvironment,
        ACTION_SCALE_UP,
        ACTION_SCALE_DOWN,
        ACTION_HOLD,
    )
    from tasks import ALL_TASKS
except ModuleNotFoundError:  # pragma: no cover
    from ..environment import (
        AutoScalingEnvironment,
        ACTION_SCALE_UP,
        ACTION_SCALE_DOWN,
        ACTION_HOLD,
    )
    from ..tasks import ALL_TASKS


app = FastAPI(
    title="Auto-Scaling Infrastructure Agent - OpenEnv",
    description="OpenEnv environment for AI-driven server auto-scaling",
    version="1.0.1",
)

env = AutoScalingEnvironment()


# -----------------------------
# Request / Response Models
# -----------------------------
class ResetRequest(BaseModel):
    task_id: int = 1


class StepRequest(BaseModel):
    action: int


class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]


def _safe_json_body(request: Request) -> Dict[str, Any]:
    try:
        raw = request.scope.get("_body")
        if raw is None:
            raw = b""
        if isinstance(raw, bytes):
            text = raw.decode("utf-8", errors="ignore").strip()
        else:
            text = str(raw).strip()
        if not text:
            return {}
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


# -----------------------------
# UI Routes (IMPORTANT FOR HF)
# -----------------------------
@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <html>
    <body style="font-family: Arial; padding: 40px;">
        <h1>Auto-Scaling Infrastructure Agent</h1>
        <p>API is running.</p>
        <p><a href="/docs">Open API Docs</a></p>
        <p><a href="/tasks">View Tasks</a></p>
    </body>
    </html>
    """


@app.get("/web", response_class=HTMLResponse)
def web():
    return """
    <html>
    <body style="font-family: Arial; padding: 40px;">
        <h1>Auto-Scaling Infrastructure Agent</h1>
        <p>OpenEnv environment is live.</p>

        <h3>Available Endpoints:</h3>
        <ul>
            <li><a href="/health">/health</a></li>
            <li><a href="/tasks">/tasks</a></li>
            <li><a href="/docs">/docs</a></li>
        </ul>

        <p>Use /docs to test API.</p>
    </body>
    </html>
    """


# -----------------------------
# API Routes
# -----------------------------
@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/metadata")
def metadata():
    return {
        "name": "auto-scaling-infrastructure-agent",
        "description": "OpenEnv environment for AI-driven server auto-scaling",
        "version": "1.0.1",
    }


@app.get("/schema")
def schema():
    return {
        "action": {
            "type": "object",
            "properties": {
                "action": {"type": "integer", "enum": [0, 1, 2]},
            },
            "required": ["action"],
        },
        "observation": {
            "type": "object",
            "additionalProperties": True,
        },
        "state": {
            "type": "object",
            "additionalProperties": True,
        },
    }


@app.post("/mcp")
async def mcp(request: Request):
    """Minimal MCP JSON-RPC endpoint.

    OpenEnv's runtime validator expects POST /mcp to return a JSON-RPC payload
    with {"jsonrpc": "2.0"}. This environment does not expose MCP tools, so we
    return an empty tool list and safe errors for tool calls.
    """

    body = _safe_json_body(request)
    req_id = body.get("id")
    method = body.get("method")

    # Validator may send an empty JSON object: {}
    if not method:
        return {"jsonrpc": "2.0", "id": req_id, "result": {}}

    if method == "openenv/session/create":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"session_id": uuid.uuid4().hex},
        }

    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": []}}

    if method == "tools/call":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": "No tools available"},
        }

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"Unknown method: {method}"},
    }


@app.post("/reset")
async def reset(request: Request):
    body = _safe_json_body(request)
    task_id = body.get("task_id", 1)

    try:
        task_id = int(task_id)
    except (TypeError, ValueError):
        task_id = 1

    if task_id not in (1, 2, 3):
        raise HTTPException(status_code=400, detail="task_id must be 1, 2, or 3")

    obs = env.reset(task_id=task_id)
    return {"observation": obs, "task_id": task_id}


@app.post("/step")
async def step(request: Request):
    body = _safe_json_body(request)
    action = body.get("action", ACTION_HOLD)
    try:
        action = int(action)
    except (TypeError, ValueError):
        action = ACTION_HOLD

    if action not in (ACTION_SCALE_UP, ACTION_SCALE_DOWN, ACTION_HOLD):
        raise HTTPException(status_code=400, detail="Invalid action")

    if env.task is None:
        raise HTTPException(status_code=400, detail="Call /reset before /step")

    obs, reward, done, info = env.step(action)

    return StepResponse(
        observation=obs,
        reward=reward,
        done=done,
        info=info,
    )


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


# -----------------------------
# Run Server
# -----------------------------
def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()