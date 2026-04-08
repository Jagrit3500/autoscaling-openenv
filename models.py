from pydantic import BaseModel
from typing import Any, Dict, List


class ResetRequest(BaseModel):
    task_id: int = 1


class StepRequest(BaseModel):
    action: int


class ObservationResponse(BaseModel):
    observation: Dict[str, Any]


class ResetResponse(BaseModel):
    observation: Dict[str, Any]
    task_id: int


class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]


class TasksResponse(BaseModel):
    tasks: List[Dict[str, Any]]


class HealthResponse(BaseModel):
    status: str