import json
from urllib import request, error


class AutoScalingEnvClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def _get(self, path: str):
        req = request.Request(f"{self.base_url}{path}", method="GET")
        with request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _post(self, path: str, payload: dict):
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def health(self):
        return self._get("/health")

    def tasks(self):
        return self._get("/tasks")

    def reset(self, task_id: int = 1):
        return self._post("/reset", {"task_id": task_id})

    def step(self, action: int):
        return self._post("/step", {"action": action})

    def state(self):
        return self._get("/state")