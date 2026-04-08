"""
environment.py - Core simulation engine for Auto-Scaling Infrastructure Agent

OpenEnv interface:
    reset(task_id)  -> start a fresh episode
    step(action)    -> agent acts, world updates, reward returned
    state()         -> current observation snapshot

Simulation model:
    - A cluster of servers handles incoming HTTP requests
    - Each server handles task.instance_capacity_rps at healthy load
    - CPU = (rps / total_capacity) * 100  +  queue_pressure
      CPU is NOT capped at 100 - it reflects true overload signal
    - Queue grows when rps > capacity, drains gradually via drain_factor
    - Scale up adds a server after boot_delay_steps (realism)
    - Scale down removes a server immediately
    - Booting servers are billed (cloud charges from provisioning)
    - Memory tracks CPU smoothly (added to state for richer observation)

Actions:
    0 -> scale_up    add 1 server (boots after delay)
    1 -> scale_down  remove 1 server immediately
    2 -> hold        do nothing
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

from tasks import Task, get_task, empty_episode_info


# 
# Constants
# 

# Per-server capacity is now defined per task via instance_capacity_rps
# to model different cloud instance types (t3.micro / t3.medium / c5.large).
# This fallback constant is no longer used in simulation logic.
CAPACITY_PER_INSTANCE: float = 120.0   # kept for backward compatibility only
CPU_HARD_CAP: float = 200.0
QUEUE_DRAIN_FACTOR: float = 0.35
MEMORY_BASE: float = 35.0
MEMORY_PER_CPU_RATIO: float = 0.5

ACTION_SCALE_UP: int = 0
ACTION_SCALE_DOWN: int = 1
ACTION_HOLD: int = 2

ACTION_NAMES: Dict[int, str] = {
    ACTION_SCALE_UP: "scale_up",
    ACTION_SCALE_DOWN: "scale_down",
    ACTION_HOLD: "hold",
}

VALID_ACTIONS: List[int] = list(ACTION_NAMES.keys())


# 
# Environment
# 

class AutoScalingEnvironment:
    """
    Simulates a server cluster under variable HTTP traffic.

    Usage:
        env = AutoScalingEnvironment()
        obs = env.reset(task_id=1)
        done = False
        while not done:
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
    """

    def __init__(self) -> None:
        self.task: Optional[Task] = None
        self.task_id: Optional[int] = None

        self.current_step: int = 0
        self.current_instances: int = 0
        self.pending_scale_ups: List[int] = []
        self.queue_length: float = 0.0
        self.cost_so_far: float = 0.0
        self.cpu_usage: float = 0.0
        self.memory_usage: float = 0.0
        self.requests_per_second: float = 0.0
        self.done: bool = False

        self.last_action: int = ACTION_HOLD
        self.last_action_name: str = ACTION_NAMES[ACTION_HOLD]

        self.consecutive_critical_steps: int = 0
        self.sla_violation_count: int = 0
        self.critical_violation_count: int = 0
        self.unnecessary_scaleups: int = 0
        self.unnecessary_scaledowns: int = 0
        self.recent_scale_down_step: Optional[int] = None
        self.termination_reason: str = "running"

        self._info: dict = empty_episode_info()

    # 
    # reset()
    # 

    def reset(self, task_id: int = 1) -> Dict[str, Any]:
        """
        Start a fresh episode for the given task.

        Args:
            task_id: 1 (easy), 2 (medium), or 3 (hard)

        Returns:
            Initial state observation dictionary.
        """
        self.task = get_task(task_id)
        self.task_id = task_id

        self.current_step = 0
        self.current_instances = self.task.initial_instances
        self.pending_scale_ups = []
        self.queue_length = 0.0
        self.cost_so_far = 0.0
        self.done = False

        self.last_action = ACTION_HOLD
        self.last_action_name = ACTION_NAMES[ACTION_HOLD]

        self.consecutive_critical_steps = 0
        self.sla_violation_count = 0
        self.critical_violation_count = 0
        self.unnecessary_scaleups = 0
        self.unnecessary_scaledowns = 0
        self.recent_scale_down_step = None
        self.termination_reason = "running"
        self._info = empty_episode_info()

        self.requests_per_second = float(self.task.traffic_pattern[0])
        self._recompute_metrics()

        return self.state()

    # 
    # step()
    # 

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Apply agent action, advance simulation one step.

        Args:
            action: 0=scale_up, 1=scale_down, 2=hold

        Returns:
            (observation, reward, done, info)
        """
        if self.task is None:
            raise RuntimeError("Call reset() before step().")

        if self.done:
            return self.state(), 0.0, True, self._build_info()

        if action not in ACTION_NAMES:
            raise ValueError(
                f"Invalid action {action}. "
                f"Valid: {VALID_ACTIONS} -> {ACTION_NAMES}"
            )

        self.last_action = action
        self.last_action_name = ACTION_NAMES[action]

        prev_cpu = self.cpu_usage
        prev_queue = self.queue_length

        reward = 0.0

        # 1. Apply action
        reward += self._apply_action(action)

        # 2. Activate booting servers that are ready
        self._activate_pending_instances()

        # 3. Update traffic for this step
        self.requests_per_second = float(
            self.task.traffic_pattern[self.current_step]
        )

        # 4. Recompute CPU, queue, memory
        self._recompute_metrics()

        # 5. Charge cost - active + booting servers (cloud bills from provisioning)
        billable = self.current_instances + len(self.pending_scale_ups)
        self.cost_so_far += billable * self.task.cost_per_instance_per_step

        # 6. Compute step reward
        reward += self._compute_step_reward(prev_cpu, prev_queue, action)

        # 7. Update violation counters
        self._update_violations()

        # 8. Check termination
        self._check_termination()

        # 9. Advance step
        if not self.done:
            self.current_step += 1
            if self.current_step >= self.task.max_steps:
                self.done = True
                self.termination_reason = "success"

        return self.state(), round(reward, 4), self.done, self._build_info()

    # 
    # state()
    # 

    def state(self) -> Dict[str, Any]:
        """
        Return the current environment observation.

        Includes real-time metrics AND task constraints so any
        LLM-based agent can reason about limits without extra context.
        """
        if self.task is None:
            raise RuntimeError("Call reset() before state().")

        step = min(self.current_step, self.task.max_steps - 1)
        current_rps = float(self.task.traffic_pattern[step])

        return {
            # Identity
            "task_id": self.task.task_id,
            "task_name": self.task.name,
            "difficulty": self.task.difficulty,

            # Core metrics
            "cpu_usage": round(self.cpu_usage, 2),
            "memory_usage": round(self.memory_usage, 2),
            "queue_length": round(self.queue_length, 2),
            "requests_per_second": round(current_rps, 2),

            # Infrastructure
            "current_instances": self.current_instances,
            "pending_instances": len(self.pending_scale_ups),
            "max_instances": self.task.max_instances,

            # Cost
            "cost_so_far": round(self.cost_so_far, 4),
            "budget": self.task.budget,
            "budget_remaining": round(self.task.budget - self.cost_so_far, 4),

            # Progress
            "time_step": self.current_step,
            "max_steps": self.task.max_steps,
            "steps_remaining": self.task.max_steps - self.current_step,

            # Task constraints
            "sla_cpu_limit": self.task.sla_cpu_limit,
            "sla_queue_limit": self.task.sla_queue_limit,
            "critical_cpu_threshold": self.task.critical_cpu_threshold,
            "critical_queue_threshold": self.task.critical_queue_threshold,

            # Danger signal
            "consecutive_critical_steps": self.consecutive_critical_steps,

            # Last action
            "last_action": self.last_action_name,
        }

    # 
    # Internal: metrics
    # 

    def _recompute_metrics(self) -> None:
        """
        Recompute CPU, queue, and memory from current traffic and instances.

        CPU model:
            cpu = (rps / capacity) * 100 + queue_pressure
            NOT capped at 100 - true overload magnitude is visible to agent.
            queue_pressure adds up to 20% when queue is backed up.
            Hard cap at CPU_HARD_CAP=200 prevents float explosion only.

        Queue model:
            overflow = max(0, rps - capacity)          queue grows
            drain    = spare_capacity * drain_factor   gradual realistic drain
            queue    = max(0, queue + overflow - drain)

        Memory model:
            memory = base + (cpu * ratio), clamped 0-100
        """
        total_capacity = max(1.0, self.current_instances * self.task.instance_capacity_rps)

        # CPU
        raw_cpu = (self.requests_per_second / total_capacity) * 100.0
        queue_pressure = min(self.queue_length / 20.0, 20.0)
        self.cpu_usage = max(0.0, min(raw_cpu + queue_pressure, CPU_HARD_CAP))

        # Queue
        overflow = max(0.0, self.requests_per_second - total_capacity)
        spare = max(0.0, total_capacity - self.requests_per_second)
        drain = spare * QUEUE_DRAIN_FACTOR
        self.queue_length = max(0.0, self.queue_length + overflow - drain)

        # Memory
        raw_memory = MEMORY_BASE + (self.cpu_usage * MEMORY_PER_CPU_RATIO)
        self.memory_usage = max(0.0, min(raw_memory, 100.0))

    # 
    # Internal: action
    # 

    def _apply_action(self, action: int) -> float:
        """Apply action to infrastructure. Returns immediate reward component."""
        reward = 0.0

        if action == ACTION_HOLD:
            return reward

        if action == ACTION_SCALE_UP:
            total_future = self.current_instances + len(self.pending_scale_ups)
            if total_future >= self.task.max_instances:
                reward -= 0.3
            else:
                activation_step = self.current_step + self.task.boot_delay_steps
                self.pending_scale_ups.append(activation_step)

                if (self.cpu_usage < self.task.sla_cpu_limit - 10 and
                        self.queue_length < self.task.sla_queue_limit * 0.5):
                    self.unnecessary_scaleups += 1
                    reward -= 0.2
                else:
                    reward += 0.3

        elif action == ACTION_SCALE_DOWN:
            if self.current_instances <= 1:
                reward -= 0.3
            else:
                self.current_instances -= 1
                self.recent_scale_down_step = self.current_step

                if (self.cpu_usage < 55.0 and
                        self.queue_length < self.task.sla_queue_limit * 0.25):
                    reward += 0.2

        return reward

    def _activate_pending_instances(self) -> None:
        """Move booting servers that have finished their delay to active pool."""
        ready = [s for s in self.pending_scale_ups if s <= self.current_step]
        if ready:
            self.current_instances = min(
                self.current_instances + len(ready),
                self.task.max_instances
            )
        self.pending_scale_ups = [
            s for s in self.pending_scale_ups if s > self.current_step
        ]

    # 
    # Internal: reward
    # 

    def _compute_step_reward(
        self,
        prev_cpu: float,
        prev_queue: float,
        action: int,
    ) -> float:
        """
        Compute reward for this step based on system health.

        Components:
            CPU health        +0.3 healthy | -0.1 SLA warn | -0.4 critical
            Queue health      +0.1 ok | -0.1 warn | -0.2 critical
            Cost efficiency   +0.1 within budget | proportional penalty over
            Improvement bonus +0.05 if CPU or queue improved
            Scale-down guard  -0.5 if scale down caused overload within 2 steps
        """
        reward = 0.0
        task = self.task

        # CPU health
        if self.cpu_usage <= task.sla_cpu_limit:
            reward += 0.3
        elif self.cpu_usage <= task.critical_cpu_threshold:
            reward -= 0.1
        else:
            reward -= 0.4

        # Queue health
        if self.queue_length <= task.sla_queue_limit:
            reward += 0.1
        elif self.queue_length <= task.critical_queue_threshold:
            reward -= 0.1
        else:
            reward -= 0.2

        # Cost efficiency
        if self.cost_so_far <= task.budget:
            reward += 0.1
        else:
            overspend = (self.cost_so_far - task.budget) / task.budget
            reward -= 0.3 * overspend

        # Improvement momentum
        if self.queue_length < prev_queue:
            reward += 0.05
        if prev_cpu > task.sla_cpu_limit and self.cpu_usage < prev_cpu:
            reward += 0.05

        # Scale-down caused overload penalty
        if self.recent_scale_down_step is not None:
            steps_since = self.current_step - self.recent_scale_down_step
            if 0 < steps_since <= 2:
                overloaded = (
                    self.cpu_usage > task.critical_cpu_threshold or
                    self.queue_length > task.critical_queue_threshold
                )
                if overloaded:
                    self.unnecessary_scaledowns += 1
                    reward -= 0.5
                    self.recent_scale_down_step = None

        return reward

    # 
    # Internal: violations and termination
    # 

    def _update_violations(self) -> None:
        """Track SLA and critical violations for grading and termination."""
        if (self.cpu_usage > self.task.sla_cpu_limit or
                self.queue_length > self.task.sla_queue_limit):
            self.sla_violation_count += 1

        critical = (
            self.cpu_usage > self.task.critical_cpu_threshold or
            self.queue_length > self.task.critical_queue_threshold
        )
        if critical:
            self.critical_violation_count += 1
            self.consecutive_critical_steps += 1
        else:
            self.consecutive_critical_steps = 0

    def _check_termination(self) -> None:
        """
        Check failure conditions (priority order):
            1. Budget exceeded beyond 10% buffer  -> budget_exceeded
            2. Sustained critical state           -> critical_overload
        Success handled in step() after advancing current_step.
        """
        if self.cost_so_far > self.task.budget * self.task.budget_failure_multiplier:
            self.done = True
            self.termination_reason = "budget_exceeded"
            return

        if self.consecutive_critical_steps >= self.task.max_consecutive_critical_steps:
            self.done = True
            self.termination_reason = "critical_overload"

    # 
    # Internal: info
    # 

    def _build_info(self) -> Dict[str, Any]:
        """
        Build metadata dict returned on every step().
        Fully populated when done=True.
        Used by graders.py for multi-dimensional scoring.
        """
        steps = max(1, self.current_step if self.done else self.current_step + 1)
        safe_steps = max(0, steps - self.sla_violation_count)
        uptime = (safe_steps / steps) * 100.0

        return {
            "total_cost": round(self.cost_so_far, 4),
            "sla_violation_count": int(self.sla_violation_count),
            "critical_violation_count": int(self.critical_violation_count),
            "uptime_percentage": round(uptime, 2),
            "unnecessary_scaleups": int(self.unnecessary_scaleups),
            "unnecessary_scaledowns": int(self.unnecessary_scaledowns),
            "final_instances": int(self.current_instances),
            "steps_completed": int(steps),
            "termination_reason": self.termination_reason,
        }

    # 
    # Utility
    # 

    def get_task_metadata(self) -> Dict[str, Any]:
        """Return full task config as a plain dict."""
        return asdict(self.task)

    def render(self) -> None:
        """Print a human-readable step summary with ASCII CPU bar."""
        if self.task is None:
            print("No task loaded. Call reset() first.")
            return
        s = self.state()
        bar_len = 20
        filled = int(min(s["cpu_usage"], 100.0) / 100.0 * bar_len)
        bar = "" * filled + "" * (bar_len - filled)
        print(
            f"Step {s['time_step']:>3}/{self.task.max_steps} | "
            f"CPU [{bar}] {s['cpu_usage']:>6.1f}% | "
            f"Mem {s['memory_usage']:>5.1f}% | "
            f"Inst: {s['current_instances']}(+{s['pending_instances']}) | "
            f"Queue: {s['queue_length']:>6.1f} | "
            f"RPS: {s['requests_per_second']:>5.0f} | "
            f"Cost: {s['cost_so_far']:.2f}/{self.task.budget}"
        )


# 
# Self-test  (run: python environment.py)
# 

if __name__ == "__main__":
    print("=" * 65)
    print("  Auto-Scaling Environment - Self-Test (rule-based agent)")
    print("=" * 65)

    for task_id in [1, 2, 3]:
        env = AutoScalingEnvironment()
        obs = env.reset(task_id=task_id)

        print(f"\nTask {task_id} [{env.task.difficulty.upper()}]: {env.task.name}")
        print(f"  Initial -> CPU={obs['cpu_usage']}% | "
              f"Instances={obs['current_instances']} | "
              f"RPS={obs['requests_per_second']}")

        done = False
        total_reward = 0.0
        step_count = 0

        while not done:
            # Improved self-test agent - mirrors the proactive thresholds
            # used by baseline.py so the self-test output reflects the
            # actual agent quality rather than a naive reactive policy.
            total = obs["current_instances"] + obs["pending_instances"]
            sla   = obs["sla_cpu_limit"]
            sla_q = obs["sla_queue_limit"]

            if obs["cpu_usage"] > sla * 0.55 or obs["queue_length"] > sla_q * 0.25:
                if total < obs["max_instances"]:
                    action = ACTION_SCALE_UP
                else:
                    action = ACTION_HOLD
            elif (
                obs["cpu_usage"] < 35.0
                and obs["queue_length"] < sla_q * 0.15
                and obs["current_instances"] > 1
            ):
                action = ACTION_SCALE_DOWN
            else:
                action = ACTION_HOLD

            obs, reward, done, info = env.step(action)
            total_reward += reward
            step_count += 1

            if step_count % 10 == 0 or done:
                env.render()

        print(f"\n  Result")
        print(f"   Steps completed : {info['steps_completed']} / {env.task.max_steps}")
        print(f"   Termination     : {info['termination_reason']}")
        print(f"   Total reward    : {total_reward:.2f}")
        print(f"   Total cost      : {info['total_cost']} / {env.task.budget}")
        print(f"   Uptime          : {info['uptime_percentage']}%")
        print(f"   SLA violations  : {info['sla_violation_count']}")
        print(f"   Critical steps  : {info['critical_violation_count']}")
        print(f"   Unnecessary    : {info['unnecessary_scaleups']}")
        print(f"   Unnecessary    : {info['unnecessary_scaledowns']}")

    print("\n" + "=" * 65)
    print("  Self-test complete.")
    print("=" * 65)