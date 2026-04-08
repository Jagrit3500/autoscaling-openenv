---
title: Auto-Scaling Infrastructure Agent
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
short_description: "OpenEnv auto-scaling RL environment"
---

# Auto-Scaling Infrastructure Agent

> An environment where delayed effects, cost tradeoffs, and traffic uncertainty make scaling a genuine sequential decision problem.

**OpenEnv environment for AI-driven cloud server management**  
Meta x Scaler OpenEnv AI Hackathon

## What this is

A production-inspired reinforcement learning environment where an agent controls a cloud server cluster in real time. The agent observes CPU load, request queue, traffic rate, instance count, and budget - then decides every step whether to `scale_up`, `scale_down`, or `hold`.

This is a genuine sequential decision problem. Each action has delayed consequences: a server provisioned now arrives in 2 steps, a budget depleted now cannot be recovered, and a CPU that redlines for 4 consecutive steps ends the episode. A reactive agent that waits for CPU to hit 85% before acting will always fail - by the time it reacts, the capacity it requested has not arrived yet. The optimal policy requires anticipation, not reaction.

## Why this is RL-native

| Property | This environment |
|---|---|
| Delayed consequences | Scale-up takes 2 steps to activate |
| Partial observability | No future traffic visibility |
| Non-stationarity | Traffic changes dynamically |
| Resource constraints | Budget is limited |
| Credit assignment | Actions have delayed impact |
| Exploration vs exploitation | Tradeoff between cost and performance |

A rule-based agent scores about 0.49 on the leaderboard. An RL agent with lookahead should significantly exceed this.

## Tasks

### Task 1 - Single Spike Recovery
Baseline score: 0.86

### Task 2 - Traffic Wave Management
Baseline score: 0.44

### Task 3 - Adaptive Scaling Under Uncertainty
Baseline score: 0.36

## How scoring works

Each episode produces a 6-dimensional score:

| Dimension | What it measures |
|---|---|
| Completion | Fraction of steps completed |
| Uptime | Percent of steps below SLA CPU |
| SLA | SLA violation frequency |
| Cost | Budget efficiency |
| Stability | Critical overload frequency |
| Scaling efficiency | Unnecessary scaling |

Crash penalty: -0.3

Final score:

0.20 x Task1 + 0.35 x Task2 + 0.45 x Task3

Baseline leaderboard score: 0.49

## Quick start

```bash
pip install -r requirements.txt
python baseline.py
python -m uvicorn server.app:app --host 0.0.0.0 --port 7860
```

## Action space

| Action | Value |
|---|---|
| scale_up | 0 |
| scale_down | 1 |
| hold | 2 |

## Observation space

cpu_usage, queue_length, requests_per_second,  
current_instances, pending_instances,  
budget_remaining, time_step, ...

## Project structure

```text
autoscaling_env/
|-- tasks.py
|-- environment.py
|-- graders.py
|-- baseline.py
|-- inference.py
|-- server/
|   |-- __init__.py
|   `-- app.py
|-- openenv.yaml
|-- Dockerfile
`-- README.md
```

## Design decisions

- CPU > 100% allowed  
- Boot delay = 2 steps  
- Budget buffer = 10%  
- Seeded randomness for reproducibility  

## Hackathon track

Meta x Scaler OpenEnv AI Hackathon