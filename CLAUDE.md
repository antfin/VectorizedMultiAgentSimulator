# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VMAS (Vectorized Multi-Agent Simulator) is a PyTorch-based framework for GPU-accelerated parallel multi-agent simulation. It runs batches of environments simultaneously using tensor operations, primarily for multi-agent reinforcement learning research. Licensed under GPLv3.

## Common Commands

### Install (development)
```bash
python setup.py develop
pip install pre-commit && pre-commit install
```

### Run tests
```bash
# Full suite (Linux needs xvfb-run for rendering)
pytest tests/
xvfb-run -s "-screen 0 1024x768x24" pytest tests/  # Linux

# Single test file
pytest tests/test_vmas.py

# Single test (scenario-specific)
pytest tests/test_scenarios/test_navigation.py

# With coverage
pytest tests/ --cov=. --cov-report=xml
```

### Lint and format
```bash
pre-commit run --all-files
```
This runs: ufmt (black 22.3.0 + usort 1.0.3) for formatting, flake8 with bugbear/comprehensions plugins. Config is in `setup.cfg`. Line length: 79 (flake8), 120 (pep8).

## Architecture

### Core Simulation (`vmas/simulator/`)

- **`core.py`** — Central module. `TorchVectorizedObject` is the base class enabling GPU-accelerated batch operations. Contains `World`, `Entity`, `Agent`, `Landmark`, shape classes (`Box`, `Sphere`, `Line`, `Capsule`), and state/action abstractions. Everything operates on batched PyTorch tensors with a `batch_dim` dimension.
- **`physics.py`** — Collision detection and contact physics.
- **`scenario.py`** — `BaseScenario` abstract class that all scenarios implement. Required methods: `make_world()`, `reset_world_at()`, `observation()`, `reward()`.
- **`dynamics/`** — Agent motion models: holonomic, differential drive, kinematic bicycle, drone, etc. All extend `Dynamics` from `common.py`.
- **`sensors.py`** — Sensor abstractions (e.g., LiDAR).
- **`joints.py`** — Rigid body joint constraints.
- **`rendering.py`** — Pyglet-based 2D rendering.

### Environment Wrappers (`vmas/simulator/environment/`)

- **`environment.py`** — Main `Environment` class wrapping the simulation loop.
- **`gym/`** — Wrappers for OpenAI Gym, Gymnasium, and vectorized Gymnasium.
- **`rllib.py`** — Ray RLlib integration.

### Scenarios (`vmas/scenarios/`)

41+ scenarios organized into: main scenarios (21), `debug/` (11), and `mpe/` (9, OpenAI Multi-Particle Environment ports). Each is a Python file with a `Scenario` class extending `BaseScenario`. Loaded dynamically by name via `make_env()`.

### Entry Point

```python
from vmas import make_env
env = make_env(scenario="waterfall", num_envs=32, device="cpu",
               continuous_actions=True, wrapper="gymnasium")
```

`make_env()` in `vmas/make_env.py` is the factory for creating environments with optional wrapper selection.

## Key Design Patterns

- **Batch dimension**: All tensors carry a leading batch dimension (`num_envs`). Operations must be vectorized across this dimension.
- **Device-aware**: All objects track their `device` (CPU/GPU) via `TorchVectorizedObject`.
- **Actions**: List or dict of per-agent tensors. Continuous (forces/velocities) or discrete (integer indices).
- **Scenario interface**: Scenarios define world setup, observations, and rewards. Optional hooks: `info()`, `extra_render()`, `process_action()`, `pre_step()`, `post_step()`.

## Style Conventions

- Formatting: Black (v22.3.0), import sorting via usort
- Flake8 ignores: E203, E402, W503, W504, E501
- Docstrings: Google convention (pydocstyle)
- Copyright headers on all source files
