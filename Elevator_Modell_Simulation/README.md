# Elevator Model Simulation

Runs a trained MaskablePPO reinforcement learning model to control 3 elevators in a 10-floor building. The model was trained for 1318 episodes using `sb3-contrib` with action masking.

## Getting Started

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

### Install dependencies

```bash
cd Elevator_Modell_Simulation
uv sync
```

### Run with Gymnasium (recommended)

**Headless (no display needed):**

```bash
uv run python run_gym.py
```

**With Pygame visualization:**

```bash
uv run python run_gym.py --render human
```

**Options:**

```
--render human|rgb_array   Render mode (omit for headless)
--model PATH               Path to trained model (default: modell/ppo_elevator_episode_1318.zip)
--max-guests N             Number of guests to simulate (default: 200)
```

### Run with legacy Main.py

```bash
uv run python Main.py --headless    # headless
uv run python Main.py               # with tkinter + pygame GUI
```

## Architecture

The simulation uses a proper `gymnasium.Env` interface (`elevator_env.py`):

| Component | File | Role |
|-----------|------|------|
| Gym Environment | `elevator_env.py` | `ElevatorEnv` — standard Gymnasium API with optional Pygame rendering |
| Runner | `run_gym.py` | Loads model, runs inference loop, prints stats |
| Elevator | `Elevator.py` | Action execution, boarding/dropoff logic |
| Guest | `Guest.py` | Guest lifecycle — spawn, wait, ride, work, leave |
| Visualization | `Visualization.py` | Post-simulation matplotlib charts |
| Legacy entry | `Main.py` | Original SimPy + Pygame loop (kept for compatibility) |

## Model Details

- **Algorithm:** MaskablePPO (Proximal Policy Optimization with action masking)
- **Training:** 1318 episodes, single-elevator, curriculum learning
- **Inference:** Applied independently to each of 3 elevators
- **Action space:** `[wait, up, down]` per elevator
- **Observation:** Elevator position, passenger count, destination histogram, waiting guests per floor

## Performance

| Metric | SCAN (baseline) | RL (PPO) | Improvement |
|--------|----------------|----------|-------------|
| Avg wait time | 47.4s | 24.1s | -49% |
| Avg ride time | 30.6s | 48.6s | +59% |
| Avg total time | 78.0s | 72.6s | -7% |

The RL model halves waiting time at the cost of longer rides, netting ~7% improvement in total guest experience.
