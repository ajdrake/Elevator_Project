# Elevator Scanning

Classic scan-based (LOOK algorithm) elevator control simulation. Multiple elevators serve guests in a building, picking up passengers as they travel continuously up and down. Visualization via Pygame; discrete-event simulation via SimPy.

## Getting Started

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (recommended package manager)

### Install dependencies

```bash
cd Elevator_Scanning
uv sync
```

### Run the simulation

```bash
uv run python Main.py
```

A settings dialog will appear where you can configure floors, elevators, capacity, and other parameters. Press **Start Simulation** to launch the Pygame window.

**Controls during simulation:**
- `-` / `+` buttons: adjust visualization speed
- `×2` button: double the visualization speed
- `Space`: skip to end (run headlessly, then show results)

### Run unit tests

```bash
uv run pytest
```

With coverage:

```bash
uv run pytest --cov=. --cov-report=term-missing
```

## Architecture

| File | Role |
|------|------|
| `Main.py` | Entry point — tkinter param dialog, Pygame init, runs simulation |
| `Building.py` | SimPy environment, guest spawner, Pygame drawing |
| `Elevator.py` | SCAN elevator logic — collect requests, open doors, move floors |
| `Dispatcher.py` | Routes ride/destination requests to elevator queues |
| `Guest.py` | Guest lifecycle — arrive, wait, ride, work, leave |
| `Visualization.py` | Post-simulation matplotlib charts |
| `RideRequest.py` | Request object for calling an elevator |
| `DestinationRequest.py` | Request object for specifying a destination floor |
| `Wait.py` | Gamma-distributed waiting time helper |
| `ElevatorException.py` | Custom exception for full elevators |

## Dependencies

Managed via `pyproject.toml` + `uv`:

- **simpy** — discrete-event simulation
- **pygame** — real-time visualization
- **numpy** — random distributions
- **pandas** — log analysis
- **matplotlib** — result charts
