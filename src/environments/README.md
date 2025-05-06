# Traffic Signal Control Environments

This directory contains implementations of environment classes for traffic signal control using reinforcement learning. The environments follow the Gymnasium interface, allowing seamless integration with various RL algorithms.

## Available Environments

### IntersectionEnv

A lightweight, standalone implementation of a single traffic intersection without external dependencies.

**Key Features:**
- Fast execution for rapid prototyping and algorithm development
- Simplified but representative traffic dynamics with probabilistic arrivals/departures
- Configurable traffic patterns and intersection parameters
- Built-in visualization for debugging and demonstration

**Use Cases:**
- Algorithm development and initial testing
- Hyperparameter tuning
- Educational purposes
- Rapid experimentation

### SUMOIntersectionEnv

A high-fidelity traffic simulation environment using SUMO (Simulation of Urban MObility).

**Key Features:**
- Realistic vehicle dynamics (acceleration, braking, lane-changing)
- Detailed traffic light phasing and intersection geometry
- Support for real-world road networks and traffic patterns
- Comprehensive performance metrics (emissions, fuel consumption, etc.)

**Use Cases:**
- Final experiments for publication-quality results
- Testing in realistic traffic conditions
- Transfer learning to real-world applications
- Evaluation with complex traffic patterns

## Data Generation and Processing

### State Representation

Both environments provide a consistent state representation:

- **Queue Lengths**: Number of vehicles waiting at each approach (N,S,E,W)
- **Waiting Times**: Average waiting time at each approach
- **Traffic Densities**: Normalized traffic density for each approach
- **Signal Phase**: Current traffic light phase (integer)
- **Phase Duration**: Time spent in current phase (seconds)

### Action Space

Both environments use a simplified action space:

- **Binary Action**: 0 = maintain current phase, 1 = change to next phase
- **Constraints**: Minimum green time enforced before allowing phase changes
- **Yellow Phase**: Automatic yellow phase insertion when changing between green phases

### Reward Function

Multi-objective reward function based on traffic efficiency metrics:

```
reward = w₁ × queue_length + w₂ × waiting_time + w₃ × throughput + w₄ × phase_switches
```

Where:
- w₁: Negative weight for queue length (e.g., -1.0)
- w₂: Negative weight for waiting time (e.g., -0.5)
- w₃: Positive weight for throughput (e.g., 1.0)
- w₄: Negative weight for phase switches (e.g., -2.0)

## Traffic Data

### SUMO Configuration

Traffic data is defined through SUMO configuration files in `src/data/simulation/networks/`:

- **Network Files**: Define intersection geometry and traffic light logic
  - `single_intersection.nod.xml`: Node (intersection) definitions
  - `single_intersection.edg.xml`: Edge (road) definitions
  - `single_intersection.con.xml`: Connection definitions
  - `single_intersection.tll.xml`: Traffic light logic

- **Route Files**: Define traffic demand profiles
  - `low_traffic.rou.xml`: Low traffic demand scenario
  - `high_traffic.rou.xml`: High traffic demand scenario
  - `variable_demand.rou.xml`: Time-varying traffic patterns

- **Simulation Configurations**:
  - `low_traffic.sumocfg`: Configuration for low traffic scenario
  - `high_traffic.sumocfg`: Configuration for high traffic scenario
  - `variable_demand.sumocfg`: Configuration for variable demand scenario

### Traffic Generation

Traffic data is generated through:

1. **Predefined Patterns**: The route files define vehicle types and flows with:
   - Start and end times for each flow
   - Probability-based vehicle generation
   - Origin-destination pairs
   - Vehicle type distributions

2. **On-the-fly Generation**: During simulation, SUMO:
   - Inserts vehicles based on defined probabilities
   - Simulates realistic vehicle movement
   - Handles traffic light responses
   - Collects detailed metrics

3. **Generate Traffic Data Script**: `src/data/simulation/generate_traffic_data.py`
   - Creates traffic scenarios programmatically
   - Allows for parametric traffic pattern definition
   - Outputs route and configuration files for SUMO

## Usage Example

```python
# Using the lightweight environment
from src.environments import IntersectionEnv

env = IntersectionEnv(config={
    "arrival_rates": {"north": 0.2, "south": 0.2, "east": 0.3, "west": 0.3},
    "reward_weights": {"queue_length": -1.0, "wait_time": -0.5, "throughput": 1.0}
})

# Using the SUMO environment
from src.environments import SUMOIntersectionEnv

env = SUMOIntersectionEnv(
    config_file="src/data/simulation/networks/variable_demand.sumocfg",
    render_mode="human",
    config={"reward_weights": {"queue_length": -2.0, "wait_time": -1.0}}
)

# Standard Gymnasium interface
observation, info = env.reset()
for _ in range(1000):
    action = policy(observation)  # Your RL policy here
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

## Environment Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `max_time_steps` | Maximum steps per episode | 1000 |
| `yellow_time` | Duration of yellow phase in seconds | 2 |
| `min_green_time` | Minimum green time before allowing phase change | 5 |
| `arrival_rates` | Probability of vehicle arrival per direction | N/S: 0.2, E/W: 0.3 |
| `reward_weights` | Weights for different reward components | queue: -1.0, wait: -0.5, throughput: 1.0, switch: -2.0 |

## Performance Metrics

The environments provide detailed metrics for analysis:

- **Queue Lengths**: Average and maximum queue length per approach
- **Waiting Times**: Average and total waiting time
- **Throughput**: Number of vehicles that completed their trips
- **Emissions**: CO2, NOx emissions (SUMO environment only)
- **Fuel Consumption**: Total fuel used (SUMO environment only)
- **Travel Time**: Average travel time through the intersection

These metrics are available in the `info` dictionary returned by the `step()` method. 