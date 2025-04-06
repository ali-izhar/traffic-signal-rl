# Traffic Signal Control

Implementation of the paper "Adaptive Traffic Signal Control with Reinforcement Learning".

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set SUMO_HOME environment variable (Windows)
$env:SUMO_HOME = "C:\Program Files (x86)\Eclipse\Sumo"  # Adjust path as needed
```

## Data Generation

The simulation data is created using SUMO (Simulation of Urban MObility):

```bash
# Generate the intersection network
cd data/simulation/networks
netconvert -c single_intersection.netccfg

# Generate traffic scenarios
cd ../..
python simulation/generate_traffic_data.py
```

This creates three traffic scenarios:
- `low_traffic.rou.xml`: Light traffic conditions
- `high_traffic.rou.xml`: Heavy traffic conditions
- `variable_demand.rou.xml`: Time-varying traffic patterns

## Training Models

```bash
# Train DQN agent on SUMO environment (50 episodes)
python scripts/train.py --algorithm dqn --env_type sumo --episodes 50

# Train A2C agent
python scripts/train.py --algorithm a2c --env_type sumo --episodes 50

# Train with different hyperparameters
python scripts/train.py --algorithm dqn --env_type sumo --episodes 50 --gamma 0.99 --learning_rate 0.0001
```

Models will be saved to the `logs` directory with timestamps.

## Quick Testing

```bash
# Test environment with random agent vs fixed-time controller
python scripts/test_sumo_env.py --episodes 2 --render
```

## Evaluation

```bash
# Evaluate all methods on the SUMO environment
python scripts/evaluate.py --env_type sumo --methods dqn fixed actuated --episodes 5

# Evaluate a specific model
python scripts/evaluate.py --env_type sumo --methods dqn --model_path logs/dqn_sumo_TIMESTAMP/best_model.pt --episodes 5

# Visualize during evaluation
python scripts/evaluate.py --env_type sumo --methods dqn --episodes 2 --render
```

Results will be saved to the `results` directory, including performance metrics, comparisons, and visualizations.

## Key Findings

- Random control surprisingly outperforms fixed-timing on unbalanced traffic
- DQN can learn effective policies with ~50 episodes of training
- Performance improvements over traditional methods:
  - ~45% reduction in queue lengths
  - ~60% reduction in waiting times
  - ~2x increase in throughput

The intersection code supports three environments:
- Simple simulated environment (`--env_type single`)
- Multi-intersection environment (`--env_type multi`)
- SUMO-based realistic environment (`--env_type sumo`)
