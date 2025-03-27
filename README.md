# Adaptive Traffic Signal Control using Reinforcement Learning

## Overview
This paper implements adaptive traffic signal control through reinforcement learning. We use multiple RL algorithms to optimize traffic flow at both single and multiple intersections, comparing their performance against traditional control methods.

![Traffic Signal Control Demo](paper/images/traffic_signal_demo.png)

## Features
- Single intersection and multi-intersection traffic environments
- Implementation of Q-learning, DQN, A2C, and PPO algorithms
- Advanced state representations capturing queue lengths, waiting times, and traffic density
- Multi-objective reward function balancing throughput, delay, and signal stability
- Visualization tools for traffic flow and performance metrics
- Support for both centralized and decentralized control in multi-intersection scenarios

## Project Structure
```
.
├── paper/                 # Research paper and images
├── src/
│   ├── agents/            # RL agent implementations
│   │   ├── a2c_agent.py   # A2C algorithm
│   │   ├── dqn_agent.py   # DQN algorithm
│   │   └── ppo_agent.py   # PPO algorithm
│   ├── config/            # Configuration files
│   ├── data/              # Data storage
│   ├── demos/             # Demo scripts and visualizations
│   ├── environments/      # Traffic environments
│   │   ├── intersection_env.py  # Single intersection
│   │   └── traffic_env.py       # Multiple intersections
│   ├── logs/              # Training logs
│   ├── models/            # Neural network models
│   ├── scripts/           # Training and evaluation scripts
│   ├── tests/             # Test files
│   └── utils/             # Utility functions
└── requirements.txt       # Dependencies
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup
1. Clone the repository:
```bash
git clone https://github.com/your-username/traffic-signal-rl.git
cd traffic-signal-rl
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix or MacOS
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Demo
To see a visualization of traffic signal control:
```bash
python src/demos/traffic_signal_demo.py
```

This will show how a reinforcement learning agent controls a traffic signal at a single intersection, with visualizations of traffic flow and performance metrics.

### Training a Model
To train an RL agent on the traffic environment:
```bash
python src/scripts/train.py --algorithm dqn --episodes 1000
```

Supported algorithms: `qlearning`, `dqn`, `a2c`, `ppo`

### Running Experiments
For full experimental evaluation with different algorithms and scenarios:
```bash
python src/scripts/evaluate.py --scenario single_intersection
```

Supported scenarios: `single_intersection`, `grid_2x2`, `corridor`

### Visualizing Results
```bash
python src/scripts/visualize.py --log_dir src/logs/dqn_experiment
```

## Implementation Details

### Environment
- State space: queue lengths, waiting times, traffic density, signal phase
- Action space: keep current phase or switch to next phase
- Reward: weighted sum of queue lengths, waiting times, throughput, and signal changes

### Network Architectures
- DQN: 3-layer feed-forward network with layer normalization
- Actor-Critic: Shared feature extractor with separate policy and value heads

### Training Infrastructure
- Experience replay with prioritization
- N-step returns for better credit assignment
- Parallel environment sampling

Refer to the paper for detailed descriptions of network architectures, hyperparameters, and the reward function design.

## Citation
If you use this code in your research, please cite:

```bibtex
@article{
  author = {Ali, Izhar and Haileyesus, Million},
  title = {Adaptive Traffic Signal Control with Reinforcement Learning},
  journal = {},
  year = {2025},
}
```

## License
[MIT License](LICENSE)

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.