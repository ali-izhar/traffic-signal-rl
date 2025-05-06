# Traffic Signal Control with Reinforcement Learning

This repository implements adaptive traffic signal control using reinforcement learning (RL) techniques to optimize traffic flow at signalized intersections.

## 📋 Project Overview

This project explores the application of reinforcement learning for adaptive traffic signal control. Traditional traffic control methods like fixed-timing and actuated control are often suboptimal in dynamic traffic conditions. Reinforcement learning offers a data-driven approach to optimize signal timing in real-time, leading to reduced congestion, shorter waiting times, and improved throughput.

The implementation includes:

- SUMO-based traffic simulation environments
- Multiple RL algorithms (DQN, A2C, PPO, Q-learning)
- Traditional baseline controllers (fixed-timing, actuated, Webster)
- Comprehensive evaluation framework
- Visualization tools

## 🚦 Project Structure

```
traffic-signal-rl/
├── src/                    # Source code
│   ├── agents/             # RL agent implementations
│   │   ├── a2c_agent.py    # Advantage Actor-Critic implementation
│   │   ├── dqn_agent.py    # Deep Q-Network implementation
│   │   ├── ppo_agent.py    # Proximal Policy Optimization implementation
│   │   ├── qlearning.py    # Tabular Q-learning implementation
│   │   └── base_agent.py   # Abstract base class for all agents
│   ├── environments/       # Traffic environments
│   │   ├── sumo_env.py     # SUMO-based traffic simulation environment
│   │   └── utils/          # Environment utility functions
│   ├── utils/              # Utility functions
│   │   ├── metrics.py      # Performance metrics calculation
│   │   ├── logger.py       # Logging utilities
│   │   ├── visualization.py # Visualization tools
│   │   └── replay_buffer.py # Experience replay for DQN
│   ├── data/               # Simulation data
│   │   └── simulation/     # SUMO configuration files
│   │       └── networks/   # Road network definitions
│   ├── config/             # Configuration files
│   │   ├── hyperparameters.yaml # Algorithm-specific hyperparameters
│   │   └── evaluation_config.yaml # Evaluation settings
│   ├── scripts/            # Training and evaluation scripts
│   │   ├── train.py        # Main training script
│   │   ├── evaluate.py     # Evaluation script
│   │   ├── compare_models.py # Model comparison tool
│   │   └── analyze_results.py # Results analysis
│   └── demos/              # Interactive demonstrations
├── logs/                   # Training logs
├── results/                # Evaluation results
└── tests/                  # Test scripts
```

## 📚 Module Descriptions

### Agents

- **`a2c_agent.py`**: Implements the Advantage Actor-Critic algorithm with a shared network architecture for feature extraction, separate actor (policy) and critic (value function) networks, and Generalized Advantage Estimation (GAE).
- **`dqn_agent.py`**: Deep Q-Network implementation with double DQN, prioritized experience replay, and dueling network architecture.
- **`ppo_agent.py`**: Proximal Policy Optimization with clipped objective function for stable policy updates.
- **`qlearning.py`**: Traditional tabular Q-learning for discrete state spaces, serving as a baseline.
- **`base_agent.py`**: Abstract base class defining the interface for all RL agents.

### Environments

- **`sumo_env.py`**: A wrapper for the SUMO traffic simulator that provides a reinforcement learning interface (observation space, action space, reward function) for traffic signal control.

### Utils

- **`metrics.py`**: Functions for calculating performance metrics like average waiting time, queue length, throughput, and emissions.
- **`logger.py`**: Utilities for logging training progress and evaluation results.
- **`visualization.py`**: Tools for visualizing traffic states, learning curves, and performance comparisons.
- **`replay_buffer.py`**: Implementation of experience replay buffer with prioritization for DQN.

### Configuration

- **`hyperparameters.yaml`**: Contains algorithm-specific hyperparameters for each RL method.
- **`evaluation_config.yaml`**: Settings for evaluation scenarios and metrics.

### Scripts

- **`train.py`**: Main script for training RL agents with configurable hyperparameters and environments.
- **`evaluate.py`**: Script for evaluating trained agents across different traffic scenarios.
- **`compare_models.py`**: Tool for comparing performance of different control strategies.
- **`analyze_results.py`**: Script for generating metrics and visualizations from evaluation results.

## 🔧 Installation

```bash
# Clone repository
git clone https://github.com/username/traffic-signal-rl.git
cd traffic-signal-rl

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify SUMO installation
sumo --version
```

## 🚀 Usage

### Training

The `train.py` script supports various algorithms and traffic scenarios:

```bash
# Train DQN agent on variable demand scenario
python src/scripts/train.py --algorithm dqn --env_type sumo --sumo_config src/data/simulation/networks/variable_demand.sumocfg --episodes 1000 --batch_size 512 --learning_rate 5e-4 --target_update 10 --save_dir logs/dqn/variable
```

### Evaluation

To evaluate trained agents across different traffic scenarios:

```bash
# Evaluate trained DQN agent
python src/scripts/evaluate.py --algorithm dqn --env_type sumo --sumo_config src/data/simulation/networks/variable_demand.sumocfg --model_path logs/dqn/variable/checkpoints/best_model.pt --episodes 100 --render true --save_video true --output_dir evaluation/dqn
```

For comparing multiple agents:

```bash
# Compare all control strategies
python src/scripts/compare_models.py --config_path src/config/evaluation_config.yaml --output_dir comparison
```

## 📊 Performance Metrics

The framework provides comprehensive evaluation metrics:

- Average waiting time
- Average queue length
- Throughput
- Travel time
- CO2 emissions (estimated)

## 📈 Visualization

Visualization tools include:

- Learning curves
- Performance comparisons
- Traffic state visualizations
- Statistical significance tests

View learning curves in real-time:

```bash
# Run TensorBoard to view learning curves
tensorboard --logdir=logs
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request