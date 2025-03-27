# Traffic Signal Control with Reinforcement Learning

This repository implements adaptive traffic signal control using reinforcement learning (RL) techniques, as described in the paper "Adaptive Traffic Signal Control with Reinforcement Learning".

## 📋 Project Overview

This project explores the application of reinforcement learning for adaptive traffic signal control. Traditional traffic control methods like fixed-timing and actuated control are often suboptimal in dynamic traffic conditions. Reinforcement learning offers a data-driven approach to optimize signal timing in real-time, leading to reduced congestion, shorter waiting times, and improved throughput.

The implementation includes:

- Single and multi-intersection traffic environments
- Multiple RL algorithms (DQN, A2C, PPO, Q-learning)
- Traditional baseline controllers (fixed-timing, actuated)
- Comprehensive evaluation framework
- Visualization tools

## 🚦 Environments

### Single Intersection

The `IntersectionEnv` simulates a four-way intersection with configurable traffic patterns. The agent controls the traffic signals, deciding when to change the signal phase to optimize traffic flow.

### Multi-Intersection Network

The `TrafficMultiEnv` extends the simulation to multiple connected intersections, supporting both centralized and decentralized control strategies.

## 🤖 Agents

The following control methods are implemented:

### RL Agents

- **DQN (Deep Q-Network)**: Includes double DQN, prioritized experience replay, and n-step returns
- **A2C (Advantage Actor-Critic)**: Policy gradient method with baseline value function
- **PPO (Proximal Policy Optimization)**: More stable policy updates with clipped objective
- **Q-Learning**: Tabular approach for baseline comparison

### Traditional Controllers

- **Fixed-timing**: Predefined cycle lengths and green splits
- **Actuated**: Vehicle-actuated control with extension logic
- **Webster**: Adaptive signal timing based on the Webster formula

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
```

## 🚀 Usage

### Training

To train an agent on a single intersection:

```bash
python src/scripts/train.py --algorithm dqn --env_type single --episodes 500
```

For multi-intersection scenarios:

```bash
python src/scripts/train.py --algorithm a2c --env_type multi --topology 2x2_grid --control_mode decentralized
```

### Evaluation

To evaluate agents across different traffic scenarios:

```bash
python src/scripts/evaluate.py --methods dqn a2c ppo fixed actuated --env_type single --scenario normal
```

For visualizing traffic state:

```bash
python src/demos/traffic_signal_demo.py
```

## 📊 Results

The framework provides comprehensive evaluation metrics:

- Average waiting time
- Average queue length
- Throughput
- Travel time
- CO2 emissions (estimated)

Visualization tools include:

- Learning curves
- Performance comparisons
- Traffic state visualizations
- Statistical significance tests

## 📂 Project Structure

```
traffic-signal-rl/
├── paper/                  # Paper and related materials
├── src/                    # Source code
│   ├── agents/             # RL agent implementations
│   ├── environments/       # Traffic environments
│   ├── utils/              # Utility functions
│   ├── demos/              # Interactive demonstrations
│   ├── scripts/            # Training and evaluation scripts
│   └── config/             # Configuration files
├── logs/                   # Training logs
├── results/                # Evaluation results
└── tests/                  # Test scripts
```

## ⚙️ Configuration

Configuration files are located in the `src/config` directory:

- `config.yaml`: Environment and experiment settings
- `hyperparameters.yaml`: Algorithm-specific hyperparameters

## 🔍 Experiments

The paper demonstrates several experiments:

1. **Single intersection with varying traffic conditions**
2. **2×2 grid with coordinated traffic flow**
3. **Variable demand scenario simulating peak/off-peak hours**


## 📝 Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{author2023adaptive,
  title={Adaptive Traffic Signal Control with Reinforcement Learning},
  author={Ali, Izhar and Haileyesus, Million},
  journal={Journal Name},
  year={2025},
  volume={},
  pages={}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request