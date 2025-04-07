# Traffic Signal Control with Reinforcement Learning

## Setup Instructions

```
# Clone the repository
git clone https://github.com/yourusername/traffic-signal-rl.git
cd traffic-signal-rl

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Verify SUMO installation
sumo --version
```

## Training Procedure

### 1. Baseline Controllers (for benchmarking)

```bash
# Fixed timing controller
python src/scripts/train.py --algorithm fixed --env_type sumo --sumo_config src/data/simulation/networks/variable_demand.sumocfg --episodes 50 --save_dir logs/baseline

# Actuated controller
python src/scripts/train.py --algorithm actuated --env_type sumo --sumo_config src/data/simulation/networks/variable_demand.sumocfg --episodes 50 --save_dir logs/baseline

# Webster controller
python src/scripts/train.py --algorithm webster --env_type sumo --sumo_config src/data/simulation/networks/variable_demand.sumocfg --episodes 50 --save_dir logs/baseline
```

### 2. Training RL Agents

#### Q-Learning (discrete state space)

```bash
# Train on low traffic
python src/scripts/train.py --algorithm qlearning --env_type sumo --sumo_config src/data/simulation/networks/low_traffic.sumocfg --episodes 500 --max_steps 1000 --learning_rate 0.1 --gamma 0.95 --epsilon_start 1.0 --epsilon_end 0.01 --epsilon_decay 0.995 --save_dir logs/qlearning/low

# Train on high traffic
python src/scripts/train.py --algorithm qlearning --env_type sumo --sumo_config src/data/simulation/networks/high_traffic.sumocfg --episodes 500 --max_steps 1000 --learning_rate 0.1 --gamma 0.95 --epsilon_start 1.0 --epsilon_end 0.01 --epsilon_decay 0.995 --save_dir logs/qlearning/high

# Train on variable demand
python src/scripts/train.py --algorithm qlearning --env_type sumo --sumo_config src/data/simulation/networks/variable_demand.sumocfg --episodes 500 --max_steps 1000 --learning_rate 0.1 --gamma 0.95 --epsilon_start 1.0 --epsilon_end 0.01 --epsilon_decay 0.995 --save_dir logs/qlearning/variable
```

#### Deep Q-Network (DQN)

```bash
# Train on low traffic
python src/scripts/train.py --algorithm dqn --env_type sumo --sumo_config src/data/simulation/networks/low_traffic.sumocfg --episodes 1000 --max_steps 1000 --batch_size 512 --learning_rate 5e-4 --target_update 10 --save_dir logs/dqn/low

# Train on high traffic
python src/scripts/train.py --algorithm dqn --env_type sumo --sumo_config src/data/simulation/networks/high_traffic.sumocfg --episodes 1000 --max_steps 1000 --batch_size 512 --learning_rate 5e-4 --target_update 10 --save_dir logs/dqn/high

# Train on variable demand
python src/scripts/train.py --algorithm dqn --env_type sumo --sumo_config src/data/simulation/networks/variable_demand.sumocfg --episodes 1000 --max_steps 1000 --batch_size 512 --learning_rate 5e-4 --target_update 10 --save_dir logs/dqn/variable
```

#### Advantage Actor-Critic (A2C)

```bash
# Train on low traffic
python src/scripts/train.py --algorithm a2c --env_type sumo --sumo_config src/data/simulation/networks/low_traffic.sumocfg --episodes 1000 --max_steps 1000 --learning_rate 5e-4 --gamma 0.95 --save_dir logs/a2c/low

# Train on high traffic
python src/scripts/train.py --algorithm a2c --env_type sumo --sumo_config src/data/simulation/networks/high_traffic.sumocfg --episodes 1000 --max_steps 1000 --learning_rate 5e-4 --gamma 0.95 --save_dir logs/a2c/high

# Train on variable demand
python src/scripts/train.py --algorithm a2c --env_type sumo --sumo_config src/data/simulation/networks/variable_demand.sumocfg --episodes 1000 --max_steps 1000 --learning_rate 5e-4 --gamma 0.95 --save_dir logs/a2c/variable
```

#### Proximal Policy Optimization (PPO)

```bash
# Train on low traffic
python src/scripts/train.py --algorithm ppo --env_type sumo --sumo_config src/data/simulation/networks/low_traffic.sumocfg --episodes 1000 --max_steps 1000 --batch_size 512 --learning_rate 5e-4 --gamma 0.95 --save_dir logs/ppo/low

# Train on high traffic
python src/scripts/train.py --algorithm ppo --env_type sumo --sumo_config src/data/simulation/networks/high_traffic.sumocfg --episodes 1000 --max_steps 1000 --batch_size 512 --learning_rate 5e-4 --gamma 0.95 --save_dir logs/ppo/high

# Train on variable demand
python src/scripts/train.py --algorithm ppo --env_type sumo --sumo_config src/data/simulation/networks/variable_demand.sumocfg --episodes 1000 --max_steps 1000 --batch_size 512 --learning_rate 5e-4 --gamma 0.95 --save_dir logs/ppo/variable
```

## Evaluation Procedure

```bash
# Evaluate trained agents (specify path to best model checkpoint)
python src/scripts/evaluate.py --algorithm dqn --env_type sumo --sumo_config src/data/simulation/networks/variable_demand.sumocfg --model_path logs/dqn/variable/checkpoints/best_model.pt --episodes 100 --render true --save_video true --output_dir evaluation/dqn

# Compare all agents
python src/scripts/compare_models.py --config_path src/config/evaluation_config.yaml --output_dir comparison
```

## Analysis and Visualization

```bash
# Generate metrics plots
python src/scripts/analyze_results.py --log_dirs logs/baseline logs/dqn/variable logs/a2c/variable logs/ppo/variable --output_dir analysis

# Run TensorBoard to view learning curves
tensorboard --logdir=logs
```

## Paper-Specific Instructions

1. For the final results in the paper, run all agents for at least 1000 episodes
2. Test on all traffic scenarios (low, high, variable) to demonstrate adaptability
3. Ensure to save model checkpoints every 50 episodes 
4. Run evaluation with 100 episodes for statistically significant results
5. Use the `--seed` parameter (e.g., `--seed 42`) to ensure reproducibility
6. For optimal performance on a GPU machine:
   - Set `--batch_size 1024` for deeper networks
   - Enable mixed precision with PyTorch's native AMP
   - Adjust learning rate scheduling with cosine decay
   - Monitor GPU utilization and memory usage

The existing config files are sufficient for all experiments, but memory usage should be monitored on the GPU machine when scaling to larger batch sizes.
