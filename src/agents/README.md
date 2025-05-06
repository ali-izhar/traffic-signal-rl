# Reinforcement Learning Agents for Traffic Signal Control

This directory contains implementations of several reinforcement learning agents for traffic signal control. Each agent implements different algorithms with varying complexity and capabilities.

## Q-Learning Agent

**Algorithm:** Tabular Q-learning with discretized state space

**How it works:**
- Discretizes continuous state variables into buckets using adaptive binning
- Maintains a table mapping state-action pairs to Q-values
- Uses epsilon-greedy exploration strategy with annealing
- Updates Q-values via temporal difference learning: Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]
- Optional double Q-learning to reduce overestimation bias
- Optional experience replay for more stable learning

**Input:**
- State: Array of continuous values (queue lengths, waiting times, densities, phase info)
- Reward: Scalar value representing traffic efficiency
- Action: Integer representing previous action taken

**Output:**
- Action: Integer (0: maintain current phase, 1: change phase)
- Metrics: Dictionary with TD error, Q-value, epsilon, table size

## DQN Agent

**Algorithm:** Deep Q-Network with experience replay and target networks

**How it works:**
- Approximates Q-values with deep neural networks instead of tables
- Uses separate target network to stabilize learning
- Implements experience replay buffer to break correlations between samples
- Supports prioritized experience replay for focusing on important transitions
- Implements double DQN to reduce overestimation bias
- Optional dueling architecture for better value estimation
- N-step returns for more efficient learning
- Supports different loss functions (MSE, Huber)

**Input:**
- State: Continuous vector (queue lengths, waiting times, densities, phase info)
- Reward: Scalar value representing traffic efficiency
- Action: Integer representing previous action taken

**Output:**
- Action: Integer (0: maintain current phase, 1: change phase)
- Metrics: Dictionary with loss, Q-values, exploration rate

## A2C Agent

**Algorithm:** Advantage Actor-Critic with shared representation

**How it works:**
- Uses two networks: actor (policy) and critic (value function)
- Actor suggests actions, critic evaluates state values
- Shared feature extraction layers for both networks
- Uses Generalized Advantage Estimation (GAE) for more stable training
- Updates via policy gradient with entropy regularization
- Supports visualization and tensorboard logging

**Input:**
- State: Continuous vector (queue lengths, waiting times, densities, phase info)
- Reward: Scalar value representing traffic efficiency

**Output:**
- Action: Integer (0: maintain current phase, 1: change phase)
- Metrics: Dictionary with actor loss, critic loss, entropy

## PPO Agent

**Algorithm:** Proximal Policy Optimization with clipped objective

**How it works:**
- On-policy algorithm with clipped surrogate objective to limit policy changes
- Actor-critic architecture with separate policy and value networks
- Uses Generalized Advantage Estimation (GAE) for advantage computation
- Implements multiple epochs of training on each batch of data
- Uses KL divergence monitoring for early stopping
- Normalizes advantages for more stable training
- Implements learning rate scheduling

**Input:**
- State: Continuous vector (queue lengths, waiting times, densities, phase info)
- Reward: Scalar value representing traffic efficiency

**Output:**
- Action: Integer (0: maintain current phase, 1: change phase)
- Metrics: Dictionary with policy loss, value loss, KL divergence, clip fraction

## Data Preprocessing

All agents expect traffic state data with the following structure:
- Queue lengths: Number of vehicles waiting at each approach (N,S,E,W)
- Waiting times: Average waiting time at each approach (N,S,E,W)
- Densities: Traffic density at each approach (N,S,E,W)
- Current phase: Current traffic light phase (integer)
- Phase duration: Time spent in current phase (seconds)

## Common Agent Interface

All agents implement a common interface:
```python
agent.act(state, eval_mode=False) → action
agent.learn(state, action, reward, next_state, done) → metrics
agent.save(filepath) → None
agent.load(filepath) → None
```

This consistent interface allows for easy swapping and comparison of different algorithms.
