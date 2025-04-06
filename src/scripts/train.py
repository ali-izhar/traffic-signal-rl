#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Training Script for Traffic Signal Control

This script trains RL agents on the traffic signal control environments.

Environment Selection Guide:
- IntersectionEnv (--env_type single): Best for rapid prototyping, algorithm
  development, and hyperparameter tuning. Use for initial experiments.

- SUMOIntersectionEnv (--env_type sumo): Recommended for final results and
  publication-quality experiments. Provides the most realistic traffic
  simulation but requires SUMO to be installed.

- TrafficMultiEnv (--env_type multi): Use for experiments involving multiple
  intersections and coordination strategies. Supports both centralized and
  decentralized control modes.

USAGE:
------
# Train a DQN agent
python src/scripts/train.py --algorithm dqn --env_type single --scenario normal

# Train a PPO agent
python src/scripts/train.py --algorithm ppo --env_type single --scenario variable

# Evaluate a baseline controller
python src/scripts/train.py --algorithm fixed --env_type single --scenario unbalanced

# Use the SUMO environment for more realistic simulation
python src/scripts/train.py --algorithm a2c --env_type sumo --sumo_config data/simulation/networks/single_intersection.sumocfg
"""

from datetime import datetime
from tqdm import tqdm

import os
import sys
import argparse
import random
import json

import numpy as np
import matplotlib.pyplot as plt
import torch

# Add src directory to path
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Import environments
from environments.intersection_env import IntersectionEnv
from environments.traffic_env import TrafficMultiEnv
from environments.sumo_env import SUMOIntersectionEnv

# Import all agent implementations
from agents.dqn_agent import DQNAgent
from agents.a2c_agent import A2CAgent
from agents.ppo_agent import PPOAgent
from agents.qlearning_agent import QLearningAgent
from agents.baseline_controllers import (
    FixedTimingController,
    ActuatedController,
    WebsterController,
)

from utils.logger import Logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train RL agents for traffic signal control"
    )

    # Algorithm selection
    parser.add_argument(
        "--algorithm",
        type=str,
        default="dqn",
        choices=["a2c", "dqn", "ppo", "qlearning", "fixed", "actuated", "webster"],
        help="RL algorithm or baseline controller to use",
    )

    # Environment settings
    parser.add_argument(
        "--env_type",
        type=str,
        default="single",
        choices=["single", "multi", "sumo"],
        help="Environment type: single intersection or multi-intersection or SUMO",
    )
    parser.add_argument(
        "--topology",
        type=str,
        default="2x2_grid",
        choices=["2x2_grid", "corridor"],
        help="Network topology for multi-intersection environment",
    )
    parser.add_argument(
        "--sumo_config",
        type=str,
        default="data/simulation/networks/single_intersection.sumocfg",
        help="Path to SUMO configuration file (for SUMO environment)",
    )
    parser.add_argument(
        "--control_mode",
        type=str,
        default="decentralized",
        choices=["centralized", "decentralized"],
        help="Control mode for multi-intersection environment",
    )

    # Training parameters
    parser.add_argument(
        "--episodes", type=int, default=500, help="Number of training episodes"
    )
    parser.add_argument(
        "--max_steps", type=int, default=1000, help="Maximum steps per episode"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.0005, help="Learning rate"
    )
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor")
    parser.add_argument(
        "--epsilon_start", type=float, default=1.0, help="Starting exploration rate"
    )
    parser.add_argument(
        "--epsilon_end", type=float, default=0.01, help="Final exploration rate"
    )
    parser.add_argument(
        "--epsilon_decay",
        type=float,
        default=0.995,
        help="Exploration rate decay factor",
    )

    # Model settings
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=256,
        help="Hidden layer size for neural networks",
    )
    parser.add_argument(
        "--target_update",
        type=int,
        default=10,
        help="Target network update frequency (episodes)",
    )

    # Saving and logging
    parser.add_argument(
        "--save_dir",
        type=str,
        default="../logs",
        help="Directory to save models and logs",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
        help="Episode interval for logging stats",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=50,
        help="Episode interval for saving models",
    )
    parser.add_argument(
        "--eval_interval", type=int, default=20, help="Episode interval for evaluation"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )

    # Scenario selection
    parser.add_argument(
        "--scenario",
        type=str,
        default="normal",
        choices=["normal", "unbalanced", "variable"],
        help="Traffic scenario to use",
    )

    # Config file paths
    parser.add_argument(
        "--config_path",
        type=str,
        default="src/config/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--hyperparams_path",
        type=str,
        default="src/config/hyperparameters.yaml",
        help="Path to hyperparameters file",
    )

    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # For multi-GPU


def create_environment(args):
    """Create the appropriate environment based on arguments."""
    if args.env_type == "single":
        env_config = {
            "max_time_steps": args.max_steps,
            "arrival_rates": {"north": 0.2, "south": 0.2, "east": 0.3, "west": 0.3},
            "reward_weights": {
                "queue_length": -1.0,
                "wait_time": -0.5,
                "throughput": 1.0,
                "switch_penalty": -2.0,
            },
            "random_seed": args.seed,
        }
        env = IntersectionEnv(config=env_config)
    elif args.env_type == "multi":
        env_config = {
            "topology": args.topology,
            "max_time_steps": args.max_steps,
            "control_mode": args.control_mode,
            "arrival_rates": {"default": 0.2, "peak": 0.4},
            "reward_weights": {
                "queue_length": -1.0,
                "wait_time": -0.5,
                "throughput": 1.0,
                "switch_penalty": -2.0,
            },
            "random_seed": args.seed,
        }
        env = TrafficMultiEnv(config=env_config)
    elif args.env_type == "sumo":
        env_config = {
            "max_time_steps": args.max_steps,
            "reward_weights": {
                "queue_length": -1.0,
                "wait_time": -0.5,
                "throughput": 1.0,
                "switch_penalty": -2.0,
            },
            "random_seed": args.seed,
        }
        env = SUMOIntersectionEnv(config_file=args.sumo_config, config=env_config)
    else:
        raise ValueError(f"Unsupported environment type: {args.env_type}")

    return env


def create_agent(args, env):
    """Create the appropriate agent based on arguments."""

    # Extract state and action dimensions from environment
    if args.env_type == "single":
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
    elif args.env_type == "multi":
        if args.control_mode == "centralized":
            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.shape[0]
        else:
            # For decentralized control, just use dimensions from one intersection
            first_id = list(env.intersections.keys())[0]
            state_dim = env.observation_spaces[first_id].shape[0]
            action_dim = env.action_spaces[first_id].n
    elif args.env_type == "sumo":
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
    else:
        raise ValueError(f"Unsupported environment type: {args.env_type}")

    # Create agent based on selected algorithm
    if args.algorithm == "dqn":
        agent_config = {
            "state_dim": state_dim,
            "action_dim": action_dim,
            "hidden_size": args.hidden_size,
            "learning_rate": args.learning_rate,
            "gamma": args.gamma,
            "epsilon_start": args.epsilon_start,
            "epsilon_end": args.epsilon_end,
            "epsilon_decay": args.epsilon_decay,
            "buffer_size": 10000,
            "batch_size": args.batch_size,
            "target_update_freq": args.target_update,
        }
        agent = DQNAgent(**agent_config)
    elif args.algorithm == "qlearning":
        agent_config = {
            "state_dim": state_dim,
            "action_dim": action_dim,
            "learning_rate": args.learning_rate,
            "gamma": args.gamma,
            "epsilon_start": args.epsilon_start,
            "epsilon_end": args.epsilon_end,
            "epsilon_decay": args.epsilon_decay,
            "discretization": 5,  # Number of buckets for continuous state variables
        }
        agent = QLearningAgent(**agent_config)
    elif args.algorithm == "a2c":
        agent_config = {
            "state_dim": state_dim,
            "action_dim": action_dim,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "lr": args.learning_rate,
            "gamma": args.gamma,
            "gae_lambda": 0.95,
            "entropy_coef": 0.01,
            "value_coef": 0.5,
            "max_grad_norm": 1.0,
            "hidden_sizes": [256, 128, 64],
            "checkpoint_dir": os.path.join(args.save_dir, "checkpoints")
        }
        agent = A2CAgent(**agent_config)
    elif args.algorithm == "ppo":
        agent_config = {
            "state_dim": state_dim,
            "action_dim": action_dim,
            "hidden_size": args.hidden_size,
            "learning_rate": args.learning_rate,
            "gamma": args.gamma,
            "clip_ratio": 0.2,
            "value_coef": 0.5,
            "entropy_coef": 0.01,
            "gae_lambda": 0.95,
            "batch_size": args.batch_size,
        }
        agent = PPOAgent(**agent_config)
    elif args.algorithm == "fixed":
        agent_config = {
            "cycle_length": 60,
            "green_splits": [0.5, 0.5],  # Equal time for N-S and E-W
            "yellow_time": 3,
            "min_phase_time": 5,
        }
        agent = FixedTimingController(**agent_config)
    elif args.algorithm == "actuated":
        agent_config = {
            "min_green": 5,
            "max_green": 30,
            "extension_time": 2,
            "gap_threshold": 2,
            "yellow_time": 3,
        }
        agent = ActuatedController(**agent_config)
    elif args.algorithm == "webster":
        agent_config = {
            "saturation_flow": 1800,
            "lost_time_per_phase": 2,
            "min_cycle": 30,
            "max_cycle": 120,
        }
        agent = WebsterController(**agent_config)
    else:
        raise ValueError(f"Unsupported algorithm: {args.algorithm}")

    return agent


def setup_logging(args):
    """Set up directories and logging for the training run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.algorithm}_{args.env_type}_{args.scenario}_{timestamp}"

    log_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(log_dir, exist_ok=True)
    
    # Create checkpoints directory
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save configuration
    with open(os.path.join(log_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    # Initialize logger
    logger = Logger(log_dir=log_dir)

    return log_dir, logger, checkpoint_dir


def evaluate(agent, env, num_episodes=5):
    """
    Evaluate agent performance without exploration.

    Args:
        agent: The agent to evaluate
        env: The environment to evaluate on
        num_episodes: Number of evaluation episodes

    Returns:
        Dictionary containing evaluation metrics
    """
    rewards = []
    throughputs = []
    queue_lengths = []

    for _ in range(num_episodes):
        # Reset environment
        state, _ = env.reset()
        episode_reward = 0
        episode_throughput = 0
        episode_queue_sum = 0
        step_count = 0
        done = False
        truncated = False

        # Disable exploration for RL agents
        if hasattr(agent, "epsilon"):
            old_epsilon = agent.epsilon
            agent.epsilon = 0.0

        while not (done or truncated):
            # Select action without exploration
            action = agent.act(state)

            # Take action
            next_state, reward, done, truncated, info = env.step(action)

            # Update metrics
            episode_reward += reward
            episode_throughput += info.get("throughput", 0)
            episode_queue_sum += info.get(
                "total_queue", sum(state[:4])
            )  # Assuming first 4 elements are queue lengths
            step_count += 1

            # Update state
            state = next_state

        # Restore exploration rate for RL agents
        if hasattr(agent, "epsilon"):
            agent.epsilon = old_epsilon

        # Store episode metrics
        rewards.append(episode_reward)
        throughputs.append(episode_throughput)
        queue_lengths.append(episode_queue_sum / step_count if step_count > 0 else 0)

    # Calculate averages
    avg_reward = sum(rewards) / num_episodes
    avg_throughput = sum(throughputs) / num_episodes
    avg_queue_length = sum(queue_lengths) / num_episodes

    return {
        "reward": avg_reward,
        "throughput": avg_throughput,
        "queue_length": avg_queue_length,
    }


def train(args):
    """Main training loop."""
    # Set random seed
    set_seed(args.seed)

    # Create environment
    env = create_environment(args)

    # Setup logging
    log_dir, logger, checkpoint_dir = setup_logging(args)
    print(f"Logs will be saved to: {log_dir}")

    # Update checkpoint directory in agent config if needed
    if args.algorithm == "a2c":
        agent_config = {
            "state_dim": env.observation_space.shape[0],
            "action_dim": env.action_space.n,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "lr": args.learning_rate,
            "gamma": args.gamma,
            "gae_lambda": 0.95,
            "entropy_coef": 0.01,
            "value_coef": 0.5,
            "max_grad_norm": 1.0,
            "hidden_sizes": [256, 128, 64],
            "checkpoint_dir": checkpoint_dir
        }
        agent = A2CAgent(**agent_config)
    else:
        agent = create_agent(args, env)

    # Training metrics
    best_reward = float("-inf")
    episode_rewards = []
    episode_throughputs = []
    episode_queue_lengths = []

    # For baseline (non-learning) controllers, we only need to evaluate them
    if args.algorithm in ["fixed", "actuated", "webster"]:
        print(f"Evaluating baseline controller: {args.algorithm}")
        eval_metrics = evaluate(agent, env, num_episodes=args.episodes)

        print(
            f"Evaluation metrics: Reward={eval_metrics['reward']:.2f}, "
            f"Throughput={eval_metrics['throughput']:.2f}, "
            f"Avg Queue Length={eval_metrics['queue_length']:.2f}"
        )

        # Log metrics
        logger.log_metrics(
            {
                "eval_reward": eval_metrics["reward"],
                "eval_throughput": eval_metrics["throughput"],
                "eval_queue_length": eval_metrics["queue_length"],
            },
            step=0,
        )

        return agent, log_dir

    # Training loop for RL agents
    print(f"Starting training with {args.algorithm} for {args.episodes} episodes...")
    progress_bar = tqdm(range(args.episodes), desc="Training")

    for episode in progress_bar:
        # Reset environment
        state, _ = env.reset()
        episode_reward = 0
        episode_throughput = 0
        episode_queue_sum = 0
        step_count = 0

        # Episode loop
        done = False
        truncated = False
        
        # Lists to store episode experiences
        states, actions, rewards, next_states, dones = [], [], [], [], []

        while not (done or truncated):
            # Select action
            action = agent.act(state)

            # Take action
            next_state, reward, done, truncated, info = env.step(action)

            # Store experience
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

            # Update metrics
            episode_reward += reward
            episode_throughput += info.get("throughput", 0)
            episode_queue_sum += info.get("total_queue", sum(state[:4]))
            step_count += 1

            # Update state
            state = next_state

            # For A2C, update after every step
            if args.algorithm == "a2c":
                agent.update(states, actions, rewards, next_states, dones)
                # Clear experience lists
                states, actions, rewards, next_states, dones = [], [], [], [], []
            else:
                # For other agents, use their learn method
                agent.learn(state, action, reward, next_state, done)

        # Calculate average queue length
        avg_queue_length = episode_queue_sum / step_count if step_count > 0 else 0

        # Store episode metrics
        episode_rewards.append(episode_reward)
        episode_throughputs.append(episode_throughput)
        episode_queue_lengths.append(avg_queue_length)

        # Update progress bar
        progress_bar.set_postfix(
            {
                "reward": f"{episode_reward:.2f}",
                "throughput": f"{episode_throughput}",
                "queue": f"{avg_queue_length:.2f}",
            }
        )

        # Log metrics
        logger.log_metrics(
            {
                "episode_reward": episode_reward,
                "episode_throughput": episode_throughput,
                "episode_queue_length": avg_queue_length,
                "epsilon": agent.epsilon if hasattr(agent, "epsilon") else 0,
            },
            step=episode,
        )

        # Periodic evaluation
        if (episode + 1) % args.eval_interval == 0:
            eval_metrics = evaluate(agent, env)

            # Log evaluation metrics
            logger.log_metrics(
                {
                    "eval_reward": eval_metrics["reward"],
                    "eval_throughput": eval_metrics["throughput"],
                    "eval_queue_length": eval_metrics["queue_length"],
                },
                step=episode,
            )

            # Update best model if current is better
            if eval_metrics["reward"] > best_reward:
                best_reward = eval_metrics["reward"]
                # Save model if it has a save method
                if hasattr(agent, "save"):
                    agent.save(os.path.join(checkpoint_dir, "best_model.pt"))
                    print(
                        f"Episode {episode+1}: New best model saved with reward {best_reward:.2f}"
                    )

        # Save model checkpoint
        if (episode + 1) % args.save_interval == 0:
            agent.save(os.path.join(checkpoint_dir, f"model_episode_{episode+1}.pt"))

        # Save training curves
        plt.figure(figsize=(15, 5))

        # Reward plot
        plt.subplot(1, 3, 1)
        plt.plot(episode_rewards)
        plt.title("Episode Rewards")
        plt.xlabel("Episode")
        plt.ylabel("Reward")

        # Throughput plot
        plt.subplot(1, 3, 2)
        plt.plot(episode_throughputs)
        plt.title("Episode Throughput")
        plt.xlabel("Episode")
        plt.ylabel("Throughput")

        # Queue length plot
        plt.subplot(1, 3, 3)
        plt.plot(episode_queue_lengths)
        plt.title("Average Queue Length")
        plt.xlabel("Episode")
        plt.ylabel("Queue Length")

        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, f"training_curves_ep{episode+1}.png"))
        plt.close()

    # Save final model
    agent.save(os.path.join(checkpoint_dir, "final_model.pt"))

    # Final evaluation
    final_metrics = evaluate(agent, env, num_episodes=10)

    # Log final metrics
    logger.log_metrics(
        {
            "final_eval_reward": final_metrics["reward"],
            "final_eval_throughput": final_metrics["throughput"],
            "final_eval_queue_length": final_metrics["queue_length"],
        },
        step=args.episodes,
    )

    # Save final training curves
    plt.figure(figsize=(15, 5))

    # Reward plot
    plt.subplot(1, 3, 1)
    plt.plot(episode_rewards)
    plt.title("Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")

    # Throughput plot
    plt.subplot(1, 3, 2)
    plt.plot(episode_throughputs)
    plt.title("Episode Throughput")
    plt.xlabel("Episode")
    plt.ylabel("Throughput")

    # Queue length plot
    plt.subplot(1, 3, 3)
    plt.plot(episode_queue_lengths)
    plt.title("Average Queue Length")
    plt.xlabel("Episode")
    plt.ylabel("Queue Length")

    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "final_training_curves.png"))
    plt.close()

    print(f"Training completed. Final models and logs saved to {log_dir}")
    print(
        f"Final evaluation metrics: Reward={final_metrics['reward']:.2f}, "
        f"Throughput={final_metrics['throughput']:.2f}, "
        f"Avg Queue Length={final_metrics['queue_length']:.2f}"
    )

    return agent, log_dir


if __name__ == "__main__":
    args = parse_args()
    train(args)
