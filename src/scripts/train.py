#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Training Script for Traffic Signal Control

This script trains reinforcement learning agents on the traffic signal control
environments. It supports different algorithms and environment configurations.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
from datetime import datetime
import random
import json
from tqdm import tqdm
import gymnasium as gym

# Add src directory to path
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Import environments
from environments.intersection_env import IntersectionEnv
from environments.traffic_env import TrafficMultiEnv
from environments.sumo_env import SUMOIntersectionEnv

# Import agent implementations (will be implemented later)
from agents.dqn_agent import DQNAgent

# from agents.a2c_agent import A2CAgent
# from agents.ppo_agent import PPOAgent

# Import utilities
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
        choices=["qlearning", "dqn", "a2c", "ppo"],
        help="RL algorithm to use",
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
        # Q-learning implementation will be added
        raise NotImplementedError("Q-learning agent not implemented yet")
    elif args.algorithm == "a2c":
        # A2C implementation will be added
        raise NotImplementedError("A2C agent not implemented yet")
    elif args.algorithm == "ppo":
        # PPO implementation will be added
        raise NotImplementedError("PPO agent not implemented yet")
    else:
        raise ValueError(f"Unsupported algorithm: {args.algorithm}")

    return agent


def setup_logging(args):
    """Set up directories and logging for the training run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.algorithm}_{args.env_type}_{timestamp}"

    log_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(log_dir, exist_ok=True)

    # Save configuration
    with open(os.path.join(log_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    # Initialize logger
    logger = Logger(log_dir=log_dir)

    return log_dir, logger


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

        # Disable exploration
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

        # Restore exploration rate
        agent.epsilon = old_epsilon

        # Store episode metrics
        rewards.append(episode_reward)
        throughputs.append(episode_throughput)
        queue_lengths.append(episode_queue_sum / step_count)

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

    # Create agent
    agent = create_agent(args, env)

    # Setup logging
    log_dir, logger = setup_logging(args)
    print(f"Logs will be saved to: {log_dir}")

    # Training metrics
    best_reward = float("-inf")
    episode_rewards = []
    episode_throughputs = []
    episode_queue_lengths = []

    # Training loop
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

        while not (done or truncated):
            # Select action
            action = agent.act(state)

            # Take action
            next_state, reward, done, truncated, info = env.step(action)

            # Store experience and learn
            agent.learn(state, action, reward, next_state, done)

            # Update metrics
            episode_reward += reward
            episode_throughput += info.get("throughput", 0)
            episode_queue_sum += info.get(
                "total_queue", sum(state[:4])
            )  # Assuming first 4 elements are queue lengths
            step_count += 1

            # Update state
            state = next_state

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
                "epsilon": agent.epsilon,
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
                agent.save(os.path.join(log_dir, "best_model.pt"))
                print(
                    f"Episode {episode+1}: New best model saved with reward {best_reward:.2f}"
                )

        # Periodic saving
        if (episode + 1) % args.save_interval == 0:
            agent.save(os.path.join(log_dir, f"model_ep{episode+1}.pt"))

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
    agent.save(os.path.join(log_dir, "final_model.pt"))

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
