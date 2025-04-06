#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Test script for SUMO Environment"""

from pathlib import Path

import argparse
import sys
import random
import numpy as np
import matplotlib.pyplot as plt


# Ensure the src directory is in the Python path
script_dir = Path(__file__).resolve().parent
src_dir = script_dir.parent
sys.path.insert(0, str(src_dir))

from environments import SUMOIntersectionEnv
from agents.baseline_controllers import FixedTimingController


def run_random_agent(config_file, episodes=3, steps_per_episode=1000, render_mode=None):
    """Run a random agent in the SUMO environment and collect statistics.

    Args:
        config_file: Path to SUMO configuration file
        episodes: Number of episodes to run
        steps_per_episode: Maximum steps per episode
        render_mode: Rendering mode (None, 'human')
    """
    # Create environment with custom configuration
    env_config = {
        "max_time_steps": steps_per_episode,
        "reward_weights": {
            "queue_length": -0.5,
            "wait_time": -0.2,
            "throughput": 2.0,
            "switch_penalty": -1.0,
        },
    }

    env = SUMOIntersectionEnv(
        config_file=config_file, render_mode=render_mode, config=env_config
    )

    # Statistics
    episode_rewards = []
    episode_queue_lengths = []
    episode_waiting_times = []
    episode_throughputs = []

    for episode in range(episodes):
        print(f"Episode {episode+1}/{episodes}")
        observation, info = env.reset()

        episode_reward = 0
        episode_queues = []
        episode_waits = []
        episode_through = 0

        for step in range(steps_per_episode):
            # Random action: 0 = keep current phase, 1 = switch phase
            action = random.randint(0, 1)

            # Take step in environment
            observation, reward, terminated, truncated, info = env.step(action)

            # Update statistics
            episode_reward += reward
            episode_queues.append(sum(info["queue_lengths"].values()))
            avg_wait = sum(info["waiting_times"].values()) / len(info["waiting_times"])
            episode_waits.append(avg_wait)

            # Print progress every 100 steps
            if step % 100 == 0:
                print(
                    f"  Step {step}/{steps_per_episode}, "
                    f"Reward: {reward:.2f}, "
                    f"Queue: {sum(info['queue_lengths'].values())}, "
                    f"Phase: {info['phase']}"
                )

            if terminated or truncated:
                break

        # Collect episode statistics
        episode_rewards.append(episode_reward)
        episode_queue_lengths.append(np.mean(episode_queues))
        episode_waiting_times.append(np.mean(episode_waits))
        episode_throughputs.append(info["cumulative_throughput"])

        print(
            f"Episode {episode+1} complete. "
            f"Reward: {episode_reward:.2f}, "
            f"Avg Queue: {np.mean(episode_queues):.2f}, "
            f"Avg Wait: {np.mean(episode_waits):.2f}, "
            f"Throughput: {info['cumulative_throughput']}"
        )

    # Close environment
    env.close()

    # Return collected statistics
    return {
        "rewards": episode_rewards,
        "queue_lengths": episode_queue_lengths,
        "waiting_times": episode_waiting_times,
        "throughputs": episode_throughputs,
    }


def run_fixed_time_controller(
    config_file, episodes=3, steps_per_episode=1000, render_mode=None
):
    """Run a fixed-time controller in the SUMO environment and collect statistics.

    Args:
        config_file: Path to SUMO configuration file
        episodes: Number of episodes to run
        steps_per_episode: Maximum steps per episode
        render_mode: Rendering mode (None, 'human')
    """
    # Create environment with custom configuration
    env_config = {
        "max_time_steps": steps_per_episode,
    }

    env = SUMOIntersectionEnv(
        config_file=config_file, render_mode=render_mode, config=env_config
    )

    # Statistics
    episode_rewards = []
    episode_queue_lengths = []
    episode_waiting_times = []
    episode_throughputs = []

    for episode in range(episodes):
        print(f"Episode {episode+1}/{episodes}")
        observation, info = env.reset()

        # Create a fresh controller for each episode instead of resetting
        controller = FixedTimingController(
            cycle_length=60, green_splits=[0.5, 0.5], yellow_time=3
        )

        episode_reward = 0
        episode_queues = []
        episode_waits = []
        episode_through = 0

        for step in range(steps_per_episode):
            # Get action from controller
            action = controller.act(observation)

            # Take step in environment
            observation, reward, terminated, truncated, info = env.step(action)

            # Update statistics
            episode_reward += reward
            episode_queues.append(sum(info["queue_lengths"].values()))
            avg_wait = sum(info["waiting_times"].values()) / len(info["waiting_times"])
            episode_waits.append(avg_wait)

            # Print progress every 100 steps
            if step % 100 == 0:
                print(
                    f"  Step {step}/{steps_per_episode}, "
                    f"Reward: {reward:.2f}, "
                    f"Queue: {sum(info['queue_lengths'].values())}, "
                    f"Phase: {info['phase']}"
                )

            if terminated or truncated:
                break

        # Collect episode statistics
        episode_rewards.append(episode_reward)
        episode_queue_lengths.append(np.mean(episode_queues))
        episode_waiting_times.append(np.mean(episode_waits))
        episode_throughputs.append(info["cumulative_throughput"])

        print(
            f"Episode {episode+1} complete. "
            f"Reward: {episode_reward:.2f}, "
            f"Avg Queue: {np.mean(episode_queues):.2f}, "
            f"Avg Wait: {np.mean(episode_waits):.2f}, "
            f"Throughput: {info['cumulative_throughput']}"
        )

    # Close environment
    env.close()

    # Return collected statistics
    return {
        "rewards": episode_rewards,
        "queue_lengths": episode_queue_lengths,
        "waiting_times": episode_waiting_times,
        "throughputs": episode_throughputs,
    }


def plot_comparison(random_stats, fixed_stats):
    """Plot a comparison of performance between random and fixed-time controllers.

    Args:
        random_stats: Statistics from random agent
        fixed_stats: Statistics from fixed-time controller
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Create x values (episode numbers) starting from 1
    episodes = list(range(1, len(random_stats["rewards"]) + 1))

    # Plot rewards
    axes[0, 0].plot(episodes, random_stats["rewards"], "o-", label="Random")
    axes[0, 0].plot(episodes, fixed_stats["rewards"], "o-", label="Fixed-Time")
    axes[0, 0].set_title("Total Reward per Episode")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].legend()

    # Plot queue lengths
    axes[0, 1].plot(episodes, random_stats["queue_lengths"], "o-", label="Random")
    axes[0, 1].plot(episodes, fixed_stats["queue_lengths"], "o-", label="Fixed-Time")
    axes[0, 1].set_title("Average Queue Length per Episode")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Queue Length")
    axes[0, 1].legend()

    # Plot waiting times
    axes[1, 0].plot(episodes, random_stats["waiting_times"], "o-", label="Random")
    axes[1, 0].plot(episodes, fixed_stats["waiting_times"], "o-", label="Fixed-Time")
    axes[1, 0].set_title("Average Waiting Time per Episode")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Waiting Time (s)")
    axes[1, 0].legend()

    # Plot throughputs
    axes[1, 1].plot(episodes, random_stats["throughputs"], "o-", label="Random")
    axes[1, 1].plot(episodes, fixed_stats["throughputs"], "o-", label="Fixed-Time")
    axes[1, 1].set_title("Total Throughput per Episode")
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("Throughput (vehicles)")
    axes[1, 1].legend()

    # Add a summary table for average values
    plt.figtext(
        0.5,
        0.01,
        f"Summary (averages):\n"
        f"Random - Reward: {np.mean(random_stats['rewards']):.2f}, "
        f"Queue: {np.mean(random_stats['queue_lengths']):.2f}, "
        f"Wait: {np.mean(random_stats['waiting_times']):.2f}, "
        f"Throughput: {np.mean(random_stats['throughputs']):.2f}\n"
        f"Fixed - Reward: {np.mean(fixed_stats['rewards']):.2f}, "
        f"Queue: {np.mean(fixed_stats['queue_lengths']):.2f}, "
        f"Wait: {np.mean(fixed_stats['waiting_times']):.2f}, "
        f"Throughput: {np.mean(fixed_stats['throughputs']):.2f}",
        ha="center",
        fontsize=12,
        bbox=dict(facecolor="white", alpha=0.5),
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for the text
    plt.savefig("controller_comparison.png")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Test SUMO Environment with simple controllers"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="data/simulation/networks/single_intersection.sumocfg",
        help="Path to SUMO configuration file",
    )
    parser.add_argument(
        "--episodes", type=int, default=3, help="Number of episodes to run"
    )
    parser.add_argument(
        "--steps", type=int, default=1000, help="Maximum steps per episode"
    )
    parser.add_argument(
        "--render", action="store_true", help="Enable rendering (SUMO-GUI)"
    )

    args = parser.parse_args()

    render_mode = "human" if args.render else None

    print("\n=== Running Random Agent ===")
    random_stats = run_random_agent(
        config_file=args.config,
        episodes=args.episodes,
        steps_per_episode=args.steps,
        render_mode=render_mode,
    )

    print("\n=== Running Fixed-Time Controller ===")
    fixed_stats = run_fixed_time_controller(
        config_file=args.config,
        episodes=args.episodes,
        steps_per_episode=args.steps,
        render_mode=render_mode,
    )

    # Plot comparison
    plot_comparison(random_stats, fixed_stats)


if __name__ == "__main__":
    main()
