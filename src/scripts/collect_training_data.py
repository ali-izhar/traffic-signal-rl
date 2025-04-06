#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Training Data Collection Script"""

import os
import sys
import argparse
import numpy as np
import random

# Add src directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import environments
from environments.sumo_env import SUMOIntersectionEnv
from environments.intersection_env import IntersectionEnv
from scripts.data_collection_wrapper import DataCollectionWrapper


def collect_data_with_random_agent(env, num_episodes=10):
    """Collect data using a random agent.

    Args:
        env: The wrapped environment
        num_episodes: Number of episodes to run
    """
    # Run episodes
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0

        while not (done or truncated):
            # Select random action
            action = env.action_space.sample()

            # Take action in environment
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward

            # Print progress occasionally
            if env.step_count % 100 == 0:
                print(
                    f"Episode {episode+1}, Step {env.step_count}, Reward: {reward:.2f}"
                )

        print(f"Episode {episode+1} complete. Total reward: {episode_reward:.2f}")

    # Make sure all data is saved
    env.close()


def setup_sumo_env(config_file, save_dir, save_format, experiment_name, seed=None):
    """Set up the SUMO environment with data collection wrapper.

    Args:
        config_file: Path to SUMO configuration file
        save_dir: Directory to save data
        save_format: Format for saving data
        experiment_name: Name for the experiment
        seed: Random seed

    Returns:
        Wrapped environment
    """
    # Create base environment
    base_env = SUMOIntersectionEnv(config_file=config_file)

    # Wrap with data collection
    wrapped_env = DataCollectionWrapper(
        env=base_env,
        save_dir=save_dir,
        save_format=save_format,
        collection_frequency=1,  # Save after each episode
        episode_buffer_size=5,  # Buffer up to 5 episodes
        experiment_name=experiment_name,
        seed=seed,
    )

    return wrapped_env


def setup_simple_env(save_dir, save_format, experiment_name, seed=None):
    """Set up the simplified environment with data collection wrapper.

    Args:
        save_dir: Directory to save data
        save_format: Format for saving data
        experiment_name: Name for the experiment
        seed: Random seed

    Returns:
        Wrapped environment
    """
    # Create base environment
    config = {
        "max_time_steps": 500,
        "arrival_rates": {"north": 0.2, "south": 0.2, "east": 0.3, "west": 0.3},
        "random_seed": seed,
    }
    base_env = IntersectionEnv(config=config)

    # Wrap with data collection
    wrapped_env = DataCollectionWrapper(
        env=base_env,
        save_dir=save_dir,
        save_format=save_format,
        collection_frequency=1,  # Save after each episode
        episode_buffer_size=5,  # Buffer up to 5 episodes
        experiment_name=experiment_name,
        seed=seed,
    )

    return wrapped_env


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Collect training data from RL environments"
    )
    parser.add_argument(
        "--env_type",
        type=str,
        choices=["sumo", "simple"],
        default="simple",
        help="Environment type to use",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="src/data/simulation/networks/low_traffic.sumocfg",
        help="Path to SUMO configuration file",
    )
    parser.add_argument(
        "--save_dir", type=str, default="processed", help="Directory to save data"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["csv", "hdf5", "json"],
        default="csv",
        help="Format for saving data",
    )
    parser.add_argument(
        "--episodes", type=int, default=5, help="Number of episodes to run"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    args = parser.parse_args()

    # Set random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)

    # Create experiment name
    experiment_name = f"{args.env_type}_data_{args.format}"

    # Set up environment based on type
    if args.env_type == "sumo":
        env = setup_sumo_env(
            config_file=args.config_file,
            save_dir=args.save_dir,
            save_format=args.format,
            experiment_name=experiment_name,
            seed=args.seed,
        )
        print(f"Created SUMO environment with configuration: {args.config_file}")
    else:
        env = setup_simple_env(
            save_dir=args.save_dir,
            save_format=args.format,
            experiment_name=experiment_name,
            seed=args.seed,
        )
        print("Created simplified environment")

    # Collect data
    print(f"Starting data collection for {args.episodes} episodes...")
    collect_data_with_random_agent(env, args.episodes)

    print(
        f"Data collection complete. Data saved to: {os.path.join(os.getcwd(), args.save_dir, experiment_name)}"
    )


if __name__ == "__main__":
    main()
