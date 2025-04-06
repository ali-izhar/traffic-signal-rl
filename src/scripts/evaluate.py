#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Evaluation Script for Traffic Signal Control"""

from datetime import datetime
from tqdm import tqdm

import os
import sys
import argparse
import yaml
import json

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

# Add src directory to path
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Import environments
from environments.intersection_env import IntersectionEnv
from environments.traffic_env import TrafficMultiEnv
from environments.sumo_env import SUMOIntersectionEnv

# Import agents
from agents.dqn_agent import DQNAgent
from agents.a2c_agent import A2CAgent
from agents.ppo_agent import PPOAgent
from agents.qlearning_agent import QLearningAgent
from agents.baseline_controllers import (
    FixedTimingController,
    ActuatedController,
    WebsterController,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate traffic signal control methods"
    )

    # Environment settings
    parser.add_argument(
        "--env_type",
        type=str,
        default="single",
        choices=["single", "multi", "sumo"],
        help="Environment type: single intersection, multi-intersection, or SUMO",
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
    parser.add_argument(
        "--scenario",
        type=str,
        default="normal",
        choices=["normal", "unbalanced", "variable"],
        help="Traffic scenario to evaluate",
    )

    # Agent settings
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["dqn", "a2c", "ppo", "qlearning", "fixed", "actuated", "webster"],
        help="Methods to evaluate (space-separated list)",
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default="../logs",
        help="Directory containing trained models",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Direct path to model file (overrides models_dir)",
    )

    # Evaluation parameters
    parser.add_argument(
        "--episodes", type=int, default=10, help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--max_steps", type=int, default=1000, help="Maximum steps per episode"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--render", action="store_true", help="Render the environment during evaluation"
    )
    parser.add_argument(
        "--output_dir", type=str, default="../results", help="Directory to save results"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file",
    )

    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def create_environment(args, config):
    """Create the appropriate environment based on arguments and config."""
    if args.env_type == "single":
        env_config = config["environments"]["single"]

        # Update arrival rates based on scenario
        scenario_config = config["scenarios"][args.scenario]
        if args.scenario in ["normal", "unbalanced"]:
            env_config["arrival_rates"] = scenario_config["arrival_rates"]

        # For variable scenario, start with the first phase
        if args.scenario == "variable":
            env_config["arrival_rates"] = scenario_config["phases"][0]["arrival_rates"]
            env_config["variable_demand"] = True
            env_config["demand_phases"] = scenario_config["phases"]

        env = IntersectionEnv(config=env_config)
    elif args.env_type == "sumo":
        # SUMO environment uses a different initialization
        env_config = {
            "max_time_steps": args.max_steps,
            "reward_weights": {
                "queue_length": -1.0,
                "wait_time": -0.5,
                "throughput": 1.0,
                "switch_penalty": -2.0,
            },
        }

        env = SUMOIntersectionEnv(
            config_file=args.sumo_config,
            render_mode="human" if args.render else None,
            config=env_config,
        )
    else:
        env_config = config["environments"]["multi"]
        env_config["topology"] = args.topology
        env_config["control_mode"] = args.control_mode

        # Update arrival rates based on scenario
        scenario_config = config["scenarios"][args.scenario]
        if args.scenario in ["normal", "unbalanced"]:
            env_config["arrival_rates"] = scenario_config["arrival_rates"]

        # For variable scenario, start with the first phase
        if args.scenario == "variable":
            env_config["arrival_rates"] = scenario_config["phases"][0]["arrival_rates"]
            env_config["variable_demand"] = True
            env_config["demand_phases"] = scenario_config["phases"]

        env = TrafficMultiEnv(config=env_config)

    return env


def load_agent(method, env, args, config):
    """Load an agent based on method name."""
    state_dim = env.observation_space.shape[0]

    if hasattr(env, "action_space"):
        action_dim = env.action_space.n
    else:
        # For multi-intersection with decentralized control, get from first intersection
        first_id = list(env.intersections.keys())[0]
        action_dim = env.action_spaces[first_id].n

    # Path to model checkpoint
    if method in ["dqn", "a2c", "ppo", "qlearning"]:
        model_path = args.model_path or os.path.join(
            args.models_dir, f"{method}_{args.env_type}", "best_model.pt"
        )

    # Load appropriate agent
    if method == "dqn":
        agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)
        try:
            agent.load(model_path)
        except FileNotFoundError:
            print(f"Warning: Model file {model_path} not found. Using untrained agent.")

    elif method == "a2c":
        agent = A2CAgent(state_dim=state_dim, action_dim=action_dim)
        try:
            agent.load(model_path)
        except FileNotFoundError:
            print(f"Warning: Model file {model_path} not found. Using untrained agent.")

    elif method == "ppo":
        agent = PPOAgent(state_dim=state_dim, action_dim=action_dim)
        try:
            agent.load(model_path)
        except FileNotFoundError:
            print(f"Warning: Model file {model_path} not found. Using untrained agent.")

    elif method == "qlearning":
        agent = QLearningAgent(state_dim=state_dim, action_dim=action_dim)
        try:
            agent.load(model_path)
        except FileNotFoundError:
            print(f"Warning: Model file {model_path} not found. Using untrained agent.")

    elif method == "fixed":
        # Get fixed timing params from config
        fixed_params = config["baselines"][0]["parameters"]
        agent = FixedTimingController(
            cycle_length=fixed_params["cycle_length"],
            green_splits=fixed_params["green_splits"],
        )

    elif method == "actuated":
        # Get actuated params from config
        actuated_params = config["baselines"][1]["parameters"]
        agent = ActuatedController(
            min_green=actuated_params["min_green"],
            max_green=actuated_params["max_green"],
            extension_time=actuated_params["extension_time"],
            gap_threshold=actuated_params["gap_threshold"],
        )

    elif method == "webster":
        agent = WebsterController()

    else:
        raise ValueError(f"Unsupported method: {method}")

    return agent


def evaluate_agent(agent, env, args, config):
    """Evaluate an agent on the environment.

    Args:
        agent: The agent to evaluate
        env: The environment to evaluate on
        args: Command line arguments
        config: Configuration dictionary

    Returns:
        Dictionary containing evaluation metrics
    """
    # Metrics to track
    episode_rewards = []
    episode_waiting_times = []
    episode_queue_lengths = []
    episode_throughputs = []
    episode_emissions = []
    episode_travel_times = []
    episode_switches = []

    # Run evaluation episodes
    for episode in range(args.episodes):
        # Reset environment
        state, _ = env.reset()
        episode_reward = 0
        step_waiting_times = []
        step_queue_lengths = []
        step_throughputs = []
        switch_count = 0
        prev_action = None

        # Episode loop
        done = False
        truncated = False

        # Progress tracking
        pbar = tqdm(total=args.max_steps, desc=f"Episode {episode+1}/{args.episodes}")

        while not (done or truncated):
            # Select action (without exploration)
            if hasattr(agent, "epsilon"):
                # For DQN-style agents, temporarily set epsilon to 0
                old_epsilon = agent.epsilon
                agent.epsilon = 0.0
                action = agent.act(state)
                agent.epsilon = old_epsilon
            elif (
                hasattr(agent, "act")
                and "deterministic" in agent.act.__code__.co_varnames
            ):
                # For policy gradient agents with deterministic parameter
                action = agent.act(state, deterministic=True)
            else:
                # For other controllers
                action = agent.act(state)

            # Count phase switches
            if prev_action is not None and action != prev_action and action == 1:
                switch_count += 1
            prev_action = action

            # Take action
            next_state, reward, done, truncated, info = env.step(action)

            # Render if requested
            if args.render:
                env.render()

            # Update metrics
            episode_reward += reward

            # Extract metrics from state and info
            # Queue lengths are typically the first 4 elements of the state
            current_queue = (
                sum(state[:4]) if len(state) >= 4 else info.get("total_queue", 0)
            )
            step_queue_lengths.append(current_queue)

            # Waiting times are typically the next 4 elements after queue lengths
            current_waiting = (
                sum(state[4:8])
                if len(state) >= 8
                else info.get("total_waiting_time", 0)
            )
            step_waiting_times.append(current_waiting)

            # Throughput is typically in the info dict
            current_throughput = info.get("throughput", 0)
            step_throughputs.append(current_throughput)

            # Update state
            state = next_state

            # Update progress bar
            pbar.update(1)

            # Break if max steps reached
            if len(step_queue_lengths) >= args.max_steps:
                break

        pbar.close()

        # Calculate episode metrics
        avg_waiting_time = np.mean(step_waiting_times)
        avg_queue_length = np.mean(step_queue_lengths)
        total_throughput = np.sum(step_throughputs)

        # Store episode metrics
        episode_rewards.append(episode_reward)
        episode_waiting_times.append(avg_waiting_time)
        episode_queue_lengths.append(avg_queue_length)
        episode_throughputs.append(total_throughput)
        episode_switches.append(switch_count)

        # Estimate emissions and travel time (simplified)
        # In a real implementation, these would be calculated from detailed vehicle data
        estimated_emissions = avg_queue_length * 10  # g CO2 (simplified model)
        estimated_travel_time = avg_waiting_time + 30  # seconds (simplified model)

        episode_emissions.append(estimated_emissions)
        episode_travel_times.append(estimated_travel_time)

        print(f"Episode {episode+1} metrics:")
        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Avg Queue Length: {avg_queue_length:.2f}")
        print(f"  Avg Waiting Time: {avg_waiting_time:.2f}")
        print(f"  Total Throughput: {total_throughput}")
        print(f"  Signal Switches: {switch_count}")
        print()

    # Calculate overall metrics
    metrics = {
        "reward": {"mean": np.mean(episode_rewards), "std": np.std(episode_rewards)},
        "waiting_time": {
            "mean": np.mean(episode_waiting_times),
            "std": np.std(episode_waiting_times),
        },
        "queue_length": {
            "mean": np.mean(episode_queue_lengths),
            "std": np.std(episode_queue_lengths),
        },
        "throughput": {
            "mean": np.mean(episode_throughputs),
            "std": np.std(episode_throughputs),
        },
        "emissions": {
            "mean": np.mean(episode_emissions),
            "std": np.std(episode_emissions),
        },
        "travel_time": {
            "mean": np.mean(episode_travel_times),
            "std": np.std(episode_travel_times),
        },
        "switches": {
            "mean": np.mean(episode_switches),
            "std": np.std(episode_switches),
        },
        "raw_data": {
            "rewards": episode_rewards,
            "waiting_times": episode_waiting_times,
            "queue_lengths": episode_queue_lengths,
            "throughputs": episode_throughputs,
            "emissions": episode_emissions,
            "travel_times": episode_travel_times,
            "switches": episode_switches,
        },
    }

    return metrics


def save_results(results, args):
    """Save evaluation results to files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(
        args.output_dir, f"eval_{args.env_type}_{args.scenario}_{timestamp}"
    )
    os.makedirs(results_dir, exist_ok=True)

    # Save configuration
    with open(os.path.join(results_dir, "evaluation_config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    # Helper function to convert numpy types to native Python types
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj

    # Convert results to serializable format
    serializable_results = convert_to_serializable(results)

    # Save results as JSON
    with open(os.path.join(results_dir, "results.json"), "w") as f:
        json.dump(serializable_results, f, indent=4)

    # Create summary CSV
    summary_data = []
    for method, metrics in results.items():
        row = {"method": method}
        for metric, values in metrics.items():
            if metric != "raw_data":
                row[f"{metric}_mean"] = float(values["mean"])
                row[f"{metric}_std"] = float(values["std"])
        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(results_dir, "summary.csv"), index=False)

    # Create plots
    create_comparison_plots(results, results_dir, args)

    print(f"Results saved to {results_dir}")
    return results_dir


def create_comparison_plots(results, output_dir, args):
    """Create comparative plots for all evaluated methods."""
    methods = list(results.keys())
    metrics = ["waiting_time", "queue_length", "throughput", "emissions", "travel_time"]

    # Prepare data for plotting
    for metric in metrics:
        plt.figure(figsize=(10, 6))

        means = [results[method][metric]["mean"] for method in methods]
        stds = [results[method][metric]["std"] for method in methods]

        # Create bar plot
        bars = plt.bar(methods, means, yerr=stds, capsize=5)

        # Add value labels on top of each bar
        for bar, mean in zip(bars, means):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1 * max(means),
                f"{mean:.2f}",
                ha="center",
                fontsize=9,
            )

        # Set labels and title
        plt.xlabel("Control Method")
        plt.ylabel(metric.replace("_", " ").title())
        plt.title(
            f"{metric.replace('_', ' ').title()} Comparison - {args.scenario.title()} Scenario"
        )

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha="right")

        # Adjust layout
        plt.tight_layout()

        # Save figure
        plt.savefig(os.path.join(output_dir, f"{metric}_comparison.png"), dpi=300)
        plt.close()

    # Create multi-metric comparison plot
    plt.figure(figsize=(15, 10))

    # Normalize metrics to [0,1] scale for comparison
    normalized_data = {}
    for metric in metrics:
        values = [results[method][metric]["mean"] for method in methods]
        min_val = min(values)
        max_val = max(values)

        # For metrics where lower is better (waiting time, queue length, etc.)
        if metric in ["waiting_time", "queue_length", "emissions", "travel_time"]:
            if max_val > min_val:
                normalized_data[metric] = [
                    1 - (v - min_val) / (max_val - min_val) for v in values
                ]
            else:
                normalized_data[metric] = [1 for _ in values]
        # For metrics where higher is better (throughput)
        else:
            if max_val > min_val:
                normalized_data[metric] = [
                    (v - min_val) / (max_val - min_val) for v in values
                ]
            else:
                normalized_data[metric] = [1 for _ in values]

    # Plot radar chart
    num_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop

    # Set up subplot
    ax = plt.subplot(111, polar=True)

    # Plot each method
    for i, method in enumerate(methods):
        values = [normalized_data[metric][i] for metric in metrics]
        values += values[:1]  # Close the loop

        ax.plot(angles, values, linewidth=2, linestyle="solid", label=method)
        ax.fill(angles, values, alpha=0.1)

    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)

    # Add legend
    plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))

    plt.title(
        f"Performance Comparison - {args.scenario.title()} Scenario",
        position=(0.5, 1.1),
        fontsize=15,
    )

    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "radar_comparison.png"), dpi=300)
    plt.close()


def main():
    """Main evaluation function."""
    # Parse arguments
    args = parse_args()

    # Load configuration
    config = load_config(args.config_path)

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Store results for all methods
    all_results = {}

    # Evaluate each method
    for method in args.methods:
        print(f"\n{'-'*40}")
        print(f"Evaluating method: {method}")
        print(f"{'-'*40}\n")

        # Create environment
        env = create_environment(args, config)

        # Load agent
        agent = load_agent(method, env, args, config)

        # Evaluate agent
        metrics = evaluate_agent(agent, env, args, config)

        # Store results
        all_results[method] = metrics

        # Close environment
        env.close()

    # Save results
    results_dir = save_results(all_results, args)

    # Print summary
    print("\nEvaluation Summary:")
    print(f"{'-'*60}")
    print(
        f"{'Method':<12} {'Reward':<12} {'Queue':<12} {'Wait Time':<12} {'Throughput':<12}"
    )
    print(f"{'-'*60}")

    for method, metrics in all_results.items():
        reward = metrics["reward"]["mean"]
        queue = metrics["queue_length"]["mean"]
        wait = metrics["waiting_time"]["mean"]
        throughput = metrics["throughput"]["mean"]

        print(
            f"{method:<12} {reward:<12.2f} {queue:<12.2f} {wait:<12.2f} {throughput:<12.2f}"
        )

    print(f"{'-'*60}")
    print(f"Detailed results saved to: {results_dir}")


if __name__ == "__main__":
    main()
