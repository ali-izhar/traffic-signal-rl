#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Training Data Analysis Script"""

import os
import argparse
import json
import glob
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set plotting style
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_context("paper")


def load_episode_data(data_dir, format_type):
    """Load episode-level data from the specified directory.

    Args:
        data_dir: Path to the experiment directory
        format_type: Format of the data files (csv, hdf5, json)

    Returns:
        DataFrame containing episode data
    """
    if format_type == "csv":
        # Find all episode CSV files
        episode_files = glob.glob(os.path.join(data_dir, "episodes_*.csv"))
        if not episode_files:
            raise FileNotFoundError(f"No episode data files found in {data_dir}")

        # Load and concatenate all files
        dfs = []
        for file in episode_files:
            df = pd.read_csv(file)
            dfs.append(df)

        if dfs:
            return pd.concat(dfs, ignore_index=True)
        else:
            return pd.DataFrame()

    elif format_type == "hdf5":
        # Find all HDF5 files
        h5_files = glob.glob(os.path.join(data_dir, "data_*.h5"))
        if not h5_files:
            raise FileNotFoundError(f"No HDF5 data files found in {data_dir}")

        # Load episode data from all files
        dfs = []
        for file in h5_files:
            with h5py.File(file, "r") as f:
                if "episodes" in f:
                    # Convert HDF5 datasets to a dictionary
                    data_dict = {
                        key: f["episodes"][key][:] for key in f["episodes"].keys()
                    }
                    df = pd.DataFrame(data_dict)
                    dfs.append(df)

        if dfs:
            return pd.concat(dfs, ignore_index=True)
        else:
            return pd.DataFrame()

    elif format_type == "json":
        # Find all episode JSON files
        json_files = glob.glob(os.path.join(data_dir, "episodes_*.json"))
        if not json_files:
            raise FileNotFoundError(f"No episode data files found in {data_dir}")

        # Load and concatenate all files
        dfs = []
        for file in json_files:
            with open(file, "r") as f:
                data = json.load(f)
                df = pd.DataFrame(data)
                dfs.append(df)

        if dfs:
            return pd.concat(dfs, ignore_index=True)
        else:
            return pd.DataFrame()

    else:
        raise ValueError(f"Unsupported format type: {format_type}")


def load_step_data(data_dir, format_type, max_episodes=None):
    """Load step-level data from the specified directory.

    Args:
        data_dir: Path to the experiment directory
        format_type: Format of the data files (csv, hdf5, json)
        max_episodes: Maximum number of episodes to load (for memory management)

    Returns:
        DataFrame containing step data
    """
    if format_type == "csv":
        # Find all step CSV files
        step_files = glob.glob(os.path.join(data_dir, "steps_*.csv"))
        if not step_files:
            raise FileNotFoundError(f"No step data files found in {data_dir}")

        # Load and concatenate files
        dfs = []
        for file in step_files:
            df = pd.read_csv(file)

            # Filter by episode if needed
            if max_episodes is not None:
                unique_episodes = df["episode_ids"].unique()
                if len(unique_episodes) > max_episodes:
                    selected_episodes = unique_episodes[:max_episodes]
                    df = df[df["episode_ids"].isin(selected_episodes)]

            dfs.append(df)

            # Check if we've loaded enough episodes
            if max_episodes is not None and len(dfs) >= max_episodes:
                break

        if dfs:
            return pd.concat(dfs, ignore_index=True)
        else:
            return pd.DataFrame()

    elif format_type == "hdf5":
        # Find all HDF5 files
        h5_files = glob.glob(os.path.join(data_dir, "data_*.h5"))
        if not h5_files:
            raise FileNotFoundError(f"No HDF5 data files found in {data_dir}")

        # Load step data from files
        dfs = []
        loaded_episodes = 0

        for file in h5_files:
            with h5py.File(file, "r") as f:
                if "steps" in f:
                    # Get episode IDs in this file
                    episode_ids = f["steps"]["episode_ids"][:]
                    unique_episodes = np.unique(episode_ids)

                    # Determine which episodes to load
                    if (
                        max_episodes is not None
                        and loaded_episodes + len(unique_episodes) > max_episodes
                    ):
                        # Calculate how many more episodes we need
                        episodes_to_load = max_episodes - loaded_episodes
                        selected_episodes = unique_episodes[:episodes_to_load]
                        mask = np.isin(episode_ids, selected_episodes)

                        # Create filtered data dictionary
                        data_dict = {}
                        for key in f["steps"].keys():
                            if key not in ["queue_lengths", "waiting_times"]:
                                data_dict[key] = f["steps"][key][:][mask]

                        # Handle special cases for nested dictionaries
                        for key in ["queue_lengths", "waiting_times"]:
                            if key in f["steps"]:
                                if isinstance(f["steps"][key][0], str):
                                    # JSON strings
                                    json_strings = f["steps"][key][:][mask]
                                    data_dict[key] = [
                                        json.loads(s) for s in json_strings
                                    ]
                                else:
                                    # Numeric data
                                    data_dict[key] = f["steps"][key][:][mask]

                        df = pd.DataFrame(data_dict)
                        dfs.append(df)
                        loaded_episodes += len(selected_episodes)
                        break
                    else:
                        # Create data dictionary for all episodes in this file
                        data_dict = {}
                        for key in f["steps"].keys():
                            if key not in ["queue_lengths", "waiting_times"]:
                                data_dict[key] = f["steps"][key][:]

                        # Handle special cases for nested dictionaries
                        for key in ["queue_lengths", "waiting_times"]:
                            if key in f["steps"]:
                                if isinstance(f["steps"][key][0], str):
                                    # JSON strings
                                    json_strings = f["steps"][key][:]
                                    data_dict[key] = [
                                        json.loads(s) for s in json_strings
                                    ]
                                else:
                                    # Numeric data
                                    data_dict[key] = f["steps"][key][:]

                        df = pd.DataFrame(data_dict)
                        dfs.append(df)
                        loaded_episodes += len(unique_episodes)

                # Check if we've loaded enough episodes
                if max_episodes is not None and loaded_episodes >= max_episodes:
                    break

        if dfs:
            return pd.concat(dfs, ignore_index=True)
        else:
            return pd.DataFrame()

    elif format_type == "json":
        # Find all step JSON files
        step_files = glob.glob(os.path.join(data_dir, "steps_*.json"))
        if not step_files:
            raise FileNotFoundError(f"No step data files found in {data_dir}")

        # Load and concatenate files
        dfs = []
        loaded_episodes = 0

        for file in step_files:
            with open(file, "r") as f:
                data = json.load(f)
                df = pd.DataFrame(data)

                # Filter by episode if needed
                if max_episodes is not None:
                    unique_episodes = df["episode_ids"].unique()
                    episodes_remaining = max_episodes - loaded_episodes

                    if episodes_remaining <= 0:
                        break

                    if len(unique_episodes) > episodes_remaining:
                        selected_episodes = unique_episodes[:episodes_remaining]
                        df = df[df["episode_ids"].isin(selected_episodes)]

                    loaded_episodes += len(df["episode_ids"].unique())

                dfs.append(df)

        if dfs:
            return pd.concat(dfs, ignore_index=True)
        else:
            return pd.DataFrame()

    else:
        raise ValueError(f"Unsupported format type: {format_type}")


def calculate_episode_statistics(episode_df):
    """Calculate statistics from episode data.

    Args:
        episode_df: DataFrame containing episode data

    Returns:
        Dictionary of statistics
    """
    if episode_df.empty:
        return {}

    stats = {}

    # Basic statistics
    for column in [
        "episode_rewards",
        "episode_lengths",
        "episode_throughputs",
        "episode_avg_queue_lengths",
        "episode_avg_waiting_times",
        "episode_total_switches",
    ]:
        if column in episode_df.columns:
            data = episode_df[column]
            stats[column] = {
                "mean": data.mean(),
                "median": data.median(),
                "min": data.min(),
                "max": data.max(),
                "std": data.std(),
            }

    return stats


def plot_episode_metrics(episode_df, output_dir):
    """Create plots for episode-level metrics.

    Args:
        episode_df: DataFrame containing episode data
        output_dir: Directory to save plots
    """
    if episode_df.empty:
        print("No episode data to plot")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Plot episode rewards
    plt.figure(figsize=(10, 6))
    sns.lineplot(x="episode_ids", y="episode_rewards", data=episode_df)
    plt.title("Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "episode_rewards.png"))
    plt.close()

    # Plot episode throughputs
    if "episode_throughputs" in episode_df.columns:
        plt.figure(figsize=(10, 6))
        sns.lineplot(x="episode_ids", y="episode_throughputs", data=episode_df)
        plt.title("Episode Throughputs")
        plt.xlabel("Episode")
        plt.ylabel("Total Throughput")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "episode_throughputs.png"))
        plt.close()

    # Plot average queue lengths
    if "episode_avg_queue_lengths" in episode_df.columns:
        plt.figure(figsize=(10, 6))
        sns.lineplot(x="episode_ids", y="episode_avg_queue_lengths", data=episode_df)
        plt.title("Average Queue Lengths")
        plt.xlabel("Episode")
        plt.ylabel("Average Queue Length")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "episode_queue_lengths.png"))
        plt.close()

    # Plot average waiting times
    if "episode_avg_waiting_times" in episode_df.columns:
        plt.figure(figsize=(10, 6))
        sns.lineplot(x="episode_ids", y="episode_avg_waiting_times", data=episode_df)
        plt.title("Average Waiting Times")
        plt.xlabel("Episode")
        plt.ylabel("Average Waiting Time")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "episode_waiting_times.png"))
        plt.close()

    # Plot total switches
    if "episode_total_switches" in episode_df.columns:
        plt.figure(figsize=(10, 6))
        sns.lineplot(x="episode_ids", y="episode_total_switches", data=episode_df)
        plt.title("Total Signal Switches")
        plt.xlabel("Episode")
        plt.ylabel("Number of Switches")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "episode_switches.png"))
        plt.close()

    # Plot multiple metrics together
    plt.figure(figsize=(18, 12))

    plt.subplot(2, 2, 1)
    sns.lineplot(x="episode_ids", y="episode_rewards", data=episode_df)
    plt.title("Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")

    if "episode_throughputs" in episode_df.columns:
        plt.subplot(2, 2, 2)
        sns.lineplot(x="episode_ids", y="episode_throughputs", data=episode_df)
        plt.title("Episode Throughputs")
        plt.xlabel("Episode")
        plt.ylabel("Total Throughput")

    if "episode_avg_queue_lengths" in episode_df.columns:
        plt.subplot(2, 2, 3)
        sns.lineplot(x="episode_ids", y="episode_avg_queue_lengths", data=episode_df)
        plt.title("Average Queue Lengths")
        plt.xlabel("Episode")
        plt.ylabel("Average Queue Length")

    if "episode_avg_waiting_times" in episode_df.columns:
        plt.subplot(2, 2, 4)
        sns.lineplot(x="episode_ids", y="episode_avg_waiting_times", data=episode_df)
        plt.title("Average Waiting Times")
        plt.xlabel("Episode")
        plt.ylabel("Average Waiting Time")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "episode_summary.png"))
    plt.close()


def analyze_step_data(step_df, output_dir):
    """Analyze step-level data and create visualizations.

    Args:
        step_df: DataFrame containing step data
        output_dir: Directory to save analysis results
    """
    if step_df.empty:
        print("No step data to analyze")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Handle nested queue length data if stored as JSON strings
    queue_data = step_df["queue_lengths"]
    if isinstance(queue_data.iloc[0], str):
        # Convert JSON strings to dictionaries
        queue_data = queue_data.apply(json.loads)

        # Extract individual directions
        step_df["north_queue"] = queue_data.apply(lambda x: x.get("north", 0))
        step_df["south_queue"] = queue_data.apply(lambda x: x.get("south", 0))
        step_df["east_queue"] = queue_data.apply(lambda x: x.get("east", 0))
        step_df["west_queue"] = queue_data.apply(lambda x: x.get("west", 0))
        step_df["total_queue"] = (
            step_df["north_queue"]
            + step_df["south_queue"]
            + step_df["east_queue"]
            + step_df["west_queue"]
        )

    # Handle nested waiting time data if stored as JSON strings
    waiting_data = step_df["waiting_times"]
    if isinstance(waiting_data.iloc[0], str):
        # Convert JSON strings to dictionaries
        waiting_data = waiting_data.apply(json.loads)

        # Extract individual directions
        step_df["north_wait"] = waiting_data.apply(lambda x: x.get("north", 0))
        step_df["south_wait"] = waiting_data.apply(lambda x: x.get("south", 0))
        step_df["east_wait"] = waiting_data.apply(lambda x: x.get("east", 0))
        step_df["west_wait"] = waiting_data.apply(lambda x: x.get("west", 0))
        step_df["total_wait"] = (
            step_df["north_wait"]
            + step_df["south_wait"]
            + step_df["east_wait"]
            + step_df["west_wait"]
        )

    # Plot queue lengths over time for a specific episode
    episode_to_plot = step_df["episode_ids"].iloc[0]
    episode_data = step_df[step_df["episode_ids"] == episode_to_plot]

    # Plot queue lengths by direction
    if all(
        col in step_df.columns
        for col in ["north_queue", "south_queue", "east_queue", "west_queue"]
    ):
        plt.figure(figsize=(12, 6))
        sns.lineplot(x="steps", y="north_queue", data=episode_data, label="North")
        sns.lineplot(x="steps", y="south_queue", data=episode_data, label="South")
        sns.lineplot(x="steps", y="east_queue", data=episode_data, label="East")
        sns.lineplot(x="steps", y="west_queue", data=episode_data, label="West")
        plt.title(f"Queue Lengths by Direction (Episode {episode_to_plot})")
        plt.xlabel("Step")
        plt.ylabel("Queue Length")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "queue_lengths_by_direction.png"))
        plt.close()

    # Plot waiting times by direction
    if all(
        col in step_df.columns
        for col in ["north_wait", "south_wait", "east_wait", "west_wait"]
    ):
        plt.figure(figsize=(12, 6))
        sns.lineplot(x="steps", y="north_wait", data=episode_data, label="North")
        sns.lineplot(x="steps", y="south_wait", data=episode_data, label="South")
        sns.lineplot(x="steps", y="east_wait", data=episode_data, label="East")
        sns.lineplot(x="steps", y="west_wait", data=episode_data, label="West")
        plt.title(f"Waiting Times by Direction (Episode {episode_to_plot})")
        plt.xlabel("Step")
        plt.ylabel("Waiting Time")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "waiting_times_by_direction.png"))
        plt.close()

    # Plot rewards over time
    plt.figure(figsize=(12, 6))
    sns.lineplot(x="steps", y="rewards", data=episode_data)
    plt.title(f"Rewards over Time (Episode {episode_to_plot})")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rewards_over_time.png"))
    plt.close()

    # Plot cumulative throughput
    if "throughputs" in step_df.columns:
        episode_data["cumulative_throughput"] = episode_data["throughputs"].cumsum()

        plt.figure(figsize=(12, 6))
        sns.lineplot(x="steps", y="cumulative_throughput", data=episode_data)
        plt.title(f"Cumulative Throughput (Episode {episode_to_plot})")
        plt.xlabel("Step")
        plt.ylabel("Cumulative Throughput")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "cumulative_throughput.png"))
        plt.close()

    # Plot phase changes
    if "phases" in step_df.columns:
        plt.figure(figsize=(12, 6))
        sns.lineplot(x="steps", y="phases", data=episode_data, drawstyle="steps-post")
        plt.title(f"Traffic Signal Phases (Episode {episode_to_plot})")
        plt.xlabel("Step")
        plt.ylabel("Phase")
        plt.yticks([0, 1, 2, 3], ["N-S Green", "E-W Green", "N-S Yellow", "E-W Yellow"])
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "signal_phases.png"))
        plt.close()

    # Plot action distribution
    if "actions" in step_df.columns:
        plt.figure(figsize=(10, 6))
        sns.countplot(x="actions", data=step_df)
        plt.title("Action Distribution")
        plt.xlabel("Action (0: Keep, 1: Switch)")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "action_distribution.png"))
        plt.close()

    # Plot queue length distribution
    if "total_queue" in step_df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(step_df["total_queue"], kde=True)
        plt.title("Total Queue Length Distribution")
        plt.xlabel("Total Queue Length")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "queue_length_distribution.png"))
        plt.close()

    # Save detailed statistics as CSV
    if all(
        col in step_df.columns
        for col in ["north_queue", "south_queue", "east_queue", "west_queue"]
    ):
        queue_stats = pd.DataFrame(
            {
                "direction": ["North", "South", "East", "West", "Total"],
                "mean": [
                    step_df["north_queue"].mean(),
                    step_df["south_queue"].mean(),
                    step_df["east_queue"].mean(),
                    step_df["west_queue"].mean(),
                    step_df["total_queue"].mean(),
                ],
                "median": [
                    step_df["north_queue"].median(),
                    step_df["south_queue"].median(),
                    step_df["east_queue"].median(),
                    step_df["west_queue"].median(),
                    step_df["total_queue"].median(),
                ],
                "max": [
                    step_df["north_queue"].max(),
                    step_df["south_queue"].max(),
                    step_df["east_queue"].max(),
                    step_df["west_queue"].max(),
                    step_df["total_queue"].max(),
                ],
            }
        )
        queue_stats.to_csv(
            os.path.join(output_dir, "queue_statistics.csv"), index=False
        )


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Analyze traffic signal control training data"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing the experiment data",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["csv", "hdf5", "json"],
        default="csv",
        help="Format of the data files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save analysis results (default: data_dir/analysis)",
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="Maximum number of episodes to analyze",
    )

    args = parser.parse_args()

    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.data_dir, "analysis")

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading data from {args.data_dir}...")

    try:
        # Load episode data
        episode_df = load_episode_data(args.data_dir, args.format)
        print(f"Loaded data for {len(episode_df)} episodes")

        # Calculate statistics
        stats = calculate_episode_statistics(episode_df)

        # Save statistics to file
        stats_file = os.path.join(args.output_dir, "statistics.json")
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=4)

        print(f"Statistics saved to {stats_file}")

        # Create episode-level plots
        print("Generating episode-level visualizations...")
        plot_episode_metrics(episode_df, args.output_dir)

        # Load and analyze step data (limited to max_episodes if specified)
        print("Loading and analyzing step-level data...")
        step_df = load_step_data(args.data_dir, args.format, args.max_episodes)
        analyze_step_data(step_df, args.output_dir)

        print(f"Analysis complete. Results saved to {args.output_dir}")

    except Exception as e:
        print(f"Error analyzing data: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
