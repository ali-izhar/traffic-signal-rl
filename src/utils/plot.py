#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Traffic Signal Control Visualization

This module provides functions to visualize traffic signal control results,
including performance metrics and traffic states.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
from typing import Dict, List, Tuple, Any, Optional, Union
import matplotlib.ticker as ticker


def plot_learning_curves(
    training_data: Dict[str, List],
    output_path: Optional[str] = None,
    title: str = "Learning Curves",
    figsize: Tuple[int, int] = (15, 10),
    smoothing: int = 10,
):
    """
    Plot learning curves from training data.

    Args:
        training_data: Dictionary containing training metrics over episodes
        output_path: Path to save the figure (if None, just display)
        title: Title for the figure
        figsize: Figure size (width, height)
        smoothing: Window size for smoothing curves
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Function to apply smoothing
    def smooth(y, window):
        if len(y) < window:
            return y
        box = np.ones(window) / window
        return np.convolve(y, box, mode="valid")

    # Episode numbers
    if "rewards" in training_data and len(training_data["rewards"]) > 0:
        num_episodes = len(training_data["rewards"])
        x = np.arange(num_episodes)
        x_smooth = np.arange(smoothing - 1, num_episodes)
    else:
        return  # Empty data

    # Plot rewards
    if "rewards" in training_data:
        rewards = np.array(training_data["rewards"])
        ax = axes[0, 0]
        ax.plot(x, rewards, alpha=0.3, color="blue")
        if len(rewards) >= smoothing:
            smoothed_rewards = smooth(rewards, smoothing)
            ax.plot(x_smooth, smoothed_rewards, linewidth=2, color="blue")
        ax.set_title("Episode Rewards")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Reward")
        ax.grid(alpha=0.3)

    # Plot queue lengths
    if "queue_lengths" in training_data:
        queue_lengths = np.array(training_data["queue_lengths"])
        ax = axes[0, 1]
        ax.plot(x, queue_lengths, alpha=0.3, color="red")
        if len(queue_lengths) >= smoothing:
            smoothed_queue = smooth(queue_lengths, smoothing)
            ax.plot(x_smooth, smoothed_queue, linewidth=2, color="red")
        ax.set_title("Average Queue Lengths")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Queue Length")
        ax.grid(alpha=0.3)

    # Plot waiting times
    if "waiting_times" in training_data:
        waiting_times = np.array(training_data["waiting_times"])
        ax = axes[1, 0]
        ax.plot(x, waiting_times, alpha=0.3, color="green")
        if len(waiting_times) >= smoothing:
            smoothed_waiting = smooth(waiting_times, smoothing)
            ax.plot(x_smooth, smoothed_waiting, linewidth=2, color="green")
        ax.set_title("Average Waiting Times")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Waiting Time")
        ax.grid(alpha=0.3)

    # Plot throughput
    if "throughputs" in training_data:
        throughputs = np.array(training_data["throughputs"])
        ax = axes[1, 1]
        ax.plot(x, throughputs, alpha=0.3, color="purple")
        if len(throughputs) >= smoothing:
            smoothed_throughput = smooth(throughputs, smoothing)
            ax.plot(x_smooth, smoothed_throughput, linewidth=2, color="purple")
        ax.set_title("Episode Throughput")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Throughput")
        ax.grid(alpha=0.3)

    # Add exploration rate if available
    if "exploration_rates" in training_data:
        exploration_rates = np.array(training_data["exploration_rates"])
        ax_exploration = axes[1, 1].twinx()
        ax_exploration.plot(
            x, exploration_rates, alpha=0.7, color="orange", linestyle="--"
        )
        ax_exploration.set_ylabel("Exploration Rate", color="orange")
        ax_exploration.tick_params(axis="y", labelcolor="orange")

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_comparison(
    results: Dict[str, Dict[str, Dict[str, float]]],
    output_path: Optional[str] = None,
    metrics: List[str] = None,
    title: str = "Performance Comparison",
    figsize: Tuple[int, int] = (15, 10),
):
    """
    Plot comparison of different methods across metrics.

    Args:
        results: Results dictionary with methods as keys
        output_path: Path to save the figure (if None, just display)
        metrics: List of metrics to compare (if None, use all common metrics)
        title: Title for the figure
        figsize: Figure size (width, height)
    """
    methods = list(results.keys())

    # Determine metrics to plot
    if metrics is None:
        # Find common metrics across all methods
        metrics_sets = [set(m.keys()) for m in results.values()]
        metrics_sets = [s - {"raw_data"} for s in metrics_sets]  # Remove raw data
        common_metrics = set.intersection(*metrics_sets)
        metrics = sorted(list(common_metrics))

    # Number of plots
    n_metrics = len(metrics)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_metrics == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]

        # Extract values and errors
        means = []
        errors = []
        for method in methods:
            if metric in results[method]:
                means.append(results[method][metric]["mean"])
                errors.append(results[method][metric]["std"])
            else:
                means.append(0)
                errors.append(0)

        # Create bar chart
        bars = ax.bar(methods, means, yerr=errors, capsize=5)

        # Add value labels
        for bar, mean in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05 * max(means),
                f"{mean:.2f}",
                ha="center",
                fontsize=9,
            )

        # Set labels and title
        ax.set_title(metric.replace("_", " ").title())
        ax.set_xlabel("Method")
        ax.grid(alpha=0.3)

        # Rotate x-tick labels if many methods
        if len(methods) > 4:
            ax.set_xticklabels(methods, rotation=45, ha="right")

    # Hide unused subplots
    for i in range(n_metrics, n_rows * n_cols):
        axes[i].axis("off")

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_radar_comparison(
    results: Dict[str, Dict[str, Dict[str, float]]],
    output_path: Optional[str] = None,
    metrics: List[str] = None,
    higher_is_better: Dict[str, bool] = None,
    title: str = "Multi-metric Comparison",
    figsize: Tuple[int, int] = (10, 8),
):
    """
    Create a radar plot comparing methods across multiple metrics.

    Args:
        results: Results dictionary with methods as keys
        output_path: Path to save the figure (if None, just display)
        metrics: List of metrics to compare (if None, use all common metrics)
        higher_is_better: Dictionary indicating whether higher values are better for each metric
        title: Title for the figure
        figsize: Figure size (width, height)
    """
    if higher_is_better is None:
        higher_is_better = {
            "waiting_time": False,
            "queue_length": False,
            "throughput": True,
            "reward": True,
            "travel_time": False,
            "emissions": False,
            "fuel_consumption": False,
            "switches": False,
        }

    methods = list(results.keys())

    # Determine metrics to plot
    if metrics is None:
        # Find common metrics across all methods
        metrics_sets = [set(m.keys()) for m in results.values()]
        metrics_sets = [s - {"raw_data"} for s in metrics_sets]  # Remove raw data
        common_metrics = set.intersection(*metrics_sets)
        metrics = sorted(list(common_metrics))

    # Extract values and normalize
    values = {}
    for metric in metrics:
        metric_values = []
        for method in methods:
            if metric in results[method]:
                metric_values.append(results[method][metric]["mean"])
            else:
                metric_values.append(0)

        # Skip if all values are the same
        if len(set(metric_values)) <= 1:
            continue

        min_val = min(metric_values)
        max_val = max(metric_values)
        range_val = max_val - min_val

        # Normalize values between 0 and 1
        if range_val > 0:
            if higher_is_better.get(metric, False):
                # Higher is better: 1 is best
                values[metric] = [(v - min_val) / range_val for v in metric_values]
            else:
                # Lower is better: 0 is best, invert the normalization
                values[metric] = [1 - (v - min_val) / range_val for v in metric_values]
        else:
            values[metric] = [0.5] * len(methods)

    # If no metrics with variation, return
    if not values:
        return

    metrics = list(values.keys())

    # Number of metrics
    n_metrics = len(metrics)

    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, polar=True)

    # Compute angle for each metric
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()

    # Make the plot circular by adding the first point at the end
    metrics = metrics + [metrics[0]]
    angles = angles + [angles[0]]

    # Plot each method
    for i, method in enumerate(methods):
        method_values = [values[metric][i] for metric in metrics[:-1]]
        method_values = method_values + [method_values[0]]  # Close the loop

        ax.plot(angles, method_values, linewidth=2, label=method)
        ax.fill(angles, method_values, alpha=0.1)

    # Set labels and ticks
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace("_", " ").title() for m in metrics[:-1]])

    # Add legend
    ax.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))

    plt.title(title, fontsize=16, y=1.08)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_traffic_state(
    state: np.ndarray,
    phase: int,
    queues: Dict[str, int],
    output_path: Optional[str] = None,
    title: str = "Traffic State Visualization",
    figsize: Tuple[int, int] = (8, 8),
):
    """
    Visualize the current traffic state at an intersection.

    Args:
        state: Current state vector
        phase: Current signal phase
        queues: Dictionary of queue lengths
        output_path: Path to save the figure (if None, just display)
        title: Title for the figure
        figsize: Figure size (width, height)
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Draw intersection
    ax.plot([-2, 2], [0, 0], "k-", lw=3)  # East-West road
    ax.plot([0, 0], [-2, 2], "k-", lw=3)  # North-South road

    # Add road labels
    ax.text(2.1, 0, "E", fontsize=12, ha="left", va="center")
    ax.text(-2.1, 0, "W", fontsize=12, ha="right", va="center")
    ax.text(0, 2.1, "N", fontsize=12, ha="center", va="bottom")
    ax.text(0, -2.1, "S", fontsize=12, ha="center", va="top")

    # Draw traffic lights
    light_size = 0.2
    # North traffic light
    if phase == 0:  # N-S green
        ax.add_patch(Circle((0, 1), light_size, fc="green"))
    else:  # N-S red
        ax.add_patch(Circle((0, 1), light_size, fc="red"))

    # South traffic light
    if phase == 0:  # N-S green
        ax.add_patch(Circle((0, -1), light_size, fc="green"))
    else:  # N-S red
        ax.add_patch(Circle((0, -1), light_size, fc="red"))

    # East traffic light
    if phase == 1:  # E-W green
        ax.add_patch(Circle((1, 0), light_size, fc="green"))
    else:  # E-W red
        ax.add_patch(Circle((1, 0), light_size, fc="red"))

    # West traffic light
    if phase == 1:  # E-W green
        ax.add_patch(Circle((-1, 0), light_size, fc="green"))
    else:  # E-W red
        ax.add_patch(Circle((-1, 0), light_size, fc="red"))

    # Draw queues
    colors = ["blue", "green", "red", "purple", "orange", "cyan", "magenta", "yellow"]

    # North queue
    north_queue = queues.get("north", 0)
    for i in range(min(north_queue, 10)):
        y_pos = 1.2 + (i * 0.2)
        color = colors[i % len(colors)]
        ax.add_patch(Rectangle((-0.15, y_pos), 0.3, 0.15, fc=color, ec="black", lw=0.5))

    # South queue
    south_queue = queues.get("south", 0)
    for i in range(min(south_queue, 10)):
        y_pos = -1.2 - (i * 0.2) - 0.15
        color = colors[i % len(colors)]
        ax.add_patch(Rectangle((-0.15, y_pos), 0.3, 0.15, fc=color, ec="black", lw=0.5))

    # East queue
    east_queue = queues.get("east", 0)
    for i in range(min(east_queue, 10)):
        x_pos = 1.2 + (i * 0.2)
        color = colors[i % len(colors)]
        ax.add_patch(Rectangle((x_pos, -0.15), 0.15, 0.3, fc=color, ec="black", lw=0.5))

    # West queue
    west_queue = queues.get("west", 0)
    for i in range(min(west_queue, 10)):
        x_pos = -1.2 - (i * 0.2) - 0.15
        color = colors[i % len(colors)]
        ax.add_patch(Rectangle((x_pos, -0.15), 0.15, 0.3, fc=color, ec="black", lw=0.5))

    # Add text information
    phase_text = (
        "Current Phase: East-West Green"
        if phase == 1
        else "Current Phase: North-South Green"
    )
    ax.text(
        2,
        2.5,
        phase_text,
        ha="right",
        fontsize=12,
        bbox=dict(
            facecolor="white", alpha=1.0, edgecolor="black", boxstyle="round,pad=0.5"
        ),
    )

    # Add queue information
    queue_text = (
        f"Queues: N={north_queue}, S={south_queue}, E={east_queue}, W={west_queue}"
    )
    ax.text(
        2,
        -2.5,
        queue_text,
        ha="right",
        fontsize=12,
        bbox=dict(
            facecolor="white", alpha=1.0, edgecolor="black", boxstyle="round,pad=0.5"
        ),
    )

    # Add RL agent box
    agent_box = FancyBboxPatch(
        (-2.9, 2.4),
        1.4,
        0.7,
        boxstyle="round,pad=0.4",
        fc="lightgray",
        ec="black",
        alpha=0.9,
    )
    ax.add_patch(agent_box)
    ax.text(
        -2.2, 2.75, "RL Agent", ha="center", va="center", fontsize=11, fontweight="bold"
    )

    # Set limits and title
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect("equal")
    ax.set_title(title)

    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def create_heatmap(
    data: np.ndarray,
    x_labels: List[str],
    y_labels: List[str],
    output_path: Optional[str] = None,
    title: str = "Heatmap",
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = "YlGnBu",
    annot: bool = True,
):
    """
    Create a heatmap visualization.

    Args:
        data: 2D array of values
        x_labels: Labels for x-axis
        y_labels: Labels for y-axis
        output_path: Path to save the figure (if None, just display)
        title: Title for the figure
        figsize: Figure size (width, height)
        cmap: Colormap to use
        annot: Whether to annotate cells with values
    """
    plt.figure(figsize=figsize)

    # Create heatmap
    ax = sns.heatmap(
        data,
        cmap=cmap,
        annot=annot,
        fmt=".2f",
        xticklabels=x_labels,
        yticklabels=y_labels,
    )

    plt.title(title)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_multi_intersection_traffic(
    network_state: Dict[str, Any],
    phase_dict: Dict[str, int],
    queues_dict: Dict[str, Dict[str, int]],
    output_path: Optional[str] = None,
    title: str = "Multi-Intersection Traffic State",
    figsize: Tuple[int, int] = (12, 10),
):
    """
    Visualize the traffic state across multiple intersections.

    Args:
        network_state: Network state information
        phase_dict: Dictionary mapping intersection ID to current phase
        queues_dict: Dictionary mapping intersection ID to queue length dictionary
        output_path: Path to save the figure (if None, just display)
        title: Title for the figure
        figsize: Figure size (width, height)
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Extract topology information
    if "topology" in network_state:
        topology = network_state["topology"]
        if topology == "2x2_grid":
            # Create a 2x2 grid of intersections
            intersections = ["I0_0", "I0_1", "I1_0", "I1_1"]
            positions = {"I0_0": (0, 0), "I0_1": (4, 0), "I1_0": (0, 4), "I1_1": (4, 4)}

            # Draw roads
            # Horizontal roads
            ax.plot([-1, 5], [0, 0], "k-", lw=2)  # Bottom
            ax.plot([-1, 5], [4, 4], "k-", lw=2)  # Top
            # Vertical roads
            ax.plot([0, 0], [-1, 5], "k-", lw=2)  # Left
            ax.plot([4, 4], [-1, 5], "k-", lw=2)  # Right

        elif topology == "corridor":
            # Create a corridor of 3 intersections
            intersections = ["I0", "I1", "I2"]
            positions = {"I0": (0, 0), "I1": (4, 0), "I2": (8, 0)}

            # Draw roads
            ax.plot([-1, 9], [0, 0], "k-", lw=2)  # Main east-west road
            # Vertical roads
            ax.plot([0, 0], [-1, 1], "k-", lw=2)  # Left
            ax.plot([4, 4], [-1, 1], "k-", lw=2)  # Middle
            ax.plot([8, 8], [-1, 1], "k-", lw=2)  # Right

        else:
            # Default: create based on intersection IDs provided
            intersections = list(phase_dict.keys())
            positions = {}
            for i, id in enumerate(intersections):
                # Simple layout: just place them in a row
                positions[id] = (i * 4, 0)

            # Draw a simple straight road
            ax.plot([-1, len(intersections) * 4 + 1], [0, 0], "k-", lw=2)
            for i, id in enumerate(intersections):
                ax.plot([i * 4, i * 4], [-1, 1], "k-", lw=2)

    else:
        # Handle the case where network topology is not provided
        intersections = list(phase_dict.keys())
        positions = {}
        for i, id in enumerate(intersections):
            # Simple layout: just place them in a row
            positions[id] = (i * 4, 0)

        # Draw a simple straight road
        ax.plot([-1, len(intersections) * 4 + 1], [0, 0], "k-", lw=2)
        for i, id in enumerate(intersections):
            ax.plot([i * 4, i * 4], [-1, 1], "k-", lw=2)

    # Draw intersections with traffic lights and queues
    light_size = 0.2
    colors = ["blue", "green", "red", "purple", "orange", "cyan", "magenta", "yellow"]

    for id in intersections:
        pos = positions[id]
        phase = phase_dict.get(id, 0)
        queues = queues_dict.get(id, {"north": 0, "south": 0, "east": 0, "west": 0})

        # Draw traffic lights
        # North light
        if phase == 0:  # N-S green
            ax.add_patch(Circle((pos[0], pos[1] + 0.5), light_size, fc="green"))
        else:
            ax.add_patch(Circle((pos[0], pos[1] + 0.5), light_size, fc="red"))

        # South light
        if phase == 0:  # N-S green
            ax.add_patch(Circle((pos[0], pos[1] - 0.5), light_size, fc="green"))
        else:
            ax.add_patch(Circle((pos[0], pos[1] - 0.5), light_size, fc="red"))

        # East light
        if phase == 1:  # E-W green
            ax.add_patch(Circle((pos[0] + 0.5, pos[1]), light_size, fc="green"))
        else:
            ax.add_patch(Circle((pos[0] + 0.5, pos[1]), light_size, fc="red"))

        # West light
        if phase == 1:  # E-W green
            ax.add_patch(Circle((pos[0] - 0.5, pos[1]), light_size, fc="green"))
        else:
            ax.add_patch(Circle((pos[0] - 0.5, pos[1]), light_size, fc="red"))

        # Draw simplified queue representations as colored boxes with sizes
        # proportional to queue lengths
        north_queue = queues.get("north", 0)
        south_queue = queues.get("south", 0)
        east_queue = queues.get("east", 0)
        west_queue = queues.get("west", 0)

        # Scale for better visibility
        queue_scale = 0.05

        # North queue
        ax.add_patch(
            Rectangle(
                (pos[0] - 0.3, pos[1] + 0.6),
                0.6,
                north_queue * queue_scale,
                fc="blue",
                alpha=0.6,
                ec="black",
            )
        )

        # South queue
        ax.add_patch(
            Rectangle(
                (pos[0] - 0.3, pos[1] - 0.6 - south_queue * queue_scale),
                0.6,
                south_queue * queue_scale,
                fc="green",
                alpha=0.6,
                ec="black",
            )
        )

        # East queue
        ax.add_patch(
            Rectangle(
                (pos[0] + 0.6, pos[1] - 0.3),
                east_queue * queue_scale,
                0.6,
                fc="red",
                alpha=0.6,
                ec="black",
            )
        )

        # West queue
        ax.add_patch(
            Rectangle(
                (pos[0] - 0.6 - west_queue * queue_scale, pos[1] - 0.3),
                west_queue * queue_scale,
                0.6,
                fc="purple",
                alpha=0.6,
                ec="black",
            )
        )

        # Add intersection ID
        ax.text(
            pos[0],
            pos[1],
            id,
            ha="center",
            va="center",
            bbox=dict(facecolor="white", alpha=0.7, ec="black"),
        )

    # Set limits
    margin = 2
    x_min = min(pos[0] for pos in positions.values()) - margin
    x_max = max(pos[0] for pos in positions.values()) + margin
    y_min = min(pos[1] for pos in positions.values()) - margin
    y_max = max(pos[1] for pos in positions.values()) + margin

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")

    # Add legend for queue colors
    legend_elements = [
        Rectangle((0, 0), 1, 1, fc="blue", alpha=0.6, label="North Queue"),
        Rectangle((0, 0), 1, 1, fc="green", alpha=0.6, label="South Queue"),
        Rectangle((0, 0), 1, 1, fc="red", alpha=0.6, label="East Queue"),
        Rectangle((0, 0), 1, 1, fc="purple", alpha=0.6, label="West Queue"),
    ]

    ax.legend(handles=legend_elements, loc="upper right")

    # Add title
    ax.set_title(title)

    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
