#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate Traffic Signal Demo Image

This script generates a static image from the traffic signal demo for use in the
paper introduction. It creates a figure with three components:
1. A visualization of the intersection with traffic signals and queues
2. A plot showing the agent's learning curve (rewards over time)
3. A plot showing the reduction in average queue length over time
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch

# Add src to path to import local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the demo environment
from demos.traffic_signal_demo import TrafficSignalEnv


def create_demo_visualization():
    """Create a static visualization of the traffic signal control demo"""
    # Create figure
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1])

    # Intersection visualization
    ax_intersection = plt.subplot(gs[0, :])

    # Performance plots
    ax_reward = plt.subplot(gs[1, 0])
    ax_queue = plt.subplot(gs[1, 1])

    # Create and setup the environment
    env = TrafficSignalEnv()

    # Set up a busy traffic scenario
    env.queue_n = 8
    env.queue_s = 6
    env.queue_e = 12
    env.queue_w = 10
    env.phase = 1  # E-W green
    env.time = 45

    # Render the intersection
    render_intersection(env, ax_intersection)

    # Simulate learning curves
    episodes = 100
    x = np.arange(episodes)

    # Simulated reward improvement curve (negative rewards getting less negative)
    rewards = -500 + 300 * (1 - np.exp(-0.03 * x)) + np.random.normal(0, 20, episodes)
    rewards_smooth = np.convolve(rewards, np.ones(5) / 5, mode="same")

    # Simulated queue length reduction
    queue_lengths = 15 - 10 * (1 - np.exp(-0.04 * x)) + np.random.normal(0, 1, episodes)
    queue_smooth = np.convolve(queue_lengths, np.ones(5) / 5, mode="same")

    # Plot simulated learning curves
    ax_reward.plot(x, rewards_smooth, "b-")
    ax_reward.set_title("Episode Rewards")
    ax_reward.set_xlabel("Episode")
    ax_reward.set_ylabel("Total Reward")
    ax_reward.grid(True, alpha=0.3)

    ax_queue.plot(x, queue_smooth, "r-")
    ax_queue.set_title("Average Queue Length")
    ax_queue.set_xlabel("Episode")
    ax_queue.set_ylabel("Queue Length")
    ax_queue.grid(True, alpha=0.3)

    # Add comparison with fixed-time
    ax_queue.axhline(
        y=queue_smooth[-1] * 1.5,
        color="gray",
        linestyle="--",
        alpha=0.7,
        label="Fixed-time",
    )
    ax_queue.legend()

    # Add annotations
    ax_reward.annotate(
        "Learning progress",
        xy=(episodes * 0.8, rewards_smooth[int(episodes * 0.8)]),
        xytext=(episodes * 0.6, rewards_smooth[int(episodes * 0.8)] - 80),
        arrowprops=dict(facecolor="black", shrink=0.05, width=1.5, headwidth=8),
    )

    ax_queue.annotate(
        "Queue reduction",
        xy=(episodes * 0.7, queue_smooth[int(episodes * 0.7)]),
        xytext=(episodes * 0.5, queue_smooth[int(episodes * 0.7)] + 3),
        arrowprops=dict(facecolor="black", shrink=0.05, width=1.5, headwidth=8),
    )

    plt.tight_layout()

    # Save the figure
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "paper", "images"
    )
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "traffic_signal_demo.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")

    print(f"Demo image saved to: {output_path}")

    return fig


def render_intersection(env, ax):
    """Render the traffic intersection"""
    ax.clear()

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
    if env.phase == 0:  # N-S green
        ax.add_patch(Circle((0, 1), light_size, fc="green"))
    else:  # N-S red
        ax.add_patch(Circle((0, 1), light_size, fc="red"))

    # South traffic light
    if env.phase == 0:  # N-S green
        ax.add_patch(Circle((0, -1), light_size, fc="green"))
    else:  # N-S red
        ax.add_patch(Circle((0, -1), light_size, fc="red"))

    # East traffic light
    if env.phase == 1:  # E-W green
        ax.add_patch(Circle((1, 0), light_size, fc="green"))
    else:  # E-W red
        ax.add_patch(Circle((1, 0), light_size, fc="red"))

    # West traffic light
    if env.phase == 1:  # E-W green
        ax.add_patch(Circle((-1, 0), light_size, fc="green"))
    else:  # E-W red
        ax.add_patch(Circle((-1, 0), light_size, fc="red"))

    # Draw queues as stacked cars
    car_size = 0.1

    # North queue - add some variability to car appearance
    for i in range(min(env.queue_n, 10)):  # Show max 10 cars
        y_pos = 1.2 + (i * 0.2)
        color = plt.cm.tab10(random.randint(0, 9))
        ax.add_patch(Rectangle((-0.15, y_pos), 0.3, 0.15, fc=color, ec="black", lw=0.5))

    # South queue
    for i in range(min(env.queue_s, 10)):
        y_pos = -1.2 - (i * 0.2) - 0.15
        color = plt.cm.tab10(random.randint(0, 9))
        ax.add_patch(Rectangle((-0.15, y_pos), 0.3, 0.15, fc=color, ec="black", lw=0.5))

    # East queue
    for i in range(min(env.queue_e, 10)):
        x_pos = 1.2 + (i * 0.2)
        color = plt.cm.tab10(random.randint(0, 9))
        ax.add_patch(Rectangle((x_pos, -0.15), 0.15, 0.3, fc=color, ec="black", lw=0.5))

    # West queue
    for i in range(min(env.queue_w, 10)):
        x_pos = -1.2 - (i * 0.2) - 0.15
        color = plt.cm.tab10(random.randint(0, 9))
        ax.add_patch(Rectangle((x_pos, -0.15), 0.15, 0.3, fc=color, ec="black", lw=0.5))

    # Set plot limits and labels
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect("equal")
    ax.set_title(f"Traffic Signal Control - Step {env.time}")

    # Add current phase and decision
    phase_text = (
        "Current Phase: East-West Green"
        if env.phase == 1
        else "Current Phase: North-South Green"
    )
    ax.text(
        2.9,
        2.7,
        phase_text,
        ha="right",
        fontsize=12,
        bbox=dict(
            facecolor="white", alpha=1.0, edgecolor="black", boxstyle="round,pad=0.5"
        ),
    )

    # Add agent's action information
    reward = -(env.queue_n + env.queue_s + env.queue_e + env.queue_w)
    action_text = f"Queue Total: {env.queue_n + env.queue_s + env.queue_e + env.queue_w} | Reward: {reward}"
    ax.text(
        2.9,
        -2.7,
        action_text,
        ha="right",
        fontsize=12,
        bbox=dict(
            facecolor="white", alpha=1.0, edgecolor="black", boxstyle="round,pad=0.5"
        ),
    )

    # Set empty tick labels
    ax.set_xticks([])
    ax.set_yticks([])

    # Add RL agent box - move to top-left corner with better styling
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


if __name__ == "__main__":
    fig = create_demo_visualization()
    plt.show()
