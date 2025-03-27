#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Traffic Signal Control Environment - Single Intersection

This module implements a Gymnasium environment for a single traffic intersection
control problem, as described in the paper "Adaptive Traffic Signal Control with
Reinforcement Learning".

The environment follows the Gymnasium interface and can be used with various RL
algorithms including Q-learning, DQN, and A2C.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import random
from typing import Dict, Tuple, Optional, Any


class IntersectionEnv(gym.Env):
    """
    A Gymnasium environment for traffic signal control at a single intersection.

    This environment simulates a four-way intersection with configurable traffic
    patterns. The agent controls the traffic signals, deciding when to change
    the signal phase to optimize traffic flow.

    State space:
        - Queue lengths for each approach (N, S, E, W)
        - Waiting times for each approach
        - Traffic density on incoming/outgoing roads
        - Current signal phase and elapsed time
        - Optional historical traffic patterns

    Action space:
        - Discrete: Change to specific phase configuration
        - Or: Keep current phase or switch to next phase

    Reward:
        Weighted sum of various traffic metrics:
        - Queue lengths (negative)
        - Waiting times (negative)
        - Throughput (positive)
        - Number of signal switches (negative)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self, render_mode: Optional[str] = None, config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the intersection environment.

        Args:
            render_mode: Mode for rendering the environment
            config: Configuration parameters
        """
        # Default configuration
        self.config = {
            "max_time_steps": 1000,
            "arrival_rates": {"north": 0.2, "south": 0.2, "east": 0.3, "west": 0.3},
            "departure_rates": {
                "base": 0.4,
                "max_departure": 3,  # Max vehicles that can depart per step
            },
            "max_queue_length": 30,
            "max_wait_time": 100,
            "use_density": True,
            "use_waiting_time": True,
            "reward_weights": {
                "queue_length": -1.0,
                "wait_time": -0.5,
                "throughput": 1.0,
                "switch_penalty": -2.0,
            },
            "yellow_time": 2,  # Duration of yellow phase in time steps
            "min_green_time": 5,  # Minimum green time before allowing phase change
            "random_seed": None,
        }

        # Update configuration if provided
        if config is not None:
            self.config.update(config)

        # Set random seed
        if self.config["random_seed"] is not None:
            np.random.seed(self.config["random_seed"])
            random.seed(self.config["random_seed"])

        # Traffic state variables
        self.queues = {"north": 0, "south": 0, "east": 0, "west": 0}

        self.waiting_times = {
            "north": np.zeros(self.config["max_queue_length"]),
            "south": np.zeros(self.config["max_queue_length"]),
            "east": np.zeros(self.config["max_queue_length"]),
            "west": np.zeros(self.config["max_queue_length"]),
        }

        self.densities = {"north": 0.0, "south": 0.0, "east": 0.0, "west": 0.0}

        # Signal state variables
        # Phases: 0 = N-S green, E-W red; 1 = N-S red, E-W green; 2 = N-S yellow, E-W red; 3 = N-S red, E-W yellow
        self.current_phase = 0
        self.phase_time = 0  # Time elapsed in current phase
        self.yellow_phase_active = False

        # Performance metrics
        self.total_wait_time = 0
        self.total_throughput = 0
        self.total_switches = 0

        # Time variables
        self.time_step = 0
        self.done = False

        # Define action space: 0 = keep current phase, 1 = switch phase
        self.action_space = spaces.Discrete(2)

        # Define observation space
        observation_low = np.array(
            # Queue lengths (N,S,E,W), waiting times (N,S,E,W), densities (N,S,E,W), phase, phase time
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            dtype=np.float32,
        )

        observation_high = np.array(
            [
                self.config["max_queue_length"],
                self.config["max_queue_length"],
                self.config["max_queue_length"],
                self.config["max_queue_length"],
                self.config["max_wait_time"],
                self.config["max_wait_time"],
                self.config["max_wait_time"],
                self.config["max_wait_time"],
                1.0,
                1.0,
                1.0,
                1.0,  # Densities normalized to [0,1]
                3,  # Phase (0,1,2,3)
                100,  # Phase time - arbitrary large number
            ],
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=observation_low, high=observation_high, dtype=np.float32
        )

        # Rendering setup
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.fig = None
        self.ax = None

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to an initial state.

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            Initial observation and empty info dict
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # Reset traffic state
        initial_queue_size = 0  # Start with empty queues or randomize

        self.queues = {
            "north": initial_queue_size,
            "south": initial_queue_size,
            "east": initial_queue_size,
            "west": initial_queue_size,
        }

        self.waiting_times = {
            "north": np.zeros(self.config["max_queue_length"]),
            "south": np.zeros(self.config["max_queue_length"]),
            "east": np.zeros(self.config["max_queue_length"]),
            "west": np.zeros(self.config["max_queue_length"]),
        }

        self.densities = {
            "north": random.random() * 0.3,  # Random initial density
            "south": random.random() * 0.3,
            "east": random.random() * 0.3,
            "west": random.random() * 0.3,
        }

        # Reset signal state
        self.current_phase = 0  # Start with N-S green
        self.phase_time = 0
        self.yellow_phase_active = False

        # Reset performance metrics
        self.total_wait_time = 0
        self.total_throughput = 0
        self.total_switches = 0

        # Reset time variables
        self.time_step = 0
        self.done = False

        # Return initial observation
        observation = self._get_observation()
        info = {}

        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment with the given action.

        Args:
            action: Action to take (0 = keep phase, 1 = switch phase)

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Process action
        reward = 0

        # Handle phase switching logic with yellow phase
        if self.yellow_phase_active:
            # If in yellow phase, continue counting down
            if self.phase_time >= self.config["yellow_time"]:
                # Yellow phase completed, switch to the next green phase
                if self.current_phase == 2:  # N-S yellow
                    self.current_phase = 1  # Switch to E-W green
                else:  # E-W yellow
                    self.current_phase = 0  # Switch to N-S green

                self.phase_time = 0
                self.yellow_phase_active = False
            else:
                # Continue yellow phase
                self.phase_time += 1
        else:
            # Regular green phase logic
            if action == 1 and self.phase_time >= self.config["min_green_time"]:
                # Switch phase request when minimum green time is satisfied
                if self.current_phase == 0:  # N-S green
                    self.current_phase = 2  # N-S yellow
                else:  # E-W green
                    self.current_phase = 3  # E-W yellow

                self.yellow_phase_active = True
                self.phase_time = 0
                self.total_switches += 1
                reward += self.config["reward_weights"]["switch_penalty"]
            else:
                # Continue or extend current green phase
                self.phase_time += 1

        # Process vehicle arrivals based on arrival rates
        self._process_arrivals()

        # Process vehicle departures based on current phase
        throughput = self._process_departures()
        self.total_throughput += throughput
        reward += self.config["reward_weights"]["throughput"] * throughput

        # Update waiting times for all vehicles in queues
        wait_time_penalty = self._update_waiting_times()
        self.total_wait_time += wait_time_penalty
        reward += self.config["reward_weights"]["wait_time"] * wait_time_penalty

        # Apply queue length penalty
        queue_length_sum = sum(self.queues.values())
        reward += self.config["reward_weights"]["queue_length"] * queue_length_sum

        # Update time step
        self.time_step += 1

        # Check if done
        terminated = False
        truncated = self.time_step >= self.config["max_time_steps"]

        # Get observation and info
        observation = self._get_observation()
        info = {
            "queue_lengths": self.queues.copy(),
            "waiting_times": {
                k: v[v > 0].mean() if len(v[v > 0]) > 0 else 0
                for k, v in self.waiting_times.items()
            },
            "throughput": throughput,
            "cumulative_throughput": self.total_throughput,
            "cumulative_wait_time": self.total_wait_time,
            "phase": self.current_phase,
            "total_switches": self.total_switches,
        }

        # Render if requested
        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """
        Construct the observation vector.

        Returns:
            Numpy array with the current state representation
        """
        # Queue lengths
        queue_lengths = [
            self.queues["north"],
            self.queues["south"],
            self.queues["east"],
            self.queues["west"],
        ]

        # Average waiting times for each direction
        avg_waiting_times = [
            (
                self.waiting_times["north"][self.waiting_times["north"] > 0].mean()
                if len(self.waiting_times["north"][self.waiting_times["north"] > 0]) > 0
                else 0
            ),
            (
                self.waiting_times["south"][self.waiting_times["south"] > 0].mean()
                if len(self.waiting_times["south"][self.waiting_times["south"] > 0]) > 0
                else 0
            ),
            (
                self.waiting_times["east"][self.waiting_times["east"] > 0].mean()
                if len(self.waiting_times["east"][self.waiting_times["east"] > 0]) > 0
                else 0
            ),
            (
                self.waiting_times["west"][self.waiting_times["west"] > 0].mean()
                if len(self.waiting_times["west"][self.waiting_times["west"] > 0]) > 0
                else 0
            ),
        ]

        # Traffic densities
        densities = [
            self.densities["north"],
            self.densities["south"],
            self.densities["east"],
            self.densities["west"],
        ]

        # Phase information
        phase_info = [self.current_phase, self.phase_time]

        # Combine all features
        observation = np.array(
            queue_lengths + avg_waiting_times + densities + phase_info, dtype=np.float32
        )

        return observation

    def _process_arrivals(self) -> None:
        """Process vehicle arrivals based on arrival rates."""
        # North approach arrivals
        if random.random() < self.config["arrival_rates"]["north"]:
            if self.queues["north"] < self.config["max_queue_length"]:
                self.queues["north"] += 1

        # South approach arrivals
        if random.random() < self.config["arrival_rates"]["south"]:
            if self.queues["south"] < self.config["max_queue_length"]:
                self.queues["south"] += 1

        # East approach arrivals
        if random.random() < self.config["arrival_rates"]["east"]:
            if self.queues["east"] < self.config["max_queue_length"]:
                self.queues["east"] += 1

        # West approach arrivals
        if random.random() < self.config["arrival_rates"]["west"]:
            if self.queues["west"] < self.config["max_queue_length"]:
                self.queues["west"] += 1

        # Update density based on arrivals
        self._update_density()

    def _process_departures(self) -> int:
        """
        Process vehicle departures based on current phase.

        Returns:
            Number of vehicles that departed in this step
        """
        total_departures = 0

        # Only process departures during green phases
        if self.current_phase == 0:  # N-S green
            # North departures
            north_departures = min(
                self.queues["north"],
                np.random.binomial(
                    n=self.config["departure_rates"]["max_departure"],
                    p=self.config["departure_rates"]["base"],
                ),
            )
            self.queues["north"] -= north_departures

            # South departures
            south_departures = min(
                self.queues["south"],
                np.random.binomial(
                    n=self.config["departure_rates"]["max_departure"],
                    p=self.config["departure_rates"]["base"],
                ),
            )
            self.queues["south"] -= south_departures

            total_departures = north_departures + south_departures

        elif self.current_phase == 1:  # E-W green
            # East departures
            east_departures = min(
                self.queues["east"],
                np.random.binomial(
                    n=self.config["departure_rates"]["max_departure"],
                    p=self.config["departure_rates"]["base"],
                ),
            )
            self.queues["east"] -= east_departures

            # West departures
            west_departures = min(
                self.queues["west"],
                np.random.binomial(
                    n=self.config["departure_rates"]["max_departure"],
                    p=self.config["departure_rates"]["base"],
                ),
            )
            self.queues["west"] -= west_departures

            total_departures = east_departures + west_departures

        # Yellow phases don't allow departures in this simplified model

        return total_departures

    def _update_waiting_times(self) -> float:
        """
        Update waiting times for all vehicles in queues.

        Returns:
            Total waiting time penalty for this step
        """
        total_wait_time = 0

        # Update north queue waiting times
        if self.queues["north"] > 0:
            self.waiting_times["north"][: self.queues["north"]] += 1
            total_wait_time += np.sum(
                self.waiting_times["north"][: self.queues["north"]]
            )

        # Update south queue waiting times
        if self.queues["south"] > 0:
            self.waiting_times["south"][: self.queues["south"]] += 1
            total_wait_time += np.sum(
                self.waiting_times["south"][: self.queues["south"]]
            )

        # Update east queue waiting times
        if self.queues["east"] > 0:
            self.waiting_times["east"][: self.queues["east"]] += 1
            total_wait_time += np.sum(self.waiting_times["east"][: self.queues["east"]])

        # Update west queue waiting times
        if self.queues["west"] > 0:
            self.waiting_times["west"][: self.queues["west"]] += 1
            total_wait_time += np.sum(self.waiting_times["west"][: self.queues["west"]])

        return total_wait_time

    def _update_density(self) -> None:
        """Update traffic density based on current queues and random variation."""
        # Base density on queue length with some randomness
        self.densities["north"] = min(
            1.0,
            self.queues["north"] / self.config["max_queue_length"]
            + random.random() * 0.1,
        )
        self.densities["south"] = min(
            1.0,
            self.queues["south"] / self.config["max_queue_length"]
            + random.random() * 0.1,
        )
        self.densities["east"] = min(
            1.0,
            self.queues["east"] / self.config["max_queue_length"]
            + random.random() * 0.1,
        )
        self.densities["west"] = min(
            1.0,
            self.queues["west"] / self.config["max_queue_length"]
            + random.random() * 0.1,
        )

    def render(self) -> Optional[np.ndarray]:
        """
        Render the environment.

        Returns:
            RGB array if render_mode is "rgb_array"
        """
        if self.render_mode is None:
            return None

        # Initialize figure if needed
        if self.fig is None:
            plt.ion()  # Enable interactive mode
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            self.fig.suptitle("Adaptive Traffic Signal Control", fontsize=16)

        # Clear the axis
        self.ax.clear()

        # Draw intersection
        self._draw_intersection()

        # Pause to update the figure
        plt.pause(0.1)

        if self.render_mode == "rgb_array":
            # Convert plot to RGB array
            self.fig.canvas.draw()
            image = np.array(self.fig.canvas.renderer.buffer_rgba())
            return image

        return None

    def _draw_intersection(self) -> None:
        """Draw the intersection with current state."""
        # Draw roads
        self.ax.plot([-2, 2], [0, 0], "k-", lw=3)  # East-West road
        self.ax.plot([0, 0], [-2, 2], "k-", lw=3)  # North-South road

        # Draw traffic lights
        light_size = 0.2
        light_colors = self._get_light_colors()

        # North traffic light
        self.ax.add_patch(Circle((0, 1), light_size, fc=light_colors["north"]))

        # South traffic light
        self.ax.add_patch(Circle((0, -1), light_size, fc=light_colors["south"]))

        # East traffic light
        self.ax.add_patch(Circle((1, 0), light_size, fc=light_colors["east"]))

        # West traffic light
        self.ax.add_patch(Circle((-1, 0), light_size, fc=light_colors["west"]))

        # Draw vehicles in queues
        self._draw_queues()

        # Add status text
        phase_names = ["N-S Green", "E-W Green", "N-S Yellow", "E-W Yellow"]
        phase_text = f"Phase: {phase_names[self.current_phase]} (t={self.phase_time})"
        self.ax.text(0, 2.5, phase_text, ha="center", fontsize=12)

        # Display queue lengths
        queue_text = f"Queues: N={self.queues['north']}, S={self.queues['south']}, E={self.queues['east']}, W={self.queues['west']}"
        self.ax.text(0, -2.5, queue_text, ha="center", fontsize=10)

        # Display time step
        time_text = f"Time Step: {self.time_step}"
        self.ax.text(-2.8, 2.8, time_text, ha="left", fontsize=10)

        # Display throughput
        throughput_text = f"Total Throughput: {self.total_throughput}"
        self.ax.text(-2.8, 2.5, throughput_text, ha="left", fontsize=10)

        # Set plot limits and remove ticks
        self.ax.set_xlim(-3, 3)
        self.ax.set_ylim(-3, 3)
        self.ax.set_aspect("equal")
        self.ax.set_xticks([])
        self.ax.set_yticks([])

    def _get_light_colors(self) -> Dict[str, str]:
        """Get the colors for traffic lights based on current phase."""
        light_colors = {"north": "red", "south": "red", "east": "red", "west": "red"}

        if self.current_phase == 0:  # N-S green
            light_colors["north"] = "green"
            light_colors["south"] = "green"
        elif self.current_phase == 1:  # E-W green
            light_colors["east"] = "green"
            light_colors["west"] = "green"
        elif self.current_phase == 2:  # N-S yellow
            light_colors["north"] = "yellow"
            light_colors["south"] = "yellow"
        elif self.current_phase == 3:  # E-W yellow
            light_colors["east"] = "yellow"
            light_colors["west"] = "yellow"

        return light_colors

    def _draw_queues(self) -> None:
        """Draw vehicles in queues."""
        # Vehicle dimensions
        car_width = 0.15
        car_length = 0.3

        # Color map for variety
        colors = plt.cm.tab10.colors

        # North queue
        for i in range(min(self.queues["north"], 15)):  # Limit visual queue length
            y_pos = 1.2 + i * 0.35
            color = colors[i % len(colors)]
            self.ax.add_patch(
                Rectangle(
                    (-car_width / 2, y_pos),
                    car_width,
                    car_length,
                    fc=color,
                    ec="black",
                    lw=0.5,
                )
            )

        # South queue
        for i in range(min(self.queues["south"], 15)):
            y_pos = -1.2 - i * 0.35 - car_length
            color = colors[i % len(colors)]
            self.ax.add_patch(
                Rectangle(
                    (-car_width / 2, y_pos),
                    car_width,
                    car_length,
                    fc=color,
                    ec="black",
                    lw=0.5,
                )
            )

        # East queue
        for i in range(min(self.queues["east"], 15)):
            x_pos = 1.2 + i * 0.35
            color = colors[i % len(colors)]
            self.ax.add_patch(
                Rectangle(
                    (x_pos, -car_width / 2),
                    car_length,
                    car_width,
                    fc=color,
                    ec="black",
                    lw=0.5,
                )
            )

        # West queue
        for i in range(min(self.queues["west"], 15)):
            x_pos = -1.2 - i * 0.35 - car_length
            color = colors[i % len(colors)]
            self.ax.add_patch(
                Rectangle(
                    (x_pos, -car_width / 2),
                    car_length,
                    car_width,
                    fc=color,
                    ec="black",
                    lw=0.5,
                )
            )

    def close(self) -> None:
        """Close the environment and release resources."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None


# Example usage
if __name__ == "__main__":
    # Create environment
    env = IntersectionEnv(render_mode="human")

    # Reset environment
    obs, _ = env.reset()

    # Run simulation
    done = False
    truncated = False

    while not (done or truncated):
        # Random action
        action = env.action_space.sample()

        # Step environment
        obs, reward, done, truncated, info = env.step(action)

        # Print info
        print(f"Step: {env.time_step}, Action: {action}, Reward: {reward:.2f}")
        print(f"Queue lengths: {info['queue_lengths']}")

        # Sleep for visualization
        plt.pause(0.5)

    env.close()
