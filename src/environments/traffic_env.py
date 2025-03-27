#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Traffic Signal Control Environment - Multiple Intersections

This module implements a Gymnasium environment for multi-intersection traffic
control, as described in the paper "Adaptive Traffic Signal Control with
Reinforcement Learning". It supports coordinated control of multiple
intersections in a traffic network.

The environment can be used with centralized or decentralized RL approaches.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
from typing import Dict, Tuple, List, Optional, Any, Union


class TrafficNetwork:
    """
    A traffic network with multiple connected intersections.

    This class represents the road network structure with intersections and
    connecting roads. It handles the topology and traffic flow between
    intersections.
    """

    def __init__(self, topology: str = "2x2_grid"):
        """
        Initialize the traffic network.

        Args:
            topology: Type of network topology to create
        """
        self.topology = topology
        self.graph = nx.DiGraph()
        self.intersections = {}
        self.roads = {}

        # Build the network according to the specified topology
        if topology == "2x2_grid":
            self._build_2x2_grid()
        elif topology == "corridor":
            self._build_corridor()
        else:
            raise ValueError(f"Unsupported network topology: {topology}")

    def _build_2x2_grid(self):
        """Build a 2x2 grid network with 4 intersections and 8 connecting roads."""
        # Create 4 intersections
        for i in range(2):
            for j in range(2):
                intersection_id = f"I{i}_{j}"
                pos = (i, j)
                self.intersections[intersection_id] = {
                    "position": pos,
                    "incoming_roads": [],
                    "outgoing_roads": [],
                }
                self.graph.add_node(intersection_id, pos=pos)

        # Connect intersections with roads
        # Horizontal roads (West to East)
        for i in range(2):
            self._add_road(f"I{i}_0", f"I{i}_1", f"R{i}_0_h")

        # Horizontal roads (East to West)
        for i in range(2):
            self._add_road(f"I{i}_1", f"I{i}_0", f"R{i}_1_h")

        # Vertical roads (South to North)
        for j in range(2):
            self._add_road(f"I0_{j}", f"I1_{j}", f"R0_{j}_v")

        # Vertical roads (North to South)
        for j in range(2):
            self._add_road(f"I1_{j}", f"I0_{j}", f"R1_{j}_v")

        # Add external roads
        # West boundary
        self._add_external_road("W0", f"I0_0", "RW0_in")
        self._add_external_road("W1", f"I1_0", "RW1_in")
        self._add_external_road(f"I0_0", "W0", "RW0_out")
        self._add_external_road(f"I1_0", "W1", "RW1_out")

        # East boundary
        self._add_external_road("E0", f"I0_1", "RE0_in")
        self._add_external_road("E1", f"I1_1", "RE1_in")
        self._add_external_road(f"I0_1", "E0", "RE0_out")
        self._add_external_road(f"I1_1", "E1", "RE1_out")

        # South boundary
        self._add_external_road("S0", f"I0_0", "RS0_in")
        self._add_external_road("S1", f"I0_1", "RS1_in")
        self._add_external_road(f"I0_0", "S0", "RS0_out")
        self._add_external_road(f"I0_1", "S1", "RS1_out")

        # North boundary
        self._add_external_road("N0", f"I1_0", "RN0_in")
        self._add_external_road("N1", f"I1_1", "RN1_in")
        self._add_external_road(f"I1_0", "N0", "RN0_out")
        self._add_external_road(f"I1_1", "N1", "RN1_out")

    def _build_corridor(self):
        """Build a corridor network with 3 intersections in a line."""
        # Create 3 intersections
        for i in range(3):
            intersection_id = f"I{i}"
            pos = (i, 0)
            self.intersections[intersection_id] = {
                "position": pos,
                "incoming_roads": [],
                "outgoing_roads": [],
            }
            self.graph.add_node(intersection_id, pos=pos)

        # Connect intersections with roads (both directions)
        for i in range(2):
            self._add_road(f"I{i}", f"I{i+1}", f"R{i}_{i+1}")
            self._add_road(f"I{i+1}", f"I{i}", f"R{i+1}_{i}")

        # Add external roads
        # West boundary
        self._add_external_road("W", "I0", "RW_in")
        self._add_external_road("I0", "W", "RW_out")

        # East boundary
        self._add_external_road("E", "I2", "RE_in")
        self._add_external_road("I2", "E", "RE_out")

        # North/South cross roads
        for i in range(3):
            self._add_external_road(f"N{i}", f"I{i}", f"RN{i}_in")
            self._add_external_road(f"I{i}", f"N{i}", f"RN{i}_out")
            self._add_external_road(f"S{i}", f"I{i}", f"RS{i}_in")
            self._add_external_road(f"I{i}", f"S{i}", f"RS{i}_out")

    def _add_road(self, from_id: str, to_id: str, road_id: str):
        """Add a road between two intersections."""
        self.roads[road_id] = {
            "from_intersection": from_id,
            "to_intersection": to_id,
            "length": 1.0,  # Default length
            "queue": 0,
            "capacity": 20,
            "is_external": False,
        }

        self.graph.add_edge(from_id, to_id, road_id=road_id)
        self.intersections[from_id]["outgoing_roads"].append(road_id)
        self.intersections[to_id]["incoming_roads"].append(road_id)

    def _add_external_road(self, from_id: str, to_id: str, road_id: str):
        """Add an external road (from/to outside the network)."""
        if from_id.startswith(("N", "S", "E", "W")):
            # External source
            self.roads[road_id] = {
                "from_intersection": from_id,
                "to_intersection": to_id,
                "length": 1.0,
                "queue": 0,
                "capacity": 30,
                "is_external": True,
                "is_source": True,
            }

            # Don't add external nodes to the graph
            if to_id in self.intersections:
                self.intersections[to_id]["incoming_roads"].append(road_id)

        else:
            # External sink
            self.roads[road_id] = {
                "from_intersection": from_id,
                "to_intersection": to_id,
                "length": 1.0,
                "queue": 0,
                "capacity": 30,
                "is_external": True,
                "is_source": False,
            }

            if from_id in self.intersections:
                self.intersections[from_id]["outgoing_roads"].append(road_id)

    def reset(self):
        """Reset all road queues to initial state."""
        for road_id in self.roads:
            self.roads[road_id]["queue"] = 0

    def get_incoming_roads(self, intersection_id: str) -> List[str]:
        """Get list of incoming roads for an intersection."""
        return self.intersections[intersection_id]["incoming_roads"]

    def get_outgoing_roads(self, intersection_id: str) -> List[str]:
        """Get list of outgoing roads for an intersection."""
        return self.intersections[intersection_id]["outgoing_roads"]

    def get_queue(self, road_id: str) -> int:
        """Get the current queue length for a road."""
        return self.roads[road_id]["queue"]

    def set_queue(self, road_id: str, value: int):
        """Set the queue length for a road."""
        self.roads[road_id]["queue"] = max(
            0, min(value, self.roads[road_id]["capacity"])
        )

    def increment_queue(self, road_id: str, amount: int = 1) -> bool:
        """
        Increment the queue length for a road.

        Returns:
            True if successful, False if queue is at capacity
        """
        if self.roads[road_id]["queue"] < self.roads[road_id]["capacity"]:
            self.roads[road_id]["queue"] += amount
            return True
        return False

    def decrement_queue(self, road_id: str, amount: int = 1) -> int:
        """
        Decrement the queue length for a road.

        Returns:
            Actual amount decremented
        """
        actual_amount = min(amount, self.roads[road_id]["queue"])
        self.roads[road_id]["queue"] -= actual_amount
        return actual_amount

    def get_total_queue(self) -> int:
        """Get the total queue length across all roads."""
        return sum(road["queue"] for road in self.roads.values())

    def plot_network(self, ax=None, show_queues=True):
        """Plot the network graph with current traffic state."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))

        # Extract positions from nodes for drawing
        pos = nx.get_node_attributes(self.graph, "pos")

        # Draw intersections
        nx.draw_networkx_nodes(
            self.graph,
            pos,
            nodelist=[n for n in self.graph.nodes() if n.startswith("I")],
            node_color="lightblue",
            node_size=500,
            edgecolors="black",
            ax=ax,
        )

        # Draw roads
        edge_colors = []
        edge_widths = []

        for u, v, data in self.graph.edges(data=True):
            road_id = data.get("road_id")
            if road_id in self.roads:
                # Calculate color and width based on queue length
                queue_ratio = (
                    self.roads[road_id]["queue"] / self.roads[road_id]["capacity"]
                )
                if queue_ratio < 0.3:
                    edge_colors.append("green")
                elif queue_ratio < 0.7:
                    edge_colors.append("orange")
                else:
                    edge_colors.append("red")

                # Width based on queue
                edge_widths.append(1 + 3 * queue_ratio)
            else:
                edge_colors.append("gray")
                edge_widths.append(1)

        # Draw edges with arrows
        nx.draw_networkx_edges(
            self.graph,
            pos,
            edgelist=[(u, v) for u, v, _ in self.graph.edges(data=True)],
            edge_color=edge_colors,
            width=edge_widths,
            arrowsize=15,
            connectionstyle="arc3,rad=0.1",
            ax=ax,
        )

        # Draw intersection labels
        nx.draw_networkx_labels(
            self.graph, pos, font_size=10, font_weight="bold", ax=ax
        )

        # Draw queue numbers on roads if requested
        if show_queues:
            edge_labels = {}
            for u, v, data in self.graph.edges(data=True):
                road_id = data.get("road_id")
                if road_id in self.roads:
                    edge_labels[(u, v)] = str(self.roads[road_id]["queue"])

            nx.draw_networkx_edge_labels(
                self.graph,
                pos,
                edge_labels=edge_labels,
                font_color="black",
                font_size=8,
                ax=ax,
            )

        ax.set_title("Traffic Network State")
        ax.axis("off")
        return ax


class Intersection:
    """
    A single intersection with traffic signals in the network.

    This class represents an intersection with traffic signals, queues on
    incoming roads, and logic for signal control and vehicle movements.
    """

    def __init__(self, intersection_id: str, network: TrafficNetwork):
        """
        Initialize an intersection.

        Args:
            intersection_id: Unique identifier for this intersection
            network: Reference to the traffic network
        """
        self.id = intersection_id
        self.network = network

        # Get incoming and outgoing roads from network
        self.incoming_roads = network.get_incoming_roads(intersection_id)
        self.outgoing_roads = network.get_outgoing_roads(intersection_id)

        # Define phases
        # A phase maps certain incoming roads to outgoing roads
        # For simplicity, we model NS and EW phases
        self.phases = self._setup_phases()

        # Current phase (0: NS, 1: EW, 2: NS yellow, 3: EW yellow)
        self.current_phase = 0
        self.phase_time = 0
        self.yellow_phase_active = False

        # Signal phase configuration
        self.min_green_time = 5
        self.yellow_time = 2

        # Stats
        self.throughput = 0
        self.total_waiting_time = 0
        self.last_action = 0  # 0: keep, 1: switch

    def _setup_phases(self) -> List[Dict[str, List[str]]]:
        """
        Set up signal phases for the intersection.

        Phases define which incoming roads have green signals and map to
        which outgoing roads traffic can flow to.

        Returns:
            List of phase dictionaries
        """
        phases = []

        # Identify North-South and East-West roads based on IDs
        # This is a simplification - in a real implementation this would be
        # based on the actual geometry of the roads
        ns_incoming = []
        ew_incoming = []

        # Categorize roads
        for road_id in self.incoming_roads:
            road = self.network.roads[road_id]
            # Very simplistic - just using road naming convention
            if "v" in road_id or "N" in road_id or "S" in road_id:
                ns_incoming.append(road_id)
            else:
                ew_incoming.append(road_id)

        # Phase 0: North-South Green
        ns_phase = {"green_roads": ns_incoming, "red_roads": ew_incoming}
        phases.append(ns_phase)

        # Phase 1: East-West Green
        ew_phase = {"green_roads": ew_incoming, "red_roads": ns_incoming}
        phases.append(ew_phase)

        return phases

    def step(self, action: int, departure_rate: float = 0.4) -> int:
        """
        Take a step at this intersection based on the action.

        Args:
            action: 0 = keep current phase, 1 = switch phase
            departure_rate: Rate at which vehicles depart when signal is green

        Returns:
            Number of vehicles that moved through the intersection
        """
        throughput = 0

        # Handle phase switching with yellow phase
        if self.yellow_phase_active:
            # If in yellow phase, continue counting down
            if self.phase_time >= self.yellow_time:
                # Yellow phase completed, switch to the next green phase
                if self.current_phase == 2:  # NS yellow
                    self.current_phase = 1  # Switch to EW green
                else:  # EW yellow
                    self.current_phase = 0  # Switch to NS green

                self.phase_time = 0
                self.yellow_phase_active = False
            else:
                # Continue yellow phase
                self.phase_time += 1
        else:
            # Regular green phase logic
            if action == 1 and self.phase_time >= self.min_green_time:
                # Switch phase request when minimum green time is satisfied
                if self.current_phase == 0:  # NS green
                    self.current_phase = 2  # NS yellow
                else:  # EW green
                    self.current_phase = 3  # EW yellow

                self.yellow_phase_active = True
                self.phase_time = 0
                self.last_action = 1
            else:
                # Continue or extend current green phase
                self.phase_time += 1
                self.last_action = 0

        # Process vehicle movements (only during green phases)
        if self.current_phase == 0 or self.current_phase == 1:  # Green phases
            active_phase = self.current_phase
            green_roads = self.phases[active_phase]["green_roads"]

            # Process each incoming green road
            for road_id in green_roads:
                # Probabilistic departure model
                max_departure = min(3, self.network.get_queue(road_id))
                departures = 0

                # Binomial model for departures
                if max_departure > 0:
                    departures = min(
                        max_departure,
                        np.random.binomial(n=max_departure, p=departure_rate),
                    )

                if departures > 0:
                    # Process the departures
                    actual_departed = self.network.decrement_queue(road_id, departures)
                    throughput += actual_departed

                    # Distribute to outgoing roads (random assignment for simplicity)
                    if self.outgoing_roads:
                        for _ in range(actual_departed):
                            # Random outgoing road selection
                            target_road = random.choice(self.outgoing_roads)
                            self.network.increment_queue(target_road)

        # Update stats
        self.throughput += throughput

        # Calculate waiting time penalty - all vehicles in red roads
        waiting_time = 0
        if self.current_phase == 0 or self.current_phase == 2:  # NS green or yellow
            red_roads = self.phases[1]["green_roads"]  # EW roads are red
        else:  # EW green or yellow
            red_roads = self.phases[0]["green_roads"]  # NS roads are red

        for road_id in red_roads:
            waiting_time += self.network.get_queue(road_id)

        self.total_waiting_time += waiting_time

        return throughput

    def get_state(self) -> np.ndarray:
        """
        Get the state representation for this intersection.

        Returns:
            Numpy array with intersection state
        """
        # Include queue lengths for all incoming roads
        queue_lengths = [
            self.network.get_queue(road_id) for road_id in self.incoming_roads
        ]

        # Phase information
        phase_info = [self.current_phase, self.phase_time]

        # Combine all features
        state = np.array(queue_lengths + phase_info, dtype=np.float32)

        return state

    def reset(self):
        """Reset the intersection to initial state."""
        self.current_phase = 0
        self.phase_time = 0
        self.yellow_phase_active = False
        self.throughput = 0
        self.total_waiting_time = 0
        self.last_action = 0


class TrafficMultiEnv(gym.Env):
    """
    A Gymnasium environment for multi-intersection traffic signal control.

    This environment simulates multiple intersections in a traffic network,
    allowing for coordinated traffic signal control using RL.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self, render_mode: Optional[str] = None, config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the traffic environment.

        Args:
            render_mode: Mode for rendering the environment
            config: Configuration parameters
        """
        # Default configuration
        self.config = {
            "topology": "2x2_grid",
            "max_time_steps": 1000,
            "arrival_rates": {"default": 0.2, "peak": 0.4},
            "departure_rate": 0.4,
            "control_mode": "decentralized",  # or "centralized"
            "reward_weights": {
                "queue_length": -1.0,
                "wait_time": -0.5,
                "throughput": 1.0,
                "switch_penalty": -2.0,
            },
            "random_seed": None,
        }

        # Update configuration if provided
        if config is not None:
            self.config.update(config)

        # Set random seed
        if self.config["random_seed"] is not None:
            np.random.seed(self.config["random_seed"])
            random.seed(self.config["random_seed"])

        # Create network
        self.network = TrafficNetwork(topology=self.config["topology"])

        # Create intersections
        self.intersections = {}
        for intersection_id in self.network.intersections:
            self.intersections[intersection_id] = Intersection(
                intersection_id, self.network
            )

        # Time variables
        self.time_step = 0
        self.done = False

        # Performance metrics
        self.total_throughput = 0
        self.total_wait_time = 0
        self.total_queue = 0

        # Define action and observation spaces based on control mode
        if self.config["control_mode"] == "centralized":
            # Centralized control: one action per intersection
            self.action_space = spaces.MultiDiscrete([2] * len(self.intersections))

            # Combined observation space for all intersections
            # Each intersection: queue lengths + phase info
            max_incoming_roads = max(
                len(self.network.get_incoming_roads(i_id))
                for i_id in self.intersections
            )

            obs_dim = max_incoming_roads + 2  # queue lengths + phase + phase_time
            obs_high = np.array([30] * max_incoming_roads + [3, 100])  # Example limits

            self.observation_space = spaces.Box(
                low=np.zeros(obs_dim * len(self.intersections)),
                high=np.tile(obs_high, len(self.intersections)),
                dtype=np.float32,
            )
        else:
            # Decentralized control: define spaces per intersection
            self.action_spaces = {}
            self.observation_spaces = {}

            for i_id, intersection in self.intersections.items():
                # Action space: 0 = keep, 1 = switch
                self.action_spaces[i_id] = spaces.Discrete(2)

                # Observation space: queue lengths + phase info
                n_incoming = len(intersection.incoming_roads)
                obs_dim = n_incoming + 2  # queue lengths + phase + phase_time

                obs_high = np.array([30] * n_incoming + [3, 100])  # Example limits

                self.observation_spaces[i_id] = spaces.Box(
                    low=np.zeros(obs_dim), high=obs_high, dtype=np.float32
                )

        # For gym compatibility, set a single action/observation space
        # if in decentralized mode, this is just a placeholder
        if self.config["control_mode"] == "decentralized":
            first_id = list(self.intersections.keys())[0]
            self.action_space = self.action_spaces[first_id]
            self.observation_space = self.observation_spaces[first_id]

        # Rendering setup
        self.render_mode = render_mode
        self.fig = None
        self.axes = None

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Union[Dict[str, np.ndarray], np.ndarray]:
        """
        Reset the environment to an initial state.

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            Initial observation
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # Reset network
        self.network.reset()

        # Reset intersections
        for intersection in self.intersections.values():
            intersection.reset()

        # Reset time and metrics
        self.time_step = 0
        self.total_throughput = 0
        self.total_wait_time = 0
        self.total_queue = 0
        self.done = False

        # Add initial traffic
        self._generate_initial_traffic()

        # Get initial observation
        observation = self._get_observation()
        info = {}

        return observation, info

    def step(
        self, action: Union[np.ndarray, Dict[str, int]]
    ) -> Tuple[
        Union[Dict[str, np.ndarray], np.ndarray], float, bool, bool, Dict[str, Any]
    ]:
        """
        Take a step in the environment.

        Args:
            action: Action(s) to take, either an array for centralized or dict for decentralized

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Process action based on control mode
        if self.config["control_mode"] == "centralized":
            if not isinstance(action, np.ndarray):
                action = np.array(action)

            # Map actions to intersections
            intersection_actions = {}
            for i, i_id in enumerate(sorted(self.intersections.keys())):
                intersection_actions[i_id] = int(action[i])
        else:
            # Decentralized mode - action is a dict mapping intersection IDs to actions
            intersection_actions = action

        # Process vehicle arrivals
        self._process_arrivals()

        # Process intersection steps
        step_throughput = 0
        for i_id, action in intersection_actions.items():
            intersection = self.intersections[i_id]
            throughput = intersection.step(
                action, departure_rate=self.config["departure_rate"]
            )
            step_throughput += throughput

        self.total_throughput += step_throughput

        # Update total queue
        self.total_queue = self.network.get_total_queue()

        # Calculate reward
        reward = self._calculate_reward(step_throughput, intersection_actions)

        # Update time step
        self.time_step += 1

        # Check if done
        terminated = False
        truncated = self.time_step >= self.config["max_time_steps"]

        # Get observation and info
        observation = self._get_observation()
        info = {
            "throughput": step_throughput,
            "cumulative_throughput": self.total_throughput,
            "total_queue": self.total_queue,
            "intersection_states": {
                i_id: {
                    "phase": intersection.current_phase,
                    "phase_time": intersection.phase_time,
                    "throughput": intersection.throughput,
                }
                for i_id, intersection in self.intersections.items()
            },
        }

        # Render if requested
        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info

    def _generate_initial_traffic(self):
        """Generate initial traffic in the network."""
        # Add some vehicles to incoming external roads
        for road_id, road in self.network.roads.items():
            if road.get("is_external", False) and road.get("is_source", False):
                # External source road
                queue = np.random.randint(0, 5)  # 0-4 vehicles initially
                self.network.set_queue(road_id, queue)

    def _process_arrivals(self):
        """Process vehicle arrivals at network boundaries."""
        # Generate arrivals at external source roads
        for road_id, road in self.network.roads.items():
            if road.get("is_external", False) and road.get("is_source", False):
                # External source road - add vehicles based on arrival rate
                if random.random() < self.config["arrival_rates"]["default"]:
                    self.network.increment_queue(road_id)

    def _calculate_reward(
        self, throughput: int, intersection_actions: Dict[str, int]
    ) -> Union[float, Dict[str, float]]:
        """
        Calculate reward(s) based on the environment state.

        Args:
            throughput: Total throughput in this step
            intersection_actions: Actions taken by each intersection

        Returns:
            Single reward for centralized control or dict of rewards for decentralized
        """
        if self.config["control_mode"] == "centralized":
            # Global reward based on overall performance
            reward = 0
            reward += self.config["reward_weights"]["throughput"] * throughput
            reward += self.config["reward_weights"]["queue_length"] * self.total_queue

            # Switching penalty
            switches = sum(
                1
                for i_id, intersection in self.intersections.items()
                if intersection.last_action == 1
            )
            reward += self.config["reward_weights"]["switch_penalty"] * switches

            return reward
        else:
            # Individual rewards for each intersection
            rewards = {}

            for i_id, intersection in self.intersections.items():
                local_reward = 0

                # Throughput reward
                local_reward += (
                    self.config["reward_weights"]["throughput"]
                    * intersection.throughput
                )

                # Queue length penalty - using incoming roads
                queue_sum = sum(
                    self.network.get_queue(road_id)
                    for road_id in intersection.incoming_roads
                )
                local_reward += (
                    self.config["reward_weights"]["queue_length"] * queue_sum
                )

                # Switch penalty
                if intersection.last_action == 1:
                    local_reward += self.config["reward_weights"]["switch_penalty"]

                rewards[i_id] = local_reward

            return rewards

    def _get_observation(self) -> Union[Dict[str, np.ndarray], np.ndarray]:
        """
        Get current observation from the environment.

        Returns:
            Observation dict or array depending on control mode
        """
        if self.config["control_mode"] == "centralized":
            # Combined observation for all intersections
            all_states = []

            for i_id in sorted(self.intersections.keys()):  # Sort for consistency
                intersection = self.intersections[i_id]
                state = intersection.get_state()
                all_states.append(state)

            return np.concatenate(all_states)
        else:
            # Individual observation for each intersection
            observations = {}

            for i_id, intersection in self.intersections.items():
                state = intersection.get_state()
                observations[i_id] = state

            return observations

    def render(self) -> Optional[np.ndarray]:
        """
        Render the current state of the environment.

        Returns:
            RGB array if render_mode is "rgb_array"
        """
        if self.render_mode is None:
            return None

        # Initialize figure if needed
        if self.fig is None:
            plt.ion()  # Enable interactive mode
            self.fig, self.axes = plt.subplots(
                1, 2, figsize=(15, 7), gridspec_kw={"width_ratios": [2, 1]}
            )
            self.fig.suptitle("Traffic Network Control Simulation", fontsize=16)

        # Clear axes
        for ax in self.axes:
            ax.clear()

        # Plot network on first axis
        self.network.plot_network(ax=self.axes[0], show_queues=True)

        # Plot stats on second axis
        self._plot_stats(ax=self.axes[1])

        # Update the figure
        plt.tight_layout()
        plt.pause(0.1)

        if self.render_mode == "rgb_array":
            # Convert plot to RGB array
            self.fig.canvas.draw()
            image = np.array(self.fig.canvas.renderer.buffer_rgba())
            return image

        return None

    def _plot_stats(self, ax=None):
        """Plot current statistics about the environment."""
        if ax is None:
            return

        # Create a list of metrics to display
        metrics = [
            f"Time Step: {self.time_step}",
            f"Total Queue: {self.total_queue}",
            f"Total Throughput: {self.total_throughput}",
            f"\nIntersection States:",
        ]

        # Add info for each intersection
        for i_id, intersection in self.intersections.items():
            phase_names = ["NS Green", "EW Green", "NS Yellow", "EW Yellow"]
            phase_name = phase_names[intersection.current_phase]

            metrics.append(f"{i_id}: {phase_name} (t={intersection.phase_time})")
            metrics.append(f"  Throughput: {intersection.throughput}")

        # Display as text
        ax.text(
            0.05,
            0.95,
            "\n".join(metrics),
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
        )

        ax.set_title("Simulation Statistics")
        ax.axis("off")

    def close(self):
        """Close the environment and release resources."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.axes = None


# Example usage
if __name__ == "__main__":
    # Create environment
    env = TrafficMultiEnv(
        render_mode="human",
        config={"topology": "2x2_grid", "control_mode": "decentralized"},
    )

    # Reset environment
    obs, _ = env.reset()

    # Run simulation with random actions
    for _ in range(100):
        # For decentralized control, create actions for each intersection
        actions = {}
        for i_id in env.intersections:
            actions[i_id] = random.randint(0, 1)

        # Step environment
        obs, reward, done, truncated, info = env.step(actions)

        # Print info
        print(
            f"Step: {env.time_step}, Throughput: {info['throughput']}, Total Queue: {info['total_queue']}"
        )

        # Slow down rendering for visibility
        plt.pause(0.5)

        if done or truncated:
            break

    env.close()
