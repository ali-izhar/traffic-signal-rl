#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SUMO Traffic Signal Control Environment

This module implements a Gymnasium environment for traffic signal control
that interfaces with the SUMO traffic simulator.

Key features:
- High-fidelity traffic simulation using SUMO (Simulation of Urban MObility)
- Realistic vehicle dynamics, including acceleration, braking, and lane-changing
- Detailed intersection geometry and traffic light phasing
- Support for real-world road networks and traffic patterns
- Collection of comprehensive traffic metrics (queue lengths, waiting times, emissions)

Recommended use:
- Final experiments for publication results
- Realistic validation of RL algorithms
- Experiments requiring detailed vehicle-level dynamics
- Studies involving complex traffic patterns and intersection geometries
- Transfer learning between simulation and real-world applications

This environment provides the most realistic simulation of the three environments
but requires SUMO to be installed and has slower execution speed compared to
the simplified IntersectionEnv.
"""

import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional, Any, List
import xml.etree.ElementTree as ET

# Check if SUMO_HOME is defined
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import traci
import sumolib


class SUMOIntersectionEnv(gym.Env):
    """
    A Gymnasium environment for traffic signal control using SUMO simulator.

    This environment interfaces with SUMO through TraCI to simulate traffic
    at a controlled intersection. It provides RL-compatible observation and
    action spaces consistent with the environment described in the paper.

    State space:
        - Queue lengths for each approach (N, S, E, W)
        - Waiting times for each approach
        - Traffic density on incoming/outgoing roads
        - Current signal phase and elapsed time

    Action space:
        - Discrete: Change to specific phase configuration

    Reward:
        Weighted sum of various traffic metrics:
        - Queue lengths (negative)
        - Waiting times (negative)
        - Throughput (positive)
        - Number of signal switches (negative)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}

    def __init__(
        self,
        config_file: str,
        render_mode: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the SUMO environment.

        Args:
            config_file: Path to SUMO configuration file (.sumocfg)
            render_mode: Mode for rendering the environment
            config: Configuration parameters
        """
        # Default configuration
        self.config = {
            "max_time_steps": 3600,
            "gui": False,  # Use SUMO GUI
            "reward_weights": {
                "queue_length": -1.0,
                "wait_time": -0.5,
                "throughput": 1.0,
                "switch_penalty": -2.0,
            },
            "yellow_time": 2,  # Duration of yellow phase in time steps
            "min_green_time": 5,  # Minimum green time before allowing phase change
            "delta_time": 1,  # Simulation step size in seconds
            "tls_id": "center",  # Traffic light ID
            "incoming_lanes": [
                "north_to_center_0",
                "north_to_center_1",
                "south_to_center_0",
                "south_to_center_1",
                "east_to_center_0",
                "east_to_center_1",
                "west_to_center_0",
                "west_to_center_1",
            ],
            "outgoing_lanes": [
                "center_to_north_0",
                "center_to_north_1",
                "center_to_south_0",
                "center_to_south_1",
                "center_to_east_0",
                "center_to_east_1",
                "center_to_west_0",
                "center_to_west_1",
            ],
            "phase_definitions": {
                0: {"name": "North-South green", "states": "GGrrGGrr"},
                1: {"name": "East-West green", "states": "rrGGrrGG"},
                2: {"name": "North-South yellow", "states": "yyrryyrr"},
                3: {"name": "East-West yellow", "states": "rryyrryy"},
            },
        }

        # Update configuration if provided
        if config is not None:
            self.config.update(config)

        # Store SUMO configuration file path
        self.config_file = config_file
        self.render_mode = render_mode

        # SUMO connection status
        self.sumo = None
        self.simulation_running = False

        # Traffic light information
        self.tls_id = self.config["tls_id"]
        self.phase_definitions = self.config["phase_definitions"]
        self.current_phase = 0
        self.phase_time = 0
        self.yellow_phase_active = False
        self.last_phase_change = 0

        # Performance metrics
        self.total_wait_time = 0
        self.total_throughput = 0
        self.total_switches = 0
        self.prev_vehicle_count = 0

        # Time variables
        self.time_step = 0

        # Define action space: 0 = keep current phase, 1 = switch phase
        self.action_space = spaces.Discrete(2)

        # Define observation space
        num_directions = 4  # N, S, E, W
        self.observation_space = spaces.Box(
            low=np.zeros(
                2 * num_directions + num_directions + 2
            ),  # Queue, wait, density, phase, phase_time
            high=np.array(
                [100] * num_directions
                + [300] * num_directions
                + [1.0] * num_directions
                + [3.0, 100.0]
            ),
            dtype=np.float32,
        )

        # Map lanes to directions for observation aggregation
        self.lane_to_direction = {}
        for lane in self.config["incoming_lanes"]:
            if "north" in lane:
                self.lane_to_direction[lane] = "north"
            elif "south" in lane:
                self.lane_to_direction[lane] = "south"
            elif "east" in lane:
                self.lane_to_direction[lane] = "east"
            elif "west" in lane:
                self.lane_to_direction[lane] = "west"

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to an initial state.

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            Initial observation and info dict
        """
        if seed is not None:
            np.random.seed(seed)

        # Close existing SUMO connection if active
        if self.simulation_running and self.sumo is not None:
            traci.close()
            self.simulation_running = False

        # Start SUMO
        sumo_cmd = ["sumo"]
        if self.render_mode == "human":
            sumo_cmd = ["sumo-gui"]

        # Convert path to absolute path if needed
        config_path = (
            os.path.abspath(self.config_file)
            if not os.path.isabs(self.config_file)
            else self.config_file
        )

        sumo_cmd.extend(
            [
                "-c",
                config_path,
                "--no-warnings",
                "--no-step-log",
                "--random",
                "--start",
                "--quit-on-end",
            ]
        )

        print(f"Starting SUMO with command: {' '.join(sumo_cmd)}")
        traci.start(sumo_cmd)
        self.sumo = traci
        self.simulation_running = True

        # Get traffic light information
        tls_ids = self.sumo.trafficlight.getIDList()
        print(f"Available traffic lights: {tls_ids}")

        if not tls_ids:
            print("ERROR: No traffic lights found in the network!")
            self.sumo.close()
            self.simulation_running = False
            raise ValueError("No traffic lights found in SUMO network")

        # Use the first traffic light if the specified one doesn't exist
        if self.tls_id not in tls_ids:
            self.tls_id = tls_ids[0]
            print(
                f"Specified traffic light '{self.config['tls_id']}' not found, using '{self.tls_id}' instead"
            )

        # Get the traffic light's program
        try:
            tl_logic = self.sumo.trafficlight.getAllProgramLogics(self.tls_id)[0]
            phases = tl_logic.phases

            # Update phase definitions based on actual traffic light configuration
            self.phase_definitions = {}

            # Find main green phases and yellow phases
            green_phases = []
            yellow_phases = []

            for i, phase in enumerate(phases):
                state = phase.state
                print(f"Phase {i}: {state} (duration: {phase.duration})")

                # Classify phase based on state (G = green, y = yellow, r = red)
                if "G" in state and "y" not in state:
                    green_phases.append((i, state))
                elif "y" in state:
                    yellow_phases.append((i, state))

            # If we have at least one green phase, use it
            if green_phases:
                for i, (phase_idx, state) in enumerate(green_phases):
                    self.phase_definitions[i * 2] = {
                        "name": f"Green phase {i}",
                        "states": state,
                    }

                # Add yellow phases if available
                if yellow_phases:
                    for i, (phase_idx, state) in enumerate(
                        yellow_phases[: len(green_phases)]
                    ):
                        self.phase_definitions[i * 2 + 1] = {
                            "name": f"Yellow phase {i}",
                            "states": state,
                        }
                else:
                    # Create default yellow phases if none exist
                    for i, (_, green_state) in enumerate(green_phases):
                        yellow_state = green_state.replace("G", "y")
                        self.phase_definitions[i * 2 + 1] = {
                            "name": f"Yellow phase {i}",
                            "states": yellow_state,
                        }
            else:
                # No green phases found, creating default
                print("WARNING: No green phases found, creating default phases")
                num_signals = len(phases[0].state) if phases else 8
                self.phase_definitions = {
                    0: {
                        "name": "Default green 1",
                        "states": "G" * (num_signals // 2) + "r" * (num_signals // 2),
                    },
                    1: {
                        "name": "Default yellow 1",
                        "states": "y" * (num_signals // 2) + "r" * (num_signals // 2),
                    },
                    2: {
                        "name": "Default green 2",
                        "states": "r" * (num_signals // 2) + "G" * (num_signals // 2),
                    },
                    3: {
                        "name": "Default yellow 2",
                        "states": "r" * (num_signals // 2) + "y" * (num_signals // 2),
                    },
                }

            print("Using phase definitions:")
            for phase_id, phase_info in self.phase_definitions.items():
                print(
                    f"  Phase {phase_id}: {phase_info['name']} - {phase_info['states']}"
                )

        except Exception as e:
            print(f"Error analyzing traffic light program: {e}")
            print("Using default phase definitions")

        # Reset state
        self.current_phase = 0
        self.phase_time = 0
        self.yellow_phase_active = False
        self.last_phase_change = 0

        # Reset metrics
        self.total_wait_time = 0
        self.total_throughput = 0
        self.total_switches = 0
        self.prev_vehicle_count = len(self.sumo.vehicle.getIDList())

        # Reset time
        self.time_step = 0

        # Set initial traffic light phase
        try:
            self.sumo.trafficlight.setRedYellowGreenState(
                self.tls_id, self.phase_definitions[self.current_phase]["states"]
            )
            print(
                f"Successfully set initial traffic light state to: {self.phase_definitions[self.current_phase]['states']}"
            )
        except Exception as e:
            print(f"Error setting traffic light state: {e}")
            print(f"Current phases available: {[p.state for p in phases]}")
            print("Using SUMO's default traffic light program")

        # Advance simulation by one step to apply initial state
        self.sumo.simulationStep()

        # Get observation
        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment with the given action.

        Args:
            action: Action to take (0 = keep phase, 1 = switch phase)

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Save previous state for reward calculation
        prev_waiting_time = self._get_total_waiting_time()

        # Process action (switch phase or keep current)
        reward = self._apply_action(action)

        # Advance the simulation
        self.sumo.simulationStep()
        self.time_step += 1
        self.phase_time += 1

        # Calculate additional reward components
        current_waiting_time = self._get_total_waiting_time()
        wait_time_diff = current_waiting_time - prev_waiting_time

        # Current throughput (vehicles that have completed their routes)
        current_vehicle_count = len(self.sumo.vehicle.getIDList())
        throughput = max(
            0,
            self.prev_vehicle_count
            - current_vehicle_count
            + self._get_new_vehicles_count(),
        )
        self.prev_vehicle_count = current_vehicle_count
        self.total_throughput += throughput

        # Update total waiting time
        self.total_wait_time += wait_time_diff

        # Calculate reward components
        reward += self.config["reward_weights"]["throughput"] * throughput
        reward += self.config["reward_weights"]["wait_time"] * wait_time_diff
        reward += (
            self.config["reward_weights"]["queue_length"]
            * self._get_total_queue_length()
        )

        # Get new observation and info
        observation = self._get_observation()
        info = self._get_info()

        # Check termination conditions
        terminated = False
        truncated = self.time_step >= self.config["max_time_steps"]

        if terminated or truncated:
            if self.simulation_running:
                self.sumo.close()
                self.simulation_running = False

        return observation, reward, terminated, truncated, info

    def _apply_action(self, action: int) -> float:
        """
        Apply the action to the traffic light controller.

        Args:
            action: 0 = keep current phase, 1 = switch to next phase

        Returns:
            Immediate reward for this action
        """
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

                # Apply the new phase
                self.sumo.trafficlight.setRedYellowGreenState(
                    self.tls_id, self.phase_definitions[self.current_phase]["states"]
                )

                self.phase_time = 0
                self.yellow_phase_active = False
                self.last_phase_change = self.time_step
            # No else needed - yellow continues automatically
        else:
            # Regular green phase logic
            if action == 1 and self.phase_time >= self.config["min_green_time"]:
                # Switch phase request when minimum green time is satisfied
                if self.current_phase == 0:  # N-S green
                    self.current_phase = 2  # N-S yellow
                else:  # E-W green
                    self.current_phase = 3  # E-W yellow

                # Apply the new phase
                self.sumo.trafficlight.setRedYellowGreenState(
                    self.tls_id, self.phase_definitions[self.current_phase]["states"]
                )

                self.yellow_phase_active = True
                self.phase_time = 0
                self.total_switches += 1
                self.last_phase_change = self.time_step
                reward += self.config["reward_weights"]["switch_penalty"]
            # No else needed - green continues if no switch requested

        return reward

    def _get_observation(self) -> np.ndarray:
        """
        Construct the observation vector from the SUMO simulation state.

        Returns:
            Numpy array with the current state representation
        """
        # Get queue lengths for each direction
        queue_lengths = self._get_queue_lengths()

        # Get waiting times for each direction
        waiting_times = self._get_waiting_times()

        # Get traffic densities for each direction
        densities = self._get_densities()

        # Phase information
        phase_info = [float(self.current_phase), float(self.phase_time)]

        # Combine all features
        observation = np.array(
            list(queue_lengths.values())
            + list(waiting_times.values())
            + list(densities.values())
            + phase_info,
            dtype=np.float32,
        )

        return observation

    def _get_info(self) -> Dict[str, Any]:
        """
        Get additional information about the environment state.

        Returns:
            Dictionary with additional information
        """
        queue_lengths = self._get_queue_lengths()
        waiting_times = self._get_waiting_times()

        return {
            "queue_lengths": queue_lengths,
            "waiting_times": waiting_times,
            "cumulative_throughput": self.total_throughput,
            "cumulative_wait_time": self.total_wait_time,
            "phase": self.current_phase,
            "total_switches": self.total_switches,
            "time_step": self.time_step,
        }

    def _get_queue_lengths(self) -> Dict[str, float]:
        """
        Get the number of vehicles waiting at each approach.

        Returns:
            Dictionary mapping directions to queue lengths
        """
        queue_lengths = {"north": 0, "south": 0, "east": 0, "west": 0}

        # For each lane, count vehicles that are standing still (speed < 0.1 m/s)
        for lane_id in self.config["incoming_lanes"]:
            direction = self.lane_to_direction[lane_id]
            lane_vehicles = self.sumo.lane.getLastStepVehicleIDs(lane_id)

            for vehicle_id in lane_vehicles:
                if self.sumo.vehicle.getSpeed(vehicle_id) < 0.1:
                    queue_lengths[direction] += 1

        return queue_lengths

    def _get_waiting_times(self) -> Dict[str, float]:
        """
        Get the average waiting time of vehicles at each approach.

        Returns:
            Dictionary mapping directions to average waiting times
        """
        total_waiting_time = {"north": 0, "south": 0, "east": 0, "west": 0}
        vehicle_count = {"north": 0, "south": 0, "east": 0, "west": 0}

        # For each lane, sum the waiting times of vehicles
        for lane_id in self.config["incoming_lanes"]:
            direction = self.lane_to_direction[lane_id]
            lane_vehicles = self.sumo.lane.getLastStepVehicleIDs(lane_id)

            for vehicle_id in lane_vehicles:
                # Waiting time is time spent with speed < 0.1 m/s
                wait_time = self.sumo.vehicle.getAccumulatedWaitingTime(vehicle_id)
                total_waiting_time[direction] += wait_time
                vehicle_count[direction] += 1

        # Calculate average waiting time for each direction
        avg_waiting_time = {}
        for direction in total_waiting_time:
            if vehicle_count[direction] > 0:
                avg_waiting_time[direction] = (
                    total_waiting_time[direction] / vehicle_count[direction]
                )
            else:
                avg_waiting_time[direction] = 0.0

        return avg_waiting_time

    def _get_densities(self) -> Dict[str, float]:
        """
        Get the traffic density (vehicles per unit length) at each approach.

        Returns:
            Dictionary mapping directions to densities
        """
        densities = {"north": 0, "south": 0, "east": 0, "west": 0}

        # For each lane, calculate density as vehicles per meter
        for lane_id in self.config["incoming_lanes"]:
            direction = self.lane_to_direction[lane_id]
            lane_vehicles = self.sumo.lane.getLastStepVehicleNumber(lane_id)
            lane_length = self.sumo.lane.getLength(lane_id)

            if lane_length > 0:
                # Normalize by max possible vehicles (assuming 7.5m per vehicle)
                max_vehicles = lane_length / 7.5
                density = min(1.0, lane_vehicles / max_vehicles)
                densities[direction] += density

        # Normalize densities by number of lanes in each direction
        num_lanes = {"north": 0, "south": 0, "east": 0, "west": 0}
        for lane_id in self.config["incoming_lanes"]:
            direction = self.lane_to_direction[lane_id]
            num_lanes[direction] += 1

        for direction in densities:
            if num_lanes[direction] > 0:
                densities[direction] /= num_lanes[direction]

        return densities

    def _get_total_waiting_time(self) -> float:
        """
        Get the total waiting time of all vehicles in the simulation.

        Returns:
            Total waiting time in seconds
        """
        total_waiting_time = 0
        for vehicle_id in self.sumo.vehicle.getIDList():
            total_waiting_time += self.sumo.vehicle.getAccumulatedWaitingTime(
                vehicle_id
            )

        return total_waiting_time

    def _get_total_queue_length(self) -> int:
        """
        Get the total number of vehicles waiting at the intersection.

        Returns:
            Total queue length
        """
        queue_lengths = self._get_queue_lengths()
        return sum(queue_lengths.values())

    def _get_new_vehicles_count(self) -> int:
        """
        Get the number of new vehicles that entered the simulation.

        Returns:
            Number of new vehicles
        """
        return self.sumo.simulation.getDepartedNumber()

    def close(self) -> None:
        """Close the SUMO connection."""
        if self.simulation_running:
            self.sumo.close()
            self.simulation_running = False

    def render(self) -> Optional[np.ndarray]:
        """
        Render the environment.

        In "human" mode, the environment is already visualized using SUMO-GUI.
        In "rgb_array" mode, we could return a screenshot from SUMO-GUI, but
        this is not implemented yet.

        Returns:
            None (human mode) or RGB array (rgb_array mode)
        """
        if self.render_mode == "human":
            # SUMO-GUI is already rendering
            return None

        elif self.render_mode == "rgb_array":
            # Not implemented yet - would need to capture screenshot from SUMO-GUI
            return np.zeros((600, 800, 3), dtype=np.uint8)

        return None
