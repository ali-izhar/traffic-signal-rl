#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np

from .base import Agent

# Phase codes based on environment.net.xml
PHASE_NS_GREEN = 0  # action 0 code 00
PHASE_NS_YELLOW = 1
PHASE_NSL_GREEN = 2  # action 1 code 01
PHASE_NSL_YELLOW = 3
PHASE_EW_GREEN = 4  # action 2 code 10
PHASE_EW_YELLOW = 5
PHASE_EWL_GREEN = 6  # action 3 code 11
PHASE_EWL_YELLOW = 7


class FixedTimingController(Agent):
    """Simple fixed-time controller that cycles through phases with preset durations"""

    def __init__(self, input_dim, output_dim, phase_durations=None):
        super().__init__(input_dim, output_dim)

        # Default phase durations (in simulation steps)
        if phase_durations is None:
            # Default green durations for each signal phase in order
            self._phase_durations = {
                PHASE_NS_GREEN: 30,  # North-South green
                PHASE_NSL_GREEN: 15,  # North-South left green
                PHASE_EW_GREEN: 30,  # East-West green
                PHASE_EWL_GREEN: 15,  # East-West left green
            }
        else:
            self._phase_durations = phase_durations

        # Internal state
        self._current_phase = PHASE_NS_GREEN
        self._time_in_phase = 0
        self._phase_sequence = [
            PHASE_NS_GREEN,
            PHASE_NSL_GREEN,
            PHASE_EW_GREEN,
            PHASE_EWL_GREEN,
        ]
        self._current_phase_idx = 0

    def act(self, state, epsilon=0):
        """Select action based on fixed timing"""
        # Increment time in current phase
        self._time_in_phase += 1

        # Check if it's time to change phase
        current_green_phase = self._phase_sequence[self._current_phase_idx]
        if self._time_in_phase >= self._phase_durations[current_green_phase]:
            # Move to next phase
            self._current_phase_idx = (self._current_phase_idx + 1) % len(
                self._phase_sequence
            )
            self._time_in_phase = 0

        # Return the current phase index as the action (0, 1, 2, 3)
        return self._current_phase_idx

    def learn(self, state, action, reward, next_state, done=False):
        """Nothing to learn for fixed-time controller"""
        pass

    def save(self, path):
        """Save configuration"""
        os.makedirs(path, exist_ok=True)
        np.save(
            os.path.join(path, "fixed_timing_config.npy"),
            {
                "phase_durations": self._phase_durations,
                "phase_sequence": self._phase_sequence,
            },
        )

    def load(self, path):
        """Load configuration"""
        config_path = os.path.join(path, "fixed_timing_config.npy")
        if os.path.isfile(config_path):
            config = np.load(config_path, allow_pickle=True).item()
            self._phase_durations = config["phase_durations"]
            self._phase_sequence = config["phase_sequence"]
        else:
            raise FileNotFoundError(f"No config file found at {config_path}")


class ActuatedController(Agent):
    """Actuated controller that extends or cuts short green phases based on traffic demand"""

    def __init__(
        self,
        input_dim,
        output_dim,
        min_green=5,
        max_green=60,
        extension_time=5,
        yellow_time=4,
    ):
        super().__init__(input_dim, output_dim)

        self._min_green = min_green
        self._max_green = max_green
        self._extension_time = extension_time
        self._yellow_time = yellow_time

        # Internal state
        self._current_phase = PHASE_NS_GREEN
        self._time_in_phase = 0
        self._phase_sequence = [
            PHASE_NS_GREEN,
            PHASE_NSL_GREEN,
            PHASE_EW_GREEN,
            PHASE_EWL_GREEN,
        ]
        self._current_phase_idx = 0
        self._is_yellow = False
        self._yellow_countdown = 0

    def act(self, state, epsilon=0):
        """Select action based on traffic conditions"""
        # Increment time in current phase
        self._time_in_phase += 1

        # If in yellow phase, count down
        if self._is_yellow:
            self._yellow_countdown -= 1
            if self._yellow_countdown <= 0:
                # Switch to next green phase
                self._current_phase_idx = (self._current_phase_idx + 1) % len(
                    self._phase_sequence
                )
                self._time_in_phase = 0
                self._is_yellow = False

            # Important: Don't return phase codes directly - return action index (0-3)
            # During yellow, still return the current action index (not yellow phase)
            return self._current_phase_idx

        # Evaluate if we should extend the green or switch to yellow
        demand = self._calculate_demand(state, self._current_phase_idx)

        # If minimum green time has passed
        if self._time_in_phase >= self._min_green:
            # If no demand or max green reached, switch to yellow
            if demand == 0 or self._time_in_phase >= self._max_green:
                self._is_yellow = True
                self._yellow_countdown = self._yellow_time
                return self._current_phase_idx  # Return action index, not phase code

        # Continue with current green phase
        return self._current_phase_idx

    def _calculate_demand(self, state, phase_idx):
        """Calculate demand for current phase based on state"""
        # For this simple implementation, we'll use a threshold approach
        # Reshape state to match the lanes (assuming shape as described in the SUMO environment)
        # This is tailored to the traffic simulation format described in the original code

        # Get lane groups for current phase
        if phase_idx == 0:  # NS through
            lane_groups = [2, 6]  # North and South through lanes
        elif phase_idx == 1:  # NS left
            lane_groups = [3, 7]  # North and South left lanes
        elif phase_idx == 2:  # EW through
            lane_groups = [0, 4]  # East and West through lanes
        elif phase_idx == 3:  # EW left
            lane_groups = [1, 5]  # East and West left lanes
        else:
            return 0

        # Calculate demand (number of cars) in current phase's lanes
        demand = 0
        for lane_group in lane_groups:
            for i in range(10):  # 10 cells per lane
                cell_idx = lane_group * 10 + i
                if cell_idx < len(state) and state[cell_idx] > 0:
                    # Give higher weight to cars closer to intersection
                    demand += 1 * (10 - i) / 10

        return demand

    def _convert_to_action(self, phase_idx, is_yellow):
        """Convert phase index and yellow status to action number"""
        if is_yellow:
            return phase_idx * 2 + 1  # Yellow phases are odd numbers
        else:
            return phase_idx * 2  # Green phases are even numbers

    def learn(self, state, action, reward, next_state, done=False):
        """Nothing to learn for actuated controller"""
        pass

    def save(self, path):
        """Save configuration"""
        os.makedirs(path, exist_ok=True)
        np.save(
            os.path.join(path, "actuated_config.npy"),
            {
                "min_green": self._min_green,
                "max_green": self._max_green,
                "extension_time": self._extension_time,
                "yellow_time": self._yellow_time,
            },
        )

    def load(self, path):
        """Load configuration"""
        config_path = os.path.join(path, "actuated_config.npy")
        if os.path.isfile(config_path):
            config = np.load(config_path, allow_pickle=True).item()
            self._min_green = config["min_green"]
            self._max_green = config["max_green"]
            self._extension_time = config["extension_time"]
            self._yellow_time = config["yellow_time"]
        else:
            raise FileNotFoundError(f"No config file found at {config_path}")


class WebsterController(Agent):
    """Webster's method for traffic signal timing based on flow rates"""

    def __init__(
        self,
        input_dim,
        output_dim,
        saturation_flow_rate=1900,  # vehicles per hour of green per lane
        yellow_time=4,
        lost_time_per_phase=2,
        update_interval=300,  # steps between timing updates
    ):
        super().__init__(input_dim, output_dim)

        self._saturation_flow_rate = saturation_flow_rate
        self._yellow_time = yellow_time
        self._lost_time_per_phase = lost_time_per_phase
        self._update_interval = update_interval

        # Flow tracking
        self._flow_counts = {
            PHASE_NS_GREEN: 0,
            PHASE_NSL_GREEN: 0,
            PHASE_EW_GREEN: 0,
            PHASE_EWL_GREEN: 0,
        }
        self._previous_state = None
        self._steps_since_update = 0

        # Phase timing
        self._phase_durations = {
            PHASE_NS_GREEN: 30,
            PHASE_NSL_GREEN: 15,
            PHASE_EW_GREEN: 30,
            PHASE_EWL_GREEN: 15,
        }

        # Internal state
        self._current_phase = PHASE_NS_GREEN
        self._time_in_phase = 0
        self._phase_sequence = [
            PHASE_NS_GREEN,
            PHASE_NSL_GREEN,
            PHASE_EW_GREEN,
            PHASE_EWL_GREEN,
        ]
        self._current_phase_idx = 0
        self._is_yellow = False
        self._yellow_countdown = 0

    def act(self, state, epsilon=0):
        """Select action based on Webster calculations"""
        # Update flow count if we have previous state
        if self._previous_state is not None:
            self._update_flow_counts(self._previous_state, state)
        self._previous_state = state.copy()

        # Increment counters
        self._time_in_phase += 1
        self._steps_since_update += 1

        # Check if it's time to recalculate phase durations
        if self._steps_since_update >= self._update_interval:
            self._calculate_webster_timings()
            self._steps_since_update = 0

        # If in yellow phase, count down
        if self._is_yellow:
            self._yellow_countdown -= 1
            if self._yellow_countdown <= 0:
                # Switch to next green phase
                self._current_phase_idx = (self._current_phase_idx + 1) % len(
                    self._phase_sequence
                )
                self._time_in_phase = 0
                self._is_yellow = False

            # Important: Don't return phase codes directly - return action index (0-3)
            # During yellow, still return the current action index (not yellow phase)
            return self._current_phase_idx

        # Check if it's time to change phase
        current_green_phase = self._phase_sequence[self._current_phase_idx]
        if self._time_in_phase >= self._phase_durations[current_green_phase]:
            # Switch to yellow
            self._is_yellow = True
            self._yellow_countdown = self._yellow_time
            return self._current_phase_idx  # Return action index, not phase code

        # Continue with current green phase
        return self._current_phase_idx

    def _update_flow_counts(self, previous_state, current_state):
        """Update vehicle counts based on vehicles that passed through"""
        # Count vehicles that were in the first cell of each approach and are now gone
        for phase, lane_groups in {
            PHASE_NS_GREEN: [2, 6],  # North and South through
            PHASE_NSL_GREEN: [3, 7],  # North and South left
            PHASE_EW_GREEN: [0, 4],  # East and West through
            PHASE_EWL_GREEN: [1, 5],  # East and West left
        }.items():
            for lane_group in lane_groups:
                # First cell index (closest to intersection)
                cell_idx = lane_group * 10
                # Check if a vehicle disappeared from the first cell
                if (
                    cell_idx < len(previous_state)
                    and previous_state[cell_idx] > 0
                    and current_state[cell_idx] == 0
                ):
                    self._flow_counts[phase] += 1

    def _calculate_webster_timings(self):
        """Calculate optimal cycle length and green splits using Webster's method"""
        # Calculate critical v/c ratio for each phase
        critical_ratios = {}
        total_critical_ratio = 0

        for phase, count in self._flow_counts.items():
            flow_rate = count * 3600 / self._update_interval  # Convert to hourly flow
            critical_ratio = flow_rate / self._saturation_flow_rate
            critical_ratios[phase] = max(
                0.1, critical_ratio
            )  # Minimum ratio to prevent zero times
            total_critical_ratio += critical_ratio

        # Reset flow counts for next period
        self._flow_counts = {phase: 0 for phase in self._flow_counts}

        # Calculate cycle length using Webster's formula
        lost_time = len(self._phase_sequence) * self._lost_time_per_phase

        # Cap total critical ratio to prevent cycle length from going to infinity
        total_critical_ratio = min(0.9, total_critical_ratio)

        if total_critical_ratio > 0:
            cycle_length = (1.5 * lost_time + 5) / (1 - total_critical_ratio)

            # Constrain cycle length to reasonable bounds
            cycle_length = max(40, min(cycle_length, 150))

            # Calculate green times for each phase
            effective_green_time = cycle_length - lost_time

            for phase in self._phase_sequence:
                if total_critical_ratio > 0:  # Avoid division by zero
                    green_time = (
                        critical_ratios[phase] / total_critical_ratio
                    ) * effective_green_time
                    # Ensure minimum green time
                    green_time = max(5, green_time)
                    # Set new phase duration
                    self._phase_durations[phase] = int(green_time)

    def _convert_to_action(self, phase_idx, is_yellow):
        """Convert phase index and yellow status to action number"""
        if is_yellow:
            return phase_idx * 2 + 1  # Yellow phases are odd numbers
        else:
            return phase_idx * 2  # Green phases are even numbers

    def learn(self, state, action, reward, next_state, done=False):
        """Nothing to learn for Webster controller"""
        pass

    def save(self, path):
        """Save configuration"""
        os.makedirs(path, exist_ok=True)
        np.save(
            os.path.join(path, "webster_config.npy"),
            {
                "saturation_flow_rate": self._saturation_flow_rate,
                "yellow_time": self._yellow_time,
                "lost_time_per_phase": self._lost_time_per_phase,
                "update_interval": self._update_interval,
                "phase_durations": self._phase_durations,
                "phase_sequence": self._phase_sequence,
            },
        )

    def load(self, path):
        """Load configuration"""
        config_path = os.path.join(path, "webster_config.npy")
        if os.path.isfile(config_path):
            config = np.load(config_path, allow_pickle=True).item()
            self._saturation_flow_rate = config["saturation_flow_rate"]
            self._yellow_time = config["yellow_time"]
            self._lost_time_per_phase = config["lost_time_per_phase"]
            self._update_interval = config["update_interval"]
            self._phase_durations = config["phase_durations"]
            self._phase_sequence = config["phase_sequence"]
        else:
            raise FileNotFoundError(f"No config file found at {config_path}")
