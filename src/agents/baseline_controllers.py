#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Baseline Traffic Signal Controllers

This module implements traditional traffic signal control methods (fixed-timing and
actuated) that serve as baselines for comparison with RL approaches.
"""

import numpy as np
from typing import Dict, Union, List, Any, Tuple


class FixedTimingController:
    """
    Fixed-timing traffic signal controller.

    This controller operates on a pre-defined cycle with fixed green times for each phase.
    It represents the simplest form of traffic signal control commonly used in practice.
    """

    def __init__(
        self,
        cycle_length: int = 60,
        green_splits: List[float] = None,
        yellow_time: int = 2,
        min_phase_time: int = 5,
    ):
        """
        Initialize fixed-timing controller.

        Args:
            cycle_length: Total cycle length in time steps
            green_splits: List of proportions for green time allocation to each phase
                        (must sum to 1.0, defaults to equal splits if None)
            yellow_time: Duration of yellow phase in time steps
            min_phase_time: Minimum green time for any phase
        """
        self.cycle_length = cycle_length
        self.yellow_time = yellow_time
        self.min_phase_time = min_phase_time

        # Default to equal splits if not provided
        if green_splits is None:
            # For two phases (N-S and E-W), split 50-50
            self.green_splits = [0.5, 0.5]
        else:
            # Ensure splits sum to 1.0
            total = sum(green_splits)
            self.green_splits = [split / total for split in green_splits]

        # Calculate phase durations
        self.phase_durations = []
        for split in self.green_splits:
            duration = max(
                int(split * (cycle_length - yellow_time * len(self.green_splits))),
                self.min_phase_time,
            )
            self.phase_durations.append(duration)

        # Adjust durations to exactly match cycle length
        total_duration = sum(self.phase_durations) + yellow_time * len(
            self.green_splits
        )
        if total_duration != cycle_length:
            # Adjust the longest phase to make the total match cycle_length
            idx = np.argmax(self.phase_durations)
            self.phase_durations[idx] += cycle_length - total_duration

        # State tracking
        self.current_cycle_time = 0
        self.current_phase = 0
        self.phase_time = 0
        self.in_yellow = False

    def act(self, state: np.ndarray) -> int:
        """
        Determine action based on fixed timing.

        Args:
            state: Current state (ignored in fixed-timing)

        Returns:
            Action (0: keep current phase, 1: switch phase)
        """
        # Track time in the current phase
        self.phase_time += 1
        self.current_cycle_time = (self.current_cycle_time + 1) % self.cycle_length

        # Check if we're in yellow phase
        if self.in_yellow:
            # If yellow time completed, switch to next phase
            if self.phase_time >= self.yellow_time:
                self.current_phase = (self.current_phase + 1) % len(
                    self.phase_durations
                )
                self.phase_time = 0
                self.in_yellow = False
            return 0  # Always keep yellow until it's done

        # Check if current green phase is completed
        if self.phase_time >= self.phase_durations[self.current_phase]:
            # Start yellow phase
            self.in_yellow = True
            self.phase_time = 0
            return 1  # Switch to yellow

        # Otherwise stay in current phase
        return 0

    def learn(self, state, action, reward, next_state, done):
        """
        Placeholder for compatibility with RL agents.
        Fixed-timing controller doesn't learn.
        """
        return {"td_error": 0.0}

    def save(self, filepath):
        """Placeholder for compatibility with RL agents."""
        pass

    def load(self, filepath):
        """Placeholder for compatibility with RL agents."""
        pass


class ActuatedController:
    """
    Vehicle-actuated traffic signal controller.

    This controller extends green phases when vehicles are detected and
    terminates them when no vehicles are detected for a certain period,
    within min/max green time constraints.
    """

    def __init__(
        self,
        min_green: int = 5,
        max_green: int = 30,
        yellow_time: int = 3,
        extension_time: int = 2,
        gap_threshold: int = 2,
    ):
        """
        Initialize actuated controller.

        Args:
            min_green: Minimum green time for any phase in time steps
            max_green: Maximum green time for any phase in time steps
            yellow_time: Duration of yellow phase in time steps
            extension_time: Time to extend green on vehicle detection
            gap_threshold: Queue size threshold for extension
        """
        self.min_green = min_green
        self.max_green = max_green
        self.yellow_time = yellow_time
        self.extension_time = extension_time
        self.gap_threshold = gap_threshold

        # State tracking
        self.current_phase = (
            0  # 0: N-S green, 1: E-W green, 2: N-S yellow, 3: E-W yellow
        )
        self.phase_time = 0
        self.yellow_active = False

    def act(self, state: np.ndarray) -> int:
        """
        Determine action based on vehicle detection.

        Args:
            state: Current state including queue lengths

        Returns:
            Action (0: keep current phase, 1: switch phase)
        """
        # Increment phase time
        self.phase_time += 1

        # Extract queue information from state
        # Assuming state structure from the paper:
        # [queue_n, queue_s, queue_e, queue_w, wait_n, wait_s, wait_e, wait_w, ...]
        queue_n = state[0]
        queue_s = state[1]
        queue_e = state[2]
        queue_w = state[3]

        # NS and EW directions queue totals
        ns_queue = queue_n + queue_s
        ew_queue = queue_e + queue_w

        # Check if in yellow phase
        if self.yellow_active:
            if self.phase_time >= self.yellow_time:
                # Yellow phase completed, switch to next green
                if self.current_phase == 0:
                    self.current_phase = 1  # Switch to E-W green
                else:
                    self.current_phase = 0  # Switch to N-S green
                self.phase_time = 0
                self.yellow_active = False
            return 0  # Stay in yellow until complete

        # Handle green phases with actuation
        if self.current_phase == 0:  # N-S green
            # Check if minimum green time satisfied
            if self.phase_time < self.min_green:
                return 0  # Keep phase

            # Check if maximum green time reached
            if self.phase_time >= self.max_green:
                self.yellow_active = True
                self.phase_time = 0
                return 1  # Switch to yellow

            # Actuated decision - extend if vehicles detected with threshold
            if ns_queue > self.gap_threshold:
                return 0  # Extend green
            elif ew_queue > 0:  # Vehicles waiting in cross direction
                self.yellow_active = True
                self.phase_time = 0
                return 1  # Switch to yellow
            else:
                return 0  # No vehicles anywhere, maintain green

        else:  # E-W green (phase 1)
            # Check if minimum green time satisfied
            if self.phase_time < self.min_green:
                return 0  # Keep phase

            # Check if maximum green time reached
            if self.phase_time >= self.max_green:
                self.yellow_active = True
                self.phase_time = 0
                return 1  # Switch to yellow

            # Actuated decision - extend if vehicles detected
            if ew_queue > self.gap_threshold:
                return 0  # Extend green
            elif ns_queue > 0:  # Vehicles waiting in cross direction
                self.yellow_active = True
                self.phase_time = 0
                return 1  # Switch to yellow
            else:
                return 0  # No vehicles anywhere, maintain green

    def learn(self, state, action, reward, next_state, done):
        """
        Placeholder for compatibility with RL agents.
        Actuated controller doesn't learn.
        """
        return {"td_error": 0.0}

    def save(self, filepath):
        """Placeholder for compatibility with RL agents."""
        pass

    def load(self, filepath):
        """Placeholder for compatibility with RL agents."""
        pass


class WebsterController:
    """
    Webster's traffic signal controller.

    This controller calculates optimal cycle length and green splits based on
    traffic flow measured by the Webster formula, widely used in traffic engineering.
    """

    def __init__(
        self,
        yellow_time: int = 3,
        lost_time_per_phase: int = 2,
        saturation_flow: int = 1800,  # vehicles per hour of green time
        update_frequency: int = 300,  # Update timings every n steps
        min_cycle: int = 30,
        max_cycle: int = 120,
    ):
        """
        Initialize Webster controller.

        Args:
            yellow_time: Duration of yellow phase in time steps
            lost_time_per_phase: Startup and clearance lost time
            saturation_flow: Maximum flow rate during green
            update_frequency: How often to recalculate timings
            min_cycle: Minimum allowable cycle length
            max_cycle: Maximum allowable cycle length
        """
        self.yellow_time = yellow_time
        self.lost_time_per_phase = lost_time_per_phase
        self.saturation_flow = saturation_flow
        self.update_frequency = update_frequency
        self.min_cycle = min_cycle
        self.max_cycle = max_cycle

        # State tracking
        self.current_phase = 0  # 0: N-S green, 1: E-W green, 2: yellow
        self.phase_time = 0
        self.steps_since_update = 0
        self.yellow_active = False

        # Traffic flow tracking
        self.flow_counts = {"north": [], "south": [], "east": [], "west": []}

        # Current timing plan
        self.cycle_length = 60  # Default
        self.green_splits = [0.5, 0.5]  # Default
        self.phase_durations = [25, 25]  # Default (allowing for yellow)

    def update_timings(self):
        """Update signal timings using Webster's formula based on collected flow data."""
        # Calculate average flow rates
        flows = {}
        for direction, counts in self.flow_counts.items():
            flows[direction] = np.mean(counts) if counts else 0

        # Clear flow counts for next period
        for direction in self.flow_counts:
            self.flow_counts[direction] = []

        # Calculate flow ratios (flow / saturation flow)
        y_ns = max(flows.get("north", 0), flows.get("south", 0)) / self.saturation_flow
        y_ew = max(flows.get("east", 0), flows.get("west", 0)) / self.saturation_flow

        # Calculate critical flow ratio sum
        Y = y_ns + y_ew

        # Calculate lost time
        L = self.lost_time_per_phase * 2  # Two phases

        # Webster's optimal cycle length formula
        if Y < 0.9:  # Avoid division by zero or negative values
            Co = (1.5 * L + 5) / (1 - Y)
            # Apply min/max constraints
            self.cycle_length = min(max(int(Co), self.min_cycle), self.max_cycle)
        else:
            # High congestion - use max cycle
            self.cycle_length = self.max_cycle

        # Calculate effective green times for each phase
        total_effective_green = self.cycle_length - L
        g_ns = total_effective_green * (y_ns / Y)
        g_ew = total_effective_green * (y_ew / Y)

        # Calculate actual green times
        g_ns_actual = max(10, int(g_ns))
        g_ew_actual = max(10, int(g_ew))

        # Recalculate cycle length to match calculated green times
        self.cycle_length = g_ns_actual + g_ew_actual + L

        # Set phase durations
        self.phase_durations = [g_ns_actual, g_ew_actual]

        # Calculate green splits for reference
        self.green_splits = [
            g_ns_actual / self.cycle_length,
            g_ew_actual / self.cycle_length,
        ]

    def act(self, state: np.ndarray) -> int:
        """
        Determine action based on Webster's method and traffic data.

        Args:
            state: Current state including queue lengths

        Returns:
            Action (0: keep current phase, 1: switch phase)
        """
        # Increment counters
        self.phase_time += 1
        self.steps_since_update += 1

        # Extract queue information from state
        # Assuming state structure from the paper
        queue_n = state[0]
        queue_s = state[1]
        queue_e = state[2]
        queue_w = state[3]

        # Track flow (approximated by queue changes)
        self.flow_counts["north"].append(queue_n)
        self.flow_counts["south"].append(queue_s)
        self.flow_counts["east"].append(queue_e)
        self.flow_counts["west"].append(queue_w)

        # Update timing plan periodically
        if self.steps_since_update >= self.update_frequency:
            self.update_timings()
            self.steps_since_update = 0

        # Check if in yellow phase
        if self.yellow_active:
            if self.phase_time >= self.yellow_time:
                # Yellow phase completed, switch to next green
                if self.current_phase == 0:
                    self.current_phase = 1  # Switch to E-W green
                else:
                    self.current_phase = 0  # Switch to N-S green
                self.phase_time = 0
                self.yellow_active = False
            return 0  # Stay in yellow until complete

        # Handle green phases based on calculated durations
        if self.phase_time >= self.phase_durations[self.current_phase]:
            # Current phase completed, switch to yellow
            self.yellow_active = True
            self.phase_time = 0
            return 1  # Switch to yellow

        # Otherwise maintain current phase
        return 0

    def learn(self, state, action, reward, next_state, done):
        """
        Placeholder for compatibility with RL agents.
        Webster controller uses its own update mechanism.
        """
        return {"td_error": 0.0}

    def save(self, filepath):
        """Placeholder for compatibility with RL agents."""
        pass

    def load(self, filepath):
        """Placeholder for compatibility with RL agents."""
        pass
