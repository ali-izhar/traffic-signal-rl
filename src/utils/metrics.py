#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Traffic Signal Control Metrics

This module provides functions to calculate and analyze performance metrics
for traffic signal control systems.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Any, Tuple
from scipy import stats


def calculate_metrics(
    episode_data: Dict[str, List],
    include_emissions: bool = True,
    include_fuel: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    Calculate comprehensive performance metrics from episode data.

    Args:
        episode_data: Dictionary containing lists of metrics over time/episodes
                    (waiting_times, queue_lengths, throughputs, etc.)
        include_emissions: Whether to include emissions metrics
        include_fuel: Whether to include fuel consumption metrics

    Returns:
        Dictionary of metrics with their means and standard deviations
    """
    metrics = {}

    # Process waiting time metrics
    if "waiting_times" in episode_data:
        waiting_times = np.array(episode_data["waiting_times"])
        metrics["waiting_time"] = {
            "mean": np.mean(waiting_times),
            "std": np.std(waiting_times),
            "median": np.median(waiting_times),
            "min": np.min(waiting_times),
            "max": np.max(waiting_times),
            "p95": np.percentile(waiting_times, 95),
        }

    # Process queue length metrics
    if "queue_lengths" in episode_data:
        queue_lengths = np.array(episode_data["queue_lengths"])
        metrics["queue_length"] = {
            "mean": np.mean(queue_lengths),
            "std": np.std(queue_lengths),
            "median": np.median(queue_lengths),
            "min": np.min(queue_lengths),
            "max": np.max(queue_lengths),
            "p95": np.percentile(queue_lengths, 95),
        }

    # Process throughput metrics
    if "throughputs" in episode_data:
        throughputs = np.array(episode_data["throughputs"])
        metrics["throughput"] = {
            "mean": np.mean(throughputs),
            "std": np.std(throughputs),
            "median": np.median(throughputs),
            "min": np.min(throughputs),
            "max": np.max(throughputs),
            "p95": np.percentile(throughputs, 95),
        }

    # Process reward metrics
    if "rewards" in episode_data:
        rewards = np.array(episode_data["rewards"])
        metrics["reward"] = {
            "mean": np.mean(rewards),
            "std": np.std(rewards),
            "median": np.median(rewards),
            "min": np.min(rewards),
            "max": np.max(rewards),
            "p95": np.percentile(rewards, 95),
        }

    # Process travel time metrics
    if "travel_times" in episode_data:
        travel_times = np.array(episode_data["travel_times"])
        metrics["travel_time"] = {
            "mean": np.mean(travel_times),
            "std": np.std(travel_times),
            "median": np.median(travel_times),
            "min": np.min(travel_times),
            "max": np.max(travel_times),
            "p95": np.percentile(travel_times, 95),
        }

    # Process phase switch metrics
    if "switches" in episode_data:
        switches = np.array(episode_data["switches"])
        metrics["switches"] = {
            "mean": np.mean(switches),
            "std": np.std(switches),
            "median": np.median(switches),
            "min": np.min(switches),
            "max": np.max(switches),
            "total": np.sum(switches),
        }

    # Calculate emissions if queue length data is available and emissions requested
    if include_emissions and "queue_lengths" in episode_data:
        # Simple emissions model based on queue lengths
        # In a real implementation, this would use detailed vehicle acceleration/deceleration data
        queue_lengths = np.array(episode_data["queue_lengths"])
        idling_emissions = queue_lengths * 2.5  # g CO2 per vehicle per time step

        metrics["emissions"] = {
            "mean": np.mean(idling_emissions),
            "std": np.std(idling_emissions),
            "median": np.median(idling_emissions),
            "min": np.min(idling_emissions),
            "max": np.max(idling_emissions),
            "total": np.sum(idling_emissions),
        }

    # Calculate fuel consumption if queue length data is available and fuel requested
    if include_fuel and "queue_lengths" in episode_data:
        # Simple fuel consumption model based on queue lengths
        # In a real implementation, this would use detailed vehicle data
        queue_lengths = np.array(episode_data["queue_lengths"])
        idling_fuel = queue_lengths * 0.15  # ml per vehicle per time step

        metrics["fuel_consumption"] = {
            "mean": np.mean(idling_fuel),
            "std": np.std(idling_fuel),
            "median": np.median(idling_fuel),
            "min": np.min(idling_fuel),
            "max": np.max(idling_fuel),
            "total": np.sum(idling_fuel),
        }

    return metrics


def normalize_metrics(
    metrics: Dict[str, Dict[str, float]],
    reference_method: str = None,
    higher_is_better: Dict[str, bool] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Normalize metrics for comparison between different methods.

    Args:
        metrics: Metrics dictionary with methods as keys
        reference_method: Method to use as reference (if None, use min/max)
        higher_is_better: Dictionary indicating whether higher values are better for each metric
                        (defaults to higher is better for throughput only)

    Returns:
        Dictionary of normalized metrics
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

    normalized = {}

    # Extract all methods
    methods = list(metrics.keys())

    # No normalization needed for single method
    if len(methods) <= 1:
        return metrics

    # For each metric type
    metric_types = set()
    for method in methods:
        metric_types.update(metrics[method].keys())

    for metric_type in metric_types:
        # Skip metrics that don't exist for all methods
        if not all(metric_type in metrics[method] for method in methods):
            continue

        # Get values for all methods
        values = {method: metrics[method][metric_type]["mean"] for method in methods}

        # Determine min and max
        min_val = min(values.values())
        max_val = max(values.values())

        # Avoid division by zero
        if min_val == max_val:
            # All methods have the same value
            normalized_values = {method: 1.0 for method in methods}
        else:
            # Normalize
            if higher_is_better.get(metric_type, False):
                # Higher is better: 1.0 for highest, 0.0 for lowest
                normalized_values = {
                    method: (values[method] - min_val) / (max_val - min_val)
                    for method in methods
                }
            else:
                # Lower is better: 1.0 for lowest, 0.0 for highest
                normalized_values = {
                    method: 1.0 - (values[method] - min_val) / (max_val - min_val)
                    for method in methods
                }

        # Store normalized values
        for method in methods:
            if method not in normalized:
                normalized[method] = {}
            normalized[method][metric_type] = normalized_values[method]

    return normalized


def calculate_statistical_significance(
    results: Dict[str, Dict[str, Dict[str, Union[float, np.ndarray]]]],
    metric: str = "waiting_time",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Calculate statistical significance of differences between methods.

    Args:
        results: Results dictionary with methods as keys
        metric: Metric to analyze
        alpha: Significance level

    Returns:
        DataFrame with p-values for each method pair
    """
    methods = list(results.keys())
    n_methods = len(methods)

    # Create empty dataframe for p-values
    p_values = pd.DataFrame(index=methods, columns=methods)

    # Perform t-tests for each pair of methods
    for i, method1 in enumerate(methods):
        for j, method2 in enumerate(methods):
            if i == j:
                p_values.loc[method1, method2] = 1.0
                continue

            # Extract raw data
            if "raw_data" in results[method1] and "raw_data" in results[method2]:
                data1 = np.array(results[method1]["raw_data"][f"{metric}s"])
                data2 = np.array(results[method2]["raw_data"][f"{metric}s"])

                # Perform t-test
                _, p_value = stats.ttest_ind(data1, data2, equal_var=False)
                p_values.loc[method1, method2] = p_value

    # Mark significant differences
    significant = pd.DataFrame(
        (p_values < alpha).astype(int), index=methods, columns=methods
    )

    return p_values, significant


def calculate_improvement_percentage(
    baseline_metrics: Dict[str, float],
    method_metrics: Dict[str, float],
    higher_is_better: Dict[str, bool] = None,
) -> Dict[str, float]:
    """
    Calculate percentage improvement over baseline for each metric.

    Args:
        baseline_metrics: Metrics for baseline method
        method_metrics: Metrics for the method to compare
        higher_is_better: Dictionary indicating whether higher values are better for each metric

    Returns:
        Dictionary of percentage improvements
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

    improvements = {}

    for metric, baseline_value in baseline_metrics.items():
        if metric in method_metrics:
            method_value = method_metrics[metric]

            # Skip if baseline value is zero to avoid division by zero
            if baseline_value == 0:
                improvements[metric] = float("inf") if method_value > 0 else 0.0
                continue

            # Calculate improvement percentage
            if higher_is_better.get(metric, False):
                # Higher is better
                improvements[metric] = (
                    (method_value - baseline_value) / baseline_value
                ) * 100
            else:
                # Lower is better
                improvements[metric] = (
                    (baseline_value - method_value) / baseline_value
                ) * 100

    return improvements


def estimate_co2_emissions(
    queue_lengths: np.ndarray,
    phase_switches: np.ndarray = None,
    vehicle_type: str = "average",
) -> Dict[str, float]:
    """
    Estimate CO2 emissions based on queue lengths and phase switches.

    Args:
        queue_lengths: Array of queue lengths over time
        phase_switches: Array of phase switch events
        vehicle_type: Type of vehicle for emission calculations

    Returns:
        Dictionary with emission metrics
    """
    # Emission factors (g CO2 per second)
    emission_factors = {"car": 2.5, "truck": 7.5, "bus": 10.0, "average": 3.5}

    # Acceleration emission factor (additional g CO2 per acceleration event)
    acceleration_factors = {"car": 15.0, "truck": 40.0, "bus": 50.0, "average": 20.0}

    factor = emission_factors.get(vehicle_type, emission_factors["average"])
    acc_factor = acceleration_factors.get(vehicle_type, acceleration_factors["average"])

    # Idling emissions from queue lengths
    idling_emissions = queue_lengths * factor

    # Acceleration emissions from phase switches (simplified model)
    acceleration_emissions = 0
    if phase_switches is not None:
        # Estimate number of accelerating vehicles after each phase switch
        # For simplicity, assume half the queue length accelerates
        for i, switch in enumerate(phase_switches):
            if switch and i < len(queue_lengths):
                acceleration_emissions += (queue_lengths[i] / 2) * acc_factor

    # Total emissions
    total_emissions = np.sum(idling_emissions) + acceleration_emissions

    return {
        "idling_emissions": np.sum(idling_emissions),
        "acceleration_emissions": acceleration_emissions,
        "total_emissions": total_emissions,
        "average_emissions_per_step": np.mean(idling_emissions),
    }
