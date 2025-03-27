#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data Manager for Traffic Signal Control

This module provides utilities for managing experimental data,
including saving, loading, and processing results.
"""

import os
import json
import numpy as np
import pandas as pd
import pickle
from typing import Dict, List, Any, Union, Optional, Tuple
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class DataManager:
    """
    Data management utility for traffic signal control experiments.

    This class provides methods to save, load, and process experimental results,
    including metrics calculation and visualization.
    """

    def __init__(self, base_dir: str = "../results"):
        """
        Initialize the data manager.

        Args:
            base_dir: Base directory for storing results
        """
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def create_experiment_dir(self, experiment_name: str = None) -> str:
        """
        Create a directory for a new experiment.

        Args:
            experiment_name: Name of the experiment (if None, use timestamp)

        Returns:
            Path to the created directory
        """
        if experiment_name is None:
            # Generate a name based on timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"experiment_{timestamp}"

        # Create experiment directory
        experiment_dir = os.path.join(self.base_dir, experiment_name)
        os.makedirs(experiment_dir, exist_ok=True)

        return experiment_dir

    def save_experiment_config(
        self, config: Dict[str, Any], experiment_dir: str
    ) -> str:
        """
        Save experiment configuration.

        Args:
            config: Configuration dictionary
            experiment_dir: Directory to save the configuration

        Returns:
            Path to the saved configuration file
        """
        config_path = os.path.join(experiment_dir, "config.json")

        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)

        return config_path

    def save_results(
        self,
        results: Dict[str, Any],
        experiment_dir: str,
        filename: str = "results.json",
    ) -> str:
        """
        Save experiment results.

        Args:
            results: Results dictionary
            experiment_dir: Directory to save the results
            filename: Name of the results file

        Returns:
            Path to the saved results file
        """
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = self._make_serializable(results)

        results_path = os.path.join(experiment_dir, filename)

        with open(results_path, "w") as f:
            json.dump(serializable_results, f, indent=4)

        return results_path

    def _make_serializable(self, obj: Any) -> Any:
        """
        Convert object to JSON serializable format.

        Args:
            obj: Object to convert

        Returns:
            JSON serializable version of the object
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return [self._make_serializable(item) for item in obj]
        else:
            return obj

    def load_results(
        self, experiment_dir: str, filename: str = "results.json"
    ) -> Dict[str, Any]:
        """
        Load experiment results.

        Args:
            experiment_dir: Directory containing the results
            filename: Name of the results file

        Returns:
            Results dictionary
        """
        results_path = os.path.join(experiment_dir, filename)

        with open(results_path, "r") as f:
            results = json.load(f)

        return results

    def load_config(
        self, experiment_dir: str, filename: str = "config.json"
    ) -> Dict[str, Any]:
        """
        Load experiment configuration.

        Args:
            experiment_dir: Directory containing the configuration
            filename: Name of the configuration file

        Returns:
            Configuration dictionary
        """
        config_path = os.path.join(experiment_dir, filename)

        with open(config_path, "r") as f:
            config = json.load(f)

        return config

    def save_model(
        self, model: Any, experiment_dir: str, filename: str = "model.pkl"
    ) -> str:
        """
        Save a model.

        Args:
            model: Model to save
            experiment_dir: Directory to save the model
            filename: Name of the model file

        Returns:
            Path to the saved model file
        """
        model_path = os.path.join(experiment_dir, filename)

        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        return model_path

    def load_model(self, experiment_dir: str, filename: str = "model.pkl") -> Any:
        """
        Load a model.

        Args:
            experiment_dir: Directory containing the model
            filename: Name of the model file

        Returns:
            Loaded model
        """
        model_path = os.path.join(experiment_dir, filename)

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        return model

    def save_training_history(
        self,
        history: Dict[str, List],
        experiment_dir: str,
        filename: str = "training_history.json",
    ) -> str:
        """
        Save training history.

        Args:
            history: Training history dictionary
            experiment_dir: Directory to save the history
            filename: Name of the history file

        Returns:
            Path to the saved history file
        """
        # Convert numpy arrays to lists for JSON serialization
        serializable_history = self._make_serializable(history)

        history_path = os.path.join(experiment_dir, filename)

        with open(history_path, "w") as f:
            json.dump(serializable_history, f, indent=4)

        return history_path

    def load_training_history(
        self, experiment_dir: str, filename: str = "training_history.json"
    ) -> Dict[str, List]:
        """
        Load training history.

        Args:
            experiment_dir: Directory containing the history
            filename: Name of the history file

        Returns:
            Training history dictionary
        """
        history_path = os.path.join(experiment_dir, filename)

        with open(history_path, "r") as f:
            history = json.load(f)

        return history

    def create_summary_csv(
        self, results: Dict[str, Dict[str, Dict[str, float]]], output_path: str
    ) -> pd.DataFrame:
        """
        Create a CSV summary of results.

        Args:
            results: Results dictionary
            output_path: Path to save the CSV file

        Returns:
            DataFrame of the summary
        """
        # Extract metrics from results
        summary_data = []

        for method, metrics in results.items():
            row = {"method": method}

            for metric_name, values in metrics.items():
                if metric_name != "raw_data":
                    for stat_name, value in values.items():
                        row[f"{metric_name}_{stat_name}"] = value

            summary_data.append(row)

        # Create DataFrame
        summary_df = pd.DataFrame(summary_data)

        # Save to CSV
        summary_df.to_csv(output_path, index=False)

        return summary_df

    def create_latex_table(
        self,
        results: Dict[str, Dict[str, Dict[str, float]]],
        metrics: List[str] = None,
        methods: List[str] = None,
        highlight_best: bool = True,
        caption: str = "Performance Comparison",
        label: str = "tab:results",
    ) -> str:
        """
        Create a LaTeX table of results.

        Args:
            results: Results dictionary
            metrics: List of metrics to include (if None, use all)
            methods: List of methods to include (if None, use all)
            highlight_best: Whether to highlight the best result in each metric
            caption: Table caption
            label: Table label

        Returns:
            LaTeX table string
        """
        if methods is None:
            methods = list(results.keys())

        if metrics is None:
            # Find common metrics
            all_metrics = set()
            for method in methods:
                all_metrics.update(results[method].keys())
            metrics = [m for m in all_metrics if m != "raw_data"]

        # Default higher is better for each metric
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

        # Start building the table
        table = []
        table.append("\\begin{table}")
        table.append("\\centering")
        table.append("\\caption{" + caption + "}")
        table.append("\\label{" + label + "}")

        # Table header
        table.append("\\begin{tabular}{l" + "c" * len(metrics) + "}")
        table.append("\\toprule")

        # Metrics header
        header = ["Method"] + [metric.replace("_", " ").title() for metric in metrics]
        table.append(" & ".join(header) + " \\\\")
        table.append("\\midrule")

        # Find best values for each metric
        best_values = {}
        for metric in metrics:
            values = []
            for method in methods:
                if metric in results[method]:
                    values.append(results[method][metric]["mean"])

            if values:
                if higher_is_better.get(metric, True):
                    best_values[metric] = max(values)
                else:
                    best_values[metric] = min(values)

        # Add rows for each method
        for method in methods:
            row = [method]
            for metric in metrics:
                if metric in results[method]:
                    value = results[method][metric]["mean"]
                    std = results[method][metric]["std"]

                    # Format value with standard deviation
                    value_str = f"{value:.2f} $\\pm$ {std:.2f}"

                    # Highlight best value if requested
                    if (
                        highlight_best
                        and metric in best_values
                        and value == best_values[metric]
                    ):
                        value_str = "\\textbf{" + value_str + "}"

                    row.append(value_str)
                else:
                    row.append("-")

            table.append(" & ".join(row) + " \\\\")

        # Table footer
        table.append("\\bottomrule")
        table.append("\\end{tabular}")
        table.append("\\end{table}")

        return "\n".join(table)

    def export_figures(
        self, figures_dir: str, output_dir: str, formats: List[str] = ["png", "pdf"]
    ) -> List[str]:
        """
        Export figures in multiple formats.

        Args:
            figures_dir: Directory containing the figures
            output_dir: Directory to save the exported figures
            formats: List of formats to export

        Returns:
            List of exported figure paths
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Get all figure files
        figure_files = []
        for ext in ["png", "jpg", "jpeg"]:
            figure_files.extend(list(Path(figures_dir).glob(f"*.{ext}")))

        # Export each figure
        exported_paths = []
        for figure_path in figure_files:
            figure_name = figure_path.stem

            # Load figure
            img = plt.imread(figure_path)

            # Save in each format
            for format in formats:
                output_path = os.path.join(output_dir, f"{figure_name}.{format}")
                plt.imsave(output_path, img)
                exported_paths.append(output_path)

        return exported_paths

    def aggregate_results(
        self,
        result_dirs: List[str],
        metrics: List[str] = None,
        methods: List[str] = None,
    ) -> Dict[str, Dict[str, List[float]]]:
        """
        Aggregate results from multiple experiments.

        Args:
            result_dirs: List of directories containing results
            metrics: List of metrics to include (if None, use all)
            methods: List of methods to include (if None, use all)

        Returns:
            Aggregated results dictionary
        """
        all_results = []

        # Load results from each directory
        for dir_path in result_dirs:
            try:
                results = self.load_results(dir_path)
                all_results.append(results)
            except Exception as e:
                print(f"Error loading results from {dir_path}: {e}")

        if not all_results:
            return {}

        # Determine methods and metrics if not provided
        if methods is None:
            methods = set()
            for results in all_results:
                methods.update(results.keys())
            methods = sorted(list(methods))

        if metrics is None:
            metrics = set()
            for results in all_results:
                for method in methods:
                    if method in results:
                        metrics.update(
                            k for k in results[method].keys() if k != "raw_data"
                        )
            metrics = sorted(list(metrics))

        # Aggregate results
        aggregated = {}
        for method in methods:
            aggregated[method] = {}

            for metric in metrics:
                values = []

                for results in all_results:
                    if method in results and metric in results[method]:
                        values.append(results[method][metric]["mean"])

                if values:
                    aggregated[method][metric] = {
                        "values": values,
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "min": np.min(values),
                        "max": np.max(values),
                        "median": np.median(values),
                    }

        return aggregated

    def calculate_significance(
        self,
        aggregated_results: Dict[str, Dict[str, Dict[str, Any]]],
        metric: str,
        alpha: float = 0.05,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculate statistical significance of differences between methods.

        Args:
            aggregated_results: Aggregated results dictionary
            metric: Metric to analyze
            alpha: Significance level

        Returns:
            DataFrames with p-values and significance indicators
        """
        from scipy import stats

        methods = list(aggregated_results.keys())
        n_methods = len(methods)

        # Create empty DataFrames for p-values and significance
        p_values = pd.DataFrame(index=methods, columns=methods)
        significant = pd.DataFrame(index=methods, columns=methods)

        # Fill with default values
        p_values.fillna(1.0, inplace=True)
        significant.fillna(0, inplace=True)

        # Perform t-tests
        for i, method1 in enumerate(methods):
            for j, method2 in enumerate(methods):
                if i == j:
                    continue

                if (
                    metric in aggregated_results[method1]
                    and metric in aggregated_results[method2]
                ):

                    values1 = aggregated_results[method1][metric]["values"]
                    values2 = aggregated_results[method2][metric]["values"]

                    if len(values1) > 1 and len(values2) > 1:
                        # Perform t-test
                        _, p_value = stats.ttest_ind(values1, values2, equal_var=False)
                        p_values.loc[method1, method2] = p_value
                        significant.loc[method1, method2] = int(p_value < alpha)

        return p_values, significant
