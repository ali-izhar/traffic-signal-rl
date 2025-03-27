#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Logger for Reinforcement Learning Training

This module provides logging utilities for tracking training metrics
and visualizing them with TensorBoard.
"""

import os
import time
import json
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, List, Union

# TensorBoard support
try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("TensorBoard not available. Install with: pip install tensorboard")


class Logger:
    """
    Logger for tracking and visualizing training metrics.

    This class supports:
    - TensorBoard logging
    - JSON file logging
    - Console output
    """

    def __init__(self, log_dir: str, use_tensorboard: bool = True):
        """
        Initialize the logger.

        Args:
            log_dir: Directory to save logs
            use_tensorboard: Whether to use TensorBoard for logging
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # Initialize TensorBoard if available and requested
        self.use_tensorboard = use_tensorboard and TENSORBOARD_AVAILABLE
        if self.use_tensorboard:
            self.writer = SummaryWriter(log_dir=os.path.join(log_dir, "tensorboard"))

        # Initialize JSON log file
        self.json_log_path = os.path.join(log_dir, "metrics.json")
        self.metrics_history = {}

        # Record start time
        self.start_time = time.time()

        print(f"Logger initialized. Logs will be saved to: {log_dir}")
        if self.use_tensorboard:
            print(
                f"TensorBoard logs available. Run 'tensorboard --logdir={log_dir}' to view"
            )

    def log_metrics(self, metrics: Dict[str, Any], step: int):
        """
        Log metrics to all enabled outputs.

        Args:
            metrics: Dictionary of metrics to log
            step: Current step/episode number
        """
        # Log to TensorBoard
        if self.use_tensorboard:
            for key, value in metrics.items():
                if isinstance(value, (int, float, np.number)):
                    self.writer.add_scalar(key, value, step)
                elif isinstance(value, (list, np.ndarray)) and len(value) > 0:
                    if isinstance(value[0], (int, float, np.number)):
                        self.writer.add_histogram(key, np.array(value), step)

        # Log to metrics history
        for key, value in metrics.items():
            if key not in self.metrics_history:
                self.metrics_history[key] = []

            # Convert numpy values to Python native types for JSON serialization
            if isinstance(value, np.number):
                value = value.item()
            elif isinstance(value, np.ndarray):
                value = value.tolist()

            # Add step and timestamp
            entry = {
                "step": step,
                "value": value,
                "time": time.time() - self.start_time,
            }
            self.metrics_history[key].append(entry)

        # Periodically save JSON metrics
        if step % 10 == 0:
            self._save_json_metrics()

    def log_hyperparameters(self, hyperparams: Dict[str, Any]):
        """
        Log hyperparameters.

        Args:
            hyperparams: Dictionary of hyperparameters
        """
        # Save as JSON
        hp_path = os.path.join(self.log_dir, "hyperparameters.json")
        with open(hp_path, "w") as f:
            json.dump(hyperparams, f, indent=4)

        # Log to TensorBoard if available
        if self.use_tensorboard:
            # Convert hyperparameters to compatible format
            hparams_dict = {}
            metrics_dict = {}

            for key, value in hyperparams.items():
                if isinstance(value, (str, bool, int, float)):
                    hparams_dict[key] = value

            self.writer.add_hparams(hparams_dict, metrics_dict)

    def log_model_graph(self, model, input_size):
        """
        Log model graph to TensorBoard.

        Args:
            model: PyTorch model
            input_size: Input size for the model
        """
        if self.use_tensorboard:
            try:
                import torch

                device = next(model.parameters()).device
                dummy_input = torch.zeros(1, *input_size, device=device)
                self.writer.add_graph(model, dummy_input)
            except Exception as e:
                print(f"Failed to log model graph: {e}")

    def log_figure(self, tag: str, figure, step: int):
        """
        Log matplotlib figure to TensorBoard.

        Args:
            tag: Name for the figure
            figure: Matplotlib figure
            step: Current step/episode
        """
        if self.use_tensorboard:
            self.writer.add_figure(tag, figure, step)

        # Save figure as image
        figure_dir = os.path.join(self.log_dir, "figures")
        os.makedirs(figure_dir, exist_ok=True)
        figure_path = os.path.join(figure_dir, f"{tag}_{step}.png")
        figure.savefig(figure_path)

    def _save_json_metrics(self):
        """Save metrics history to JSON file."""
        with open(self.json_log_path, "w") as f:
            json.dump(self.metrics_history, f, indent=4)

    def get_metric_history(self, metric_name: str) -> List[Dict[str, Any]]:
        """
        Get history of a specific metric.

        Args:
            metric_name: Name of the metric

        Returns:
            List of metric entries with step, value, and time
        """
        return self.metrics_history.get(metric_name, [])

    def close(self):
        """Close logger and save final metrics."""
        self._save_json_metrics()

        if self.use_tensorboard:
            self.writer.close()

        print(f"Logger closed. All metrics saved to {self.log_dir}")

    def __del__(self):
        """Ensure resources are properly closed."""
        try:
            self.close()
        except:
            pass
