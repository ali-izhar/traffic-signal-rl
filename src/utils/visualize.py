"""Visualization Utilities for Traffic Signal Control"""

from typing import List, Optional

import os
import matplotlib.pyplot as plt
import numpy as np


class Visualization:
    """Visualization class for generating and saving plots and data.

    This class handles the creation of performance plots and saving of
    corresponding data files for later analysis.

    Attributes:
        _path: Directory path where plots and data should be saved
        _dpi: Resolution (dots per inch) for saved plots
    """

    def __init__(self, path: str, dpi: int = 300) -> None:
        """Initialize the visualization object.

        Args:
            path: Directory path where plots and data should be saved
            dpi: Resolution (dots per inch) for saved plots
        """
        self._path = path
        self._dpi = dpi

        # Create the directory if it doesn't exist
        os.makedirs(path, exist_ok=True)

    def save_data_and_plot(
        self,
        data: List[float],
        filename: str,
        xlabel: str,
        ylabel: str,
        title: Optional[str] = None,
    ) -> None:
        """Generate a performance plot and save the data to a text file.

        Creates a line plot of the provided data, with appropriate labels and formatting.
        Also saves the raw data to a text file for further analysis.

        Args:
            data: List of numerical values to plot
            filename: Base name for plot image and data file (without extension)
            xlabel: Label for x-axis
            ylabel: Label for y-axis
            title: Optional title for the plot

        Raises:
            ValueError: If data is empty
        """
        if not data:
            raise ValueError("Cannot create plot: empty data")

        # Calculate min and max for better y-axis limits
        min_val = min(data)
        max_val = max(data)
        y_margin = (
            0.05 * abs(max_val - min_val) if min_val != max_val else 0.05 * abs(min_val)
        )

        # Set up plot styling
        plt.rcParams.update({"font.size": 24})  # Set bigger font size
        plt.figure(figsize=(20, 11.25))

        # Create the plot
        plt.plot(data, linewidth=2)
        plt.ylabel(ylabel, fontweight="bold")
        plt.xlabel(xlabel, fontweight="bold")

        # Add title if provided
        if title:
            plt.title(title, fontweight="bold")

        # Set plot formatting
        plt.margins(0)
        plt.ylim(min_val - y_margin, max_val + y_margin)
        plt.grid(True, linestyle="--", alpha=0.7)

        # Save the plot
        plot_path = os.path.join(self._path, f"plot_{filename}.png")
        plt.savefig(plot_path, dpi=self._dpi, bbox_inches="tight")
        plt.close("all")

        # Save the data to a text file
        data_path = os.path.join(self._path, f"plot_{filename}_data.txt")
        try:
            with open(data_path, "w") as file:
                for value in data:
                    file.write(f"{value}\n")
            print(f"Saved plot to {plot_path} and data to {data_path}")
        except IOError as e:
            print(f"Error saving data file: {e}")

    def save_multiple_plots(
        self,
        datasets: List[List[float]],
        labels: List[str],
        filename: str,
        xlabel: str,
        ylabel: str,
        title: Optional[str] = None,
        colors: Optional[List[str]] = None,
    ) -> None:
        """Generate a plot with multiple data series and save to image file.

        Creates a line plot with multiple series, useful for comparing different
        algorithms or configurations.

        Args:
            datasets: List of data series to plot
            labels: List of labels for each data series
            filename: Base name for plot image (without extension)
            xlabel: Label for x-axis
            ylabel: Label for y-axis
            title: Optional title for the plot
            colors: Optional list of colors for each data series

        Raises:
            ValueError: If datasets is empty or lengths don't match
        """
        if not datasets:
            raise ValueError("Cannot create plot: no datasets provided")
        if len(datasets) != len(labels):
            raise ValueError("Number of datasets must match number of labels")

        # Set up default colors if not provided
        if not colors or len(colors) < len(datasets):
            colors = plt.cm.tab10(np.linspace(0, 1, len(datasets)))

        # Set up plot styling
        plt.rcParams.update({"font.size": 24})
        plt.figure(figsize=(20, 11.25))

        # Plot each data series
        for i, (data, label) in enumerate(zip(datasets, labels)):
            plt.plot(data, label=label, color=colors[i], linewidth=2)

        # Add title if provided
        if title:
            plt.title(title, fontweight="bold")

        # Set plot formatting
        plt.ylabel(ylabel, fontweight="bold")
        plt.xlabel(xlabel, fontweight="bold")
        plt.legend(loc="best")
        plt.grid(True, linestyle="--", alpha=0.7)

        # Save the plot
        plot_path = os.path.join(self._path, f"plot_{filename}.png")
        plt.savefig(plot_path, dpi=self._dpi, bbox_inches="tight")
        plt.close("all")
        print(f"Saved comparison plot to {plot_path}")
