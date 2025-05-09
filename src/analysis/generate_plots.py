"""Performance Visualization for Traffic Signal Control

- Traffic demand is modeled using Weibull distribution: f(x;k,λ) = (k/λ)(x/λ)^(k-1)e^(-(x/λ)^k)
  where k=2 (shape parameter) and λ=max_steps/√2 (scale parameter)
- Improvement metrics: Δ = (baseline - dqn)/baseline * 100%
"""

from typing import Dict, List, Optional, Union

import os
import datetime
import warnings

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from scipy.stats import weibull_min


def load_data(file_path: str) -> Optional[np.ndarray]:
    """Load numeric data from a text file, with one value per line.

    Args:
        file_path: Path to the data file

    Returns:
        NumPy array of values, or None if the file doesn't exist or has errors
    """
    try:
        with open(file_path, "r") as f:
            data = [float(line.strip()) for line in f if line.strip()]
        return np.array(data)
    except FileNotFoundError:
        print(f"Warning: Data file not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None


def plot_metric(
    data: np.ndarray,
    metric_name: str,
    model_name: str,
    output_dir: str,
    color: str,
    y_label: str,
    episode_smoothing_window: int = 5,
) -> None:
    """Generate and save a plot for training metrics across episodes.
    Includes raw metric values and a smoothed moving average.

    Args:
        data: Array of metric values per episode
        metric_name: Name of the metric being plotted
        model_name: Name/ID of the model
        output_dir: Directory to save the plot
        color: Color for the plot lines
        y_label: Label for the y-axis
        episode_smoothing_window: Window size for moving average smoothing
    """
    if data is None or data.size == 0:
        print(f"Skipping plot for {metric_name} due to missing/empty data.")
        return

    episodes = np.arange(1, len(data) + 1)
    plt.figure(figsize=(8, 5))
    sns.set_theme(style="whitegrid", palette="muted")

    # Plot raw data
    plt.plot(
        episodes,
        data,
        color=color,
        alpha=0.3,
        linewidth=1.5,
        label=f"Episodic {y_label}",
    )

    # Plot moving average if enough data points
    if len(data) >= episode_smoothing_window:
        moving_avg = np.convolve(
            data,
            np.ones(episode_smoothing_window) / episode_smoothing_window,
            mode="valid",
        )
        moving_avg_episodes = episodes[episode_smoothing_window - 1 :]
        plt.plot(
            moving_avg_episodes,
            moving_avg,
            color=color,
            linestyle="-",
            linewidth=2.5,
            label=f"{episode_smoothing_window}-Episode Moving Avg.",
        )

    # Set plot styling
    plt.xlabel("Episode", fontsize=14, fontweight="medium")
    plt.ylabel(y_label, fontsize=14, fontweight="medium")
    plt.legend(fontsize=12)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    plt.tick_params(axis="both", which="major", labelsize=12)
    ax = plt.gca()
    ax.set_facecolor("#f7f7f7")
    plt.tight_layout(pad=1.5)

    # Save the plot
    plot_filename = f"plot_{metric_name.lower().replace(' ', '_')}.png"
    output_path = os.path.join(output_dir, plot_filename)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved enhanced training plot to: {output_path}")
    plt.close()


def plot_3way_comparison(
    model_path_name: str = "model_1",
    fixed_dqn_data_dir: Optional[str] = None,
    actuated_data_dir: Optional[str] = None,
    scenario_display_name: str = "Default Test Run",
) -> None:
    """Generate comparative plots between DQN, fixed-time, and actuated control strategies.
    Creates plots showing performance metrics of different control strategies and
    calculates improvement percentages. Includes a traffic demand visualization.

    Args:
        model_path_name: Name/ID of the model directory
        fixed_dqn_data_dir: Directory containing fixed-time and DQN results
        actuated_data_dir: Directory containing actuated control results
        scenario_display_name: Human-readable name for the test scenario
    """
    print(
        f"\n--- Generating 3-Way Comparative Plots for Scenario: {scenario_display_name} ---"
    )

    # Ensure base directories exist
    if not (fixed_dqn_data_dir and os.path.isdir(fixed_dqn_data_dir)):
        print(
            f"Warning: Fixed/DQN data directory not found or not specified: {fixed_dqn_data_dir}"
        )
    if not (actuated_data_dir and os.path.isdir(actuated_data_dir)):
        print(
            f"Warning: Actuated data directory not found or not specified: {actuated_data_dir}"
        )

    # Create a combined output directory
    model_base_dir = (
        os.path.dirname(fixed_dqn_data_dir)
        if fixed_dqn_data_dir and os.path.isdir(fixed_dqn_data_dir)
        else (
            os.path.dirname(actuated_data_dir)
            if actuated_data_dir and os.path.isdir(actuated_data_dir)
            else None
        )
    )

    if not model_base_dir:
        return print(
            "Error: Cannot determine model base directory from provided paths."
        )

    comparison_scenario_name = (
        scenario_display_name.lower().replace(" ", "_").replace("/", "_")
    )
    combined_output_dir = os.path.join(
        model_base_dir, f"comparison_{comparison_scenario_name}"
    )
    os.makedirs(combined_output_dir, exist_ok=True)
    print(f"Saving comparison plots to: {combined_output_dir}")

    # Traffic demand curve parameters (Weibull distribution)
    max_steps_approx = 5400
    weibull_k = 2  # Shape parameter
    weibull_lambda = max_steps_approx / np.sqrt(2)  # Scale parameter

    # Define metrics to compare
    # Format: (metric_label, y_axis_label, dqn_file, fixed_file, actuated_file, output_name)
    comparative_metrics = [
        (
            "Total Waiting Time",
            "Total Acc. Waiting Time (s)",
            "dqn_step_wait_times.txt",
            "fixed_step_wait_times.txt",
            "actuated_step_wait_times.txt",
            "comparison_wait_time",
        ),
        (
            "Total Queue Length",
            "Total Queue Length (vehicles)",
            "dqn_step_queue_lengths.txt",
            "fixed_step_queue_lengths.txt",
            "actuated_step_queue_lengths.txt",
            "comparison_queue_length",
        ),
    ]

    summary_stats: Dict[str, Dict[str, Dict[str, Union[float, str]]]] = {}
    plot_files_generated: List[str] = []

    for (
        metric_label,
        y_label,
        dqn_suffix,
        fixed_suffix,
        actuated_suffix,
        plot_base,
    ) in comparative_metrics:
        print(f"\nProcessing: {metric_label}")

        # Load data for each control strategy
        dqn_data = (
            load_data(os.path.join(fixed_dqn_data_dir, dqn_suffix))
            if fixed_dqn_data_dir
            else None
        )
        fixed_data = (
            load_data(os.path.join(fixed_dqn_data_dir, fixed_suffix))
            if fixed_dqn_data_dir
            else None
        )
        actuated_data = (
            load_data(os.path.join(actuated_data_dir, actuated_suffix))
            if actuated_data_dir
            else None
        )

        # Collect available datasets
        available_data: Dict[str, np.ndarray] = {}
        if dqn_data is not None and dqn_data.size > 0:
            available_data["DQN"] = dqn_data
        if fixed_data is not None and fixed_data.size > 0:
            available_data["Fixed"] = fixed_data
        if actuated_data is not None and actuated_data.size > 0:
            available_data["Actuated"] = actuated_data

        if len(available_data) < 2:
            print(f"Skipping plot for {metric_label}: Need at least two datasets.")
            continue

        # Create comparative plot
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.set_theme(style="whitegrid", palette="muted")

        # Define visual styles for each strategy
        colors = {"DQN": "#1f77b4", "Fixed": "#ff7f0e", "Actuated": "#2ca02c"}
        linestyles = {"DQN": "-", "Fixed": "--", "Actuated": ":"}

        # Find common length for all datasets
        plot_stats_data = []
        min_len = min(len(data) for data in available_data.values())
        if min_len == 0:
            continue

        steps = np.arange(1, min_len + 1)
        summary_stats.setdefault(metric_label, {})

        # Calculate traffic demand curve (Weibull distribution)
        demand_pdf = weibull_min.pdf(steps, weibull_k, loc=0, scale=weibull_lambda)
        demand_pdf_scaled = (demand_pdf / (np.max(demand_pdf) + 1e-9)) * 0.9

        # Plot primary metrics for each strategy
        for mode, data in available_data.items():
            if len(data) != min_len:
                warnings.warn(f"Truncating {mode} data for {metric_label}.")

            data_truncated = data[:min_len]
            ax.plot(
                steps,
                data_truncated,
                color=colors[mode],
                linestyle=linestyles[mode],
                linewidth=2.2,
                label=mode,
            )

            # Calculate summary statistics
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                if "Waiting Time" in metric_label:
                    val = np.nansum(data_truncated)
                    unit = "Total Wait (s)"
                else:
                    val = np.nanmean(data_truncated)
                    unit = "Avg Queue"

            plot_stats_data.append([f"{val:.1f}"])
            summary_stats[metric_label][mode] = {"value": val, "unit": unit}

        # Set up plot styling
        ax.set_xlabel("Simulation Step (seconds)", fontsize=14, fontweight="medium")
        ax.set_ylabel(y_label, fontsize=14, fontweight="medium")
        ax.legend(fontsize=12, loc="upper left")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.tick_params(axis="both", which="major", labelsize=12)
        ax.set_facecolor("#f9f9f9")
        y_lim_primary = ax.get_ylim()

        # Add traffic demand curve as secondary axis
        ax2 = ax.twinx()
        ax2.plot(
            steps,
            demand_pdf_scaled * y_lim_primary[1],
            color="grey",
            linestyle=":",
            linewidth=1.5,
            alpha=0.6,
            label="Traffic Demand Intensity",
        )
        ax2.set_ylabel("Traffic Demand Intensity (Scaled)", fontsize=12, color="grey")
        ax2.tick_params(axis="y", labelcolor="grey", labelsize=10)
        ax2.set_ylim(0, y_lim_primary[1] * 1.05)
        ax2.grid(False)

        # Add demand curve legend
        handles, labels = ax2.get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=1,
            fontsize=10,
        )

        # Save the plot
        fig.tight_layout(rect=[0, 0.05, 0.95, 1])
        plot_filename = (
            f"{plot_base}_3way_{scenario_display_name.lower().replace(' ', '_')}.png"
        )
        output_path = os.path.join(combined_output_dir, plot_filename)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved 3-way comparative plot to: {output_path}")
        plot_files_generated.append(output_path)
        plt.close(fig)

    # Generate summary log with statistics
    log_content = []
    log_content.append(f"Comparison Summary for Model: {model_path_name}")
    log_content.append(f"Scenario: {scenario_display_name}")
    log_content.append(
        f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    )

    for metric, modes_data in summary_stats.items():
        if not modes_data:
            continue

        log_content.append(f"Metric: {metric}")
        baseline_val = modes_data.get("Fixed", {}).get("value", np.nan)
        actuated_val = modes_data.get("Actuated", {}).get("value", np.nan)
        dqn_val = modes_data.get("DQN", {}).get("value", np.nan)

        # Log values for each control strategy
        for mode, stats in sorted(modes_data.items()):
            log_content.append(f"  {mode}: {stats['value']:.2f} ({stats['unit']})")

        # Calculate and log improvement percentages
        if not np.isnan(dqn_val) and not np.isnan(baseline_val) and baseline_val > 1e-6:
            reduction_vs_fixed = ((baseline_val - dqn_val) / baseline_val) * 100
            log_content.append(
                f"  Improvement by DQN vs Fixed: {reduction_vs_fixed:.2f}%"
            )
        else:
            log_content.append("  Improvement by DQN vs Fixed: N/A")

        if not np.isnan(dqn_val) and not np.isnan(actuated_val) and actuated_val > 1e-6:
            reduction_vs_actuated = ((actuated_val - dqn_val) / actuated_val) * 100
            log_content.append(
                f"  Improvement by DQN vs Actuated: {reduction_vs_actuated:.2f}%"
            )
        else:
            log_content.append("  Improvement by DQN vs Actuated: N/A")

        log_content.append("")  # Add newline between metrics

    # Write summary log file
    log_filename = f"comparison_summary_stats_{scenario_display_name.lower().replace(' ', '_')}.txt"
    log_filepath = os.path.join(combined_output_dir, log_filename)
    try:
        with open(log_filepath, "w") as f:
            f.write("\n".join(log_content))
        print(f"Saved summary statistics to: {log_filepath}")
    except Exception as e:
        print(f"Error writing summary log file: {e}")

    # Print summary to console
    print("\n--- Quantitative Comparison Summary (also saved to log file) ---")
    print("\n".join(log_content[3:]))


def generate_all_plots(model_path_name: str = "model_1") -> None:
    """Generate training progress plots for a specific model.
    Creates visualizations of reward, queue length, and delay metrics
    over the course of training.

    Args:
        model_path_name: Name/ID of the model directory
    """
    print("--- Generating Training Plots ---")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "models", model_path_name)

    if not os.path.isdir(model_dir):
        print(f"Error: Model directory not found at {model_dir}")
        return

    print(f"Generating training plots for data in: {model_dir}")

    # Define metrics to plot
    metrics_to_plot = [
        {
            "name": "Reward",
            "file": "plot_reward_data.txt",
            "color": "#2c7bb6",
            "ylabel": "Cumulative Reward",
        },
        {
            "name": "Queue Length",
            "file": "plot_queue_data.txt",
            "color": "#d7191c",
            "ylabel": "Average Queue Length (vehicles)",
        },
        {
            "name": "Delay",
            "file": "plot_delay_data.txt",
            "color": "#fdae61",
            "ylabel": "Cumulative Delay (seconds)",
        },
    ]

    # Generate plot for each metric
    for metric_info in metrics_to_plot:
        data_file_path = os.path.join(model_dir, metric_info["file"])
        data = load_data(data_file_path)
        if data is not None and data.size > 0:
            plot_metric(
                data,
                metric_info["name"],
                model_path_name,
                model_dir,
                metric_info["color"],
                metric_info["ylabel"],
            )


if __name__ == "__main__":
    # Check for required packages
    try:
        import matplotlib, seaborn, scipy
    except ImportError:
        exit(
            "Error: Matplotlib/Seaborn/Scipy not installed. Use: pip install matplotlib seaborn scipy"
        )

    # Parse command line arguments
    import sys

    model_to_plot = "model_1"
    scenario_arg = None

    if len(sys.argv) > 1:
        model_to_plot = sys.argv[1]
        print(f"Targeting model: {model_to_plot}")
    if len(sys.argv) > 2:
        scenario_arg = sys.argv[2]
        print(f"Targeting specific scenario based on basename: {scenario_arg}")

    # Generate training plots
    generate_all_plots(model_path_name=model_to_plot)

    # Determine paths for comparative plotting
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_base_dir_main = os.path.join(script_dir, "models", model_to_plot)

    if scenario_arg:
        # Construct paths based on the provided scenario argument
        fixed_dqn_dir = os.path.join(
            model_base_dir_main, f"test_sumo_config_{scenario_arg}_fixed"
        )
        actuated_dir = os.path.join(
            model_base_dir_main, f"test_sumo_config_{scenario_arg}_actuated"
        )
        display_name = scenario_arg.replace("_", " ").title()

        # Check for rebuilt actuated directory if standard one doesn't exist
        if not os.path.isdir(actuated_dir):
            actuated_dir_rebuilt = os.path.join(
                model_base_dir_main, f"test_sumo_config_{scenario_arg}_actuated_rebuilt"
            )
            if os.path.isdir(actuated_dir_rebuilt):
                print(f"Using actuated directory: {actuated_dir_rebuilt}")
                actuated_dir = actuated_dir_rebuilt
            else:
                print(
                    f"Warning: Could not find standard or rebuilt actuated directory for scenario {scenario_arg}"
                )
                actuated_dir = None
    else:
        # Use default paths for standard test run
        print("No specific scenario provided, using default test run directories.")
        fixed_dqn_dir = os.path.join(model_base_dir_main, "test_sumo_config")
        actuated_dir = os.path.join(
            model_base_dir_main, "test_sumo_config_actuated_rebuilt"
        )
        display_name = "Default Test Run"

    # Generate comparative plots
    plot_3way_comparison(
        model_path_name=model_to_plot,
        fixed_dqn_data_dir=fixed_dqn_dir,
        actuated_data_dir=actuated_dir,
        scenario_display_name=display_name,
    )

    print("\nPlot generation finished.")
