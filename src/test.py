"""Traffic Signal Control Testing Script

This script tests a previously trained reinforcement learning agent on traffic signal control.
It loads a trained model, sets up the testing environment according to the configuration file,
and runs a test episode to evaluate the agent's performance.

Usage:
    python test.py [--config CONFIG_FILE] [--model MODEL_ID]

Example:
    python test.py --config custom_testing_settings.ini --model 3
"""

import os
import sys
import argparse
from shutil import copyfile
from typing import Dict, Any, Optional

from environment.testing_simulation import Simulation
from environment.generator import TrafficGenerator
from agent.model import TestModel
from utils.visualize import Visualization
from utils.utils import import_test_configuration, set_sumo, set_test_path


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test a trained reinforcement learning agent for traffic signal control"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="src/config/testing_settings.ini",
        help="Path to the testing configuration file",
    )
    parser.add_argument(
        "--model", type=int, help="Specific model ID to test (overrides config file)"
    )
    return parser.parse_args()


def setup_testing(config_file: str, model_id: Optional[int] = None) -> Dict[str, Any]:
    """Set up the testing environment based on the configuration file.

    Args:
        config_file: Path to the configuration file
        model_id: Optional model ID to override the one in the config file

    Returns:
        Dictionary containing all testing components

    Raises:
        FileNotFoundError: If the configuration file or model doesn't exist
        KeyError: If required configuration settings are missing
    """
    print(f"Loading testing configuration from {config_file}")

    try:
        # Load configuration
        config = import_test_configuration(config_file=config_file)

        # Override model ID if specified
        if model_id is not None:
            config["model_to_test"] = model_id
            print(f"Using model ID {model_id} (from command line)")
        else:
            print(f"Using model ID {config['model_to_test']} (from config file)")

        # Set up SUMO command
        sumo_cmd = set_sumo(
            config["gui"], config["sumocfg_file_name"], config["max_steps"]
        )

        # Get model and test paths
        try:
            model_path, plot_path = set_test_path(
                config["models_path_name"], config["model_to_test"]
            )
            print(f"Testing model from: {model_path}")
            print(f"Results will be saved to: {plot_path}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)

        # Initialize model
        model = TestModel(input_dim=config["num_states"], model_path=model_path)

        # Initialize traffic generator
        traffic_gen = TrafficGenerator(config["max_steps"], config["n_cars_generated"])

        # Initialize visualization
        visualization = Visualization(plot_path, dpi=96)

        # Initialize simulation
        simulation = Simulation(
            model,
            traffic_gen,
            sumo_cmd,
            config["max_steps"],
            config["green_duration"],
            config["yellow_duration"],
            config["num_states"],
            config["num_actions"],
        )

        # Return all components needed for testing
        return {
            "config": config,
            "plot_path": plot_path,
            "simulation": simulation,
            "visualization": visualization,
            "config_file": config_file,
        }

    except (KeyError, ValueError) as e:
        print(f"Error setting up testing: {e}")
        sys.exit(1)


def run_testing(setup: Dict[str, Any]) -> None:
    """Run the testing process for a single episode.

    Args:
        setup: Dictionary containing all testing components
    """
    config = setup["config"]
    plot_path = setup["plot_path"]
    simulation = setup["simulation"]
    visualization = setup["visualization"]
    config_file = setup["config_file"]

    try:
        print("\n----- Running test episode -----")

        # Run simulation for one episode
        simulation_time = simulation.run(config["episode_seed"])

        print(f"Simulation time: {simulation_time}s")
        print(f"Testing information saved at: {plot_path}")

        # Copy configuration file for reference
        dst_config = os.path.join(plot_path, os.path.basename(config_file))
        copyfile(src=config_file, dst=dst_config)

        # Generate and save plots
        generate_plots(simulation, visualization)

    except KeyboardInterrupt:
        print("\nTesting interrupted by user.")

    except Exception as e:
        print(f"Error during testing: {e}")
        sys.exit(1)


def generate_plots(simulation: Simulation, visualization: Visualization) -> None:
    """Generate and save plots of testing metrics.

    Args:
        simulation: Simulation object containing performance metrics
        visualization: Visualization object for creating plots
    """
    # Plot rewards
    visualization.save_data_and_plot(
        data=simulation.reward_episode,
        filename="reward",
        xlabel="Action step",
        ylabel="Reward",
        title="Test Performance: Reward per Action",
    )

    # Plot queue lengths
    visualization.save_data_and_plot(
        data=simulation.queue_length_episode,
        filename="queue",
        xlabel="Step",
        ylabel="Queue length (vehicles)",
        title="Test Performance: Queue Length",
    )

    # If multiple test runs available, create comparison plots
    try:
        # Compare with previous test runs if available
        # This is a placeholder for potential enhancements
        pass
    except Exception as e:
        print(f"Note: Could not generate comparison plots: {e}")


if __name__ == "__main__":
    args = parse_args()
    testing_setup = setup_testing(args.config, args.model)
    run_testing(testing_setup)
