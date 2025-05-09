"""Traffic Signal Control Training Script

Usage:
    python train.py [--config CONFIG_FILE]

Example:
    python train.py --config custom_training_settings.ini
"""

import os
import sys
import argparse
import datetime
from shutil import copyfile
from typing import Dict, Any

from environment.training_simulation import Simulation
from environment.generator import TrafficGenerator
from agent.memory import Memory
from agent.model import TrainModel
from utils.visualize import Visualization
from utils.utils import import_train_configuration, set_sumo, set_train_path


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a reinforcement learning agent for traffic signal control"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="src/config/training_settings.ini",
        help="Path to the configuration file",
    )
    return parser.parse_args()


def setup_training(config_file: str) -> Dict[str, Any]:
    """Set up the training environment based on the configuration file.

    Args:
        config_file: Path to the configuration file

    Returns:
        Dictionary containing all training components

    Raises:
        FileNotFoundError: If the configuration file doesn't exist
        KeyError: If required configuration settings are missing
    """
    print(f"Loading configuration from {config_file}")

    try:
        # Load configuration
        config = import_train_configuration(config_file=config_file)

        # Set up SUMO command
        sumo_cmd = set_sumo(
            config["gui"], config["sumocfg_file_name"], config["max_steps"]
        )

        # Create path for saving model and results
        path = set_train_path(config["models_path_name"])
        print(f"Training results will be saved to: {path}")

        # Initialize model
        model = TrainModel(
            config["num_layers"],
            config["width_layers"],
            config["batch_size"],
            config["learning_rate"],
            input_dim=config["num_states"],
            output_dim=config["num_actions"],
        )

        # Initialize replay memory
        memory = Memory(config["memory_size_max"], config["memory_size_min"])

        # Initialize traffic generator
        traffic_gen = TrafficGenerator(config["max_steps"], config["n_cars_generated"])

        # Initialize visualization
        visualization = Visualization(path, dpi=96)

        # Initialize simulation
        simulation = Simulation(
            model,
            memory,
            traffic_gen,
            sumo_cmd,
            config["gamma"],
            config["max_steps"],
            config["green_duration"],
            config["yellow_duration"],
            config["num_states"],
            config["num_actions"],
            config["training_epochs"],
        )

        # Return all components needed for training
        return {
            "config": config,
            "path": path,
            "model": model,
            "simulation": simulation,
            "visualization": visualization,
            "config_file": config_file,
        }

    except (FileNotFoundError, KeyError) as e:
        print(f"Error setting up training: {e}")
        sys.exit(1)


def run_training(setup: Dict[str, Any]) -> None:
    """Run the training process for the specified number of episodes.

    Args:
        setup: Dictionary containing all training components
    """
    config = setup["config"]
    path = setup["path"]
    model = setup["model"]
    simulation = setup["simulation"]
    visualization = setup["visualization"]
    config_file = setup["config_file"]

    episode = 0
    timestamp_start = datetime.datetime.now()
    print(f"Training started at: {timestamp_start}")

    try:
        # Training loop
        while episode < config["total_episodes"]:
            print(f"\n----- Episode {episode + 1} of {config['total_episodes']} -----")

            # Calculate epsilon for epsilon-greedy policy
            epsilon = 1.0 - (episode / config["total_episodes"])

            # Run simulation for one episode
            simulation_time, training_time = simulation.run(episode, epsilon)

            # Print timing information
            total_time = round(simulation_time + training_time, 1)
            print(
                f"Simulation time: {simulation_time}s - Training time: {training_time}s - Total: {total_time}s"
            )

            episode += 1

        # Print session information
        print(f"\n----- Start time: {timestamp_start}")
        print(f"----- End time: {datetime.datetime.now()}")
        print(f"----- Session info saved at: {path}")

        # Save model
        model.save_model(path)

        # Copy configuration file for reference
        copyfile(src=config_file, dst=os.path.join(path, os.path.basename(config_file)))

        # Generate and save plots
        generate_plots(simulation, visualization)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")

        # Save the model even if interrupted
        if episode > 0:
            print("Saving model from interrupted training...")
            model.save_model(path)
            copyfile(
                src=config_file, dst=os.path.join(path, os.path.basename(config_file))
            )
            generate_plots(simulation, visualization)

    except Exception as e:
        print(f"Error during training: {e}")
        sys.exit(1)


def generate_plots(simulation: Simulation, visualization: Visualization) -> None:
    """Generate and save plots of training metrics.

    Args:
        simulation: Simulation object containing performance metrics
        visualization: Visualization object for creating plots
    """
    # Plot cumulative negative reward
    visualization.save_data_and_plot(
        data=simulation.reward_store,
        filename="reward",
        xlabel="Episode",
        ylabel="Cumulative negative reward",
        title="Training Performance: Reward",
    )

    # Plot cumulative delay
    visualization.save_data_and_plot(
        data=simulation.cumulative_wait_store,
        filename="delay",
        xlabel="Episode",
        ylabel="Cumulative delay (s)",
        title="Training Performance: Delay",
    )

    # Plot average queue length
    visualization.save_data_and_plot(
        data=simulation.avg_queue_length_store,
        filename="queue",
        xlabel="Episode",
        ylabel="Average queue length (vehicles)",
        title="Training Performance: Queue Length",
    )


if __name__ == "__main__":
    args = parse_args()
    training_setup = setup_training(args.config)
    run_training(training_setup)
