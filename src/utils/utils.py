"""Utility Functions for Traffic Signal Control"""

import configparser
import os
import sys
from typing import Dict, List, Tuple, Any
from sumolib import checkBinary


def import_train_configuration(config_file: str) -> Dict[str, Any]:
    """Read training configuration from an INI file.

    Parses the configuration file to extract simulation parameters,
    model architecture, memory settings, and agent hyperparameters.

    Args:
        config_file: Path to the configuration file

    Returns:
        Dictionary containing all configuration parameters

    Raises:
        FileNotFoundError: If the configuration file doesn't exist
        KeyError: If required sections or options are missing
    """
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    content = configparser.ConfigParser()
    content.read(config_file)
    config = {}

    # Simulation settings
    try:
        config["gui"] = content["simulation"].getboolean("gui")
        config["total_episodes"] = content["simulation"].getint("total_episodes")
        config["max_steps"] = content["simulation"].getint("max_steps")
        config["n_cars_generated"] = content["simulation"].getint("n_cars_generated")
        config["green_duration"] = content["simulation"].getint("green_duration")
        config["yellow_duration"] = content["simulation"].getint("yellow_duration")

        # Model architecture
        config["num_layers"] = content["model"].getint("num_layers")
        config["width_layers"] = content["model"].getint("width_layers")
        config["batch_size"] = content["model"].getint("batch_size")
        config["learning_rate"] = content["model"].getfloat("learning_rate")
        config["training_epochs"] = content["model"].getint("training_epochs")

        # Memory settings
        config["memory_size_min"] = content["memory"].getint("memory_size_min")
        config["memory_size_max"] = content["memory"].getint("memory_size_max")

        # Agent parameters
        config["num_states"] = content["agent"].getint("num_states")
        config["num_actions"] = content["agent"].getint("num_actions")
        config["gamma"] = content["agent"].getfloat("gamma")

        # Directory settings
        config["models_path_name"] = content["dir"]["models_path_name"]
        config["sumocfg_file_name"] = content["dir"]["sumocfg_file_name"]
    except (KeyError, configparser.NoSectionError) as e:
        raise KeyError(f"Missing required configuration section or option: {e}")

    return config


def import_test_configuration(config_file: str) -> Dict[str, Any]:
    """Read testing configuration from an INI file.

    Parses the configuration file to extract simulation parameters,
    agent settings, and paths for testing.

    Args:
        config_file: Path to the configuration file

    Returns:
        Dictionary containing all configuration parameters

    Raises:
        FileNotFoundError: If the configuration file doesn't exist
        KeyError: If required sections or options are missing
    """
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    content = configparser.ConfigParser()
    content.read(config_file)
    config = {}

    # Simulation settings
    try:
        config["gui"] = content["simulation"].getboolean("gui")
        config["max_steps"] = content["simulation"].getint("max_steps")
        config["n_cars_generated"] = content["simulation"].getint("n_cars_generated")
        config["episode_seed"] = content["simulation"].getint("episode_seed")
        config["green_duration"] = content["simulation"].getint("green_duration")
        config["yellow_duration"] = content["simulation"].getint("yellow_duration")

        # Agent parameters
        config["num_states"] = content["agent"].getint("num_states")
        config["num_actions"] = content["agent"].getint("num_actions")

        # Directory settings
        config["sumocfg_file_name"] = content["dir"]["sumocfg_file_name"]
        config["models_path_name"] = content["dir"]["models_path_name"]
        config["model_to_test"] = content["dir"].getint("model_to_test")
    except (KeyError, configparser.NoSectionError) as e:
        raise KeyError(f"Missing required configuration section or option: {e}")

    return config


def set_sumo(gui: bool, sumocfg_file_name: str, max_steps: int) -> List[str]:
    """Configure and set up SUMO command with appropriate parameters.

    Args:
        gui: Whether to use GUI mode (True) or command-line mode (False)
        sumocfg_file_name: Name of the SUMO configuration file
        max_steps: Maximum number of simulation steps

    Returns:
        List of command-line arguments to start SUMO

    Raises:
        EnvironmentError: If SUMO_HOME environment variable is not set
    """
    # Check and set up SUMO environment
    if "SUMO_HOME" not in os.environ:
        raise EnvironmentError(
            "SUMO_HOME environment variable not set. Please declare this variable."
        )

    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)

    # Select binary based on GUI mode
    sumoBinary = checkBinary("sumo-gui" if gui else "sumo")

    # Set up command with appropriate parameters
    sumo_cmd = [
        sumoBinary,
        "-c",
        os.path.join("intersection", sumocfg_file_name),
        "--no-step-log",
        "true",
        "--waiting-time-memory",
        str(max_steps),
    ]

    return sumo_cmd


def set_train_path(models_path_name: str) -> str:
    """Create a new directory path for saving training results.

    Creates a model directory with an incremental version number
    based on previously created directories.

    Args:
        models_path_name: Name of the base directory for models

    Returns:
        Path to the new model directory
    """
    # Ensure base models directory exists
    models_path = os.path.join(os.getcwd(), models_path_name, "")
    os.makedirs(os.path.dirname(models_path), exist_ok=True)

    # Determine new model version number
    dir_content = os.listdir(models_path)
    if dir_content:
        # Extract version numbers from existing directories
        try:
            previous_versions = [
                int(name.split("_")[1])
                for name in dir_content
                if name.startswith("model_") and name.split("_")[1].isdigit()
            ]
            new_version = str(max(previous_versions) + 1)
        except (IndexError, ValueError):
            # If parsing fails, default to version 1
            new_version = "1"
    else:
        new_version = "1"

    # Create and return path to new model directory
    data_path = os.path.join(models_path, f"model_{new_version}", "")
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    return data_path


def set_test_path(models_path_name: str, model_n: int) -> Tuple[str, str]:
    """Create paths for loading a model and saving test results.

    Args:
        models_path_name: Name of the base directory for models
        model_n: Model number/version to test

    Returns:
        Tuple of (model_folder_path, test_results_path)

    Raises:
        FileNotFoundError: If the specified model directory doesn't exist
    """
    # Construct model folder path
    model_folder_path = os.path.join(
        os.getcwd(), models_path_name, f"model_{model_n}", ""
    )

    # Check if model exists
    if not os.path.isdir(model_folder_path):
        raise FileNotFoundError(
            f"Model {model_n} not found in {models_path_name} folder."
        )

    # Create test results directory
    test_path = os.path.join(model_folder_path, "test", "")
    os.makedirs(os.path.dirname(test_path), exist_ok=True)

    return model_folder_path, test_path
