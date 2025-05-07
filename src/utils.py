import configparser
from sumolib import checkBinary
import os
import sys
import random
import numpy as np
import tensorflow as tf


def set_random_seed(seed=None):
    """
    Set random seeds for reproducibility
    """
    if seed is None:
        seed = int(random.random() * 1000)

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    return seed


def import_train_configuration(config_file):
    """
    Read the config file regarding the training and import its content
    """
    content = configparser.ConfigParser()
    content.read(config_file)
    config = {}

    # Simulation configuration
    config["gui"] = content["simulation"].getboolean("gui")
    config["total_episodes"] = content["simulation"].getint("total_episodes")
    config["max_steps"] = content["simulation"].getint("max_steps")
    config["n_cars_generated"] = content["simulation"].getint("n_cars_generated")
    config["green_duration"] = content["simulation"].getint("green_duration")
    config["yellow_duration"] = content["simulation"].getint("yellow_duration")

    # Add parallel agent support if available
    if content["simulation"].has_option("parallel_agents"):
        config["parallel_agents"] = content["simulation"].getboolean("parallel_agents")
    if content["simulation"].has_option("num_cpus"):
        config["num_cpus"] = content["simulation"].getint("num_cpus")

    # Agent configuration
    config["agent_type"] = content["agent"]["type"]
    config["num_states"] = content["agent"].getint("num_states")
    config["num_actions"] = content["agent"].getint("num_actions")
    config["gamma"] = content["agent"].getfloat("gamma")

    # Hardware config if available
    if "hardware" in content:
        config["hardware"] = {}
        hardware_section = content["hardware"]

        if hardware_section.has_option("gpu"):
            config["hardware"]["gpu"] = hardware_section.getboolean("gpu")
        if hardware_section.has_option("gpu_memory_limit"):
            config["hardware"]["gpu_memory_limit"] = hardware_section.getint(
                "gpu_memory_limit"
            )
        if hardware_section.has_option("mixed_precision"):
            config["hardware"]["mixed_precision"] = hardware_section.getboolean(
                "mixed_precision"
            )
        if hardware_section.has_option("use_amp"):
            config["hardware"]["use_amp"] = hardware_section.getboolean("use_amp")
        if hardware_section.has_option("xla_optimization"):
            config["hardware"]["xla_optimization"] = hardware_section.getboolean(
                "xla_optimization"
            )
        if hardware_section.has_option("num_parallel_calls"):
            config["hardware"]["num_parallel_calls"] = hardware_section.getint(
                "num_parallel_calls"
            )

    # DQN-specific parameters
    if config["agent_type"] == "dqn" or content.has_section("dqn"):
        config["num_layers"] = content["dqn"].getint("num_layers")
        config["width_layers"] = content["dqn"].getint("width_layers")
        config["batch_size"] = content["dqn"].getint("batch_size")
        config["learning_rate"] = content["dqn"].getfloat("learning_rate")
        config["training_epochs"] = content["dqn"].getint("training_epochs")
        config["memory_size_min"] = content["dqn"].getint("memory_size_min")
        config["memory_size_max"] = content["dqn"].getint("memory_size_max")

        # Additional DQN features
        if content["dqn"].has_option("target_update_frequency"):
            config["target_update_frequency"] = content["dqn"].getint(
                "target_update_frequency"
            )
        if content["dqn"].has_option("double_dqn"):
            config["double_dqn"] = content["dqn"].getboolean("double_dqn")

    # Q-Learning specific parameters
    if config["agent_type"] == "qlearning" or content.has_section("qlearning"):
        config["qlearning_learning_rate"] = content["qlearning"].getfloat(
            "learning_rate"
        )
        config["initial_value"] = content["qlearning"].getfloat("initial_value")

    # A2C specific parameters
    if config["agent_type"] == "a2c" or content.has_section("a2c"):
        config["actor_lr"] = content["a2c"].getfloat("actor_lr")
        config["critic_lr"] = content["a2c"].getfloat("critic_lr")
        config["shared_layers"] = content["a2c"].getint("shared_layers")
        config["shared_width"] = content["a2c"].getint("shared_width")
        config["actor_layers"] = content["a2c"].getint("actor_layers")
        config["actor_width"] = content["a2c"].getint("actor_width")
        config["critic_layers"] = content["a2c"].getint("critic_layers")
        config["critic_width"] = content["a2c"].getint("critic_width")

        # Additional A2C parameters
        if content["a2c"].has_option("entropy_coef"):
            config["entropy_coef"] = content["a2c"].getfloat("entropy_coef")
        if content["a2c"].has_option("value_coef"):
            config["value_coef"] = content["a2c"].getfloat("value_coef")

    # PPO specific parameters
    if config["agent_type"] == "ppo" or content.has_section("ppo"):
        config["actor_lr"] = content["ppo"].getfloat("actor_lr")
        config["critic_lr"] = content["ppo"].getfloat("critic_lr")
        config["lambd"] = content["ppo"].getfloat("lambd")
        config["clip_ratio"] = content["ppo"].getfloat("clip_ratio")
        config["epochs"] = content["ppo"].getint("epochs")
        config["batch_size"] = content["ppo"].getint("batch_size")
        config["shared_layers"] = content["ppo"].getint("shared_layers")
        config["shared_width"] = content["ppo"].getint("shared_width")
        config["actor_layers"] = content["ppo"].getint("actor_layers")
        config["actor_width"] = content["ppo"].getint("actor_width")
        config["critic_layers"] = content["ppo"].getint("critic_layers")
        config["critic_width"] = content["ppo"].getint("critic_width")

        # Additional PPO parameters
        if content["ppo"].has_option("entropy_coef"):
            config["entropy_coef"] = content["ppo"].getfloat("entropy_coef")
        if content["ppo"].has_option("value_coef"):
            config["value_coef"] = content["ppo"].getfloat("value_coef")

    # Baseline controllers specific parameters
    if config["agent_type"] in ["fixed", "actuated", "webster"] or content.has_section(
        "baseline"
    ):
        if config["agent_type"] in ["actuated", "webster"] or content.has_section(
            "baseline"
        ):
            config["min_green"] = content["baseline"].getint("min_green")
            config["max_green"] = content["baseline"].getint("max_green")
            config["extension_time"] = content["baseline"].getint("extension_time")

        if config["agent_type"] == "webster" or content.has_section("baseline"):
            config["saturation_flow_rate"] = content["baseline"].getint(
                "saturation_flow_rate"
            )
            config["lost_time_per_phase"] = content["baseline"].getint(
                "lost_time_per_phase"
            )
            config["update_interval"] = content["baseline"].getint("update_interval")

    # Directory paths
    config["models_path_name"] = content["dir"]["models_path_name"]
    config["sumocfg_file_name"] = content["dir"]["sumocfg_file_name"]

    # Add output directory if specified
    if content["dir"].has_option("output_dir"):
        config["output_dir"] = content["dir"]["output_dir"]

    return config


def import_test_configuration(config_file):
    """
    Read the config file regarding the testing and import its content
    """
    content = configparser.ConfigParser()
    content.read(config_file)
    config = {}

    # Simulation configuration
    config["gui"] = content["simulation"].getboolean("gui")
    config["max_steps"] = content["simulation"].getint("max_steps")
    config["n_cars_generated"] = content["simulation"].getint("n_cars_generated")
    config["episode_seed"] = content["simulation"].getint("episode_seed")
    config["green_duration"] = content["simulation"].getint("green_duration")
    config["yellow_duration"] = content["simulation"].getint("yellow_duration")

    # Agent configuration
    config["agent_type"] = content["agent"]["type"]
    config["num_states"] = content["agent"].getint("num_states")
    config["num_actions"] = content["agent"].getint("num_actions")

    # Directory paths
    config["sumocfg_file_name"] = content["dir"]["sumocfg_file_name"]
    config["models_path_name"] = content["dir"]["models_path_name"]
    config["model_to_test"] = content["dir"].getint("model_to_test")

    return config


def set_sumo(gui, sumocfg_file_name, max_steps, port=None):
    """
    Configure various parameters of SUMO with optional custom port for parallel simulations
    """
    # sumo things - we need to import python modules from the $SUMO_HOME/tools directory
    if "SUMO_HOME" in os.environ:
        tools = os.path.join(os.environ["SUMO_HOME"], "tools")
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")

    # setting the cmd mode or the visual mode
    if gui == False:
        sumoBinary = checkBinary("sumo")
    else:
        sumoBinary = checkBinary("sumo-gui")

    # setting the cmd command to run sumo at simulation time
    sumo_cmd = [
        sumoBinary,
        "-c",
        os.path.join("intersection", sumocfg_file_name),
        "--no-step-log",
        "true",
        "--waiting-time-memory",
        str(max_steps),
    ]

    # Add custom port if specified for parallel simulations
    if port is not None:
        sumo_cmd.extend(["--remote-port", str(port)])
        print(f"SUMO using remote port: {port}")

    return sumo_cmd


def set_train_path(models_path_name):
    """
    Create a new model path with an incremental integer, also considering previously created model paths
    """
    models_path = os.path.join(os.getcwd(), models_path_name, "")
    os.makedirs(os.path.dirname(models_path), exist_ok=True)

    dir_content = os.listdir(models_path)
    if dir_content:
        previous_versions = [
            int(name.split("_")[1]) for name in dir_content if name.startswith("model_")
        ]
        new_version = str(max(previous_versions) + 1) if previous_versions else "1"
    else:
        new_version = "1"

    data_path = os.path.join(models_path, "model_" + new_version, "")
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    return data_path


def set_test_path(models_path_name, model_n):
    """
    Returns a model path that identifies the model number provided as argument and a newly created 'test' path
    """
    model_folder_path = os.path.join(
        os.getcwd(), models_path_name, "model_" + str(model_n), ""
    )

    if os.path.isdir(model_folder_path):
        plot_path = os.path.join(model_folder_path, "test", "")
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        return model_folder_path, plot_path
    else:
        sys.exit("The model number specified does not exist in the models folder")
