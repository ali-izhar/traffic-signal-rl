from __future__ import absolute_import
from __future__ import print_function

import os
import datetime
import multiprocessing as mp
import threading
import concurrent.futures
import time
import numpy as np
import tensorflow as tf
from shutil import copyfile

from training_simulation import Simulation
from generator import TrafficGenerator
from visualization import Visualization
from utils import import_train_configuration, set_sumo, set_train_path

# Import all agent types
from agents.dqn_agent import DQNAgent
from agents.qlearning_agent import QLearningAgent
from agents.a2c_agent import A2CAgent
from agents.ppo_agent import PPOAgent
from agents.baseline_controllers import (
    FixedTimingController,
    ActuatedController,
    WebsterController,
)


# Configure GPU for optimal performance
def configure_gpu(config=None):
    # Enable GPU memory growth
    physical_devices = tf.config.list_physical_devices("GPU")
    if len(physical_devices) > 0:
        print(f"Found {len(physical_devices)} GPU(s)")
        for device in physical_devices:
            try:
                tf.config.experimental.set_memory_growth(device, True)
                print(f"Memory growth enabled for {device}")
            except Exception as e:
                print(f"Error setting memory growth: {e}")

        # Try to limit memory growth if specified in config
        if config and config.get("hardware", {}).get("gpu_memory_limit"):
            try:
                limit = int(config["hardware"]["gpu_memory_limit"])
                tf.config.set_logical_device_configuration(
                    physical_devices[0],
                    [tf.config.LogicalDeviceConfiguration(memory_limit=limit)],
                )
                print(f"GPU memory limit set to {limit}MB")
            except Exception as e:
                print(f"Error setting memory limit: {e}")
    else:
        print("No GPUs found. Running on CPU.")

    # Enable XLA optimization if specified
    xla_setting = config and config.get("hardware", {}).get("xla_optimization", False)
    xla_enabled = False
    
    # Handle both string and boolean values
    if isinstance(xla_setting, str):
        xla_enabled = xla_setting.lower() == "true"
    elif isinstance(xla_setting, bool):
        xla_enabled = xla_setting
        
    if xla_enabled:
        tf.config.optimizer.set_jit(True)
        print("XLA optimization enabled")


# Create a specific agent based on configuration
def create_agent(agent_type, config):
    if agent_type == "dqn":
        use_prioritized = (
            config.get("agent", {}).get("prioritized_memory", "False").lower() == "true"
        )
        use_double = config.get("dqn", {}).get("double_dqn", "False").lower() == "true"

        return DQNAgent(
            input_dim=config["num_states"],
            output_dim=config["num_actions"],
            num_layers=config["num_layers"],
            width=config["width_layers"],
            batch_size=config["batch_size"],
            learning_rate=config["learning_rate"],
            gamma=config["gamma"],
            memory_size_max=config["memory_size_max"],
            memory_size_min=config["memory_size_min"],
            use_prioritized_replay=use_prioritized,
            use_double_dqn=use_double,
            target_update_freq=int(config.get("target_update_frequency", 1000)),
        )
    elif agent_type == "qlearning":
        return QLearningAgent(
            input_dim=config["num_states"],
            output_dim=config["num_actions"],
            learning_rate=config["qlearning_learning_rate"],
            gamma=config["gamma"],
            initial_value=config["initial_value"],
        )
    elif agent_type == "a2c":
        return A2CAgent(
            input_dim=config["num_states"],
            output_dim=config["num_actions"],
            actor_lr=config["actor_lr"],
            critic_lr=config["critic_lr"],
            gamma=config["gamma"],
            shared_layers=config["shared_layers"],
            shared_width=config["shared_width"],
            actor_layers=config["actor_layers"],
            actor_width=config["actor_width"],
            critic_layers=config["critic_layers"],
            critic_width=config["critic_width"],
            entropy_coef=float(config.get("entropy_coef", 0.01)),
            value_coef=float(config.get("value_coef", 0.5)),
        )
    elif agent_type == "ppo":
        return PPOAgent(
            input_dim=config["num_states"],
            output_dim=config["num_actions"],
            actor_lr=config["actor_lr"],
            critic_lr=config["critic_lr"],
            gamma=config["gamma"],
            lambd=config["lambd"],
            clip_ratio=config["clip_ratio"],
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            shared_layers=config["shared_layers"],
            shared_width=config["shared_width"],
            actor_layers=config["actor_layers"],
            actor_width=config["actor_width"],
            critic_layers=config["critic_layers"],
            critic_width=config["critic_width"],
            entropy_coef=float(config.get("entropy_coef", 0.01)),
            value_coef=float(config.get("value_coef", 0.5)),
        )
    elif agent_type == "fixed":
        return FixedTimingController(
            input_dim=config["num_states"],
            output_dim=config["num_actions"],
        )
    elif agent_type == "actuated":
        return ActuatedController(
            input_dim=config["num_states"],
            output_dim=config["num_actions"],
            min_green=config["min_green"],
            max_green=config["max_green"],
            extension_time=config["extension_time"],
            yellow_time=config["yellow_duration"],
        )
    elif agent_type == "webster":
        return WebsterController(
            input_dim=config["num_states"],
            output_dim=config["num_actions"],
            saturation_flow_rate=config["saturation_flow_rate"],
            yellow_time=config["yellow_duration"],
            lost_time_per_phase=config["lost_time_per_phase"],
            update_interval=config["update_interval"],
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


# Train a single agent
def train_agent(agent_type, config, base_path, port_offset=0):
    agent_path = os.path.join(base_path, agent_type)
    os.makedirs(agent_path, exist_ok=True)

    # Create agent
    agent = create_agent(agent_type, config)

    # Each simulation needs a unique port for SUMO
    custom_port = 8813 + port_offset
    custom_sumo_cmd = set_sumo(
        config["gui"],
        config["sumocfg_file_name"],
        config["max_steps"],
        port=custom_port,
    )

    # Create simulator components
    traffic_gen = TrafficGenerator(config["max_steps"], config["n_cars_generated"])
    visualizer = Visualization(agent_path, dpi=96)

    simulator = Simulation(
        agent,
        traffic_gen,
        custom_sumo_cmd,
        config["gamma"],
        config["max_steps"],
        config["green_duration"],
        config["yellow_duration"],
        config["num_states"],
        config["num_actions"],
    )

    # Train for the specified number of episodes
    episode = 0
    timestamp_start = datetime.datetime.now()
    print(f"\n----- Training {agent_type} agent")

    while episode < config["total_episodes"]:
        print(
            f"\n----- {agent_type}: Episode {episode + 1} of {config['total_episodes']}"
        )
        # Dynamic epsilon based on episode progress
        epsilon = max(0.05, 1.0 - (episode / config["total_episodes"]))
        simulation_time = simulator.run(episode, epsilon)
        print(
            f"{agent_type} - Episode {episode+1}: Reward: {simulator.reward_store[-1]}, "
            f"Time: {simulation_time}s, Epsilon: {epsilon:.2f}"
        )
        episode += 1

    print(f"\n----- {agent_type} training completed")
    print(f"----- Start time: {timestamp_start}")
    print(f"----- End time: {datetime.datetime.now()}")

    # Save agent and visualization
    agent.save(agent_path)
    copyfile(
        src="training_settings.ini",
        dst=os.path.join(agent_path, "training_settings.ini"),
    )

    # Save performance plots
    visualizer.save_data_and_plot(
        data=simulator.reward_store,
        filename="reward",
        xlabel="Episode",
        ylabel="Cumulative negative reward",
    )
    visualizer.save_data_and_plot(
        data=simulator.cumulative_wait_store,
        filename="delay",
        xlabel="Episode",
        ylabel="Cumulative delay (s)",
    )
    visualizer.save_data_and_plot(
        data=simulator.avg_queue_length_store,
        filename="queue",
        xlabel="Episode",
        ylabel="Average queue length (vehicles)",
    )

    return agent_type, simulator.reward_store, simulator.cumulative_wait_store


# Train multiple agents in parallel
def train_agents_parallel(agent_types, config):
    base_path = set_train_path(config["models_path_name"])

    # Determine if we're using parallel execution
    use_parallel = (
        config.get("simulation", {}).get("parallel_agents", "False").lower() == "true"
    )
    num_cpus = min(
        len(agent_types),
        int(config.get("simulation", {}).get("num_cpus", mp.cpu_count())),
    )

    results = {}

    if use_parallel:
        print(
            f"Training {len(agent_types)} agents in parallel using {num_cpus} processes"
        )
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_cpus) as executor:
            futures = {}
            for i, agent_type in enumerate(agent_types):
                future = executor.submit(train_agent, agent_type, config, base_path, i)
                futures[future] = agent_type

            for future in concurrent.futures.as_completed(futures):
                agent_type = futures[future]
                try:
                    agent_type, reward_data, wait_data = future.result()
                    results[agent_type] = (reward_data, wait_data)
                    print(f"Agent {agent_type} training completed successfully")
                except Exception as e:
                    print(f"Agent {agent_type} training failed: {e}")
    else:
        print(f"Training {len(agent_types)} agents sequentially")
        for i, agent_type in enumerate(agent_types):
            try:
                agent_type, reward_data, wait_data = train_agent(
                    agent_type, config, base_path, i
                )
                results[agent_type] = (reward_data, wait_data)
                print(f"Agent {agent_type} training completed successfully")
            except Exception as e:
                print(f"Agent {agent_type} training failed: {e}")

    # Create comparative visualization
    create_comparative_plots(results, base_path)

    return results


# Create comparative plots for all agents
def create_comparative_plots(results, base_path):
    visualization = Visualization(base_path, dpi=96)

    # Prepare data for reward comparison
    reward_data = {}
    wait_data = {}

    for agent_type, (rewards, waits) in results.items():
        reward_data[agent_type] = rewards
        wait_data[agent_type] = waits

    # Create comparative plots
    visualization.save_comparative_data_and_plot(
        data_dict=reward_data,
        filename="comparative_reward",
        xlabel="Episode",
        ylabel="Cumulative negative reward",
        title="Performance Comparison - Reward",
    )

    visualization.save_comparative_data_and_plot(
        data_dict=wait_data,
        filename="comparative_wait",
        xlabel="Episode",
        ylabel="Cumulative waiting time (s)",
        title="Performance Comparison - Waiting Time",
    )


if __name__ == "__main__":
    # Import configuration
    config = import_train_configuration(config_file="training_settings.ini")
    
    # Configure GPU for optimal performance
    configure_gpu(config)

    # Get agent type to train
    agent_type = config["agent_type"]

    # List of all agent types
    all_agent_types = ["dqn", "qlearning", "a2c", "ppo", "fixed", "actuated", "webster"]

    # If the agent type is "all", train all agents in parallel
    if agent_type.lower() == "all":
        results = train_agents_parallel(all_agent_types, config)
    else:
        # Only train the specified agent
        train_agent(agent_type, config, set_train_path(config["models_path_name"]))
