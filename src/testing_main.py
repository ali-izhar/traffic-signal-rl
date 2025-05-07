from __future__ import absolute_import
from __future__ import print_function

import os
import datetime
import multiprocessing as mp
import concurrent.futures
import time
import numpy as np
import tensorflow as tf
from shutil import copyfile

from testing_simulation import Simulation
from generator import TrafficGenerator
from visualization import Visualization
from utils import import_test_configuration, set_sumo, set_test_path, set_random_seed
from training_main import configure_gpu

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


def create_test_agent(agent_type, config, model_path=None):
    """Create and load a test agent of the specified type"""
    if agent_type == "dqn":
        agent = DQNAgent(
            input_dim=config["num_states"],
            output_dim=config["num_actions"],
        )
    elif agent_type == "qlearning":
        agent = QLearningAgent(
            input_dim=config["num_states"],
            output_dim=config["num_actions"],
        )
    elif agent_type == "a2c":
        agent = A2CAgent(
            input_dim=config["num_states"],
            output_dim=config["num_actions"],
        )
    elif agent_type == "ppo":
        agent = PPOAgent(
            input_dim=config["num_states"],
            output_dim=config["num_actions"],
        )
    elif agent_type == "fixed":
        agent = FixedTimingController(
            input_dim=config["num_states"],
            output_dim=config["num_actions"],
        )
    elif agent_type == "actuated":
        agent = ActuatedController(
            input_dim=config["num_states"],
            output_dim=config["num_actions"],
            yellow_time=config["yellow_duration"],
        )
    elif agent_type == "webster":
        agent = WebsterController(
            input_dim=config["num_states"],
            output_dim=config["num_actions"],
            yellow_time=config["yellow_duration"],
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    # Load the model for RL agents (not needed for baseline controllers)
    if agent_type in ["dqn", "qlearning", "a2c", "ppo"] and model_path is not None:
        agent.load(model_path)

    return agent


def test_single_agent(
    agent_type, config, model_path, plot_path, episode_seed, port_offset=0
):
    """Test a single agent and return metrics"""
    start_time = time.time()

    # Create and load agent
    agent = create_test_agent(agent_type, config, model_path)

    # Setup custom port for parallel testing
    custom_port = 8813 + port_offset
    sumo_cmd = set_sumo(
        config["gui"],
        config["sumocfg_file_name"],
        config["max_steps"],
        port=custom_port,
    )

    # Create auxiliary components
    traffic_gen = TrafficGenerator(config["max_steps"], config["n_cars_generated"])
    visualizer = Visualization(plot_path, dpi=96)

    simulator = Simulation(
        agent,
        traffic_gen,
        sumo_cmd,
        config["max_steps"],
        config["green_duration"],
        config["yellow_duration"],
        config["num_states"],
        config["num_actions"],
    )

    # Set random seed for reproducibility
    set_random_seed(episode_seed)

    # Run the test episode
    print(f"\n----- Testing {agent_type} agent (Episode seed: {episode_seed})")
    simulation_time = simulator.run(episode_seed)
    print(f"Simulation time: {simulation_time} s")

    # Save visualization
    visualizer.save_data_and_plot(
        data=simulator.reward_episode,
        filename=f"{agent_type}_reward",
        xlabel="Action step",
        ylabel="Reward",
    )
    visualizer.save_data_and_plot(
        data=simulator.queue_length_episode,
        filename=f"{agent_type}_queue",
        xlabel="Step",
        ylabel="Queue length (vehicles)",
    )

    # Copy config for record keeping
    copyfile(
        src="testing_settings.ini",
        dst=os.path.join(plot_path, f"{agent_type}_testing_settings.ini"),
    )

    # Collect metrics
    metrics = {
        "agent_type": agent_type,
        "reward": sum(simulator.reward_episode),
        "avg_queue": np.mean(simulator.queue_length_episode),
        "max_queue": np.max(simulator.queue_length_episode),
        "teleports": simulator.teleport_count,
        "avg_speed": simulator.avg_speed,
        "total_waiting_time": simulator.total_waiting_time,
        "co2_emission": simulator.total_co2_emission,
        "fuel_consumption": simulator.total_fuel_consumption,
        "simulation_time": simulation_time,
        "elapsed_time": time.time() - start_time,
    }

    return metrics


def test_all_agents(config, model_paths, plot_path, episode_seed, use_parallel=True):
    """Test all agent types in parallel and compare them"""
    # List of all agent types
    agent_types = ["dqn", "qlearning", "a2c", "ppo", "fixed", "actuated", "webster"]

    # Determine number of processes to use
    num_cpus = min(len(agent_types), mp.cpu_count() - 1)

    all_metrics = []

    if use_parallel and num_cpus > 1:
        print(
            f"Testing {len(agent_types)} agents in parallel using {num_cpus} processes"
        )

        # Create a pool of processes
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_cpus) as executor:
            futures = {}

            # Submit test jobs for each agent
            for i, agent_type in enumerate(agent_types):
                future = executor.submit(
                    test_single_agent,
                    agent_type,
                    config,
                    model_paths.get(agent_type),
                    plot_path,
                    episode_seed,
                    i,  # Use different port for each agent
                )
                futures[future] = agent_type

            # Collect results as they complete
            for future in concurrent.futures.as_completed(futures):
                agent_type = futures[future]
                try:
                    metrics = future.result()
                    all_metrics.append(metrics)
                    print(f"Agent {agent_type} testing completed")
                except Exception as e:
                    print(f"Agent {agent_type} testing failed: {e}")
    else:
        print("Testing agents sequentially")
        for i, agent_type in enumerate(agent_types):
            try:
                metrics = test_single_agent(
                    agent_type,
                    config,
                    model_paths.get(agent_type),
                    plot_path,
                    episode_seed,
                    i,
                )
                all_metrics.append(metrics)
                print(f"Agent {agent_type} testing completed")
            except Exception as e:
                print(f"Agent {agent_type} testing failed: {e}")

    # Create comparative visualizations
    create_comparative_plots(all_metrics, plot_path)

    return all_metrics


def create_comparative_plots(all_metrics, plot_path):
    """Create comparative visualizations for all tested agents"""
    visualizer = Visualization(plot_path, dpi=96)

    # Extract data for visualization
    reward_data = {}
    queue_data = {}
    speed_data = {}

    for metrics in all_metrics:
        agent_type = metrics["agent_type"]
        reward_data[agent_type] = [metrics["reward"]]
        queue_data[agent_type] = [metrics["avg_queue"]]
        speed_data[agent_type] = [metrics["avg_speed"]]

    # Save comparative statistics to file
    with open(os.path.join(plot_path, "comparative_metrics.txt"), "w") as f:
        f.write(
            "Agent,Reward,AvgQueue,MaxQueue,Teleports,AvgSpeed,TotalWaitTime,CO2Emission,FuelConsumption\n"
        )
        for metrics in sorted(all_metrics, key=lambda x: x["reward"], reverse=True):
            f.write(
                f"{metrics['agent_type']},{metrics['reward']:.2f},{metrics['avg_queue']:.2f},"
                f"{metrics['max_queue']:.2f},{metrics['teleports']},{metrics['avg_speed']:.2f},"
                f"{metrics['total_waiting_time']:.2f},{metrics['co2_emission']:.2f},{metrics['fuel_consumption']:.2f}\n"
            )

    # Create bar chart visualization for key metrics
    visualizer.save_comparative_bar_chart(
        data_dict={m["agent_type"]: m["reward"] for m in all_metrics},
        filename="comparative_reward_bar",
        ylabel="Total Reward",
        title="Agent Performance Comparison - Reward",
    )

    visualizer.save_comparative_bar_chart(
        data_dict={m["agent_type"]: m["avg_queue"] for m in all_metrics},
        filename="comparative_queue_bar",
        ylabel="Average Queue Length",
        title="Agent Performance Comparison - Queue Length",
    )

    visualizer.save_comparative_bar_chart(
        data_dict={m["agent_type"]: m["teleports"] for m in all_metrics},
        filename="comparative_teleports_bar",
        ylabel="Number of Teleported Vehicles",
        title="Agent Performance Comparison - Traffic Jams",
    )

    visualizer.save_comparative_bar_chart(
        data_dict={m["agent_type"]: m["avg_speed"] for m in all_metrics},
        filename="comparative_speed_bar",
        ylabel="Average Vehicle Speed (m/s)",
        title="Agent Performance Comparison - Traffic Flow",
    )


def find_latest_model_paths():
    """Find the most recent trained model for each agent type"""
    model_paths = {}
    agent_types = ["dqn", "qlearning", "a2c", "ppo", "fixed", "actuated", "webster"]

    base_models_dir = os.path.join(os.getcwd(), "models")
    if not os.path.exists(base_models_dir):
        return model_paths

    # Find latest model directory
    model_dirs = [d for d in os.listdir(base_models_dir) if d.startswith("model_")]
    if not model_dirs:
        return model_paths

    latest_model_dir = max(model_dirs, key=lambda x: int(x.split("_")[1]))
    latest_model_path = os.path.join(base_models_dir, latest_model_dir)

    # Check each agent type
    for agent_type in agent_types:
        agent_dir = os.path.join(latest_model_path, agent_type)
        if os.path.exists(agent_dir):
            model_paths[agent_type] = agent_dir

    return model_paths


if __name__ == "__main__":
    # Configure GPU optimizations
    configure_gpu()

    # Import test configuration
    config = import_test_configuration(config_file="testing_settings.ini")

    # Determine if we're testing a specific agent or all agents
    agent_type = config["agent_type"]

    if agent_type.lower() == "all":
        # Test all agents and compare
        print("Testing all agent types")

        # Find latest model for each agent type
        model_paths = find_latest_model_paths()
        if not model_paths:
            print("No trained models found. Make sure to train models first.")
            exit(1)

        # Create test plot path
        _, plot_path = set_test_path(config["models_path_name"], 0)
        test_all_agents(
            config, model_paths, plot_path, config["episode_seed"], use_parallel=True
        )

        print(f"----- Comparative testing results saved at: {plot_path}")
    else:
        # Test a single agent
        model_path, plot_path = set_test_path(
            config["models_path_name"], config["model_to_test"]
        )

        # Create agent and test
        metrics = test_single_agent(
            agent_type, config, model_path, plot_path, config["episode_seed"]
        )

        print(f"----- Testing info saved at: {plot_path}")
        print(
            f"Results: Reward={metrics['reward']:.2f}, Avg Queue={metrics['avg_queue']:.2f}, "
            f"Teleports={metrics['teleports']}, Avg Speed={metrics['avg_speed']:.2f}"
        )
