from __future__ import absolute_import
from __future__ import print_function

import os
from shutil import copyfile

from testing_simulation import Simulation
from generator import TrafficGenerator
from visualization import Visualization
from utils import import_test_configuration, set_sumo, set_test_path

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


if __name__ == "__main__":

    config = import_test_configuration(config_file="testing_settings.ini")
    sumo_cmd = set_sumo(config["gui"], config["sumocfg_file_name"], config["max_steps"])
    model_path, plot_path = set_test_path(
        config["models_path_name"], config["model_to_test"]
    )

    # Create agent based on configuration
    agent_type = config["agent_type"]

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
    if agent_type in ["dqn", "qlearning", "a2c", "ppo"]:
        agent.load(model_path)

    TrafficGen = TrafficGenerator(config["max_steps"], config["n_cars_generated"])

    Visualization = Visualization(plot_path, dpi=96)

    Simulation = Simulation(
        agent,
        TrafficGen,
        sumo_cmd,
        config["max_steps"],
        config["green_duration"],
        config["yellow_duration"],
        config["num_states"],
        config["num_actions"],
    )

    print("\n----- Test episode")
    simulation_time = Simulation.run(config["episode_seed"])  # run the simulation
    print("Simulation time:", simulation_time, "s")

    print("----- Testing info saved at:", plot_path)

    copyfile(
        src="testing_settings.ini", dst=os.path.join(plot_path, "testing_settings.ini")
    )

    Visualization.save_data_and_plot(
        data=Simulation.reward_episode,
        filename="reward",
        xlabel="Action step",
        ylabel="Reward",
    )
    Visualization.save_data_and_plot(
        data=Simulation.queue_length_episode,
        filename="queue",
        xlabel="Step",
        ylabel="Queue lenght (vehicles)",
    )
