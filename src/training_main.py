from __future__ import absolute_import
from __future__ import print_function

import os
import datetime
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


if __name__ == "__main__":

    config = import_train_configuration(config_file="training_settings.ini")
    sumo_cmd = set_sumo(config["gui"], config["sumocfg_file_name"], config["max_steps"])
    path = set_train_path(config["models_path_name"])

    # Create agent based on configuration
    agent_type = config["agent_type"]

    if agent_type == "dqn":
        agent = DQNAgent(
            input_dim=config["num_states"],
            output_dim=config["num_actions"],
            num_layers=config["num_layers"],
            width=config["width_layers"],
            batch_size=config["batch_size"],
            learning_rate=config["learning_rate"],
            gamma=config["gamma"],
            memory_size_max=config["memory_size_max"],
            memory_size_min=config["memory_size_min"],
        )
    elif agent_type == "qlearning":
        agent = QLearningAgent(
            input_dim=config["num_states"],
            output_dim=config["num_actions"],
            learning_rate=config["qlearning_learning_rate"],
            gamma=config["gamma"],
            initial_value=config["initial_value"],
        )
    elif agent_type == "a2c":
        agent = A2CAgent(
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
        )
    elif agent_type == "ppo":
        agent = PPOAgent(
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
            min_green=config["min_green"],
            max_green=config["max_green"],
            extension_time=config["extension_time"],
            yellow_time=config["yellow_duration"],
        )
    elif agent_type == "webster":
        agent = WebsterController(
            input_dim=config["num_states"],
            output_dim=config["num_actions"],
            saturation_flow_rate=config["saturation_flow_rate"],
            yellow_time=config["yellow_duration"],
            lost_time_per_phase=config["lost_time_per_phase"],
            update_interval=config["update_interval"],
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    TrafficGen = TrafficGenerator(config["max_steps"], config["n_cars_generated"])
    Visualization = Visualization(path, dpi=96)

    Simulation = Simulation(
        agent,
        TrafficGen,
        sumo_cmd,
        config["gamma"],
        config["max_steps"],
        config["green_duration"],
        config["yellow_duration"],
        config["num_states"],
        config["num_actions"],
    )

    episode = 0
    timestamp_start = datetime.datetime.now()

    while episode < config["total_episodes"]:
        print("\n----- Episode", str(episode + 1), "of", str(config["total_episodes"]))
        epsilon = 1.0 - (
            episode / config["total_episodes"]
        )  # set the epsilon for this episode according to epsilon-greedy policy
        simulation_time = Simulation.run(episode, epsilon)  # run the simulation
        print(
            "Simulation time:",
            simulation_time,
            "s - Total:",
            round(simulation_time, 1),
            "s",
        )
        episode += 1

    print("\n----- Start time:", timestamp_start)
    print("----- End time:", datetime.datetime.now())
    print("----- Session info saved at:", path)

    agent.save(path)

    copyfile(
        src="training_settings.ini", dst=os.path.join(path, "training_settings.ini")
    )

    Visualization.save_data_and_plot(
        data=Simulation.reward_store,
        filename="reward",
        xlabel="Episode",
        ylabel="Cumulative negative reward",
    )
    Visualization.save_data_and_plot(
        data=Simulation.cumulative_wait_store,
        filename="delay",
        xlabel="Episode",
        ylabel="Cumulative delay (s)",
    )
    Visualization.save_data_and_plot(
        data=Simulation.avg_queue_length_store,
        filename="queue",
        xlabel="Episode",
        ylabel="Average queue length (vehicles)",
    )
