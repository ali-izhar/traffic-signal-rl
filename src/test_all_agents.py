import subprocess
import configparser
import time
import sys
import os
import shutil
import tensorflow as tf
import multiprocessing as mp

from training_main import train_agents_parallel, configure_gpu


def check_gpu_status():
    """Check if GPU is available and print its status"""
    print("\nChecking GPU status:")

    physical_devices = tf.config.list_physical_devices("GPU")
    if physical_devices:
        for i, device in enumerate(physical_devices):
            details = tf.config.experimental.get_device_details(device)
            print(f"GPU {i}: {details.get('device_name', 'Unknown GPU')}")

        # Check VRAM
        try:
            import gpustat

            print("\nGPU Memory Status:")
            gpu_stats = gpustat.GPUStatCollection.new_query()
            for i, gpu in enumerate(gpu_stats.gpus):
                print(
                    f"GPU {i}: {gpu.name}, Memory: {gpu.memory_used}MB / {gpu.memory_total}MB"
                )
        except ImportError:
            print("gpustat not installed, skipping detailed GPU memory information")
    else:
        print("No GPU available, running on CPU only")


def create_optimized_config_for_rtx_4090():
    """Create a configuration optimized for RTX 4090 with 100 episodes and parallel training"""
    config = configparser.ConfigParser()

    # Simulation configuration with parallel agent support
    config["simulation"] = {
        "gui": "False",
        "total_episodes": "100",
        "max_steps": "1000",
        "n_cars_generated": "200",
        "green_duration": "10",
        "yellow_duration": "4",
        "parallel_agents": "True",
        "num_cpus": "7",  # Leave 1 CPU core free for system
    }

    # Use "all" as agent type to train all agents in parallel
    config["agent"] = {
        "type": "all",
        "num_states": "80",
        "num_actions": "4",
        "gamma": "0.75",
        "prioritized_memory": "True",
    }

    # Hardware optimization for RTX 4090
    config["hardware"] = {
        "gpu": "True",
        "gpu_memory_limit": "22000",  # 22GB for RTX 4090
        "mixed_precision": "True",
        "use_amp": "True",
        "xla_optimization": "True",
        "num_parallel_calls": "8",
    }

    # DQN configuration with optimizations
    config["dqn"] = {
        "num_layers": "4",
        "width_layers": "512",
        "batch_size": "512",
        "learning_rate": "0.001",
        "training_epochs": "1000",
        "memory_size_min": "1000",
        "memory_size_max": "100000",
        "target_update_frequency": "1000",
        "double_dqn": "True",
    }

    # Q-Learning configuration
    config["qlearning"] = {
        "learning_rate": "0.1",
        "initial_value": "0.0",
    }

    # A2C configuration with optimizations
    config["a2c"] = {
        "actor_lr": "0.001",
        "critic_lr": "0.002",
        "shared_layers": "2",
        "shared_width": "512",
        "actor_layers": "2",
        "actor_width": "256",
        "critic_layers": "2",
        "critic_width": "256",
        "entropy_coef": "0.01",
        "value_coef": "0.5",
    }

    # PPO configuration with optimizations
    config["ppo"] = {
        "actor_lr": "0.0003",
        "critic_lr": "0.001",
        "lambd": "0.95",
        "clip_ratio": "0.2",
        "epochs": "10",
        "batch_size": "512",
        "shared_layers": "3",
        "shared_width": "512",
        "actor_layers": "2",
        "actor_width": "256",
        "critic_layers": "2",
        "critic_width": "256",
        "entropy_coef": "0.01",
        "value_coef": "0.5",
    }

    # Baseline controllers configuration
    config["baseline"] = {
        "min_green": "5",
        "max_green": "60",
        "extension_time": "5",
        "saturation_flow_rate": "1900",
        "lost_time_per_phase": "2",
        "update_interval": "300",
    }

    # Directory configuration
    config["dir"] = {
        "models_path_name": "models",
        "sumocfg_file_name": "sumo_config.sumocfg",
        "output_dir": "results",
    }

    # Write the optimized configuration file
    with open("training_settings.ini", "w") as configfile:
        config.write(configfile)

    print("Created optimized configuration for RTX 4090 with parallel training")


def install_requirements():
    """Install required packages for GPU-accelerated training"""
    required_packages = [
        "tensorflow==2.15.0",
        "numpy>=1.22.0",
        "matplotlib>=3.5.0",
        "gpustat",
    ]

    print("Installing required packages...")
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                *required_packages,
            ],
            check=True,
        )
        print("Successfully installed required packages")
    except subprocess.CalledProcessError as e:
        print(f"Error installing packages: {e}")


def main():
    """Main function to run all agents in parallel on RTX 4090"""

    # Check if all agents should be trained
    if len(sys.argv) > 1 and sys.argv[1] == "--check-gpu":
        # Just check GPU status
        install_requirements()
        check_gpu_status()
        return

    print("Traffic Signal RL - Multi-Agent Training on RTX 4090")
    print("=" * 60)

    # Install required packages
    install_requirements()

    # Check GPU status
    check_gpu_status()

    # Create optimized configuration
    create_optimized_config_for_rtx_4090()

    # Import the optimized configuration
    config = configparser.ConfigParser()
    config.read("training_settings.ini")

    # Create a dictionary from the config
    config_dict = {}
    for section in config.sections():
        config_dict[section] = {}
        for key, val in config.items(section):
            config_dict[section][key] = val

    # Flatten the config for the training function
    flat_config = {}
    for section, items in config_dict.items():
        for key, value in items.items():
            flat_config[key] = value
    
    # Convert string values to appropriate types
    # Integers
    for key in ["num_states", "num_actions", "max_steps", "n_cars_generated", 
                "num_layers", "width_layers", "batch_size", 
                "memory_size_min", "memory_size_max", "target_update_frequency",
                "green_duration", "yellow_duration", "total_episodes",
                "shared_layers", "shared_width", "actor_layers", "actor_width",
                "critic_layers", "critic_width", "epochs",
                "min_green", "max_green", "extension_time", 
                "saturation_flow_rate", "lost_time_per_phase", "update_interval"]:
        if key in flat_config:
            flat_config[key] = int(flat_config[key])
    
    # Floats
    for key in ["learning_rate", "gamma", "initial_value", "actor_lr", "critic_lr",
                "entropy_coef", "value_coef", "lambd", "clip_ratio",
                "qlearning_learning_rate"]:
        if key in flat_config:
            flat_config[key] = float(flat_config[key])

    # Configure GPU for optimal performance with the config
    configure_gpu(flat_config)

    # Check if specific agent types are requested
    agent_types = ["dqn", "qlearning", "a2c", "ppo", "fixed", "actuated", "webster"]
    if len(sys.argv) > 1:
        requested_agents = sys.argv[1].split(",")
        agent_types = [agent for agent in requested_agents if agent in agent_types]
        print(f"Training requested agents: {', '.join(agent_types)}")
    else:
        print(f"Training all agent types: {', '.join(agent_types)}")

    # Set the number of processes based on available CPU cores and agent count
    max_processes = min(len(agent_types), mp.cpu_count() - 1)
    flat_config["num_cpus"] = str(max_processes)

    print(f"\nStarting parallel training with {max_processes} processes")
    print(f"Training for {flat_config['total_episodes']} episodes per agent")
    print("=" * 60)

    # Run the parallel training for all agents
    start_time = time.time()
    results = train_agents_parallel(agent_types, flat_config)
    end_time = time.time()

    print("\n" + "=" * 60)
    print(f"Training completed in {(end_time - start_time) / 60:.2f} minutes")
    print("Results Summary:")

    # Print summary of results
    for agent_type, (rewards, _) in results.items():
        if rewards:
            min_reward = min(rewards)
            max_reward = max(rewards)
            mean_reward = sum(rewards) / len(rewards)
            final_reward = rewards[-1]
            print(
                f"{agent_type:10}: Min: {min_reward:.2f}, Max: {max_reward:.2f}, Mean: {mean_reward:.2f}, Final: {final_reward:.2f}"
            )

    print("\nComparative plots saved in models directory")
    print("=" * 60)


if __name__ == "__main__":
    main()
