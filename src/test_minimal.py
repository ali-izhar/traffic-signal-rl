import os
import configparser
import time
import tensorflow as tf

from training_main import configure_gpu, create_agent
from training_simulation import Simulation
from generator import TrafficGenerator
from visualization import Visualization
from utils import import_train_configuration, set_sumo, set_train_path


def create_minimal_config():
    """Create a minimal configuration for testing that works on any hardware"""
    config = configparser.ConfigParser()

    # Simulation configuration with minimal settings
    config["simulation"] = {
        "gui": "False",
        "total_episodes": "2",
        "max_steps": "500",
        "n_cars_generated": "100",
        "green_duration": "10",
        "yellow_duration": "4",
        "parallel_agents": "False",
    }

    # Use "dqn" as agent type
    config["agent"] = {
        "type": "dqn",
        "num_states": "80",
        "num_actions": "4",
        "gamma": "0.75",
        "prioritized_memory": "False",
    }

    # Hardware settings with safe defaults
    config["hardware"] = {
        "gpu": "False",
        "gpu_memory_limit": "0",  # No limit
        "mixed_precision": "False",
        "use_amp": "False",
        "xla_optimization": "False",
    }

    # DQN configuration with minimal requirements
    config["dqn"] = {
        "num_layers": "2",
        "width_layers": "64",
        "batch_size": "32",
        "learning_rate": "0.001",
        "training_epochs": "100",
        "memory_size_min": "200",
        "memory_size_max": "10000",
        "target_update_frequency": "500",
        "double_dqn": "False",
    }

    # Directory configuration
    config["dir"] = {
        "models_path_name": "models_test",
        "sumocfg_file_name": "sumo_config.sumocfg",
        "output_dir": "results_test",
    }

    # Write the minimal configuration file
    with open("testing_settings.ini", "w") as configfile:
        config.write(configfile)

    print("Created minimal test configuration")

    # Convert to flat dictionary for use in training with proper types
    flat_config = {}
    
    # Copy the agent type
    flat_config["agent_type"] = config["agent"]["type"]
    
    # Integers
    int_keys = [
        "num_states", "num_actions", "max_steps", "n_cars_generated",
        "num_layers", "width_layers", "batch_size", "memory_size_min",
        "memory_size_max", "target_update_frequency", "green_duration",
        "yellow_duration", "total_episodes"
    ]
    
    # Floats
    float_keys = [
        "learning_rate", "gamma", "initial_value"
    ]
    
    # Booleans
    bool_keys = [
        "gui", "prioritized_memory", "double_dqn", "parallel_agents"
    ]
    
    # String keys that should remain strings
    str_keys = [
        "models_path_name", "sumocfg_file_name", "output_dir"
    ]
    
    # Process each section
    for section in config.sections():
        for key, value in config[section].items():
            # Handle different data types
            if key in int_keys:
                flat_config[key] = int(value)
            elif key in float_keys:
                flat_config[key] = float(value)
            elif key in bool_keys:
                flat_config[key] = value.lower() == "true"
            elif key in str_keys or key == "type":
                flat_config[key] = value
            else:
                # Default to string for unknown keys
                flat_config[key] = value
    
    # Set up hardware section explicitly
    flat_config["hardware"] = {}
    for key, value in config["hardware"].items():
        if key in ["gpu", "mixed_precision", "use_amp", "xla_optimization"]:
            flat_config["hardware"][key] = value.lower() == "true"
        elif key == "gpu_memory_limit":
            flat_config["hardware"][key] = int(value)
        else:
            flat_config["hardware"][key] = value
    
    return flat_config


def test_agent(agent_type="dqn"):
    """Run a minimal test with the specified agent"""
    print(f"\n=== Running minimal test with {agent_type} agent ===")
    
    # Create minimal config
    config = create_minimal_config()
    
    # Check for GPU
    print("\nChecking hardware:")
    physical_devices = tf.config.list_physical_devices("GPU")
    if physical_devices:
        print(f"Found {len(physical_devices)} GPU(s)")
        for device in physical_devices:
            details = tf.config.experimental.get_device_details(device)
            print(f"GPU: {details.get('device_name', 'Unknown GPU')}")
    else:
        print("No GPU found, running on CPU")
    
    # Configure GPU/CPU
    configure_gpu(config)
    
    # Create model directory
    base_path = set_train_path(config["models_path_name"])
    agent_path = os.path.join(base_path, agent_type)
    os.makedirs(agent_path, exist_ok=True)
    
    # Create agent
    agent = create_agent(agent_type, config)
    
    # Create SUMO command
    sumo_cmd = set_sumo(
        config["gui"],
        config["sumocfg_file_name"],
        config["max_steps"]
    )
    
    # Create simulation components
    traffic_gen = TrafficGenerator(
        config["max_steps"], 
        config["n_cars_generated"]
    )
    visualizer = Visualization(agent_path, dpi=96)
    
    # Create simulation
    simulator = Simulation(
        agent,
        traffic_gen,
        sumo_cmd,
        config["gamma"],
        config["max_steps"],
        config["green_duration"],
        config["yellow_duration"],
        config["num_states"],
        config["num_actions"],
    )
    
    # Run for just 2 episodes
    total_episodes = 2
    start_time = time.time()
    
    for episode in range(total_episodes):
        print(f"\n--- Episode {episode + 1}/{total_episodes} ---")
        epsilon = 0.2  # Fixed exploration rate for testing
        simulation_time = simulator.run(episode, epsilon)
        
        print(f"  Reward: {simulator.reward_store[-1]:.2f}")
        print(f"  Time: {simulation_time:.2f}s")
    
    # Print summary
    total_time = time.time() - start_time
    print(f"\n=== Test completed in {total_time:.2f} seconds ===")
    print(f"Average episode time: {total_time/total_episodes:.2f} seconds")
    print(f"Average reward: {sum(simulator.reward_store)/len(simulator.reward_store):.2f}")
    
    # Save agent
    agent.save(agent_path)
    print(f"Agent saved to {agent_path}")
    
    return True


if __name__ == "__main__":
    test_agent("dqn") 