import subprocess
import configparser
import time
import sys
import os
import shutil


def restore_training_settings():
    """Create a fresh training_settings.ini file with all required sections"""
    config = configparser.ConfigParser()

    # Create all required sections
    config["simulation"] = {
        "gui": "False",
        "total_episodes": "2",
        "max_steps": "1000",
        "n_cars_generated": "200",
        "green_duration": "10",
        "yellow_duration": "4",
    }

    config["agent"] = {
        "type": "dqn",
        "num_states": "80",
        "num_actions": "4",
        "gamma": "0.75",
    }

    config["dqn"] = {
        "num_layers": "4",
        "width_layers": "400",
        "batch_size": "100",
        "learning_rate": "0.001",
        "training_epochs": "800",
        "memory_size_min": "600",
        "memory_size_max": "50000",
    }

    config["qlearning"] = {"learning_rate": "0.1", "initial_value": "0.0"}

    config["a2c"] = {
        "actor_lr": "0.001",
        "critic_lr": "0.002",
        "shared_layers": "2",
        "shared_width": "256",
        "actor_layers": "1",
        "actor_width": "128",
        "critic_layers": "1",
        "critic_width": "128",
    }

    config["ppo"] = {
        "actor_lr": "0.0003",
        "critic_lr": "0.001",
        "lambd": "0.95",
        "clip_ratio": "0.2",
        "epochs": "10",
        "batch_size": "64",
        "shared_layers": "2",
        "shared_width": "256",
        "actor_layers": "1",
        "actor_width": "128",
        "critic_layers": "1",
        "critic_width": "128",
    }

    config["baseline"] = {
        "min_green": "5",
        "max_green": "60",
        "extension_time": "5",
        "saturation_flow_rate": "1900",
        "lost_time_per_phase": "2",
        "update_interval": "300",
    }

    config["dir"] = {
        "models_path_name": "models",
        "sumocfg_file_name": "sumo_config.sumocfg",
    }

    # Write to the training_settings.ini in the current directory (src)
    with open("training_settings.ini", "w") as configfile:
        config.write(configfile)


def update_config(agent_type):
    """Update the training_settings.ini file for the specified agent"""
    # First restore the full config file structure
    restore_training_settings()

    # Update only the agent type
    config = configparser.ConfigParser()
    config.read("training_settings.ini")
    config["agent"]["type"] = agent_type

    with open("training_settings.ini", "w") as configfile:
        config.write(configfile)

    print(f"Configuration updated for agent type: {agent_type}")


def run_training():
    """Run the training script and capture output"""
    print("Starting training...")
    start_time = time.time()

    # Use the same Python interpreter that's running this script
    python_executable = sys.executable

    try:
        # Run the script with proper environment
        env = os.environ.copy()

        # Run the training script from the current directory
        result = subprocess.run(
            [python_executable, "training_main.py"],
            capture_output=True,
            text=True,
            check=True,
            env=env,
        )
        print(result.stdout)
        print("Training completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error during training: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        return False

    end_time = time.time()
    print(f"Training took {end_time - start_time:.2f} seconds")


def main():
    # Install required packages if needed
    print("Ensuring required packages are installed...")
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "numpy",
                "tensorflow",
                "matplotlib",
            ],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error installing packages: {e}")
        return

    # Restore training settings first
    restore_training_settings()

    # Print the current directory structure
    print(f"Current directory: {os.getcwd()}")
    print("Contents of current directory:")
    subprocess.run(["dir"], shell=True)

    # Start with testing just one agent type
    agent_types = ["dqn", "qlearning", "a2c", "ppo", "fixed", "actuated", "webster"]
    results = {}

    for agent_type in agent_types:
        print(f"\n{'='*50}")
        print(f"Testing agent: {agent_type}")
        print(f"{'='*50}")

        update_config(agent_type)
        success = run_training()
        results[agent_type] = "SUCCESS" if success else "FAILED"

        # Break after testing one agent to check if it's working
        if agent_type == "dqn":
            if results[agent_type] == "SUCCESS":
                print("DQN agent test succeeded. Continuing with other agents...")
            else:
                print("DQN agent test failed. Stopping further testing.")
                break

    # Print summary
    print("\n\n")
    print(f"{'='*50}")
    print("Test Results Summary")
    print(f"{'='*50}")
    for agent_type, result in results.items():
        print(f"{agent_type}: {result}")


if __name__ == "__main__":
    main()
