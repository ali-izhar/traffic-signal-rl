"""Traffic Signal Control Agent Testing Module"""

from typing import Dict, List, Optional, Any

import os
import sys
import time
import argparse

import configparser
import numpy as np
import tensorflow as tf

# --- SUMO Configuration ---
# If 'SUMO_HOME' is not set, uncomment and set the path to your SUMO installation
# os.environ['SUMO_HOME'] = '/path/to/sumo'

try:
    if "SUMO_HOME" in os.environ:
        tools = os.path.join(os.environ["SUMO_HOME"], "tools")
        sys.path.append(tools)
    else:
        sys.exit("Please declare environment variable 'SUMO_HOME'")
    import traci
except ImportError:
    sys.exit(
        "TraCI module not found. Make sure SUMO_HOME is set and TraCI is in the Python path."
    )

# --- Global Variables (Defaults, can be overridden by settings file) ---
TLS_ID = "TL"  # Traffic light ID, must match the one in the .net.xml file

# Phase definitions (map agent actions to SUMO phase indices)
# Action to traffic light phase mapping
# Mapping of agent actions (0-3) to SUMO green phase indices:
ACTION_TO_SUMO_GREEN_PHASE = {
    0: 0,  # N-S green (Through/Right)
    1: 2,  # N-S left green
    2: 4,  # E-W green (Through/Right)
    3: 6,  # E-W left green
}
# Yellow phases are typically the SUMO phase index + 1
ACTION_TO_SUMO_YELLOW_PHASE_OFFSET = 1

# Edges for computing traffic metrics
INCOMING_LANES_FOR_METRICS = ["E2TL", "N2TL", "W2TL", "S2TL"]


def create_dqn_model(
    num_states: int,
    num_actions: int,
    num_additional_hidden_layers_from_config: int,
    width_layers: int,
) -> tf.keras.Model:
    """Create a DQN model with the specified architecture.

    Args:
        num_states: Input dimension (state space size)
        num_actions: Output dimension (action space size)
        num_additional_hidden_layers_from_config: Number of hidden layers to add
        width_layers: Width of each hidden layer (neurons)

    Returns:
        Compiled TensorFlow model for DQN agent
    """
    inputs = tf.keras.layers.Input(shape=(num_states,), name="input_layer")

    # First hidden layer
    x = tf.keras.layers.Dense(
        width_layers, activation="relu", kernel_initializer="he_uniform"
    )(inputs)

    # Add additional hidden layers
    for _ in range(num_additional_hidden_layers_from_config):
        x = tf.keras.layers.Dense(
            width_layers, activation="relu", kernel_initializer="he_uniform"
        )(x)

    # Output layer
    outputs = tf.keras.layers.Dense(
        num_actions, activation="linear", kernel_initializer="glorot_uniform"
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def load_settings(
    settings_file: str = "testing_settings.ini",
) -> Optional[Dict[str, Any]]:
    """Load simulation and agent settings from an INI file.

    Args:
        settings_file: Path to the settings INI file

    Returns:
        Dictionary containing simulation settings, or None if loading failed
    """
    config = configparser.ConfigParser()
    if not os.path.exists(settings_file):
        print(f"Error: Settings file {settings_file} not found.")
        return None

    config.read(settings_file)
    settings = {}

    try:
        # Load simulation settings
        settings["gui"] = config.getboolean("simulation", "gui")
        settings["max_steps"] = config.getint("simulation", "max_steps")
        settings["n_cars_generated"] = config.getint("simulation", "n_cars_generated")
        settings["episode_seed"] = (
            config.getint("simulation", "episode_seed")
            if config.has_option("simulation", "episode_seed")
            else None
        )
        settings["green_duration"] = config.getint("simulation", "green_duration")
        settings["yellow_duration"] = config.getint("simulation", "yellow_duration")

        # Load agent settings
        settings["num_states"] = config.getint("agent", "num_states")
        settings["num_actions"] = config.getint("agent", "num_actions")

        # Load model architecture parameters
        settings["model_num_layers"] = (
            config.getint("model", "num_layers")
            if config.has_section("model") and config.has_option("model", "num_layers")
            else 4  # Default from paper
        )
        settings["model_width_layers"] = (
            config.getint("model", "width_layers")
            if config.has_section("model")
            and config.has_option("model", "width_layers")
            else 400  # Default from paper
        )

        # Load directory settings
        settings["models_path_name"] = config.get("dir", "models_path_name")
        settings["sumocfg_file_name"] = config.get("dir", "sumocfg_file_name")
        settings["model_to_test"] = config.get("dir", "model_to_test")

    except Exception as e:
        print(f"Error parsing settings file {settings_file}: {e}")
        return None

    return settings


def get_sumo_cmd(settings: Dict[str, Any], sumocfg_path: str) -> List[str]:
    """Construct the SUMO command with appropriate parameters.

    Args:
        settings: Dictionary containing simulation settings
        sumocfg_path: Path to the SUMO configuration file

    Returns:
        List containing SUMO command-line arguments
    """
    # Choose binary based on GUI setting
    sumo_binary = "sumo-gui" if settings["gui"] else "sumo"

    # Construct command
    cmd = [sumo_binary, "-c", sumocfg_path]
    cmd += ["--no-step-log", "true"]
    cmd += ["--waiting-time-memory", str(settings["max_steps"])]  # Store waiting times
    cmd += ["--time-to-teleport", "-1"]  # Disable teleporting vehicles

    # Add random seed if specified
    if settings["episode_seed"] is not None:
        cmd += ["--seed", str(settings["episode_seed"])]

    return cmd


def get_state_from_sumo(num_states: int) -> np.ndarray:
    """Retrieve the current state of the intersection from SUMO.

    The state is represented as a binary occupancy grid of the incoming lanes,
    divided into cells of increasing size with distance from the intersection.

    Mathematical formulation:
    - Each lane is divided into 10 cells of varying length
    - Lane groups are mapped to state indices as follows:
      * West Straight/Right: 0-9
      * West Left: 10-19
      * North Straight/Right: 20-29
      * North Left: 30-39
      * East Straight/Right: 40-49
      * East Left: 50-59
      * South Straight/Right: 60-69
      * South Left: 70-79

    Args:
        num_states: Total number of state features (80 for the standard implementation)

    Returns:
        NumPy array of binary state features, shape (1, num_states)
    """
    state = np.zeros(num_states)
    car_list = traci.vehicle.getIDList()

    for car_id in car_list:
        try:
            lane_pos = traci.vehicle.getLanePosition(car_id)
            lane_id = traci.vehicle.getLaneID(car_id)

            # Inversion of lane position: 0 = close to traffic light, 750 = far from intersection
            lane_pos = 750 - lane_pos

            # Map distance to cell index (0-9)
            if lane_pos < 0:
                lane_cell = -1  # Car is not on the lane segment
            elif lane_pos < 7:
                lane_cell = 0
            elif lane_pos < 14:
                lane_cell = 1
            elif lane_pos < 21:
                lane_cell = 2
            elif lane_pos < 28:
                lane_cell = 3
            elif lane_pos < 40:
                lane_cell = 4
            elif lane_pos < 60:
                lane_cell = 5
            elif lane_pos < 100:
                lane_cell = 6
            elif lane_pos < 160:
                lane_cell = 7
            elif lane_pos < 400:
                lane_cell = 8
            elif lane_pos <= 750:
                lane_cell = 9
            else:
                lane_cell = -1  # Car is beyond detection range

            if lane_cell == -1:  # Skip invalid cell positions
                continue

            # Determine lane group and calculate state index
            lane_group = -1
            if lane_id.startswith("W2TL"):
                if lane_id == "W2TL_3":
                    lane_group = 1  # West Left (cells 10-19)
                else:
                    lane_group = 0  # West Straight/Right (cells 0-9)
            elif lane_id.startswith("N2TL"):
                if lane_id == "N2TL_3":
                    lane_group = 3  # North Left (cells 30-39)
                else:
                    lane_group = 2  # North Straight/Right (cells 20-29)
            elif lane_id.startswith("E2TL"):
                if lane_id == "E2TL_3":
                    lane_group = 5  # East Left (cells 50-59)
                else:
                    lane_group = 4  # East Straight/Right (cells 40-49)
            elif lane_id.startswith("S2TL"):
                if lane_id == "S2TL_3":
                    lane_group = 7  # South Left (cells 70-79)
                else:
                    lane_group = 6  # South Straight/Right (cells 60-69)

            if lane_group != -1:
                # Calculate state index: lane_group*10 + lane_cell
                car_position = lane_group * 10 + lane_cell

                if 0 <= car_position < num_states:
                    state[car_position] = 1  # Mark cell as occupied

        except traci.TraCIException:
            # Vehicle might have left the simulation between getIDList() and property access
            continue
        except Exception:
            continue

    # Reshape for neural network input (add batch dimension)
    return state.reshape(1, num_states)


def apply_action_to_sumo(action, previous_action, green_duration, yellow_duration):
    """Applies the selected action to the traffic light in SUMO.
    Manages yellow phase transitions."""
    if action == previous_action:  # No change in phase
        traci.simulationStep()  # Just step the simulation
        return

    # Apply yellow phase for the previous action if it was a green phase
    if previous_action is not None:
        previous_green_sumo_phase = ACTION_TO_SUMO_GREEN_PHASE.get(previous_action)
        if previous_green_sumo_phase is not None:
            yellow_sumo_phase = (
                previous_green_sumo_phase + ACTION_TO_SUMO_YELLOW_PHASE_OFFSET
            )
            try:
                # print(f"  Applying YELLOW phase {yellow_sumo_phase} for {yellow_duration}s (from action {previous_action})")
                traci.trafficlight.setPhase(TLS_ID, yellow_sumo_phase)
                for _ in range(yellow_duration):  # Run SUMO for yellow duration
                    traci.simulationStep()
            except traci.TraCIException as e:
                print(
                    f"Error setting yellow phase {yellow_sumo_phase} for {TLS_ID}: {e}"
                )
                # Potentially fall back or log error

    # Apply new green phase
    current_green_sumo_phase = ACTION_TO_SUMO_GREEN_PHASE.get(action)
    if current_green_sumo_phase is not None:
        try:
            # print(f"  Applying GREEN phase {current_green_sumo_phase} for {green_duration}s (for action {action})")
            traci.trafficlight.setPhase(TLS_ID, current_green_sumo_phase)
            # The green phase duration is maintained until the next agent decision
            # The main loop will call traci.simulationStep() repeatedly.
            # The paper states "green phase is activated for a fixed duration TG ...
            # before the next state is observed and a new action can be taken."
            # This means the agent makes a decision every TG + TY seconds.
            # The simulation steps for TG are handled outside this function by the main loop's step logic.
        except traci.TraCIException as e:
            print(
                f"Error setting green phase {current_green_sumo_phase} for {TLS_ID}: {e}"
            )
    else:
        print(
            f"Warning: Action {action} has no corresponding SUMO green phase defined."
        )


def get_current_total_waiting_time():
    total_wait_time = 0
    car_list = traci.vehicle.getIDList()
    for car_id in car_list:
        road_id = traci.vehicle.getRoadID(car_id)
        if road_id in INCOMING_LANES_FOR_METRICS:
            total_wait_time += traci.vehicle.getAccumulatedWaitingTime(car_id)
    return total_wait_time


def get_current_total_queue_length():
    queue_length = 0
    for edge_id in INCOMING_LANES_FOR_METRICS:
        queue_length += traci.edge.getLastStepHaltingNumber(edge_id)
    return queue_length


def save_metric_data(data, filename, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w") as f:
        for item in data:
            f.write(f"{item}\n")
    print(f"Saved metric data to: {filepath}")


def run_test_episode(
    settings_file="testing_settings.ini", run_mode="dqn", specified_sumocfg=None
):
    """Runs a single test episode based on the specified mode and optional specific SUMO config."""
    settings = load_settings(settings_file)
    if not settings:
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Determine the SUMO config file to use
    if specified_sumocfg:
        sumocfg_file_path = specified_sumocfg
        print(f"Using specified SUMO config: {sumocfg_file_path}")
        # Infer mode from specified config name if mode wasn't explicitly set (though cmd line arg is primary)
        if (
            "actuated" in os.path.basename(sumocfg_file_path).lower()
            and run_mode != "actuated"
        ):
            print(
                "Warning: Config file suggests actuated, but mode is not set to actuated. Running as specified mode."
            )
        elif (
            "actuated" not in os.path.basename(sumocfg_file_path).lower()
            and run_mode == "actuated"
        ):
            print(
                "Warning: Mode is actuated, but config file doesn't seem to be actuated. Proceeding."
            )
    else:
        # Original logic based on run_mode if no specific config is given
        if run_mode == "actuated":
            sumocfg_filename = "sumo_config_actuated_rebuilt.sumocfg"
            print("Running in ACTUATED mode (using netconvert-rebuilt logic).")
        elif run_mode == "fixed":
            sumocfg_filename = "sumo_config.sumocfg"
            print("Running in FIXED-TIME mode (using SUMO static logic).")
        else:  # Default to DQN mode
            run_mode = "dqn"
            sumocfg_filename = "sumo_config.sumocfg"
            print("Running in DQN AGENT mode.")
        sumocfg_file_path = os.path.join(script_dir, "intersection", sumocfg_filename)

    if not os.path.exists(sumocfg_file_path):
        # If rebuilt file doesn't exist, maybe netconvert failed? Fallback to old actuated config.
        if (
            run_mode == "actuated"
            and sumocfg_filename == "sumo_config_actuated_rebuilt.sumocfg"
        ):
            print(
                f"Warning: {sumocfg_filename} not found. Falling back to sumo_config_actuated.sumocfg"
            )
            sumocfg_filename = "sumo_config_actuated.sumocfg"
            sumocfg_file_path = os.path.join(
                script_dir, "intersection", sumocfg_filename
            )
            if not os.path.exists(sumocfg_file_path):
                return print(
                    f"Error: Fallback SUMO config file also not found: {sumocfg_file_path}"
                )
        else:
            return print(f"Error: SUMO config file not found: {sumocfg_file_path}")

    # --- Determine Output Directory based on Scenario ---
    # Use the basename of the sumocfg file (without extension) to create a unique test subfolder
    scenario_name = os.path.splitext(os.path.basename(sumocfg_file_path))[0]
    test_run_subfolder = f"test_{scenario_name}"  # e.g., test_sumo_config_tidal_fixed

    model_id_str = settings["model_to_test"]
    model_folder_name_for_results = f"model_{model_id_str}"
    output_data_dir = os.path.join(
        script_dir,
        settings["models_path_name"],
        model_folder_name_for_results,
        test_run_subfolder,  # Save results in scenario-specific folder
    )

    model = None
    if run_mode == "dqn":
        model_name = f"model_{settings['model_to_test']}"
        model_path = os.path.join(
            script_dir, settings["models_path_name"], model_name, "trained_model.h5"
        )
        if not os.path.exists(model_path):
            return print(f"Error: Trained model not found at {model_path}")
        print(f"Loading model: {model_path}")
        try:
            model = create_dqn_model(
                settings["num_states"],
                settings["num_actions"],
                settings["model_num_layers"],
                settings["model_width_layers"],
            )
            model.load_weights(model_path)
            print(
                "Successfully created model with correct architecture and loaded weights."
            )
        except Exception as e:
            print(f"Error creating model/loading weights: {e}. Trying load_model...")
            try:
                model = tf.keras.models.load_model(model_path, compile=False)
            except Exception as e2:
                return print(f"Error with load_model: {e2}")

    sumo_cmd = get_sumo_cmd(settings, sumocfg_file_path)
    print(f"Starting SUMO: {' '.join(sumo_cmd)}")

    all_step_wait_times = []
    all_step_queue_lengths = []

    try:
        traci.start(sumo_cmd)
        print("SUMO started with TraCI.")
        current_step = 0
        previous_action = None

        while current_step < settings["max_steps"]:
            if traci.simulation.getMinExpectedNumber() <= 0:
                print("All vehicles processed.")
                break

            # --- Simulation Step and Control Logic ---
            if run_mode == "dqn":
                # --- DQN Agent Logic ---
                current_state = get_state_from_sumo(settings["num_states"])
                q_values = model.predict(current_state, verbose=0)
                action = np.argmax(q_values[0])
                if (
                    current_step
                    % (settings["green_duration"] + settings["yellow_duration"])
                    < 5
                    or current_step < 10
                ):
                    print(
                        f"Step: {current_step}, SumState: {np.sum(current_state):.0f}, Qs: {q_values[0]}, Act: {action}"
                    )

                # Apply yellow if action changes
                if action != previous_action and previous_action is not None:
                    prev_green_sumo_phase = ACTION_TO_SUMO_GREEN_PHASE.get(
                        previous_action
                    )
                    if prev_green_sumo_phase is not None:
                        yellow_sumo_phase = (
                            prev_green_sumo_phase + ACTION_TO_SUMO_YELLOW_PHASE_OFFSET
                        )
                        traci.trafficlight.setPhase(TLS_ID, yellow_sumo_phase)
                        # Simulate yellow duration step-by-step
                        for _ in range(settings["yellow_duration"]):
                            if not (
                                current_step < settings["max_steps"]
                                and traci.simulation.getMinExpectedNumber() > 0
                            ):
                                break
                            traci.simulationStep()
                            # Collect metrics after each step
                            all_step_wait_times.append(get_current_total_waiting_time())
                            all_step_queue_lengths.append(
                                get_current_total_queue_length()
                            )
                            current_step += 1
                    if not (
                        current_step < settings["max_steps"]
                        and traci.simulation.getMinExpectedNumber() > 0
                    ):
                        break  # Break outer simulation loop if conditions met

                # Apply green phase
                current_green_sumo_phase = ACTION_TO_SUMO_GREEN_PHASE.get(action)
                if current_green_sumo_phase is not None:
                    traci.trafficlight.setPhase(TLS_ID, current_green_sumo_phase)
                previous_action = action

                # Simulate green duration step-by-step
                for _ in range(settings["green_duration"]):
                    if not (
                        current_step < settings["max_steps"]
                        and traci.simulation.getMinExpectedNumber() > 0
                    ):
                        break
                    traci.simulationStep()
                    # Collect metrics after each step
                    all_step_wait_times.append(get_current_total_waiting_time())
                    all_step_queue_lengths.append(get_current_total_queue_length())
                    current_step += 1
                if not (
                    current_step < settings["max_steps"]
                    and traci.simulation.getMinExpectedNumber() > 0
                ):
                    break  # Break outer simulation loop

            else:  # Fixed-Time or Actuated mode (SUMO handles TLS logic)
                # --- Baseline Logic ---
                traci.simulationStep()
                # Collect metrics after step
                all_step_wait_times.append(get_current_total_waiting_time())
                all_step_queue_lengths.append(get_current_total_queue_length())
                current_step += 1
                if current_step % 500 == 0:  # Print progress less often for baselines
                    print(
                        f"{run_mode.upper()} Step: {current_step}/{settings['max_steps']}"
                    )

            # Optional delay for GUI viewing
            if settings["gui"]:
                time.sleep(0.01)

    except traci.TraCIException as e:
        print(f"TraCI error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        try:
            if "traci" in sys.modules and traci.isLoaded():
                print("Closing TraCI.")
                traci.close()
            elif "traci" in sys.modules:
                print("Attempting close TraCI (sim not loaded/closed).")
                traci.close()
        except AttributeError:
            print("traci.isLoaded() not found, closing directly.")
            try:
                if "traci" in sys.modules:
                    traci.close()
            except Exception as ce:
                print(f"Simplified close exception: {ce}")
        except Exception as ec:
            print(f"TraCI close error: {ec}")
        print("Simulation finished.")

        # Save collected data (using the scenario-specific output_data_dir)
        if all_step_wait_times:
            prefix = f"{run_mode}_"  # Use mode for prefix
            save_metric_data(
                all_step_wait_times, f"{prefix}step_wait_times.txt", output_data_dir
            )
            save_metric_data(
                all_step_queue_lengths,
                f"{prefix}step_queue_lengths.txt",
                output_data_dir,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run SUMO with DQN agent, fixed-time, or actuated control."
    )
    parser.add_argument(
        "--mode",
        default="dqn",
        choices=["dqn", "fixed", "actuated"],
        help="Control mode to run (dqn, fixed, actuated). Default: dqn",
    )
    parser.add_argument(
        "--sumocfg",
        default=None,
        help="Path to a specific SUMO config file (.sumocfg) to use, overriding the mode default.",
    )
    script_dir_for_main = os.path.dirname(os.path.abspath(__file__))
    default_settings_in_script_dir = os.path.join(
        script_dir_for_main, "testing_settings.ini"
    )
    parser.add_argument(
        "--settings",
        default=default_settings_in_script_dir,
        help=f"Path to the settings INI file. Default: {default_settings_in_script_dir}",
    )
    args = parser.parse_args()

    settings_file_to_run = args.settings
    if not os.path.exists(settings_file_to_run):
        print(f"Error: Specified settings file not found: {settings_file_to_run}")
        alt_path_from_ws = os.path.join("DQN/TLCS", os.path.basename(args.settings))
        if os.path.exists(alt_path_from_ws):
            print(f"Found settings at alt path: {alt_path_from_ws}")
            settings_file_to_run = alt_path_from_ws
        else:
            potential_default_from_ws = "DQN/TLCS/testing_settings.ini"
            if args.settings == default_settings_in_script_dir and os.path.exists(
                potential_default_from_ws
            ):
                settings_file_to_run = potential_default_from_ws
                print(f"Using default from workspace: {potential_default_from_ws}")
            elif not os.path.exists(settings_file_to_run):
                print(
                    f"Still cannot find settings. Tried: {settings_file_to_run}, {alt_path_from_ws}"
                )
                sys.exit(1)

    # Determine the absolute path for the sumocfg if specified relative
    sumocfg_to_run = args.sumocfg
    if sumocfg_to_run and not os.path.isabs(sumocfg_to_run):
        # Assume relative to CWD or potentially script dir? For simplicity, assume relative to CWD.
        # If you always run from workspace root, relative like 'DQN/TLCS/intersection/...' is fine.
        sumocfg_to_run = os.path.abspath(sumocfg_to_run)

    print(f"Using settings file: {settings_file_to_run}")
    print(f"Running in {args.mode.upper()} mode.")
    if sumocfg_to_run:
        print(f"Using specific SUMO config: {sumocfg_to_run}")

    run_test_episode(
        settings_file_to_run, run_mode=args.mode, specified_sumocfg=sumocfg_to_run
    )
