#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Implement a wrapper around Gymnasium environments to collect data during RL training."""

from typing import Optional

import os
import time
import json
import h5py
import numpy as np
import pandas as pd
import gymnasium as gym


class DataCollectionWrapper(gym.Wrapper):
    """A wrapper that collects data during RL training."""

    def __init__(
        self,
        env: gym.Env,
        save_dir: str = "processed",
        collection_frequency: int = 1,
        save_format: str = "csv",
        episode_buffer_size: int = 10,
        save_on_episode_end: bool = True,
        experiment_name: str = None,
        seed: Optional[int] = None,
    ):
        """Initialize the data collection wrapper.

        Args:
            env: The environment to wrap
            save_dir: Directory to save collected data
            collection_frequency: How often to save data (in episodes)
            save_format: Format to save data ('csv', 'hdf5', or 'json')
            episode_buffer_size: Number of episodes to buffer before saving
            save_on_episode_end: Whether to save data at the end of each episode
            experiment_name: Name for this experiment run
            seed: Random seed for reproducibility
        """
        super().__init__(env)

        # Ensure the save directory exists
        self.save_dir = os.path.join(os.getcwd(), save_dir)
        os.makedirs(self.save_dir, exist_ok=True)

        # Configuration
        self.collection_frequency = collection_frequency
        self.save_format = save_format
        self.episode_buffer_size = episode_buffer_size
        self.save_on_episode_end = save_on_episode_end

        # Create experiment name if not provided
        if experiment_name is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"traffic_rl_{timestamp}"
        else:
            self.experiment_name = experiment_name

        # Create experiment directory
        self.experiment_dir = os.path.join(self.save_dir, self.experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)

        # Save seed for reproducibility
        self.seed = seed

        # Initialize data collection buffers
        self.reset_buffers()

        # Episode and step counters
        self.episode_count = 0
        self.total_step_count = 0

        # Save experiment configuration
        self._save_config()

    def reset_buffers(self):
        """Reset all data collection buffers."""
        # Episode-level data
        self.episode_buffer = {
            "episode_ids": [],
            "episode_rewards": [],
            "episode_lengths": [],
            "episode_throughputs": [],
            "episode_avg_queue_lengths": [],
            "episode_avg_waiting_times": [],
            "episode_total_switches": [],
        }

        # Step-level data
        self.step_buffer = {
            "episode_ids": [],
            "steps": [],
            "states": [],
            "actions": [],
            "rewards": [],
            "next_states": [],
            "dones": [],
            "truncateds": [],
            "queue_lengths": [],
            "waiting_times": [],
            "throughputs": [],
            "phases": [],
        }

    def _save_config(self):
        """Save experiment configuration."""
        config = {
            "environment": type(self.env).__name__,
            "observation_space": str(self.observation_space),
            "action_space": str(self.action_space),
            "seed": self.seed,
            "collection_frequency": self.collection_frequency,
            "save_format": self.save_format,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        # Save environment-specific configuration if available
        if hasattr(self.env, "config"):
            config["env_config"] = self.env.config

        with open(os.path.join(self.experiment_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=4)

    def reset(self, **kwargs):
        """Reset the environment and prepare for new episode data collection."""
        observation, info = self.env.reset(**kwargs)

        # Reset step counter for this episode
        self.step_count = 0

        return observation, info

    def step(self, action):
        """Take a step in the environment and record data.

        Args:
            action: The action to take

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Take action in environment
        state_before = self._get_latest_observation()
        observation, reward, terminated, truncated, info = self.env.step(action)

        # Update counters
        self.step_count += 1
        self.total_step_count += 1

        # Extract metrics to record
        queue_lengths = self._extract_queue_lengths(info)
        waiting_times = self._extract_waiting_times(info)
        throughput = self._extract_throughput(info)
        phase = self._extract_phase(info)

        # Store step data
        self.step_buffer["episode_ids"].append(self.episode_count)
        self.step_buffer["steps"].append(self.step_count)
        self.step_buffer["states"].append(state_before)
        self.step_buffer["actions"].append(action)
        self.step_buffer["rewards"].append(reward)
        self.step_buffer["next_states"].append(observation)
        self.step_buffer["dones"].append(terminated)
        self.step_buffer["truncateds"].append(truncated)
        self.step_buffer["queue_lengths"].append(queue_lengths)
        self.step_buffer["waiting_times"].append(waiting_times)
        self.step_buffer["throughputs"].append(throughput)
        self.step_buffer["phases"].append(phase)

        # If episode is done, record episode data
        if terminated or truncated:
            episode_reward = sum(self.step_buffer["rewards"][-self.step_count :])
            episode_throughput = self._extract_cumulative_throughput(info)
            avg_queue_length = np.mean(
                [
                    sum(q) if isinstance(q, dict) else q
                    for q in self.step_buffer["queue_lengths"][-self.step_count :]
                ]
            )
            avg_waiting_time = np.mean(
                [
                    sum(w.values()) if isinstance(w, dict) else w
                    for w in self.step_buffer["waiting_times"][-self.step_count :]
                ]
            )
            total_switches = self._extract_total_switches(info)

            # Store episode data
            self.episode_buffer["episode_ids"].append(self.episode_count)
            self.episode_buffer["episode_rewards"].append(episode_reward)
            self.episode_buffer["episode_lengths"].append(self.step_count)
            self.episode_buffer["episode_throughputs"].append(episode_throughput)
            self.episode_buffer["episode_avg_queue_lengths"].append(avg_queue_length)
            self.episode_buffer["episode_avg_waiting_times"].append(avg_waiting_time)
            self.episode_buffer["episode_total_switches"].append(total_switches)

            # Increment episode counter
            self.episode_count += 1

            # Save data if needed
            if (
                self.save_on_episode_end
                and self.episode_count % self.collection_frequency == 0
            ):
                self.save_data()

            # Reset buffers if we've accumulated enough episodes
            if len(self.episode_buffer["episode_ids"]) >= self.episode_buffer_size:
                self.save_data()
                self.reset_buffers()

        return observation, reward, terminated, truncated, info

    def _get_latest_observation(self):
        """Get the most recent observation (needed for recording state_before)."""
        if len(self.step_buffer["next_states"]) > 0:
            return self.step_buffer["next_states"][-1]
        return None

    def _extract_queue_lengths(self, info):
        """Extract queue length information from info dictionary."""
        if "queue_lengths" in info:
            return info["queue_lengths"]
        elif "queue_length" in info:
            return info["queue_length"]
        return 0.0

    def _extract_waiting_times(self, info):
        """Extract waiting time information from info dictionary."""
        if "waiting_times" in info:
            return info["waiting_times"]
        elif "waiting_time" in info:
            return info["waiting_time"]
        return 0.0

    def _extract_throughput(self, info):
        """Extract throughput information from info dictionary."""
        if "throughput" in info:
            return info["throughput"]
        return 0

    def _extract_cumulative_throughput(self, info):
        """Extract cumulative throughput from info dictionary."""
        if "cumulative_throughput" in info:
            return info["cumulative_throughput"]
        return sum(self.step_buffer["throughputs"][-self.step_count :])

    def _extract_phase(self, info):
        """Extract current phase from info dictionary."""
        if "phase" in info:
            return info["phase"]
        return 0

    def _extract_total_switches(self, info):
        """Extract total signal switches from info dictionary."""
        if "total_switches" in info:
            return info["total_switches"]
        return 0

    def save_data(self):
        """Save collected data to disk in the specified format."""
        # Only save if we have data
        if len(self.episode_buffer["episode_ids"]) == 0:
            return

        # Create timestamps for filenames
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        if self.save_format == "csv":
            self._save_as_csv(timestamp)
        elif self.save_format == "hdf5":
            self._save_as_hdf5(timestamp)
        elif self.save_format == "json":
            self._save_as_json(timestamp)
        else:
            raise ValueError(f"Unsupported save format: {self.save_format}")

        print(
            f"Data saved for episodes {min(self.episode_buffer['episode_ids'])} to "
            f"{max(self.episode_buffer['episode_ids'])}"
        )

    def _save_as_csv(self, timestamp):
        """Save data as CSV files."""
        # Save episode-level data
        episode_df = pd.DataFrame(self.episode_buffer)
        episode_file = os.path.join(self.experiment_dir, f"episodes_{timestamp}.csv")
        episode_df.to_csv(episode_file, index=False)

        # Save step-level data
        step_data = {
            k: v
            for k, v in self.step_buffer.items()
            if k not in ["states", "next_states"]
        }

        # Convert nested dictionaries to strings for CSV storage
        for key in ["queue_lengths", "waiting_times"]:
            if (
                key in step_data
                and len(step_data[key]) > 0
                and isinstance(step_data[key][0], dict)
            ):
                step_data[key] = [json.dumps(item) for item in step_data[key]]

        step_df = pd.DataFrame(step_data)
        step_file = os.path.join(self.experiment_dir, f"steps_{timestamp}.csv")
        step_df.to_csv(step_file, index=False)

        # Save state data separately (can be large)
        if len(self.step_buffer["states"]) > 0:
            states_array = np.array(self.step_buffer["states"])
            next_states_array = np.array(self.step_buffer["next_states"])

            states_file = os.path.join(self.experiment_dir, f"states_{timestamp}.npz")
            np.savez_compressed(
                states_file,
                states=states_array,
                next_states=next_states_array,
                episode_ids=np.array(self.step_buffer["episode_ids"]),
                steps=np.array(self.step_buffer["steps"]),
            )

    def _save_as_hdf5(self, timestamp):
        """Save data as HDF5 file."""
        file_path = os.path.join(self.experiment_dir, f"data_{timestamp}.h5")

        with h5py.File(file_path, "w") as f:
            # Create episode group
            episode_group = f.create_group("episodes")
            for key, value in self.episode_buffer.items():
                episode_group.create_dataset(key, data=np.array(value))

            # Create step group
            step_group = f.create_group("steps")

            # Save basic step data
            for key, value in self.step_buffer.items():
                if key not in [
                    "queue_lengths",
                    "waiting_times",
                    "states",
                    "next_states",
                ]:
                    step_group.create_dataset(key, data=np.array(value))

            # Save state data
            if len(self.step_buffer["states"]) > 0:
                step_group.create_dataset(
                    "states", data=np.array(self.step_buffer["states"])
                )
                step_group.create_dataset(
                    "next_states", data=np.array(self.step_buffer["next_states"])
                )

            # Save dictionary data as JSON strings
            for key in ["queue_lengths", "waiting_times"]:
                if key in self.step_buffer and len(self.step_buffer[key]) > 0:
                    if isinstance(self.step_buffer[key][0], dict):
                        json_data = [json.dumps(item) for item in self.step_buffer[key]]
                        step_group.create_dataset(
                            key,
                            data=np.array(
                                json_data, dtype=h5py.special_dtype(vlen=str)
                            ),
                        )
                    else:
                        step_group.create_dataset(
                            key, data=np.array(self.step_buffer[key])
                        )

    def _save_as_json(self, timestamp):
        """Save data as JSON files."""

        # For JSON, we need to convert numpy arrays to lists
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            return obj

        # Save episode-level data
        episode_data = {
            k: [convert_for_json(i) for i in v] for k, v in self.episode_buffer.items()
        }
        episode_file = os.path.join(self.experiment_dir, f"episodes_{timestamp}.json")
        with open(episode_file, "w") as f:
            json.dump(episode_data, f, indent=2)

        # Save step-level data (excluding states for file size)
        step_data = {
            k: [convert_for_json(i) for i in v]
            for k, v in self.step_buffer.items()
            if k not in ["states", "next_states"]
        }
        step_file = os.path.join(self.experiment_dir, f"steps_{timestamp}.json")
        with open(step_file, "w") as f:
            json.dump(step_data, f, indent=2)

        # Save state data separately
        if len(self.step_buffer["states"]) > 0:
            states_array = np.array(self.step_buffer["states"])
            next_states_array = np.array(self.step_buffer["next_states"])

            states_file = os.path.join(self.experiment_dir, f"states_{timestamp}.npz")
            np.savez_compressed(
                states_file,
                states=states_array,
                next_states=next_states_array,
                episode_ids=np.array(self.step_buffer["episode_ids"]),
                steps=np.array(self.step_buffer["steps"]),
            )

    def close(self):
        """Close the environment and save any remaining data."""
        # Save any remaining data
        if len(self.episode_buffer["episode_ids"]) > 0:
            self.save_data()

        # Close the wrapped environment
        self.env.close()


# Example usage
if __name__ == "__main__":
    from environments.intersection_env import IntersectionEnv

    # Create base environment
    base_env = IntersectionEnv()

    # Wrap with data collection
    env = DataCollectionWrapper(
        base_env,
        save_dir="processed",
        save_format="csv",
        experiment_name="test_collection",
    )

    # Reset environment
    obs, _ = env.reset()

    # Run a few episodes
    for episode in range(3):
        done = False
        truncated = False

        while not (done or truncated):
            # Random action
            action = env.action_space.sample()

            # Step environment
            obs, reward, done, truncated, info = env.step(action)

            # Print progress
            if env.step_count % 10 == 0:
                print(
                    f"Episode {episode+1}, Step {env.step_count}, Reward: {reward:.2f}"
                )

        print(f"Episode {episode+1} complete")

    # Close environment (will save any remaining data)
    env.close()

    print(f"Data saved to {os.path.join(os.getcwd(), 'processed')}")
