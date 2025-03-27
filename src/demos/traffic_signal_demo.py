#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Traffic Signal Control Demo

This demo showcases a single intersection managed by a reinforcement learning agent
(specifically DQN) as described in our paper "Adaptive Traffic Signal Control with
Reinforcement Learning".

The demo initializes a traffic simulation environment, trains a DQN agent, and
visualizes the resulting traffic flow and agent performance.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
from collections import deque
import random

# Add src to path to import local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TrafficSignalEnv:
    """Simplified traffic signal environment for the demo"""

    def __init__(self):
        # Traffic flow variables
        self.queue_n = 0  # North queue
        self.queue_s = 0  # South queue
        self.queue_e = 0  # East queue
        self.queue_w = 0  # West queue

        # Traffic signal state (0: N-S green, E-W red; 1: N-S red, E-W green)
        self.phase = 0

        # Traffic arrival rates (vehicles per step)
        self.arrival_rate_n = 0.2
        self.arrival_rate_s = 0.2
        self.arrival_rate_e = 0.3
        self.arrival_rate_w = 0.3

        # Traffic departure rates when green (vehicles per step)
        self.departure_rate = 0.4

        # Time step
        self.time = 0

        # Maximum queue length for rendering
        self.max_queue = 20

        # State and action spaces
        self.state_space = 5  # [queue_n, queue_s, queue_e, queue_w, phase]
        self.action_space = 2  # [keep current phase, switch phase]

    def reset(self):
        """Reset the environment to initial state"""
        self.queue_n = np.random.randint(0, 5)
        self.queue_s = np.random.randint(0, 5)
        self.queue_e = np.random.randint(0, 5)
        self.queue_w = np.random.randint(0, 5)
        self.phase = 0
        self.time = 0

        return self._get_state(), {}

    def step(self, action):
        """Execute action and return new state, reward, done, truncated, info"""
        # Process action (0: keep current phase, 1: switch phase)
        if action == 1:
            self.phase = 1 - self.phase  # Switch phase

        # Process arrivals based on probabilistic model
        if random.random() < self.arrival_rate_n:
            self.queue_n += 1
        if random.random() < self.arrival_rate_s:
            self.queue_s += 1
        if random.random() < self.arrival_rate_e:
            self.queue_e += 1
        if random.random() < self.arrival_rate_w:
            self.queue_w += 1

        # Process departures based on current phase
        if self.phase == 0:  # N-S green
            if self.queue_n > 0 and random.random() < self.departure_rate:
                self.queue_n -= 1
            if self.queue_s > 0 and random.random() < self.departure_rate:
                self.queue_s -= 1
        else:  # E-W green
            if self.queue_e > 0 and random.random() < self.departure_rate:
                self.queue_e -= 1
            if self.queue_w > 0 and random.random() < self.departure_rate:
                self.queue_w -= 1

        # Calculate reward - negative sum of queue lengths
        reward = -(self.queue_n + self.queue_s + self.queue_e + self.queue_w)

        # Update time step
        self.time += 1

        # Check if done
        done = self.time >= 100  # End after 100 time steps for the demo

        return self._get_state(), reward, done, False, {}

    def _get_state(self):
        """Return current state representation"""
        return np.array(
            [self.queue_n, self.queue_s, self.queue_e, self.queue_w, self.phase],
            dtype=np.float32,
        )

    def render(self, ax=None):
        """Render current state of traffic intersection"""
        if ax is None:
            return

        ax.clear()

        # Draw intersection
        ax.plot([-2, 2], [0, 0], "k-", lw=3)  # East-West road
        ax.plot([0, 0], [-2, 2], "k-", lw=3)  # North-South road

        # Draw traffic lights
        light_size = 0.2
        # North traffic light
        if self.phase == 0:  # N-S green
            ax.add_patch(plt.Circle((0, 1), light_size, fc="green"))
        else:  # N-S red
            ax.add_patch(plt.Circle((0, 1), light_size, fc="red"))

        # South traffic light
        if self.phase == 0:  # N-S green
            ax.add_patch(plt.Circle((0, -1), light_size, fc="green"))
        else:  # N-S red
            ax.add_patch(plt.Circle((0, -1), light_size, fc="red"))

        # East traffic light
        if self.phase == 1:  # E-W green
            ax.add_patch(plt.Circle((1, 0), light_size, fc="green"))
        else:  # E-W red
            ax.add_patch(plt.Circle((1, 0), light_size, fc="red"))

        # West traffic light
        if self.phase == 1:  # E-W green
            ax.add_patch(plt.Circle((-1, 0), light_size, fc="green"))
        else:  # E-W red
            ax.add_patch(plt.Circle((-1, 0), light_size, fc="red"))

        # Draw queues as stacked cars
        car_size = 0.1

        # North queue
        for i in range(min(self.queue_n, 10)):  # Show max 10 cars
            y_pos = 1.2 + (i * 0.2)
            ax.add_patch(plt.Rectangle((-0.15, y_pos), 0.3, 0.15, fc="blue"))

        # South queue
        for i in range(min(self.queue_s, 10)):
            y_pos = -1.2 - (i * 0.2) - 0.15
            ax.add_patch(plt.Rectangle((-0.15, y_pos), 0.3, 0.15, fc="blue"))

        # East queue
        for i in range(min(self.queue_e, 10)):
            x_pos = 1.2 + (i * 0.2)
            ax.add_patch(plt.Rectangle((x_pos, -0.15), 0.15, 0.3, fc="blue"))

        # West queue
        for i in range(min(self.queue_w, 10)):
            x_pos = -1.2 - (i * 0.2) - 0.15
            ax.add_patch(plt.Rectangle((x_pos, -0.15), 0.15, 0.3, fc="blue"))

        # Set plot limits and labels
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect("equal")
        ax.set_title(f"Traffic Signal Control - Step {self.time}")

        # Add queue length text
        queue_text = f"Queue Lengths: N={self.queue_n}, S={self.queue_s}, E={self.queue_e}, W={self.queue_w}"
        ax.text(0, -2.8, queue_text, ha="center", fontsize=10)

        ax.set_xticks([])
        ax.set_yticks([])


class DQNAgent:
    """Simple DQN agent for the demo"""

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        # Experience replay memory
        self.memory = deque(maxlen=2000)

        # Hyperparameters
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

        # Simple Q-table for the demo (would use neural network in real implementation)
        self.q_table = np.zeros(
            (20, 20, 20, 20, 2, 2)
        )  # [queue_n, queue_s, queue_e, queue_w, phase, action]

    def act(self, state):
        """Choose action based on epsilon-greedy policy"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        # Convert state to indices for the Q-table
        queue_n, queue_s, queue_e, queue_w, phase = state
        queue_n = min(int(queue_n), 19)
        queue_s = min(int(queue_s), 19)
        queue_e = min(int(queue_e), 19)
        queue_w = min(int(queue_w), 19)
        phase = int(phase)

        q_values = self.q_table[queue_n, queue_s, queue_e, queue_w, phase]
        return np.argmax(q_values)

    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        """Train on random batch from memory"""
        minibatch = random.sample(self.memory, min(len(self.memory), batch_size))

        for state, action, reward, next_state, done in minibatch:
            # Convert states to indices for the Q-table
            queue_n, queue_s, queue_e, queue_w, phase = state
            queue_n = min(int(queue_n), 19)
            queue_s = min(int(queue_s), 19)
            queue_e = min(int(queue_e), 19)
            queue_w = min(int(queue_w), 19)
            phase = int(phase)

            next_queue_n, next_queue_s, next_queue_e, next_queue_w, next_phase = (
                next_state
            )
            next_queue_n = min(int(next_queue_n), 19)
            next_queue_s = min(int(next_queue_s), 19)
            next_queue_e = min(int(next_queue_e), 19)
            next_queue_w = min(int(next_queue_w), 19)
            next_phase = int(next_phase)

            # Q-learning update
            if done:
                target = reward
            else:
                target = reward + self.gamma * np.max(
                    self.q_table[
                        next_queue_n,
                        next_queue_s,
                        next_queue_e,
                        next_queue_w,
                        next_phase,
                    ]
                )

            self.q_table[queue_n, queue_s, queue_e, queue_w, phase, action] = (
                1 - self.learning_rate
            ) * self.q_table[
                queue_n, queue_s, queue_e, queue_w, phase, action
            ] + self.learning_rate * target

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def run_demo():
    """Run traffic signal control demo"""
    # Create environment
    env = TrafficSignalEnv()

    # Create DQN agent
    state_size = env.state_space
    action_size = env.action_space
    agent = DQNAgent(state_size, action_size)

    # Training parameters
    episodes = 50
    batch_size = 32

    # Lists to store metrics
    episode_rewards = []
    episode_queue_lengths = []

    # Training loop
    print("Training DQN agent...")
    for e in range(episodes):
        # Reset environment
        state, _ = env.reset()
        total_reward = 0
        avg_queue_length = []

        # Episode loop
        done = False
        while not done:
            # Select action
            action = agent.act(state)

            # Take action
            next_state, reward, done, _, _ = env.step(action)

            # Store experience
            agent.remember(state, action, reward, next_state, done)

            # Update state
            state = next_state

            # Update metrics
            total_reward += reward
            avg_queue_length.append(
                np.mean([env.queue_n, env.queue_s, env.queue_e, env.queue_w])
            )

        # Train agent on batch of experiences
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

        # Record episode metrics
        episode_rewards.append(total_reward)
        episode_queue_lengths.append(np.mean(avg_queue_length))

        # Print progress
        if (e + 1) % 10 == 0:
            print(
                f"Episode {e+1}/{episodes}, Reward: {total_reward}, Avg Queue Length: {np.mean(avg_queue_length):.2f}"
            )

    print("Training complete!")

    # Create demo visualization
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1])

    # Intersection visualization
    ax_intersection = plt.subplot(gs[0, :])

    # Performance plots
    ax_reward = plt.subplot(gs[1, 0])
    ax_queue = plt.subplot(gs[1, 1])

    # Plot training metrics
    ax_reward.plot(episode_rewards, "b-")
    ax_reward.set_title("Episode Rewards")
    ax_reward.set_xlabel("Episode")
    ax_reward.set_ylabel("Total Reward")

    ax_queue.plot(episode_queue_lengths, "r-")
    ax_queue.set_title("Average Queue Length")
    ax_queue.set_xlabel("Episode")
    ax_queue.set_ylabel("Queue Length")

    # Run demo animation
    state, _ = env.reset()

    # Set to evaluation mode
    agent.epsilon = 0.0  # No exploration

    # Visualization loop
    frames = 100
    fig.suptitle(
        "Adaptive Traffic Signal Control with Reinforcement Learning", fontsize=16
    )

    def update(frame):
        # Declare nonlocal state at the beginning of the function
        nonlocal state

        # Select action
        action = agent.act(state)

        # Take action
        next_state, reward, done, _, _ = env.step(action)

        # Update state
        state = next_state

        # Render intersection
        env.render(ax_intersection)

        # Add current action text
        action_text = (
            "Action: Keep Current Phase" if action == 0 else "Action: Switch Phase"
        )
        ax_intersection.text(0, 2.5, action_text, ha="center", fontsize=12)

        # Reset if done
        if done:
            state, _ = env.reset()

        return [ax_intersection]

    # Create animation
    ani = FuncAnimation(fig, update, frames=frames, interval=200, blit=False)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

    print("Demo complete!")


if __name__ == "__main__":
    run_demo()
