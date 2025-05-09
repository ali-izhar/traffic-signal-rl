"""Training Simulation Environment for Traffic Signal Control"""

from typing import List, Tuple, Any
import random
import timeit

import numpy as np
import traci

# Phase codes based on environment.net.xml
PHASE_NS_GREEN = 0  # North-South green (through/right)
PHASE_NS_YELLOW = 1  # North-South yellow
PHASE_NSL_GREEN = 2  # North-South left green
PHASE_NSL_YELLOW = 3  # North-South left yellow
PHASE_EW_GREEN = 4  # East-West green (through/right)
PHASE_EW_YELLOW = 5  # East-West yellow
PHASE_EWL_GREEN = 6  # East-West left green
PHASE_EWL_YELLOW = 7  # East-West left yellow


class Simulation:
    """Simulation environment for reinforcement learning-based traffic signal control.

    This class manages the interaction with SUMO simulator, computes states and rewards,
    and provides functionality for training the RL agent.

    Attributes:
        _Model: Neural network model for the RL agent
        _Memory: Experience replay buffer
        _TrafficGen: Traffic generator for creating vehicle routes
        _gamma: Discount factor for future rewards
        _step: Current simulation step
        _sumo_cmd: Command to start SUMO simulation
        _max_steps: Maximum number of steps per episode
        _green_duration: Duration of green phase in seconds
        _yellow_duration: Duration of yellow phase in seconds
        _num_states: Size of the state space
        _num_actions: Size of the action space
        _reward_store: Storage for episode rewards
        _cumulative_wait_store: Storage for cumulative waiting times
        _avg_queue_length_store: Storage for average queue lengths
        _training_epochs: Number of epochs for training after each episode
    """

    def __init__(
        self,
        Model: Any,
        Memory: Any,
        TrafficGen: Any,
        sumo_cmd: List[str],
        gamma: float,
        max_steps: int,
        green_duration: int,
        yellow_duration: int,
        num_states: int,
        num_actions: int,
        training_epochs: int,
    ) -> None:
        """
        Initialize the simulation environment.

        Args:
            Model: Neural network model
            Memory: Experience replay buffer
            TrafficGen: Traffic generator for creating vehicle routes
            sumo_cmd: Command to start SUMO simulation
            gamma: Discount factor for future rewards
            max_steps: Maximum number of steps per episode
            green_duration: Duration of green phase in seconds
            yellow_duration: Duration of yellow phase in seconds
            num_states: Size of the state space
            num_actions: Size of the action space
            training_epochs: Number of training epochs after each episode
        """
        self._Model = Model
        self._Memory = Memory
        self._TrafficGen = TrafficGen
        self._gamma = gamma
        self._step = 0
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps
        self._green_duration = green_duration
        self._yellow_duration = yellow_duration
        self._num_states = num_states
        self._num_actions = num_actions
        self._reward_store = []
        self._cumulative_wait_store = []
        self._avg_queue_length_store = []
        self._training_epochs = training_epochs

    def run(self, episode: int, epsilon: float) -> Tuple[float, float]:
        """Run a complete training episode and perform training.

        The episode consists of:
        1. Generating traffic and starting simulation
        2. Collecting experience (state, action, reward, next_state)
        3. Storing experience in replay memory
        4. Training the neural network model

        Args:
            episode: Current episode number (used as random seed)
            epsilon: Exploration rate for epsilon-greedy policy

        Returns:
            simulation_time: Time taken for simulation in seconds
            training_time: Time taken for training in seconds
        """
        start_time = timeit.default_timer()

        # Generate route file and start SUMO
        self._TrafficGen.generate_routefile(seed=episode)
        traci.start(self._sumo_cmd)
        print("Simulating...")

        # Initialize episode variables
        self._step = 0
        self._waiting_times = {}
        self._sum_neg_reward = 0
        self._sum_queue_length = 0
        self._sum_waiting_time = 0
        old_total_wait = 0
        old_state = -1
        old_action = -1

        # Main simulation loop
        while self._step < self._max_steps:
            # Get current state and calculate reward
            current_state = self._get_state()
            current_total_wait = self._collect_waiting_times()
            reward = old_total_wait - current_total_wait

            # Store experience in memory (except for first step)
            if self._step != 0:
                self._Memory.add_sample((old_state, old_action, reward, current_state))

            # Choose action using epsilon-greedy policy
            action = self._choose_action(current_state, epsilon)

            # Apply yellow phase if changing action
            if self._step != 0 and old_action != action:
                self._set_yellow_phase(old_action)
                self._simulate(self._yellow_duration)

            # Apply green phase
            self._set_green_phase(action)
            self._simulate(self._green_duration)

            # Save variables for next step
            old_state = current_state
            old_action = action
            old_total_wait = current_total_wait

            # Accumulate negative rewards for statistics
            if reward < 0:
                self._sum_neg_reward += reward

        # Save episode statistics and close SUMO
        self._save_episode_stats()
        print(f"Total reward: {self._sum_neg_reward} - Epsilon: {round(epsilon, 2)}")
        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)

        # Train the model using experience replay
        print("Training...")
        start_time = timeit.default_timer()
        for _ in range(self._training_epochs):
            self._replay()
        training_time = round(timeit.default_timer() - start_time, 1)

        return simulation_time, training_time

    def _simulate(self, steps_todo: int) -> None:
        """Execute the specified number of simulation steps while gathering statistics.

        Args:
            steps_todo: Number of simulation steps to execute
        """
        # Adjust steps if we would exceed max_steps
        if (self._step + steps_todo) >= self._max_steps:
            steps_todo = self._max_steps - self._step

        # Execute steps and collect statistics
        while steps_todo > 0:
            traci.simulationStep()  # Simulate 1 step in SUMO
            self._step += 1  # Update step counter
            steps_todo -= 1

            # Update statistics
            queue_length = self._get_queue_length()
            self._sum_queue_length += queue_length
            self._sum_waiting_time += (
                queue_length  # 1 step waiting in queue = 1 second waited per car
            )

    def _collect_waiting_times(self) -> float:
        """Retrieve the waiting time of every vehicle in the incoming roads.

        The waiting time of a vehicle is the time (in seconds) spent with speed < 0.1m/s
        since the last time it was faster than 0.1m/s.

        Returns:
            Total accumulated waiting time of all vehicles in incoming roads
        """
        incoming_roads = ["E2TL", "N2TL", "W2TL", "S2TL"]
        car_list = traci.vehicle.getIDList()

        for car_id in car_list:
            wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
            road_id = traci.vehicle.getRoadID(car_id)

            if road_id in incoming_roads:
                # Track vehicle in incoming roads
                self._waiting_times[car_id] = wait_time
            elif car_id in self._waiting_times:
                # Remove vehicle that has cleared the intersection
                del self._waiting_times[car_id]

        total_waiting_time = sum(self._waiting_times.values())
        return total_waiting_time

    def _choose_action(self, state: np.ndarray, epsilon: float) -> int:
        """Select an action using an epsilon-greedy policy.

        With probability epsilon, choose a random action (exploration).
        With probability (1-epsilon), choose the best action according to the model (exploitation).

        Args:
            state: Current state representation
            epsilon: Exploration rate (0 <= epsilon <= 1)

        Returns:
            Selected action index
        """
        if random.random() < epsilon:
            # Exploration: random action
            return random.randint(0, self._num_actions - 1)
        else:
            # Exploitation: best action according to model
            return np.argmax(self._Model.predict_one(state))

    def _set_yellow_phase(self, old_action: int) -> None:
        """Activate the correct yellow light phase in SUMO based on the previous action.

        Args:
            old_action: Previous green phase action index
        """
        # Calculate yellow phase code based on previous action
        yellow_phase_code = old_action * 2 + 1
        traci.trafficlight.setPhase("TL", yellow_phase_code)

    def _set_green_phase(self, action_number: int) -> None:
        """Activate the correct green light phase in SUMO.

        Args:
            action_number: Action index (0-3) corresponding to a green phase
        """
        if action_number == 0:
            traci.trafficlight.setPhase("TL", PHASE_NS_GREEN)
        elif action_number == 1:
            traci.trafficlight.setPhase("TL", PHASE_NSL_GREEN)
        elif action_number == 2:
            traci.trafficlight.setPhase("TL", PHASE_EW_GREEN)
        elif action_number == 3:
            traci.trafficlight.setPhase("TL", PHASE_EWL_GREEN)

    def _get_queue_length(self) -> int:
        """Retrieve the number of vehicles stopped (speed < 0.1 m/s) in all incoming lanes.

        Returns:
            Total number of stopped vehicles
        """
        halt_N = traci.edge.getLastStepHaltingNumber("N2TL")
        halt_S = traci.edge.getLastStepHaltingNumber("S2TL")
        halt_E = traci.edge.getLastStepHaltingNumber("E2TL")
        halt_W = traci.edge.getLastStepHaltingNumber("W2TL")

        queue_length = halt_N + halt_S + halt_E + halt_W
        return queue_length

    def _get_state(self) -> np.ndarray:
        """Retrieve the state representation from SUMO as a binary cell occupancy grid.

        State representation:
        - Each approaching lane is divided into cells of varying size
        - The state is a binary vector where 1 indicates a cell is occupied by a vehicle
        - Total state size is 80 (8 lane groups × 10 cells)

        Returns:
            Binary state representation as a numpy array
        """
        state = np.zeros(self._num_states)
        car_list = traci.vehicle.getIDList()

        for car_id in car_list:
            lane_pos = traci.vehicle.getLanePosition(car_id)
            lane_id = traci.vehicle.getLaneID(car_id)

            # Invert lane position: 0 = close to traffic light, 750 = far from intersection
            lane_pos = 750 - lane_pos

            # Map distance to cell index (0-9)
            if lane_pos < 7:
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
                continue  # Skip vehicle if not in range

            # Determine lane group (0-7) based on lane ID
            # Lane groups: 0=W(straight/right), 1=W(left), 2=N(straight/right), etc.
            lane_group = -1
            if lane_id == "W2TL_0" or lane_id == "W2TL_1" or lane_id == "W2TL_2":
                lane_group = 0
            elif lane_id == "W2TL_3":
                lane_group = 1
            elif lane_id == "N2TL_0" or lane_id == "N2TL_1" or lane_id == "N2TL_2":
                lane_group = 2
            elif lane_id == "N2TL_3":
                lane_group = 3
            elif lane_id == "E2TL_0" or lane_id == "E2TL_1" or lane_id == "E2TL_2":
                lane_group = 4
            elif lane_id == "E2TL_3":
                lane_group = 5
            elif lane_id == "S2TL_0" or lane_id == "S2TL_1" or lane_id == "S2TL_2":
                lane_group = 6
            elif lane_id == "S2TL_3":
                lane_group = 7
            else:
                continue  # Skip vehicle if not in a relevant lane

            # Calculate state array position for the vehicle
            if lane_group >= 1 and lane_group <= 7:
                # Groups 1-7 map to positions 10-79
                car_position = int(lane_group * 10 + lane_cell)
                valid_car = True
            elif lane_group == 0:
                # Group 0 maps to positions 0-9
                car_position = lane_cell
                valid_car = True
            else:
                valid_car = False

            # Mark cell as occupied
            if valid_car and 0 <= car_position < self._num_states:
                state[car_position] = 1

        return state

    def _replay(self) -> None:
        """Train the neural network model using experience replay.

        This method:
        1. Samples a batch of experiences from memory
        2. Computes target Q-values using the Bellman equation
        3. Updates the neural network weights

        Mathematical formulation:
        Q(s,a) = r + γ * max_a' Q(s',a')
        where:
        - Q(s,a) is the Q-value for state s and action a
        - r is the immediate reward
        - γ is the discount factor
        - max_a' Q(s',a') is the maximum Q-value for the next state
        """
        # Get a batch of samples from memory
        batch = self._Memory.get_samples(self._Model.batch_size)

        if len(batch) > 0:  # Only train if batch is not empty
            # Extract states, actions, rewards, and next_states from batch
            states = np.array([val[0] for val in batch])
            next_states = np.array([val[3] for val in batch])

            # Predict Q-values for current and next states
            q_s_a = self._Model.predict_batch(states)
            q_s_a_d = self._Model.predict_batch(next_states)

            # Initialize training arrays
            x = np.zeros((len(batch), self._num_states))
            y = np.zeros((len(batch), self._num_actions))

            # Update Q-values using Bellman equation
            for i, b in enumerate(batch):
                state, action, reward, _ = b[0], b[1], b[2], b[3]
                current_q = q_s_a[i]
                # Update Q-value: Q(s,a) = r + γ * max_a' Q(s',a')
                current_q[action] = reward + self._gamma * np.amax(q_s_a_d[i])

                # Store state and updated Q-values for training
                x[i] = state
                y[i] = current_q

            # Train the neural network on the batch
            self._Model.train_batch(x, y)

    def _save_episode_stats(self) -> None:
        """Save statistics from the episode for later analysis and visualization."""
        # Store total negative reward
        self._reward_store.append(self._sum_neg_reward)

        # Store total waiting time
        self._cumulative_wait_store.append(self._sum_waiting_time)

        # Store average queue length
        self._avg_queue_length_store.append(self._sum_queue_length / self._max_steps)

    @property
    def reward_store(self) -> List[float]:
        """Get the list of rewards from all episodes."""
        return self._reward_store

    @property
    def cumulative_wait_store(self) -> List[float]:
        """Get the list of cumulative waiting times from all episodes."""
        return self._cumulative_wait_store

    @property
    def avg_queue_length_store(self) -> List[float]:
        """Get the list of average queue lengths from all episodes."""
        return self._avg_queue_length_store
