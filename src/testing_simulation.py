import traci
import numpy as np
import timeit
import tensorflow as tf

# phase codes based on environment.net.xml
PHASE_NS_GREEN = 0  # action 0 code 00
PHASE_NS_YELLOW = 1
PHASE_NSL_GREEN = 2  # action 1 code 01
PHASE_NSL_YELLOW = 3
PHASE_EW_GREEN = 4  # action 2 code 10
PHASE_EW_YELLOW = 5
PHASE_EWL_GREEN = 6  # action 3 code 11
PHASE_EWL_YELLOW = 7


class Simulation:
    def __init__(
        self,
        agent,
        TrafficGen,
        sumo_cmd,
        max_steps,
        green_duration,
        yellow_duration,
        num_states,
        num_actions,
    ):
        self._agent = agent
        self._TrafficGen = TrafficGen
        self._step = 0
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps
        self._green_duration = green_duration
        self._yellow_duration = yellow_duration
        self._num_states = num_states
        self._num_actions = num_actions
        self._reward_episode = []
        self._queue_length_episode = []

        # Optimization - Predefine numpy arrays for state
        self._current_state = np.zeros(self._num_states, dtype=np.float32)

        # Track additional metrics
        self._teleport_count = 0
        self._total_waiting_time = 0
        self._total_co2_emission = 0
        self._total_fuel_consumption = 0
        self._avg_speed = 0

    def run(self, episode):
        """
        Runs the testing simulation
        """
        start_time = timeit.default_timer()

        # first, generate the route file for this simulation and set up sumo
        self._TrafficGen.generate_routefile(seed=episode)
        traci.start(self._sumo_cmd)
        print("Simulating...")

        # inits
        self._step = 0
        self._waiting_times = {}
        self._teleport_count = 0
        self._total_waiting_time = 0
        self._total_co2_emission = 0
        self._total_fuel_consumption = 0
        self._avg_speed = 0
        old_total_wait = 0
        old_action = -1  # dummy init
        speeds = []

        while self._step < self._max_steps:
            # Track teleported vehicles
            teleports = traci.simulation.getStartingTeleportNumber()
            if teleports > self._teleport_count:
                self._teleport_count = teleports

            # Track environmental metrics
            self._total_co2_emission += sum(
                traci.edge.getCO2Emission(edge)
                for edge in ["E2TL", "N2TL", "W2TL", "S2TL"]
            )
            self._total_fuel_consumption += sum(
                traci.edge.getFuelConsumption(edge)
                for edge in ["E2TL", "N2TL", "W2TL", "S2TL"]
            )

            # Track average speed
            for veh_id in traci.vehicle.getIDList():
                speeds.append(traci.vehicle.getSpeed(veh_id))

            # get current state of the intersection
            current_state = self._get_state()

            # calculate reward of previous action: (change in cumulative waiting time between actions)
            # waiting time = seconds waited by a car since the spawn in the environment, cumulated for every car in incoming lanes
            current_total_wait = self._collect_waiting_times()
            reward = old_total_wait - current_total_wait

            # choose the light phase to activate, based on the current state of the intersection
            action = self._agent.act(current_state)

            # if the chosen phase is different from the last phase, activate the yellow phase
            if self._step != 0 and old_action != action:
                self._set_yellow_phase(old_action)
                self._simulate(self._yellow_duration)

            # execute the phase selected before
            self._set_green_phase(action)
            self._simulate(self._green_duration)

            # saving variables for later & accumulate reward
            old_action = action
            old_total_wait = current_total_wait

            self._reward_episode.append(reward)

        # Calculate average speed
        if speeds:
            self._avg_speed = sum(speeds) / len(speeds)

        # Calculate total waiting time
        self._total_waiting_time = sum(self._waiting_times.values())

        print(
            f"Test results - Reward: {sum(self._reward_episode):.2f}, Avg Queue: {sum(self._queue_length_episode)/len(self._queue_length_episode):.2f}, "
            f"Teleports: {self._teleport_count}, Avg Speed: {self._avg_speed:.2f} m/s"
        )
        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)

        return simulation_time

    def _simulate(self, steps_todo):
        """
        Proceed with the simulation in sumo
        """
        if (
            self._step + steps_todo
        ) >= self._max_steps:  # do not do more steps than the maximum allowed number of steps
            steps_todo = self._max_steps - self._step

        while steps_todo > 0:
            traci.simulationStep()  # simulate 1 step in sumo
            self._step += 1  # update the step counter
            steps_todo -= 1
            queue_length = self._get_queue_length()
            self._queue_length_episode.append(queue_length)

    def _collect_waiting_times(self):
        """
        Retrieve the waiting time of every car in the incoming roads
        """
        incoming_roads = ["E2TL", "N2TL", "W2TL", "S2TL"]
        car_list = traci.vehicle.getIDList()

        # Faster method for vehicles not in incoming roads
        vehicles_to_remove = []

        for car_id in car_list:
            wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
            road_id = traci.vehicle.getRoadID(
                car_id
            )  # get the road id where the car is located

            if (
                road_id in incoming_roads
            ):  # consider only the waiting times of cars in incoming roads
                self._waiting_times[car_id] = wait_time
            else:
                if (
                    car_id in self._waiting_times
                ):  # a car that was tracked has cleared the intersection
                    vehicles_to_remove.append(car_id)

        # Batch remove vehicles
        for car_id in vehicles_to_remove:
            del self._waiting_times[car_id]

        total_waiting_time = sum(self._waiting_times.values())
        return total_waiting_time

    def _set_yellow_phase(self, old_action):
        """
        Activate the correct yellow light combination in sumo
        """
        yellow_phase_code = (
            old_action * 2 + 1
        )  # obtain the yellow phase code, based on the old action (ref on environment.net.xml)
        traci.trafficlight.setPhase("TL", yellow_phase_code)

    def _set_green_phase(self, action_number):
        """
        Activate the correct green light combination in sumo
        """

        if action_number == 0:
            traci.trafficlight.setPhase("TL", PHASE_NS_GREEN)
        elif action_number == 1:
            traci.trafficlight.setPhase("TL", PHASE_NSL_GREEN)
        elif action_number == 2:
            traci.trafficlight.setPhase("TL", PHASE_EW_GREEN)
        elif action_number == 3:
            traci.trafficlight.setPhase("TL", PHASE_EWL_GREEN)

    def _get_queue_length(self):
        """
        Retrieve the number of cars with speed = 0 in every incoming lane
        """
        halt_N = traci.edge.getLastStepHaltingNumber("N2TL")
        halt_S = traci.edge.getLastStepHaltingNumber("S2TL")
        halt_E = traci.edge.getLastStepHaltingNumber("E2TL")
        halt_W = traci.edge.getLastStepHaltingNumber("W2TL")
        queue_length = halt_N + halt_S + halt_E + halt_W
        return queue_length

    def _get_state(self):
        """
        Retrieve the state of the intersection from sumo, in the form of cell occupancy
        Uses vectorized operations for better performance
        """
        # Reset the state array to zeros for this step
        self._current_state.fill(0)

        car_list = traci.vehicle.getIDList()

        # Batch retrieve vehicle info
        if car_list:
            lane_positions = np.array(
                [750 - traci.vehicle.getLanePosition(car) for car in car_list],
                dtype=np.float32,
            )
            lane_ids = [traci.vehicle.getLaneID(car) for car in car_list]

            # Process each car
            for i, car_id in enumerate(car_list):
                lane_pos = lane_positions[i]
                lane_id = lane_ids[i]

                # Map distance to lane cell
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
                    continue  # Skip this car

                # finding the lane where the car is located
                # x2TL_3 are the "turn left only" lanes
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
                    continue  # Skip cars not in our target lanes

                if 1 <= lane_group <= 7:
                    car_position = int(
                        str(lane_group) + str(lane_cell)
                    )  # composition of the two position IDs
                    self._current_state[car_position] = 1
                elif lane_group == 0:
                    self._current_state[lane_cell] = 1

        return self._current_state

    @property
    def queue_length_episode(self):
        return self._queue_length_episode

    @property
    def reward_episode(self):
        return self._reward_episode

    @property
    def teleport_count(self):
        return self._teleport_count

    @property
    def total_waiting_time(self):
        return self._total_waiting_time

    @property
    def avg_speed(self):
        return self._avg_speed

    @property
    def total_co2_emission(self):
        return self._total_co2_emission

    @property
    def total_fuel_consumption(self):
        return self._total_fuel_consumption
