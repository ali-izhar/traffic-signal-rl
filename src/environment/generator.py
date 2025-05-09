"""Traffic Generator for Reinforcement Learning Environment

This module generates traffic patterns for SUMO simulation. It creates route files
with vehicle definitions following a Weibull distribution to simulate realistic
traffic patterns.

Mathematical formulation of traffic generation:
- Vehicle generation times follow a Weibull distribution: f(x;k,λ) = (k/λ)(x/λ)^(k-1)e^(-(x/λ)^k)
  where k=2 (shape parameter) creates realistic traffic patterns
- Traffic distribution: 75% straight-moving vehicles, 25% turning vehicles
"""

import os
import math
import numpy as np


class TrafficGenerator:
    """Generates realistic traffic patterns for SUMO simulation.

    This class creates route files for SUMO that define when and where vehicles enter
    the network, and what routes they follow. The timing of vehicle generation
    follows a Weibull distribution to create realistic traffic patterns.

    Attributes:
        _n_cars_generated: Number of vehicles to generate per episode
        _max_steps: Maximum number of simulation steps
    """

    def __init__(self, max_steps: int, n_cars_generated: int) -> None:
        """Initialize the traffic generator.

        Args:
            max_steps: Maximum number of simulation steps
            n_cars_generated: Number of vehicles to generate per episode
        """
        self._n_cars_generated = n_cars_generated
        self._max_steps = max_steps

    def generate_routefile(self, seed: int, output_dir: str = "intersection") -> None:
        """Generate a SUMO route file with vehicle definitions for one episode.

        The method:
        1. Generates vehicle arrival times following a Weibull distribution
        2. Maps these times to the simulation steps
        3. Creates a route file with vehicle definitions
        4. Randomly assigns routes to vehicles (75% straight, 25% turning)

        Args:
            seed: Random seed for reproducible traffic generation
            output_dir: Directory to save the route file (default: "intersection")
        """
        np.random.seed(seed)  # Make tests reproducible

        # Generate vehicle arrival times using Weibull distribution (shape=2)
        # This creates a realistic traffic distribution
        timings = np.random.weibull(2, self._n_cars_generated)
        timings = np.sort(timings)

        # Rescale the distribution to fit within [0, max_steps]
        car_gen_steps = []
        min_old = math.floor(timings[1])
        max_old = math.ceil(timings[-1])
        min_new = 0
        max_new = self._max_steps

        for value in timings:
            car_gen_steps = np.append(
                car_gen_steps,
                ((max_new - min_new) / (max_old - min_old)) * (value - max_old)
                + max_new,
            )

        # Round to integer steps
        car_gen_steps = np.rint(car_gen_steps)

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        route_file_path = os.path.join(output_dir, "episode_routes.rou.xml")

        # Write the route file
        with open(route_file_path, "w") as routes:
            # Define vehicle type and routes
            self._write_route_file_header(routes)

            # Add vehicle definitions
            for car_counter, step in enumerate(car_gen_steps):
                # Determine if vehicle goes straight (75%) or turns (25%)
                straight_or_turn = np.random.uniform()

                if straight_or_turn < 0.75:  # Straight movement
                    self._add_straight_vehicle(routes, car_counter, step)
                else:  # Turning movement
                    self._add_turning_vehicle(routes, car_counter, step)

            # Close routes file
            routes.write("</routes>")

    def _write_route_file_header(self, routes_file) -> None:
        """Write the header section of the route file with vehicle type and route definitions.

        Args:
            routes_file: Open file object for writing
        """
        routes_file.write(
            """<routes>
    <vType accel="1.0" decel="4.5" id="standard_car" length="5.0" minGap="2.5" maxSpeed="25" sigma="0.5" />

    <route id="W_N" edges="W2TL TL2N"/>
    <route id="W_E" edges="W2TL TL2E"/>
    <route id="W_S" edges="W2TL TL2S"/>
    <route id="N_W" edges="N2TL TL2W"/>
    <route id="N_E" edges="N2TL TL2E"/>
    <route id="N_S" edges="N2TL TL2S"/>
    <route id="E_W" edges="E2TL TL2W"/>
    <route id="E_N" edges="E2TL TL2N"/>
    <route id="E_S" edges="E2TL TL2S"/>
    <route id="S_W" edges="S2TL TL2W"/>
    <route id="S_N" edges="S2TL TL2N"/>
    <route id="S_E" edges="S2TL TL2E"/>
"""
        )

    def _add_straight_vehicle(self, routes_file, car_counter: int, step: float) -> None:
        """Add a straight-moving vehicle to the route file.

        Args:
            routes_file: Open file object for writing
            car_counter: Vehicle counter/ID
            step: Simulation step when vehicle enters
        """
        # Choose a random straight route (1-4)
        route_straight = np.random.randint(1, 5)

        if route_straight == 1:
            # West to East
            routes_file.write(
                f'    <vehicle id="W_E_{car_counter}" type="standard_car" route="W_E" depart="{step}" departLane="random" departSpeed="10" />\n'
            )
        elif route_straight == 2:
            # East to West
            routes_file.write(
                f'    <vehicle id="E_W_{car_counter}" type="standard_car" route="E_W" depart="{step}" departLane="random" departSpeed="10" />\n'
            )
        elif route_straight == 3:
            # North to South
            routes_file.write(
                f'    <vehicle id="N_S_{car_counter}" type="standard_car" route="N_S" depart="{step}" departLane="random" departSpeed="10" />\n'
            )
        else:
            # South to North
            routes_file.write(
                f'    <vehicle id="S_N_{car_counter}" type="standard_car" route="S_N" depart="{step}" departLane="random" departSpeed="10" />\n'
            )

    def _add_turning_vehicle(self, routes_file, car_counter: int, step: float) -> None:
        """Add a turning vehicle to the route file.

        Args:
            routes_file: Open file object for writing
            car_counter: Vehicle counter/ID
            step: Simulation step when vehicle enters
        """
        # Choose a random turning route (1-8)
        route_turn = np.random.randint(1, 9)

        if route_turn == 1:
            # West to North
            routes_file.write(
                f'    <vehicle id="W_N_{car_counter}" type="standard_car" route="W_N" depart="{step}" departLane="random" departSpeed="10" />\n'
            )
        elif route_turn == 2:
            # West to South
            routes_file.write(
                f'    <vehicle id="W_S_{car_counter}" type="standard_car" route="W_S" depart="{step}" departLane="random" departSpeed="10" />\n'
            )
        elif route_turn == 3:
            # North to West
            routes_file.write(
                f'    <vehicle id="N_W_{car_counter}" type="standard_car" route="N_W" depart="{step}" departLane="random" departSpeed="10" />\n'
            )
        elif route_turn == 4:
            # North to East
            routes_file.write(
                f'    <vehicle id="N_E_{car_counter}" type="standard_car" route="N_E" depart="{step}" departLane="random" departSpeed="10" />\n'
            )
        elif route_turn == 5:
            # East to North
            routes_file.write(
                f'    <vehicle id="E_N_{car_counter}" type="standard_car" route="E_N" depart="{step}" departLane="random" departSpeed="10" />\n'
            )
        elif route_turn == 6:
            # East to South
            routes_file.write(
                f'    <vehicle id="E_S_{car_counter}" type="standard_car" route="E_S" depart="{step}" departLane="random" departSpeed="10" />\n'
            )
        elif route_turn == 7:
            # South to West
            routes_file.write(
                f'    <vehicle id="S_W_{car_counter}" type="standard_car" route="S_W" depart="{step}" departLane="random" departSpeed="10" />\n'
            )
        elif route_turn == 8:
            # South to East
            routes_file.write(
                f'    <vehicle id="S_E_{car_counter}" type="standard_car" route="S_E" depart="{step}" departLane="random" departSpeed="10" />\n'
            )
