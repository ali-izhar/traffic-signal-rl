# Traffic Signal Control Data

This directory contains the data used for training and evaluating reinforcement learning models for traffic signal control.

## Directory Structure

- `raw/`: Contains raw traffic data (if you're using real-world datasets)
- `processed/`: Contains processed data ready for training
- `simulation/`: Contains SUMO simulation configurations for generating synthetic traffic data

## Simulation Data

The `simulation/` directory contains files for generating synthetic traffic data using the SUMO traffic simulator:

### Networks

The `simulation/networks/` directory contains several configuration files that collectively define the traffic simulation:

#### Network Definition Files
- **`single_intersection.nod.xml`**: Defines the nodes (intersections) in the network. Contains 5 nodes: one central intersection with traffic lights and four outer nodes.
- **`single_intersection.edg.xml`**: Defines the edges (roads) connecting the nodes. Each approach has two lanes with a speed limit of 13.89 m/s (50 km/h).
- **`single_intersection.con.xml`**: Defines the connections between lanes at the intersection, specifying allowed turning movements (straight, right, and left).
- **`single_intersection.tll.xml`**: Defines the traffic light logic with four phases (N-S green, N-S yellow, E-W green, E-W yellow).
- **`single_intersection.netccfg`**: Configuration file for SUMO's netconvert tool that combines the node, edge, and connection files.
- **`single_intersection.net.xml`**: The compiled network file created by netconvert from the above files.

#### Traffic Demand Files
- **`low_traffic.rou.xml`**: Defines vehicle types and traffic flows for a low-traffic scenario (probabilities 0.01-0.05).
- **`high_traffic.rou.xml`**: Defines a high-traffic scenario with higher probabilities (0.10-0.25).
- **`variable_demand.rou.xml`**: Implements time-varying traffic patterns:
  - Morning peak (0-1200s): Heavy north→south and east→west flows
  - Midday (1200-2400s): Balanced flows in all directions
  - Evening peak (2400-3600s): Heavy south→north and west→east flows

#### Simulation Configuration Files
- **`single_intersection.sumocfg`**: Main configuration file for SUMO that references the network file, route files, and simulation parameters.
- **`low_traffic.sumocfg`**, **`high_traffic.sumocfg`**, **`variable_demand.sumocfg`**: Specific configurations for each traffic scenario.

### Running the Simulation

1. Install SUMO (Simulation of Urban MObility):
   - Download from: https://sumo.dlr.de/docs/Downloads.html
   - Add SUMO to your PATH

2. Generate the network file:
   ```
   cd src\data\simulation\networks
   netconvert -c single_intersection.netccfg
   ```

3. Generate traffic data:
   ```
   cd src\data\simulation
   python generate_traffic_data.py
   ```

4. Run a simulation:
   ```
   sumo-gui -c simulation/networks/variable_demand.sumocfg
   ```

### Traffic Data Generation Script

The `generate_traffic_data.py` script creates the traffic scenario files programmatically:

- **`create_low_traffic_scenario()`**: Generates low traffic XML with arrival rates of 0.05-0.10 vehicles/second.
- **`create_high_traffic_scenario()`**: Generates high traffic XML with arrival rates of 0.20-0.25 vehicles/second.
- **`create_variable_demand_scenario()`**: Creates time-varying traffic patterns with morning, midday, and evening periods.
- **`create_sumo_config()`**: Creates the SUMO configuration files for each scenario.

Each function defines vehicle types (passenger cars, trucks), routes (all possible paths through intersection), and traffic flows (probability-based vehicle generation). The traffic generation follows the classic four-step transportation planning model approach, but simplified for a single intersection.

> **Important Note**: Running the Python script only generates configuration files in the `networks` directory, not actual training data. These files are used by SUMO to create a simulation environment. The actual training data (states, actions, rewards) is generated on-the-fly during RL training when your agent interacts with the SUMO simulation via TraCI. If you need to store this interaction data for offline training or analysis, you would need to add code to your RL environment that collects these interactions and saves them to the `processed/` directory.

## Using the Data for Training

The generated traffic data can be used directly by the Reinforcement Learning environment via the SUMO-RL or TraCI interfaces. The environment will interact with SUMO to:

1. Observe traffic states (queue lengths, waiting times, etc.)
2. Execute traffic signal control actions
3. Receive rewards based on traffic performance

For RL training, the simulation provides:
- **States**: Vehicle counts, waiting times, queue lengths at each approach
- **Actions**: Traffic signal phase changes
- **Rewards**: Metrics based on traffic efficiency (reduced waiting times, shorter queues)

## Real-World Datasets (Optional)

If you want to use real-world data, place the following datasets in the `raw/` directory:

- **NGSIM**: https://ops.fhwa.dot.gov/trafficanalysistools/ngsim.htm
- **PeMS**: https://pems.dot.ca.gov/
- **UTDOT**: https://udottraffic.utah.gov/

These datasets require preprocessing scripts (available in the `scripts/` directory) to convert to the format expected by the RL environment.

## Data Format

Processed data for training follows this structure:
- State features: Queue lengths, waiting times, traffic density for each approach
- Action space: Traffic signal phase configurations
- Rewards: Negative weighted sum of queue lengths, waiting times, and phase changes 