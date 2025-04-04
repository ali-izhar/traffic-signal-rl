# Traffic Signal Control Data

This directory contains the data used for training and evaluating reinforcement learning models for traffic signal control.

## Directory Structure

- `raw/`: Contains raw traffic data (if you're using real-world datasets)
- `processed/`: Contains processed data ready for training
- `simulation/`: Contains SUMO simulation configurations for generating synthetic traffic data

## Simulation Data

The `simulation/` directory contains files for generating synthetic traffic data using the SUMO traffic simulator:

### Networks

The `simulation/networks/` directory contains:

- **Basic Intersection**: `single_intersection.net.xml` and related files defining a 4-way intersection
- **Traffic Scenarios**:
  - `low_traffic.rou.xml`: Light traffic conditions
  - `high_traffic.rou.xml`: Heavy traffic conditions
  - `variable_demand.rou.xml`: Time-varying traffic patterns (morning rush, midday, evening rush)

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

## Using the Data for Training

The generated traffic data can be used directly by the Reinforcement Learning environment via the SUMO-RL or TraCI interfaces. The environment will interact with SUMO to:

1. Observe traffic states (queue lengths, waiting times, etc.)
2. Execute traffic signal control actions
3. Receive rewards based on traffic performance

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