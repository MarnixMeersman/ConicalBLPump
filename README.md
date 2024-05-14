# Conical Boundary Layer Pump Experiment and Simulation

This repository contains the code and data for the Conical Boundary Layer Pump experiment and simulation. The project aims to understand the relationship between the cone angle and the required angular velocity to initiate a flow into the disk.

## Repository Structure

- `sim_data/`: This directory contains the simulation data in CSV format.
- `experimental_data/`: This directory contains the experimental data in CSV format.
- `sim_results_massflow.py`: This Python script is used to generate plots from the simulation data.
- `main.py`: This Python script is used to control the experiment and collect data.

## Experiment

The experiment focuses on the cone angle as it is the most defining geometrical variable. It has the largest effect on the ability to accelerate flow tangentially versus radially. As the cone angle increases, the walls of the concentric cones become steeper, meaning less of the accelerating force component is directed radially outward and more is directed upwards.

The experimental setup is schematically depicted in the following figure:

For each cone angle, we increased the speed up to 120 radians per second and recorded the data from the precision scale. This scale, with an accuracy of 0.001 grams, allowed us to detect significant weight drops at particular rotational velocities, indicating the initiation of flow.

## Simulation

The simulation data is stored in the `sim_data/` directory. The `sim_results_massflow.py` script is used to generate plots from this data.

## Running the Code

To run the code, you will need Python and pip installed on your system. You can install the required dependencies with the following command:

```bash
pip install -r requirements.txt
```

You can then run the `main.py` script to start the experiment:

```bash
python main.py
```

Or the `sim_results_massflow.py` script to generate plots from the simulation data:

```bash
python sim_results_massflow.py
```

