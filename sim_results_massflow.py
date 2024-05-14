import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots
from sklearn.linear_model import LinearRegression
import matplotlib.ticker as ticker
import seaborn as sns



# Load the SciencePlots style
plt.style.use(['science', 'high-vis', 'grid'])

# Load data from a CSV file
df = pd.read_csv('sim_data/sim_results.csv')


# Rename region 5 for clarity in the plot
df['Region'] = df['Region'].replace({5: 'Flowrate of all regions combined'})

# Set up the plotting environment
fig, axs = plt.subplots(2, 1, figsize=(6, 6))  # Create 2 subplots, sharing x-axis

# Plot each region
regions = df['Region'].unique()
for region in regions:
    subset = df[df['Region'] == region]
    if region == 'Flowrate of all regions combined':
        # Plot 'Sum' on the first subplot
        axs[0].plot(subset['Angular Velocity (rad/s)'], subset['Volumetric Flowrate (ml/min)'], label=region)
        axs[0].set_ylabel('Volumetric Flowrate (ml/min)')
        axs[0].legend()
    else:
        # Plot other regions on the second subplot
        axs[1].plot(subset['Angular Velocity (rad/s)'], subset['Volumetric Flowrate (ml/min)'], label=region)
        axs[1].set_xlabel('Angular Velocity (rad/s)')
        axs[1].set_ylabel('Volumetric Flowrate (ml/min)')
        axs[1].legend(title='Region')

# Save the plot
plt.savefig('volumetric_flowrate_vs_angular_velocity.png', dpi=1000)

# Show the plot
plt.show()