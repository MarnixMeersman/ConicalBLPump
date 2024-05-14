import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style
import matplotlib as mpl
import scienceplots
plt.style.use(['science'])
# Load experimental_data_old
data_path = 'Forces_and_moments_1.csv'
data = pd.read_csv(data_path)

# Using scienceplots styles
style.use('science')
mpl.rcParams.update({'font.size': 12})

# Plotting
fig, axs = plt.subplots(3, 1, figsize=(8, 12))
axs[0].plot(data['(ω) Rotational velocity (rad/s)'], data['TOTAL_FORCE_X'], label='Total Force X', color='b')
axs[0].set_xlabel('Rotational Velocity (rad/s)')
axs[0].set_ylabel('Total Force X')
axs[0].grid(True)
axs[0].legend()

axs[1].plot(data['(ω) Rotational velocity (rad/s)'], data['TOTAL_FORCE_Y'], label='Total Force Y', color='r')
axs[1].set_xlabel('Rotational Velocity (rad/s)')
axs[1].set_ylabel('Total Force Y')
axs[1].grid(True)
axs[1].legend()

axs[2].plot(data['(ω) Rotational velocity (rad/s)'], data['TOTAL_FORCE_Z'], label='Total Force Z', color='g')
axs[2].set_xlabel('Rotational Velocity (rad/s)')
axs[2].set_ylabel('Total Force Z')
axs[2].grid(True)
axs[2].legend()

plt.tight_layout()
plt.show()
