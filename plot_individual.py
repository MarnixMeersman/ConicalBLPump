import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from scipy.interpolate import splrep, splev
plt.style.use(['science','ieee', 'muted'])

def shift_to_zero(data):
    """Shifts all elements in the array such that the minimum value becomes 0."""
    return data - np.min(data)

def load_experiment_data(filename):
    # Load the experimental_data_old from the CSV file
    data = np.genfromtxt(filename, delimiter=',', skip_header=1)

    # Separate the columns
    times = data[:, 0]
    masses = data[:, 1] * -1
    speeds = data[:, 2]

    mass_rates = np.gradient(np.convolve(masses, np.ones(100)/100, mode='same'))

    index = np.where(times >= 0.05)[0][0]

    return times[index:], shift_to_zero(masses[index:]), speeds[index:], mass_rates[index:]

def plot_experiment_data(experiment_number):
    fig, ax1 = plt.subplots()


    for i in range(10):
        filename = f'experimental_data_old/experiment_data_{experiment_number}_{i}.csv'
        times, masses, speeds, mass_rates = load_experiment_data(filename)
        ax1.plot(times, masses, label=f'Experiment {i}', linewidth=0.7)

    # Adding a secondary x-axis to show the radial velocity
    def time_to_radial_velocity(times):
        return times * 2.5/0.1 # 2.5 rad/s per 0.1 seconds (speed increment)
    def time_to_time(times):
        return times

    # Adding legend, labels, and title to the first x-axis
    ax1.legend()
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Cumulative Expelled Volume (mL)')
    secax = ax1.secondary_xaxis('top', functions=(time_to_radial_velocity, time_to_time))
    secax.set_xlabel('Radial Velocity (rad/s)')
    plt.xlim(0, 6.3)


    plt.savefig(f'figures/individual_experiments/Volume_vs_Time_{experiment_number}.png')
    plt.close(fig)  # Close the figure to prevent display issues in some environments

for i in [35, 40, 45, 50, 55, 60]:
    plot_experiment_data(i)
