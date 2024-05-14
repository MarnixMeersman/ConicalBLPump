import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import pandas as pd
from scipy.signal import savgol_filter
plt.style.use(['science','ieee', 'high-vis'])

smooth_width = 30



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

    index = np.where(speeds >= 2.6)[0][0]

    return times[index:], shift_to_zero(masses[index:]), speeds[index:], mass_rates[index:]
def get_interpolated_points(times, masses, samples=1000, start=0, end=6.4):
    '''
    :param times: Function is verified
    :param masses:
    :param samples:
    :param start:
    :param end:
    :return:
    '''
    time_points = np.linspace(start, end, samples)
    mass_points = np.interp(time_points, times, masses)
    return time_points, mass_points

def get_mass_and_time_array(experiment_number):
    '''
    This function returns a numpy array of mass experimental_data_old for a given experiment number
    get_mass_array(35)[0][0] returns the first mass experimental_data_old for the first experiment in experiment 35
    :param experiment_number:
    :return:
    '''
    mass_lst = []
    times_lst = []
    for i in range(10):
        filename = f'experimental_data_old/experiment_data_{experiment_number}_{i}.csv'
        loaded_data = load_experiment_data(filename)
        times, masses = get_interpolated_points(loaded_data[0], loaded_data[1])
        mass_lst.append(list(masses))
        times_lst.append(list(times))

    print("Experiment Number:", experiment_number)
    print("Mass Array Shape:", np.array(mass_lst).shape)
    print("Times Array Shape:", np.array(times_lst).shape)
    return np.array(mass_lst), np.array(times_lst)

def get_mass_mean_std_array(experiment_number):
    mass_array, times_array = get_mass_and_time_array(experiment_number)
    mean = np.mean(mass_array, axis = 0)
    std = np.std(mass_array, axis = 0)
    return times_array[0], mean, std

def get_mass_derivative_mean_std_array(experiment_number):
    mass_array, times_array = get_mass_and_time_array(experiment_number)
    for row in mass_array:
        fit = np.polyfit(times_array, mass_array, 2)
        f = np.poly1d(fit)
        fx = np.poly2d(fit)
    derivative_array = np.gradient(mass_array, axis=1) # Convert to mL/s
    mean = np.mean(derivative_array, axis = 0)
    std = np.std(derivative_array, axis = 0)

    return times_array[0], mean, std



    mean = np.mean(derivative_array, axis=0)
    std = np.std(derivative_array, axis=0)

    return times_array[0], mean, std

def main():
    # plot all figures together into a single plot.
    # erroe area (from std) for each experiment shown using fill_between
    # legend for each experiment
    # x-axis is time
    # y-axis is Total Expelled Volume (mL)
    # secondary x-axis is Radial Velocity (rad/s)
    fig, ax1 = plt.subplots()
    for i in [35, 40, 45, 50, 55, 60]:
        times, mean, std = get_mass_derivative_mean_std_array(i)
        # ax1.errorbar(times, mean, yerr=std, label=f'Experiment {i}', linewidth=0.7, fmt='-o', capsize=3)
        plt.fill_between(times, mean - std, mean + std, alpha=0.2)
        plt.plot(times, mean, label=f'Cone Angle: {i}°', linewidth=0.7)  # SMOOTHED
        # plt.plot(times, mean, label=f'Cone Angle: {i}°', linewidth=0.7)  # UN-SMOOTHED
    # Function to convert time to radial velocity
    def time_to_radial_velocity(times):
        return times * 2.5 / 0.1  # 2.5 rad/s per 0.1 seconds (speed increment)

    def time_to_time(times):
        return times
    secax = ax1.secondary_xaxis('top', functions=(time_to_radial_velocity, time_to_time))
    secax.set_xlabel('Radial Velocity (rad/s)')
    ax1.legend()
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Massflow (mL/s)')
    # adjust x lim
    plt.xlim(0, 6.3)
    plt.savefig(f'figures/combined/MassFlow_Velocity.png')
    plt.close(fig)  # Close the figure to prevent display issues

    fig, ax1 = plt.subplots()
    for i in [35, 40, 45, 50, 55, 60]:
        times, mean, std = get_mass_mean_std_array(i)
        # ax1.errorbar(times, mean, yerr=std, label=f'Experiment {i}', linewidth=0.7, fmt='-o', capsize=3)
        plt.fill_between(times, mean - std, mean + std, alpha=0.2)
        plt.plot(times, mean, label=f'Cone Angle: {i}°', linewidth=0.7)
    # Function to convert time to radial velocity
    def time_to_radial_velocity(times):
        return times * 2.5 / 0.1  # 2.5 rad/s per 0.1 seconds (speed increment)

    def time_to_time(times):
        return times
    secax = ax1.secondary_xaxis('top', functions=(time_to_radial_velocity, time_to_time))
    secax.set_xlabel('Radial Velocity (rad/s)')
    ax1.legend()
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Cumalative Expelled Volume (mL)')
    # adjust x lim
    plt.xlim(0, 6.3)
    plt.savefig(f'figures/combined/Mass_Velocity.png')
    # plt.show()


    for i in [35, 40, 45, 50, 55, 60]:
        fig, ax1 = plt.subplots()
        times, mean, std = get_mass_derivative_mean_std_array(i)
        # ax1.errorbar(times, mean, yerr=std, label=f'Experiment {i}', linewidth=0.7, fmt='-o', capsize=3)
        plt.fill_between(times, mean - std, mean + std, alpha=0.2)
        plt.plot(times, mean, label=f'Cone Angle: {i}°', linewidth=0.7)  # SMOOTHED

        # Function to convert time to radial velocity
        def time_to_radial_velocity(times):
            return times * 2.5 / 0.1  # 2.5 rad/s per 0.1 seconds (speed increment)

        def time_to_time(times):
            return times

        secax = ax1.secondary_xaxis('top', functions=(time_to_radial_velocity, time_to_time))
        secax.set_xlabel('Radial Velocity (rad/s)')
        ax1.legend()
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Massflow (mL/s)')
        # adjust x lim
        plt.xlim(0, 6.3)
        plt.savefig(f'figures/combined/MassFlow_Velocity_{i}.png')
        plt.figure().clear()

        for i in [35, 40, 45, 50, 55, 60]:
            fig, ax1 = plt.subplots()
            times, mean, std = get_mass_mean_std_array(i)
            # ax1.errorbar(times, mean, yerr=std, label=f'Experiment {i}', linewidth=0.7, fmt='-o', capsize=3)
            plt.fill_between(times, mean - std, mean + std, alpha=0.2)
            plt.plot(times, mean, label=f'Cone Angle: {i}°', linewidth=0.7)  # SMOOTHED

            # plt.plot(times, mean, label=f'Cone Angle: {i}°', linewidth=0.7)  # UN-SMOOTHED

            # Function to convert time to radial velocity
            def time_to_radial_velocity(times):
                return times * 2.5 / 0.1  # 2.5 rad/s per 0.1 seconds (speed increment)

            def time_to_time(times):
                return times

            secax = ax1.secondary_xaxis('top', functions=(time_to_radial_velocity, time_to_time))
            secax.set_xlabel('Radial Velocity (rad/s)')
            ax1.legend()
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Cumalative Expelled Volume (mL)')
            # adjust x lim
            plt.xlim(0, 6.3)
            plt.savefig(f'figures/combined/Mass_Velocity_{i}.png')
            plt.figure().clear()




if __name__ == '__main__':
    main()

