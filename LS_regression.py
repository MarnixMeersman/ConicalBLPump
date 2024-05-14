import numpy as np
import numpy
import matplotlib.pyplot as plt
import scienceplots
import pandas as pd
from scipy.signal import savgol_filter
from itertools import chain
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
        masses = loaded_data[1]
        times = loaded_data[0]
        mass_lst.append(list(masses))
        times_lst.append(list(times))

    return mass_lst, times_lst

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
    for i in [35, 40, 45, 50, 55, 60]:
        masses, times = get_mass_and_time_array(i)

        xData = list(chain.from_iterable(times))
        yData = list(chain.from_iterable(masses))

        polynomialOrder = 3  # example quadratic

        # curve fit the test experimental_data_old
        fittedParameters = numpy.polyfit(xData, yData, polynomialOrder)
        print('Fitted Parameters:', fittedParameters)

        modelPredictions = numpy.polyval(fittedParameters, xData)
        absError = modelPredictions - yData

        SE = numpy.square(absError)  # squared errors
        MSE = numpy.mean(SE)  # mean squared errors
        RMSE = numpy.sqrt(MSE)  # Root Mean Squared Error, RMSE
        Rsquared = 1.0 - (numpy.var(absError) / numpy.var(yData))
        print('RMSE:', RMSE)
        print('R-squared:', Rsquared)

        print()

        ##########################################################
        # graphics output section
        def ModelAndScatterPlot(graphWidth, graphHeight):
            f = plt.figure(figsize=(graphWidth / 100.0, graphHeight / 100.0), dpi=50)
            axes = f.add_subplot(111)

            # first the raw experimental_data_old as a scatter plot
            axes.plot(xData, yData, 'D')

            # create experimental_data_old for the fitted equation plot
            xModel = numpy.linspace(min(xData), max(xData))
            yModel = numpy.polyval(fittedParameters, xModel)

            # now the model as a line plot
            axes.plot(xModel, yModel)

            axes.set_xlabel('X Data')  # X axis experimental_data_old label
            axes.set_ylabel('Y Data')  # Y axis experimental_data_old label

            plt.show()
        ModelAndScatterPlot(800, 600)





if __name__ == '__main__':
    main()

