# -*- coding: utf-8 -*-
"""
                    ____________________________________________
                    PHYS20161 - Final Assignment - Nuclear Decay
                    ____________________________________________
        The purpose of this code is to find the half-life and decay constant of
        rubidium-79 and strontium-79 with uncertainties. This is done via a best
        fit parameter searching routine which finds the best parameter values that
        would match the data provided on the behaviour of the element's nuclear
        properties

        Last Updated: 14/12/2022
        Username: u95060ss
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import Avogadro
from scipy.optimize import curve_fit

INITIAL_STRONTIUM_NUCLEI_COUNT = Avogadro*10**-6
INITIAL_RUBIDIUM_DECAY_CONSTANT = 0.0005  # SI Units S^-1
INITIAL_STRONTIUM_DECAY_CONSTANT = 0.005  # SI Units, S^-1


def read_in():
    """
    Reads in two data files and combines them into one.
    It validates the files by flagging for errors
    -------
ARGUMENTS: None

RETURNS:
    file_ab : ARRAY
        An array that combines two files, file_a and file_b.
        It contains 3 columns of information regarding Rubidium-79's decay:
        time, activity, and activity uncertainty.

    """

    np.seterr(all='warn', over='raise')
    file_ab = np.zeros((0, 3))
    file_a = np.genfromtxt('Nuclear_data_1.csv',
                           delimiter=',', skip_header=1)
    file_b = np.genfromtxt('Nuclear_data_2.csv',
                           delimiter=',', skip_header=1)

    file_ab = np.vstack((file_ab, file_a))
    file_ab = np.vstack((file_ab, file_b))

    return file_ab


def rubidium_activity(time, strontium_decay_constant, rubidium_decay_constant):
    """
    Returns the activity value for rubidium-79.

    ----------
ARGUMENTS:
    time (seconds): FLOAT, ARRAY
        Time taken for rubidium-79 to be at a certain activity level.
        Floats will be used when the user wishes to find the specific activity level
        at an exact time.
        Arrays will be used when a multitude of activity level values are required

    strontium_decay_constant (per second) : FLOAT
        The precise value of the decay constant of strontium-79.

    rubidium_decay_constant (per second): FLOAT
        The precise value of the decay constant of rubidium-79..

RETURNS:
    rubidium_decay_constant*rubidium_nuclei_count (bequerels): FLOAT, ARRAY
        The value produced here is the rubidium activity which can either be a float or an array
        depending on whether the time argument was a float or an array respectively.

    """
    rubidium_nuclei_count = INITIAL_STRONTIUM_NUCLEI_COUNT * \
        (strontium_decay_constant/(rubidium_decay_constant-strontium_decay_constant)) * \
        (math.e**(-strontium_decay_constant*time) -
         math.e**(-rubidium_decay_constant*time))
    return rubidium_decay_constant*rubidium_nuclei_count


def data_is_not_null(datapoints):
    """
    Produces a boolean value of True if the input does not have null values and False if it does.

    This is useful in conjunction with filter functions which can index where the null values in
    a datafile are.
    ----------
ARGUMENTS:
    datapoints : FLOAT
        A float taken from a file in array format.
        Represents the value of time, activity, or activity uncertainty.
RETURNS:
    bool : BOOLEAN
        Outputs a boolean value that can be used for indexing in other functions.
    """
    if np.isnan(datapoints):
        return False
    return True


def remove_null_values(file_ab):
    """
    Filters out all the null values from the desired file.
    This is done by creating a new array that will only contain rows of data
    that do not have a single null value.
    The units of the file_ab are assumed to be:
    time (hours), activity (terabequerel), activity uncertainty (terabequerel)

    They get converted to time (seconds), activity (bequerel), activity uncertainty (bequerel)
    ----------
ARGUMENTS:
    file_ab : ARRAY
        An array that combines two files, file_a and file_b.
        It contains 3 columns of information regarding Rubidium-79's decay:
        time, activity, and activity uncertainty.
RETURNS:
    null_filtered_data : ARRAY
        Contains null filtered data of file_ab.
    """
    null_filtered_data = np.zeros((0, 3))

    for row in file_ab:
        if data_is_not_null(row[0]) and data_is_not_null(row[1]) and data_is_not_null(row[2]):
            row[0] = row[0]*3600
            row[1] = row[1]*10**12
            row[2] = row[2]*10**12
            temp = np.array([row[0], row[1], row[2]])
            null_filtered_data = np.vstack((null_filtered_data, temp))
    return null_filtered_data


def remove_outliers(trial_activity, time, activity, activity_uncertainty):
    """
    Finds the outliers in the arguments given and creates a new array without them.
    ----------
ARGUMENTS:
    trial_activity (bequerels): ARRAY
        Rubidium-79's activity which is calculated using the fitted parameters
        (decay constants) produced by scipy's in-built curve_fit function.
        It is calculated while the outliers are still included so this activity value
        is used only temporarily to compare with the experimental data. This is so that
        the outliers can be removed

    time (seconds): ARRAY
        Time taken for rubidium-79 to be at a certain activity level.
        The array will be matched to an index that specifies the location of the
        non-outliers so that they can be transposed into a new array

    activity (bequerels): ARRAY
        Rubidium-79's activity measured experimentally.
        The array will be matched to an index that specifies the location of the
        non-outliers so that they can be transposed into a new array

    activity_uncertainty (bequerels): ARRAY
        The uncertainty associated with each datapoint of activity.
        The array will be matched to an index that specifies the location of the
        non-outliers so that they can be transposed into a new array
RETURNS:
    outlier_filtered_data : ARRAY
        An array containing array of file_ab that has filtered out all the null values
        and outliers including 0 and infinite values.

    """
    filtered_indexes = np.where(
        np.abs(trial_activity - activity) < 3*activity_uncertainty)
    # This also filters out zero erros because a positive number is always greater than 0
    # print(filtered_indexes)

    outlier_filtered_data = np.zeros((0, 3))

    outlier_filtered_time = time[filtered_indexes]

    outlier_filtered_activity = activity[filtered_indexes]

    outlier_filtered_activity_uncertainty = activity_uncertainty[filtered_indexes]
    # print(outlier_filtered_time.shape)
    # print(outlier_filtered_activity.shape)
    # print(outlier_filtered_activity_uncertainty.shape)
    temp = np.transpose([outlier_filtered_time, outlier_filtered_activity,
                        outlier_filtered_activity_uncertainty])
    outlier_filtered_data = np.vstack((outlier_filtered_data, temp))

    return outlier_filtered_data


def plot_rubidium_activity(time, activity,
                           activity_uncertainty,
                           strontium_decay_constant,
                           rubidium_decay_constant):
    """
    Plots rubidium-79's activity against time
    Automatically saves the plot produced as
    Rubidium-79 Activity Curve
    ----------
ARGUMENTS:
    time (seconds): FLOAT, ARRAY
        Time taken for rubidium-79 to be at a certain activity level.
        Floats will be used when the user wishes to find the specific activity level
        at an exact time.
        Arrays will be used when a multitude of activity level values are required.

    activity (bequerels): ARRAY
        Rubidium-79's activity measured experimentally.

    activity_uncertainty (bequerels): ARRAY
        The uncertainty associated with each datapoint of activity.

    strontium_decay_constant (per second): FLOAT
        The precise value of the decay constant of strontium-79.

    rubidium_decay_constant (per second): FLOAT
        The precise value of the decay constant of rubidium-79.
RETURNS: None
    """

    figure = plt.figure()
    axes = figure.add_subplot()

    axes.set_xlabel(r'$ t \,\, (\mathrm{s})$')

    axes.set_ylabel(r'$A_c \, \, (\mathrm{Bq})$')

    axes.set_title('Rubidium-79 Activity Curve')

    axes.errorbar(time, activity, activity_uncertainty, ls='', marker='o',
                  ecolor='red', markersize=5, label='Datapoints with errorbars')

    fitted_time = np.linspace(0, np.max(time), 100000)

    fitted_activity = rubidium_activity(
        fitted_time, strontium_decay_constant, rubidium_decay_constant)

    axes.plot(fitted_time, fitted_activity,
              label='Curve of activity with fitted parameters')

    axes.grid()

    axes.legend(loc='best', fancybox=True, framealpha=0)

    plt.savefig('Rubidium-79 Activity Curve', dpi=600)

    plt.show()


def chi_squared_value(independent_variable, dependent_variable, error,
                      predicted_dependent_variable, dataset):
    """
    Calculates the reduced chi-squared of the variables put in against their ideal value.
    For the purpose of this code, it will find the reduced chi-squared of the ideal rubidium-79
    curve fit against the actual uncertainties .
    ----------
ARGUMENTS:
    independent_variable : ARRAY
        The independent variable of the inputs.
        In the case of rubidium-79 activity,
        that will be time.

    dependent_variable : ARRAY
        The dependent variable of the inputs.
        In the case of rubidium-79 activity,
        that will be activity.

    error : ARRAY
        The uncertainties of the values that are being put through the function.
        They will used to reduce the chi-squared value

    predicted_dependent_variable : ARRAY
        The ideal value for the dependent variable as predicted by a model.
        In the case of rubidium-79, the equation for its activity will suffice

    dataset : ARRAY
        The filtered array that contains only useful values.
        Any outliers should have been removed beforehand.
RETURNS:None
    """
    chi_squared = np.sum(
        ((dependent_variable-predicted_dependent_variable)**2)/(error**2))

    row_numbers = len(independent_variable)
    # -1 is needed to avoid the uncertainty column to be counted as a parameter
    column_numbers = len(dataset[0])-1

    degrees_of_freedom = row_numbers-column_numbers
    reduced_chi_squared = chi_squared/degrees_of_freedom

    print(
        "The reduced chi-squared value for this plot is {:.2f}".format(reduced_chi_squared))
    return reduced_chi_squared


def half_life(final_fit, final_covariance):
    """
    Calculates and prints the half life of rubidium-79 and strontium-79.
    ----------
ARGUMENTS:
    final_fit (per second): ARRAY
        Contains values of the best fitted decay constants.

    final_covariance (per second): ARRAY
        Contains a covariance matrix which can be used
        to find uncertainties of the fitted decay constants.
RETURNS:
    rubidium_half_life (seconds): FLOAT
        The approximate time taken for rubidium to decay.

    strontium_half_life (seconds): FLOAT
        The approximate time taken for strontium to decay.
    """
    rubidium_half_life = np.log(2)/final_fit[1]
    strontium_half_life = np.log(2)/final_fit[0]

    rubidium_half_life_uncertainty = rubidium_half_life/60 * \
        (np.sqrt(final_covariance[1][1])/final_fit[1])

    strontium_half_life_uncertainty = strontium_half_life/60 * \
        (np.sqrt(final_covariance[0][0])/final_fit[0])

    print("Rubidium-79 half-life =",
          f"{rubidium_half_life/60:.3} \u00B1 {rubidium_half_life_uncertainty:.2} minutes")

    print("Strontium-79 half-life =",
          f"{strontium_half_life/60:.3} \u00B1 {strontium_half_life_uncertainty:.2} minutes")

    return rubidium_half_life, strontium_half_life


def main():
    """
    A main function calls all the other functions in order to find the
    best fitted decay constants which the other functions then use to
    find other values such as half life and reduced chi-squared.

    This function also prints out the specific activity level for the 90 minute marker
    and gives the user the option to save the filtered array as a .csv data file
    """

    file_ab = read_in()

    null_filtered_data = remove_null_values(file_ab)

    fit, _ = curve_fit(rubidium_activity, null_filtered_data[:, 0], null_filtered_data[:, 1], p0=[
        INITIAL_STRONTIUM_DECAY_CONSTANT, INITIAL_RUBIDIUM_DECAY_CONSTANT])
    # Note to demonstrators: Don't put in absolute_sigma the first time
    # since that will confuse curve_fit with the outliers

    initial_strontium_decay_constant, initial_rubidium_decay_constant = fit

    trial_activity = rubidium_activity(
        null_filtered_data[:, 0], initial_strontium_decay_constant, initial_rubidium_decay_constant)

    final_data = remove_outliers(
        trial_activity, null_filtered_data[:, 0],
        null_filtered_data[:, 1], null_filtered_data[:, 2])

    final_fit, final_covariance = curve_fit(rubidium_activity,
                                            final_data[:, 0], final_data[:, 1],
                                            p0=[*fit], absolute_sigma=True,
                                            sigma=final_data[:, 2], check_finite=True)
    plot_rubidium_activity(
        final_data[:, 0], final_data[:, 1], final_data[:, 2], final_fit[1], final_fit[0])
    # while number of outliers is greater than 0
    ideal_activity = rubidium_activity(
        final_data[:, 0], final_fit[1], final_fit[0])

    chi_squared_value(final_data[:, 0], final_data[:, 1],
                      final_data[:, 2], ideal_activity, final_data)

    half_life(final_fit, final_covariance)

    time_specific_activity = rubidium_activity(
        5400, final_fit[1], final_fit[0])

    print("Rubidium-79 decay constant = {:.3g} \u00B1 {:.2g} per second".format(
        final_fit[1], np.sqrt(final_covariance[1][1])))

    print("Strontium-79 decay constant = {:.3g} \u00B1 {:.2g} per second".format(
        final_fit[0], np.sqrt(final_covariance[0][0])))

    print("The activity level at the 90 minute marker is {:.3g} terabequerel".format(
        time_specific_activity/(10**12)))

    save_data = input(
        "Do you want to save the array for the filtered dataset as a csv file? "
        + "It will be saved as final_data.csv (yes/no) ")

    if save_data == "yes":
        np.savetxt("final_data.csv", final_data, delimiter=",")
    else:
        print("Array has not been saved")


if __name__ == "__main__":
    main()
