

"""
                                ______________________________________
                                PHYS20161 - Assignment 1 - Bouncy Ball
                                ______________________________________
                The purpose of this code is to calculate the number of bounces it takes
                for ball to stop bouncing over a certain height.
                It also finds the time taken to reach that number of bounces.

                Last Updated: 25/10/2022
                Username: u95060ss
"""

import numpy as np
GRAVITATIONAL_ACCELERATION_CONSTANT = 9.807


def calculate_height_values(initial_height, minimum_height, energy_loss_coefficient):
    """
    Returns important values of height that gets used in other functions.
    Calculates the maximum height reached after each bounce.
    -------
ARGUMENTS:
    initial_height : FLOAT
        The value of the initial height the ball drops from.

    minimum_height : FLOAT
        The value of the height at which we stop counting
        the number of bounces and any other calculations.

    energy_loss_coefficient : FLOAT
        The coefficient by which the ball loses energy
        and therefore its maximum height after each bounce.
        People can also call this, efficiency or the
        coefficient of restitution.

RETURNS:
    height_list : ARRAY
        An array with the maximum height reached after each bounce.
    """
    new_height = initial_height
    height_list = np.array([])
    while new_height > minimum_height:
        new_height *= energy_loss_coefficient
        height_list = np.append(height_list, new_height)
    print("The number of bounces before the minimum height is {}".format(
        len(height_list)-1))
    return height_list


def calculate_time_values(initial_height, height_list):
    """
    Returns the total time taken for the ball to
    get a maximum bounce height less than the minimum height from the initial height.
    -------
ARGUMENTS:
    initial_height : FLOAT
        The value of the initial height the ball drops from.

    height_list : ARRAY
        An array with the maximum height reached after each bounce.
RETURNS:
    total_time : FLOAT
        Time taken until the ball no longer bounces above the minimum height.
    """
    zeroth_bounce_time = np.sqrt(
        (2*initial_height) / GRAVITATIONAL_ACCELERATION_CONSTANT)
    bounce_time_list = np.sqrt((8*height_list[:(len(height_list)-1)]) /
                               GRAVITATIONAL_ACCELERATION_CONSTANT)
    total_time = np.sum(bounce_time_list) + zeroth_bounce_time
    print('The total time taken for all the bounces including the 0th bounce'
          + ' before the minimum height is {} seconds'.format(round(total_time, 2)))
    return total_time


def main():
    '''
    A main function that you provide all the inputs for and get a validation
    before they get used in the height and time calculating functions.

    '''
    while True:
        initial_height = input(
            'Please enter an initial height assuming units of meters:')
        try:
            initial_height = float(initial_height)
        except ValueError:
            print('Please enter a positive number')
            continue
        if initial_height <= 0:
            print('Value must be greater than 0')
            continue
        break
    while True:
        minimum_height = input(
            'Please enter the minimum height assuming units meters:')
        try:
            minimum_height = float(minimum_height)
        except ValueError:
            print('Please enter a positive number')
            continue
        if minimum_height < 0:
            print('Value must be greater than or equal to 0')
            continue
        if minimum_height >= initial_height:
            print('Minimum height must be greater than initial height')
            continue
        break
    while True:
        energy_loss_coefficient = input(
            'Please enter a coefficient of energy loss:')
        try:
            energy_loss_coefficient = float(energy_loss_coefficient)
        except ValueError:
            print('Please enter a positive number between 0 and 1')
            continue
        if energy_loss_coefficient <= 0 or energy_loss_coefficient >= 1:
            print('Value must be more than 0 and less than 1')
            continue
        break
    height_list = calculate_height_values(
        initial_height, minimum_height, energy_loss_coefficient)

    calculate_time_values(initial_height, height_list)


main()
