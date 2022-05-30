"""
The following is an algorithm performing operation on a noisy 3D spacial data. It is meant to find a best fit surface
going through the data while maintaining low loss of information. I treat it here as a logistic regression problem
utilising polynomial mapping of the feature space and gradient descent.

Only with learning rate and iterations limit set to 0.003 and 30000 respectively while having a relatively small dataset
of 40000 samples and total of 28 features after polynomial mapping the algorithm takes over 143.78 seconds to complete -
it is inefficient. Also, the lack data normalisation causes exponential function to overflow which could be fixed.
"""

import numpy as np
import matplotlib.pyplot as plt
from classification_numbers.tools.classification_functions import polynomial_feature_mapping, gradient_descent
import time


def fetch_data(x_len: int, plot_: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate data according to some function and add noise.

    :param x_len: int, number of elements to be generated in the x1 and x2 arrays - y values will have x_len**2 number
    of items
    :param plot_: bool, plot the data if true
    :return: x1, x2 and y arrays containing data about axes and values
    """
    # generate some data which include axes x1 and x2 and values y
    x1 = x2 = np.linspace(-np.pi, np.pi, x_len).reshape((1, x_len))
    y_base = np.sin(x1.T @ x2)/x1

    # generate some noise and modify the values y
    noise = np.random.normal(-0.4, .4, y_base.shape)
    y = y_base + noise

    if plot_:
        fig = plt.figure()
        ax = plt.axes(projection='3d')

        x1_m, x2_m = np.meshgrid(x1, x2)
        ax.plot_surface(x1_m, x2_m, y)

        plt.show()

    return x1, x2, y


###########################################
#             EXECUTION BLOCK             #
###########################################

start_time = time.time()

sample_dimension = 200
degree = 6
learning_rate = 0.003
iterations_lim = 30000

# create a data with random noise
ax1, ax2, values = fetch_data(sample_dimension)

# transform axes into permutations of each-others values (x_1, x_2)
ax1_m, ax2_m = np.meshgrid(ax1, ax2)
x_1 = ax1_m.flatten().reshape((sample_dimension ** 2, 1))
x_2 = ax2_m.flatten().reshape((sample_dimension ** 2, 1))

poly_vec, hypothesis = polynomial_feature_mapping(degree)

data_p = poly_vec(x_1, x_2)  # polynomial feature space

theta = gradient_descent(
    values.flatten().reshape((sample_dimension ** 2, 1)),
    data_p,
    np.ones((data_p.shape[1], 1)),
    learning_rate,
    iterations_lim
)

# find decision boundary and hypothesis surface
_, _, hypothesis_surface = hypothesis(
    ax1.flatten().reshape((sample_dimension, 1)),
    ax2.flatten().reshape((sample_dimension, 1)),
    theta
)

end_time = time.time()
total_time = end_time - start_time
print("Total time of calculations: " + str(total_time))




