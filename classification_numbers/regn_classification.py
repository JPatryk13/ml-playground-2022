"""
The following code follows an exercise sheet:
https://github.com/jdwittenauer/ipython-notebooks/blob/master/exercises/ML/ex2.pdf

I will utilize some libraries I used in the classification.py exercise and apply regularization to the cost
function.
"""
from typing import Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from classification_functions import sigmoid


def fetch_data(plot_: bool = False, print_: bool = False) -> pd.DataFrame:
    """
    The function reads and plots the data from ex2data2.txt. It returns the pandas' dataframe at the end. Accept two
    booleans which determine whether the data is going to be plotted or/and printed.

    :param plot_: bool, plot the data if true
    :param print_: bool, print the table if true
    :return: DataFrame with ex2data2.txt data
    """
    # read the ex2data1.txt file as general table containing also approval or denial as 1 or 0 respectively
    data_ = pd.read_csv("ex2data2.txt", sep=",", header=None)
    data_.columns = ["Test 1", "Test 2", "Accepted"]

    if plot_:
        sns.scatterplot(
            data=data_,
            x="Test 1", y="Test 2",
            hue="Accepted", style="Accepted",
        )
        plt.show()
    elif print_:
        print(data_)

    return data_


def polynomial_feature_mapping(degree_: int) -> tuple[Callable, Callable]:
    """
        Maps feature-space of the x_ into a polynomial feature-space. Works only with two-feature datasets [x_0, x_1,
        x_2] where all x_0 = 1. Below I overview the working of the algorithm.

        Let the vertical vector, x, represent the feature space:
        x = [
            [ 1   ]
            [ x_1 ]
            [ x_2 ]
        ]
        and we want it to be mapped into e.g. 3rd degree polynomial feature space.

        Let v(x) to be a function vector as such:
        v(x) = [
            [ x^0 ]
            [ x^1 ]
            [ x^2 ]
            [ x^3 ]
        ]

        The product of v(x_1) and v(x_2) reveals a matrix containing the polynomial terms of interest as well as some
        extra terms (higher order permutations which shall not appear in the end set):
        v(x_1) @ v(x_2).T = [
            [ 1      x_2         x_2^2           x_2^3       ]
            [ x_1    x_1*x_2     x_1*x_2^2       x_1*x_2^3   ]
            [ x_1^2  x_1^2*x_2   x_1^2*x_2^2     x_1^2*x_2^3 ]
            [ x_1^3  x_1^3*x_2   x_1^3*x_2^2     x_1^3*x_2^3 ]
        ]

        Filtering out the bottom-right half leaves the vt set which can be now rearranged:
        vt = [
            [ 1      x_2         x_2^2           x_2^3       ]
            [ x_1    x_1*x_2     x_1*x_2^2         -         ]
            [ x_1^2  x_1^2*x_2      -              -         ]
            [ x_1^3     -           -              -         ]
        ]

        Collecting items from the matrix going diagonally from bottom-left to top-right starting from the top left
        corner shall yield [1], [x_1 x_2], etc.:
        new_x = [
            [ 1         ]
            [ x_1       ]
            [ x_2       ]
            [ x_1^2     ]
            [ x_1*x_2   ]
            ...
            [ x_2^3     ]
        ]

        Limitation here is that the input can have only two features + one "bias feature". To expand the method to
        higher-dimensional feature-spaces different method shall be applied.

        :param degree_: int, degree of the resultant polynomial mapping
        :return: __poly_vector function allowing for mapping the feature space of the dataset and __hypothesis function
        which generates hypothesis surface through the data and decision boundary
        """

    def __v_func(x_: np.ndarray) -> np.ndarray:
        """
        Creates a column vector storing functions for each power of x_ - from x_^0 through x_^degree_. Input vector
        has a shape of m x 1, where m is the number of examples in the dataset (the superscripts in braces are not
        powers, they are numbers of examples in the set):
        x_ = [
            [ x^(1) ]
            [ x^(2) ]
            [ x^(3) ]
            ...
            [ x^(m) ]
        ]
        Such vector shall be transformed into a 3d array of shape 1 x 1 x m:
        x_ = [
            [[ x^(1) ]]  [[ x^(2) ]]  [[ x^(3) ]]  ...  [[ x^(m)  ]]
        ]
        Next each element of the array is raised to a power starting from 0 through degree_ (d) and stacked below the
        previous one:
        v_ =  = [[      [
            [ x^0 ]         ...
            [ x^1 ]         ...
            [ x^2 ]         ...
              ...           ...
            [ x^d ]         ...
        ]               ]]

        :param x_: m x 1 vector with data from a single feature
        :return v_: k x 1 x m 3D array with factors of x_, where k = d + 1
        """
        # initialize empty array with predefined dimensions
        v_ = np.empty((degree_ + 1, 1, len(x_)))
        # reshape the x_ vector into 3D array
        x_ = x_.reshape((1, 1, len(x_)))
        # factor x_ and stack vertically
        for deg in range(0, degree_ + 1):
            v_[deg] = pow(x_, deg)
        return v_

    def __v_mul(v_1: np.ndarray, v_2: np.ndarray) -> np.ndarray:
        """
        It multiplies two 3d matrices of shape k x 1 x m where k = degree_ + 1 and m is the number of samples. The input
        matrices are (d = degree_):
        v_1 =  = [[      [
            [ x_1^0 ]         ...
            [ x_1^1 ]         ...
            [ x_1^2 ]         ...
               ...            ...
            [ x_1^d ]         ...
        ]               ]]
        v_2 =  = [[      [
            [ x_2^0 ]         ...
            [ x_2^1 ]         ...
            [ x_2^2 ]         ...
               ...            ...
            [ x_2^d ]         ...
        ]               ]]
        The function shall return a product of those along specified axes:
        v_1 @ v_2.T = [[
            [ x_1^0 ]
            [ x_1^1 ]               [
            [ x_1^2 ]       @           [[ x_2^0  x_2^1  x_2^2  ...  x_2^d]]  ...
               ...                  ]
            [ x_1^d ]
        ] ... ]
        Therefore:
        v_mul = [[
            [ 1      x_2           x_2^2             x_2^3            ...     x_2^d         ]
            [ x_1    x_1 * x_2     x_1 * x_2^2       x_1 * x_2^3      ...     x_1 * x_2^d   ]
            [ x_1^2  x_1^2 * x_2   x_1^2 * x_2^2     x_1^2 * x_2^3    ...     x_1^2 * x_2^d ]
            [ x_1^3  x_1^3 * x_2   x_1^3 * x_2^2     x_1^3 * x_2^3    ...     x_1^3 * x_2^d ]
                                               ...
            [ x_1^d  x_1^d * x_2   x_1^d * x_2^2     x_1^d * x_2^3    ...     x_1^d * x_2^d ]
        ] ... ]
        :param v_1
        :param v_2
        :return v_mul
        """
        return np.einsum('nmk,mlk->nlk', v_1, np.transpose(v_2, (1, 0, 2)))

    def __get_diagonal(matrix_: np.ndarray, offset_: int) -> np.ndarray:
        """
        Takes a matrix and extracts a single diagonal with given offset. It also rearranges the matrix so that it fits
        later calculations. The input array must be of shape k x k x m, where k = degree_ + 1 and m is the number of
        examples:
        matrix_ = [[
            [ mx_111  mx_121  mx_131  ...  mx_1k1 ]
            [ mx_211  mx_221  mx_231  ...  mx_2k1 ]
            [ mx_311  mx_321  mx_331  ...  mx_3k1 ]
                              ...
            [ mx_k11  mx_k21  mx_k31  ...  mx_kk1 ]
        ] ... ]
        That matrix can be understood as mapping of output of the __v_mul() function where each element is a function of
        x_1 and x_2: mx_ijl = x_1^(i-1) * x_2^(j-1) of l-th example.

        Example outputs for 4 x 4 x m input array:
        diag_ = [
            [[ mx_111 ]] [[ mx_112 ]] [[ mx_113 ]]  ...  [[ mx_11m ]]
        ] (offset_ = -degree_)
        diag_ = [
            [                   [                           [
                [ mx_211 ]          [ mx_212 ]      ...         [ mx_21m ]
                [ mx_121 ]          [ mx_122 ]      ...         [ mx_12m ]
            ]                   ]                           ]
        ] (offset_ = -degree + 1)

        :param matrix_: input matrix from which the diagonal shall be taken
        :param offset_: shift of the diagonal along the axis going through top-left and bottom-right corners. Offset 0
        implies the main diagonal, while any number below 0 is related to sub-diagonals going towards the top-left
        corner
        :return diag_: obtained diagonal
        """
        # get the diagonal from the matrix after flipping it upside down
        diag_ = np.flipud(matrix_).diagonal(offset=offset_)

        return diag_

    def __poly_vector(x_1: np.ndarray, x_2: np.ndarray) -> np.ndarray:
        """
        Utilises defined functions and returns transformed dataset.

        :param x_1: m x 1 vector - feature #1
        :param x_2: m x 1 vector - feature #2
        :return: poly_v, m x n matrix, where n is the number of features in the polynomial space
        """
        # validate input
        if x_1.shape[0] != len(x_1.flatten()) or x_2.shape[0] != len(x_2.flatten()):
            raise Exception("One or both input arrays have incorrect shape. Expected shape: (m,1) or (m,)")

        # transform input data into matrices of their factors
        v_1 = __v_func(x_1)  # shape: k x 1 x m
        v_2 = __v_func(x_2)  # shape: k x 1 x m

        # multiply the v_1 and v_2
        v_mul = __v_mul(v_1, v_2)  # shape: k x k x m

        # initialise the poly_v variable with the "first diagonal" - top-left corner value
        poly_v = __get_diagonal(v_mul, -degree_)  # shape: m x 1

        # gather all the other diagonals stacking them horizontally
        for i, d in enumerate(np.arange(-degree_ + 1, 1, 1)):
            poly_v = np.hstack((poly_v, __get_diagonal(v_mul, d)))  # shapes: m x 2, m x 3, ..., m x k

        return poly_v  # shape: m x n (n = 1 + 2 + ... + k)

    def __hypothesis(x_1_range_: np.ndarray, x_2_range_: np.ndarray, theta_: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        The function takes value ranges spanning across x_1 and x_2 features and optimised theta. It then creates a
        dataset of permutations of values from x_1_range_ and x_2_range_ which then are put through the polynomial
        mapping function, __poly_vector(). Then resultant dataset is multiplied with values of theta and sigmoid
        function is applied therefore resulting in a square mesh of values of the hypothesis across the surface.
        Additionally, an array containing the polynomially-mapped set of input data along with the decision boundary is
        returned.

        Note: here I often use sqrt(m) as the number of examples in x_1_range_ and x_2_range_ while m is the number of
        permutations consisting of new_x_1 and new_x_2.

        :param x_1_range_: sqrt(m) x 1 vector, range of values of x_1 (x_1 axis)
        :param x_2_range_: sqrt(m) x 1 vector, range of values of x_2 (x_2 axis)
        :param theta_: optimised values of theta
        :return: poly_vec, decision_boundary, hypothesis_surface.
        """
        # number of features in the new polynomially mapped dataset
        feature_space_len = sum([i for i in range(1, degree_ + 2)])
        # number of elements in the given dataset
        m = len(x_1_range_) * len(x_2_range_)

        # validate input (x_1 and x_2 validated through __poly_vector)
        if theta_.shape[0] != len(theta_.flatten()):
            raise Exception("theta_ has incorrect shape. Expected shape: (m,1) or (m,)")
        elif theta_.shape[0] != feature_space_len:
            raise Exception("theta_ has " + str(theta_.shape[0]) + " elements. Required number of elements: "
                            + str(feature_space_len))

        # create meshes from given coordinates
        new_x_1, new_x_2 = np.meshgrid(x_1_range_, x_2_range_)

        # create a vector from each mesh
        new_x_1 = np.asarray(new_x_1).flatten()  # shape: m x 1
        new_x_2 = np.asarray(new_x_2).flatten()  # shape: m x 1

        # find a vector of polynomial features for each coordinate pair
        poly_vec = __poly_vector(new_x_1, new_x_2)  # shape: m x n

        # extrude the theta_ array into 3rd dimension
        theta_extruded = np.array([theta_.flatten() for _ in range(m)])  # shape: m x n

        # calculate predicted labels for each coordinate pair (hypothesis). Both theta and poly_vec extruded have shape
        # (m x n) therefore resulting in the same-shape vector after multiplication. Sum along the 1st axis gives
        # and array of shape (1 x m). The function reshape produces "continuous" distribution of label values in space
        # giving matrix of shape (sqrt(m) x sqrt(m))
        z = np.sum(theta_extruded * poly_vec, axis=1)
        hypothesis_surface = sigmoid(z).reshape((int(np.sqrt(m)), int(np.sqrt(m))))

        # initialise an array for decision boundary coordinates with a row of zeros
        decision_boundary = np.array([[0, 0]])

        # go through the dataset and if the hypothesis is close enough to .5 then save the coordinate pair to the
        # array
        delta = 0.01

        for x_1_ind, x_1_coord in enumerate(x_1_range_):
            for x_2_ind, x_2_coord in enumerate(x_2_range_):
                if (0.5 - delta) <= hypothesis_surface[x_1_ind, x_2_ind] <= (0.5 + delta):
                    decision_boundary = np.append(decision_boundary, [[x_1_coord, x_2_coord]], axis=0)

        # remove the redundant row
        decision_boundary = decision_boundary[1:]

        return poly_vec, decision_boundary, hypothesis_surface

    return __poly_vector, __hypothesis


def gradient_descent(y_: np.ndarray, x_: np.ndarray, theta_: np.ndarray, learning_rate_: float = 0.001,
                     iterations_limit_: int = 300000) -> np.ndarray:
    """
    Uses gradient descent algorithm to minimize the cost function and find respective weights for the model. It tracks
    parameters such as cost function and its gradient values for each iteration. It uses iterations_limit_: to prevent
    the loop to go on forever if it would get stuck on a cost difference higher than the threshold.

    :param y_: set of labels
    :param x_: set of features
    :param theta_: initial theta
    :param learning_rate_: float, default 0.01; rate at which the algorithm shall modify the theta value
    :param iterations_limit_: int, default 1000; safety limit that prevents the loop from going infinitely
    :return: optimised theta
    """
    # number of examples in the dataset
    m = len(y_)

    # loop goes on until it reaches the iterations_limit_ or the cost function value does not change by much
    for _ in range(iterations_limit_):
        # hypothesis
        sig_ = sigmoid(x_ @ theta_)
        # gradient descent of theta
        theta_ = theta_ - (learning_rate_ / m) * x_.T @ (sig_ - y_)

    return theta_


###########################################
#             EXECUTION BLOCK             #
###########################################

degree = 6  # degree of the polynomial feature space
learning_rate = 0.003  # learning rate of the gradient descent algorithm
iterations_lim = 300000  # number of iterations of the GD algorithm

# get data from the file
test_data = fetch_data()

# get function for polynomial mapping  of the feature space
poly_vec, hypothesis = polynomial_feature_mapping(degree)

# use the function to turn the set into polynomial space
test_data_poly = poly_vec(test_data["Test 1"].to_numpy(), test_data["Test 2"].to_numpy())
labels = test_data["Accepted"].to_numpy().reshape((len(test_data["Accepted"]), 1))

# initialise theta
initial_theta = np.ones((test_data_poly.shape[1], 1))

# find a model that fits the dataset
theta = gradient_descent(labels, test_data_poly, initial_theta, learning_rate, iterations_lim)

# get the ranges across which x_1 and x_2 span
x_1_range = np.linspace(min(test_data["Test 1"]), max(test_data["Test 1"]), 300)
x_2_range = np.linspace(min(test_data["Test 2"]), max(test_data["Test 2"]), 300)

# find decision boundary and hypothesis surface
_, decision_boundary, hypothesis_surface = hypothesis(x_1_range, x_2_range, theta)

##########################################
#             PLOTTING BLOCK             #
##########################################

X, Y = np.meshgrid(x_1_range, x_2_range)

fig = plt.figure(figsize=(12, 5), dpi=200)
plt.subplots_adjust(wspace=0.5, hspace=0.5)

# plot the data and the decision boundary
_ = fig.add_subplot(1, 2, 1)
sns.scatterplot(
    data=test_data,
    x="Test 1", y="Test 2",
    hue="Accepted", style="Accepted",
    palette=['#6C3978', '#FAE95C']
)
plt.scatter(decision_boundary[:, 0], decision_boundary[:, 1], s=5, edgecolors='tab:blue')
plt.xlabel("Test 1")
plt.ylabel("Test 2")

# plot the surface and data
ax = fig.add_subplot(1, 2, 2, projection='3d')
surf = ax.plot_surface(X, Y, hypothesis_surface)
ax.scatter(test_data["Test 1"], test_data["Test 2"], labels, c=labels)
ax.set_xlabel("Test 1")
ax.set_ylabel("Test 2")
ax.set_zlabel("Accepted")

plt.show()
