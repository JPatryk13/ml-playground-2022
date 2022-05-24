"""
The following code follows an exercise sheet:
https://github.com/jdwittenauer/ipython-notebooks/blob/master/exercises/ML/ex2.pdf

I will utilize some/all libraries I used in the classification.py exercise and apply regularization to the cost
function.
"""
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from classification_functions import sigmoid, cost_function, cost_func_scalar, cost_gradient, theta_update


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


def convert_data(data_: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Breaks input DataFrame into set of features and set of labels.

    :param data_: pd.DataFrame, input data that /w given data_.columns as features_col_names_ and labels_col_name_
    :return: tuple[np.ndarray, np.ndarray], tuple with two numpy arrays - features and labels as x_ and y_ respectively
    """
    # convert the entire dataset to numpy
    np_data_ = data_.to_numpy()
    # slice the numpy array into features and labels and add a column of ones for x_0
    x_ = np.hstack((
        np.ones((len(np_data_), 1)),
        np_data_[:, 0:-1]
    ))
    y_ = np_data_[:, -1].reshape(len(np_data_[:, -1]), 1)

    return x_, y_


def polynomial_feature_mapping(x_: np.ndarray, degree_: int = 3) -> np.ndarray:
    """
    Maps feature-space of the x_ into a polynomial feature-space. Works only with 3d feature space [x_0 = 1, x_1, x_2]

    Let the vertical vector, x, represent the feature space of x_ dataset:
    x = [
        [ 1     ]
        [ x_1   ]
        [ x_2   ]
    ]
    and we want it to be mapped into e.g. 3rd degree polynomial feature space.

    Let v1 and v2 to be a vectors such:
    v1 = [                  v2 = [
        [ 1     ]               [ 1     ]
        [ x_1   ]               [ x_2   ]
        [ x_1^2 ]               [ x_2^2 ]
        [ x_1^3 ]               [ x_2^3 ]
    ]                       ]
    We are interested in the product of v1 and v2:
    v1 @ v2.T = [
        [ 1      x_2         x_2^2           x_2^3       ]
        [ x_1    x_1*x_2     x_1*x_2^2       x_1*x_2^3   ]
        [ x_1^2  x_1^2*x_2   x_1^2*x_2^2     x_1^2*x_2^3 ]
        [ x_1^3  x_1^3*x_2   x_1^3*x_2^2     x_1^3*x_2^3 ]
    ]
    More specifically in the left top corner starting from the diagonal axis. Other terms are higher order permutations
    which shall not appear in the end set.
    [
        [ 1      x_2         x_2^2           x_2^3       ]
        [ x_1    x_1*x_2     x_1*x_2^2         -         ]
        [ x_1^2  x_1^2*x_2      -              -         ]
        [ x_1^3     -           -              -         ]
    ]
    Now we need to collect items from the matrix going diagonally from bottom left to top right. First two runs shall
    yield [1] and [x_1 x_2] in order resulting in a new feature space vector:
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
    higher-dimansional feature-spaces different method shall be applied.

    :param x_: np.ndarray, input data set with two features + one "bias feature"
    :param degree_: int, degree of the resultant polynomial mapping
    :return: np.ndarray, output data with polynomial feature-space
    """
    m = len(x_)
    n = 3

    # translate (m, n) input dataset into (n, 1, m) array, where m is the number of examples and n is the number of
    # features (default: 3)
    x_ = x_.reshape((m, n, 1))  # add extra dimension
    x_ = np.transpose(x_, (1, 2, 0))  # transpose so that each column along the 3rd dimension is one example

    # take rows with x_1 and x_2 features and create v_1 and v_2 vectors out of each iterating from x_1^0 (for v_1)
    # through x_1^degree_ - the same for v_2. Note: x_1 = x_[1] and x_2 = x_[2]
    v_1 = np.array([
        pow(x_[1], i) for i in range(0, degree_ + 1)
    ])
    v_2 = np.array([
        pow(x_[2], i) for i in range(0, degree_ + 1)
    ])

    # multiply v_1 and v_2 so that the array v has the prototype feature-space in first two dimensions and examples
    # remain within third dimension
    v = np.einsum('nmk,mlk->nlk', v_1, np.transpose(v_2, (1, 0, 2)))

    # initialise an array with a temporary row. Helps to append the data and can be removed later
    pol_x_ = np.arange(1 * m).reshape((1, 1, m))

    # iterates through items along diagonal axis going from bottom left to top right corner. Functions range() were
    # fine-tuned to give out appropriate indices in the right order
    for i in range(0, degree_ + 1):
        for j in range(i, -1, -1):
            pol_x_ = np.append(pol_x_, v[j, i - j].reshape((1, 1, m)), axis=0)

    # remove first (extra) row
    pol_x_ = pol_x_[1:len(pol_x_)]

    # convert into 2d array
    pol_x_ = np.transpose(pol_x_, (2, 0, 1)).reshape((m, len(pol_x_)))

    return pol_x_


def polynomial_feature_mapping_2(degree_: int):
    """
        Maps feature-space of the x_ into a polynomial feature-space. Works only with 3d feature space [x_0, x_1, x_2]
        where all x_0 = 1.

        Let the vertical vector, x, represent the feature space of x_ dataset:
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
        We are interested in the product of v(x_1) and v(x_2):
        v(x_1) @ v(x_2).T = [
            [ 1      x_2         x_2^2           x_2^3       ]
            [ x_1    x_1*x_2     x_1*x_2^2       x_1*x_2^3   ]
            [ x_1^2  x_1^2*x_2   x_1^2*x_2^2     x_1^2*x_2^3 ]
            [ x_1^3  x_1^3*x_2   x_1^3*x_2^2     x_1^3*x_2^3 ]
        ]
        More specifically in the left top corner starting from the diagonal axis. Other terms are higher order
        permutations which shall not appear in the end set. We can call it vt for triangle vector.
        vt = [
            [ 1      x_2         x_2^2           x_2^3       ]
            [ x_1    x_1*x_2     x_1*x_2^2         -         ]
            [ x_1^2  x_1^2*x_2      -              -         ]
            [ x_1^3     -           -              -         ]
        ]

        Now we need to collect items from the matrix going diagonally from bottom left to top right. First two runs
        shall yield [1] and [x_1 x_2] in order resulting in a new feature space vector:
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
        :return:
        """

    # column vector storing functions for each power of x - from x^0 through x^degree_
    # v_func = lambda x: np.array([
    #     [pow(x, i)] for i in range(0, degree_ + 1)
    # ])
    # def v_func(x_: np.ndarray) -> np.ndarray:
    #     v_vector = np.array([])
    #     if x_.ndim != 3:
    #         raise Exception("The input array x_ is not three dimensional.")
    #     else:
    #         [pow(x, i) for i in range(0, degree_ + 1)]
    #
    # product of v_func vectors
    # v_mul_func = lambda x_1, x_2: v_func(x_1) @ v_func(x_2).T if x_1.ndim == 2 and x_2.ndim == 2 \
    #     else np.einsum('nmk,mlk->nlk', v_func(x_1), np.transpose(v_func(x_2), (1, 0, 2)))
    #
    # get_diagonal = lambda matrix, offset: np.transpose(
    #     np.diagonal(np.flip(matrix, 1), offset=offset, axis1=1, axis2=2).reshape((
    #         np.diagonal(np.flip(matrix, 1), offset=offset, axis1=1, axis2=2).shape[0],
    #         np.diagonal(np.flip(matrix, 1), offset=offset, axis1=1, axis2=2).shape[1],
    #         1
    #     )),
    #     (1, 2, 0))

    def __v_func(x_: np.ndarray) -> np.ndarray:
        """
        Creates a column vector storing functions for each power of x_ - from x_^0 through x_^degree_. Input vector
        has a shape of m x 1, were m is the number of examples in the dataset (the superscripts in bracets are not
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

        :param x_
        :return v_
        """
        v_ = np.empty((degree_ + 1, 1, len(x_)))
        x_ = x_.reshape((1, 1, len(x_)))
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
        Takes a matrix and extracts a single diagonal with offset given. It also rearranges the matrix so that it fits
        later calculations. The input matrix must be of shape k x k x m, where k = degree_ + 1 and m is the number of
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

        :param matrix_
        :param offset_
        :return diag_
        """
        # get the diagonal from the matrix after flipping it upside down
        diag_ = np.diagonal(np.flip(matrix_, 1), offset=offset_)

        # add extra dimension to the diagonal (by default it is 2d array but for further calculations we need it to
        # be three-dimensional)
        diag_ = diag_.reshape((
            diag_.shape[0],
            diag_.shape[1],
            1
        ))

        # flip the array so that it consists of vertical vectors along the third axis
        diag_ = np.transpose(diag_, (1, 2, 0))

        return diag_

    def __poly_vector(x_1: np.ndarray, x_2: np.ndarray) -> np.ndarray:
        # validate input
        if x_1.shape[0] != len(x_1.flatten()) or x_2.shape[0] != len(x_2.flatten()):
            raise Exception("One or both input arrays have incorrect shape. Expected shape: (m,1) or (m,)")

        v_1 = __v_func(x_1)
        v_2 = __v_func(x_2)
        v_mul = __v_mul(v_1, v_2)

        poly_v = __get_diagonal(v_mul, -degree_)

        for i, d in enumerate(np.arange(-degree_ + 1, 1, 1)):
            poly_v = np.vstack((poly_v, __get_diagonal(v_mul, d)))
        return poly_v

    def __hypothesis(x_1: np.ndarray, x_2: np.ndarray, theta_: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # number of features in the new polynomially mapped dataset
        feature_space_len = sum([i for i in range(1, degree_ + 2)])
        # number of elements in the given dataset
        m = len(x_1) * len(x_2)

        # validate input (x_1 and x_2 validated through __poly_vector)
        if theta_.shape[0] != len(theta_.flatten()):
            raise Exception("theta_ has incorrect shape. Expected shape: (m,1) or (m,)")
        elif theta_.shape[0] != feature_space_len:
            raise Exception("theta_ has " + str(theta_.shape[0]) + " elements. Required number of elements: "
                            + str(feature_space_len))

        # create meshes from given coordinates
        new_x_1, new_x_2 = np.meshgrid(x_1, x_2)

        # create a vector from each mesh
        new_x_1 = np.asarray(new_x_1).flatten()
        new_x_2 = np.asarray(new_x_2).flatten()

        # find a vector of features for each coordinate pair
        poly_v = __poly_vector(new_x_1, new_x_2)

        # extrude the theta_ array into 3rd dimension
        theta_arrays = [theta_ for _ in range(m)]
        theta_extruded = np.stack(theta_arrays, axis=2)

        # calculate predicted labels for each coordinate pair (hypothesis)
        predicted_labels = np.sum(theta_extruded * poly_v, axis=0)
        print(predicted_labels)

        # combine the given coordinate pairs and predicted labels
        new_dataset = np.hstack((
            new_x_1.reshape((m, 1)),
            new_x_2.reshape((m, 1)),
            np.transpose(predicted_labels, (1, 0))
        ))

        # initialise an array for decision boundary coordinates with a row of zeros
        decision_boundary = np.array([[0, 0]])
        # go through the dataset and if the hypothesis is close enough to zero then save the coordinate pair to the
        # array
        for sample in new_dataset:
            if 0.00145 <= sample[2] <= 0.00155:
                decision_boundary = np.append(decision_boundary, [sample[0: 2]], axis=0)
        # remove the redundant row
        decision_boundary = decision_boundary[1:]

        return poly_v, decision_boundary

    return __poly_vector, __hypothesis


def reg_cost_function(y_: np.ndarray, x_: np.ndarray, theta_: np.ndarray, regn_parameter_: int,
                      no_of_iterations_: int = 1, theta_range_: tuple[float, float] = (0.0, 0.0)) -> float:
    """
    Calculates cost function for the given set of features, labels and theta using cost_function() from the
    classification.py script and extra regularization term.

    :param y_: np.ndarray, dataset labels converted to m x 1 vector
    :param x_: np.ndarray, set of features converted to m x n matrix
    :param theta_: np.ndarray, n x 1 vector containing initial theta
    :param regn_parameter_: int, a number balancing how well the model is going to be trained on the data and
    smoothening the function in order to avoid over-fitting
    :param no_of_iterations_: int, default 1; if changed from default then extrude matrices x_, y_ in 3rd dimension and
    theta becomes n x 1 x k tensor where k is no_of_iterations_
    :param theta_range_: tuple[float, float], default (0.0, 0.0); when both no_of_iterations_ and theta_range_ changed
    from default do as above
    :return: float, value of the regularized cost function
    """
    # regular cost function
    cost_ = cost_function(y_, x_, theta_)
    # regularised cost function
    return (cost_ + regn_parameter_ * theta_.T @ theta_).item()


def reg_cost_gradient(y_: np.ndarray, x_: np.ndarray, theta_: np.ndarray, regn_parameter_: int) -> float:
    """
    Calculates gradient of the cost function with regularisation term. Works in similar fashion as the
    reg_cost_function() function.

    :param y_: ...
    :param x_: ...
    :param theta_: ...
    :param regn_parameter_: ...
    :return: float, value of the gradient of the regularized cost function
    """
    # number of examples in the dataset
    m = len(y_)
    # regular gradient of the cost function
    cost_grad_ = cost_gradient(theta_, x_, y_)
    # regularised gradient of the cost function
    return cost_grad_ + (regn_parameter_ / m) * theta_


def reg_gradient_descent(y_: np.ndarray, x_: np.ndarray, theta_: np.ndarray, regn_parameter_: int,
                         learning_rate_: float = 0.01, cost_threshold_: float = 0.1, iterations_limit_: int = 1000) \
        -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Uses gradient descent algorithm to minimize the cost function and find respective weights for the model. It tracks
    parameters such as cost function and its gradient values for each iteration. It uses iterations_limit_: to prevent
    the loop to go on forever if it would get stuck on a cost difference higher than the threshold.

    :param y_: ...
    :param x_: ...
    :param theta_: ...
    :param regn_parameter_: ...
    :param learning_rate_: float, default 0.01; rate at which the algorithm shall modify the theta value
    :param cost_threshold_: float, default 0.1; characteristic value for the algorithm is the change of cost function
    gradient value from one iteration to another, therefore, the cost function change threshold. I.e. if the difference
    of the cost func. value between iterations goes below the threshold the loop shall be broken
    :param iterations_limit_: int, default 1000; safety limit that prevents the loop from going infinitely
    :return: tuple[np.ndarray, np.ndarray, np.ndarray, int], optimised weights, cost function for all iterations, cost
    gradient for all iterations and number of iterations respectively
    """
    # number of examples in the dataset
    m = len(y_)
    # term modifies theta at each iteration of the loop by a factor slightly smaller than 1
    shrinking_term_ = 1 - learning_rate_ * regn_parameter_ / m

    # initialize cost_history_ with cost function value calculated from the initial theta
    cost_history_ = np.array([reg_cost_function(y_, x_, theta_, regn_parameter_)])

    # initialize cost_grad_history_ with the value calculated from initial theta
    cost_grad_history_ = np.array(reg_cost_gradient(y_, x_, theta_, regn_parameter_))

    # track the number of iterations of the loop
    no_of_iterations_ = 0

    # loop goes on until it reaches the iterations_limit_ or the cost function value does not change by much
    for _ in range(iterations_limit_):
        no_of_iterations_ += 1
        # hypothesis
        sig_ = sigmoid(x_ @ theta_)
        # gradient descent of theta
        theta_ = theta_ * shrinking_term_ - (learning_rate_ / m) * x_.T @ (sig_ - y_)

        # append cost function value calculated from the new theta to the cost_history_ array
        cost_history_ = np.append(cost_history_, reg_cost_function(y_, x_, theta_, regn_parameter_))
        # append the cost gradient calculated from the new theta
        cost_grad_history_ = np.append(cost_grad_history_, reg_cost_gradient(y_, x_, theta_, regn_parameter_), axis=1)

        # if the difference between cost values for last two iterations is smaller than the threshold break the loop
        # - the theta shall be optimised enough by this point
        if abs(cost_history_[-2] - cost_history_[-1]) <= cost_threshold_:
            break

    return theta_, cost_history_, cost_grad_history_, no_of_iterations_


# get data from the file
test_data = fetch_data()

# convert data into three-feature x and set of labels y
x, y = convert_data(test_data)

# run polynomial mapping to get the function for vector of feature space and hypothesis function
poly_vector, poly_hypothesis = polynomial_feature_mapping_2(6)

# calculate values for the new polynomial feature space
x_pol = np.transpose(poly_vector(x[:, 1], x[:, 2]), (2, 0, 1))[:, :, 0]

# initialise theta
initial_theta = np.ones((x_pol.shape[1], 1))

# set regularisation parameter and learning rate
regn_parameter = 10
learning_rate = 0.0003

# calculate regularised model that fits the dataset
theta, cost, cost_grad, iters = reg_gradient_descent(y, x_pol, initial_theta, regn_parameter, learning_rate)

# initialise range of values for x_1 and x_2
x_1_range = np.linspace(min(x[:, 1]), max(x[:, 1]), 200)
x_2_range = np.linspace(min(x[:, 2]), max(x[:, 2]), 200)

# find decision boundary
poly_v, decision_boundary = poly_hypothesis(x_1_range, x_2_range, theta)

##########################################
#             PLOTTING BLOCK             #
##########################################

print(decision_boundary)

plt.scatter(x[:, 1], x[:, 2], c=y)
plt.scatter(decision_boundary[:, 0], decision_boundary[:, 1])
plt.show()
