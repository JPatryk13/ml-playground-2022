"""
The file contains function that were initially in the classification.py. Transferred here to get access from
regn_classification.py without executing the classification.py code.
"""

import numpy as np


def sigmoid(z_):
    """
    The function calculates value of the sigmoid function for given data.

    :param z_: scalar, vector, matrix
    :return: scalar, vector, matrix
    """
    return 1/(1 + np.exp(-z_))


def cost_function(y_, x_, theta_) -> float:
    """
    The function calculates cost function for given data x_ and weights theta_. The function is fully vectorized so that
    it performs operations on matrices only avoiding loops and, therefore, speeding up the calculation.

    :param theta_: n x 1 vector
    :param x_: m x n matrix
    :param y_: m x 1 vector
    :return: scalar
    """

    # number of examples in the set
    m = len(y_)

    # calculate hypothesis
    sig_ = sigmoid(x_ @ theta_)

    # column vector of ones
    one_vec = np.ones((m, 1), dtype=int)

    # calculate cost function
    cost_ = -(1/m) * (y_.T @ np.log(sig_) + (one_vec - y_).T @ np.log(one_vec - sig_))

    # return value of the cost function as a scalar
    return cost_


def cost_func_scalar(sig_, y_) -> float:
    """
    Calculate cost function value for given sigmoid (hypothesis) value and given label 1 or 0.

    :param sig_: float between 0 and 1
    :param y_: label, 0 or 1
    :return: value of cost function
    """

    return -y_ * np.log(sig_) - (1 - y_) * np.log(1 - sig_)


def cost_gradient(theta_, x_, y_):
    """
    Calculates gradient of the cost function using fully vectorised form of cost gradient.

    :param theta_: n x 1 vector
    :param x_: m x n matrix
    :param y_: m x 1 vector
    :return: scalar
    """

    # calculate hypothesis
    sig_ = sigmoid(x_ @ theta_)
    # number of examples in the set
    m = len(y_)
    # calculate gradient
    grad_ = (1/m) * x_.T @ (sig_ - y_)

    return grad_


def theta_update(theta_, alpha_, cost_grad_):
    """
    Calculates more optimal theta than the given one using gradient of the cost function (cost_grad_) and the learning
    rate (alpha_).

    :param theta_: n x 1 vector
    :param alpha_: scalar
    :param cost_grad_: n x 1 vector
    :return: n x 1 vector
    """
    return theta_ - alpha_ * cost_grad_
