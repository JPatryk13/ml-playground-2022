"""
The following code follows an exercise sheet:
https://github.com/jdwittenauer/ipython-notebooks/blob/master/exercises/ML/ex2.pdf

I will utilize some libraries I used in the classification_linear.py exercise and apply regularization to the cost
function.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from classification_numbers.tools.classification_functions import polynomial_feature_mapping, gradient_descent


def fetch_data(plot_: bool = False, print_: bool = False) -> pd.DataFrame:
    """
    The function reads and plots the data from ex2data2.txt. It returns the pandas' dataframe at the end. Accept two
    booleans which determine whether the data is going to be plotted or/and printed.

    :param plot_: bool, plot the data if true
    :param print_: bool, print the table if true
    :return: DataFrame with ex2data2.txt data
    """
    # read the ex2data1.txt file as general table containing also approval or denial as 1 or 0 respectively
    data_ = pd.read_csv("data/ex2data2.txt", sep=",", header=None)
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
