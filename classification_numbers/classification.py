"""
The following code follows an exercise sheet:
https://github.com/jdwittenauer/ipython-notebooks/blob/master/exercises/ML/ex2.pdf
First time using pandas and seaborn as well as solving classification problem.

22:00 02 May: I yet again encountered the issue with normalization of data I suppose. The decision boundary fits
roughly the data assuming manually tweaked correction is added to it. I am going to leave it for now in the current
state. I shall get back to it soo though.

00:00 03 May: I found a solution. Normalizing data makes the resultant decision boundary only applicable for that
data, therefore the theta is different I suppose. I had to change few things with plotting and recreate the DataFrame
with these normalized numbers. Now everything seems to work better.

00:30 03 May: Added calculation of the model accuracy and sigmoid plot along with the exam data split according to
labels. The accuracy is 76%; not bad. Comments require some improvements and typing failed in this project. Also, there
was no need really to use seaborn library here.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from classification_functions import *


def fetch_data() -> pd.DataFrame:
    """
    The function reads and plots the data from ex2data1.txt. It returns the pandas' dataframe at the end.

    :return: DataFrame
    """
    # read the ex2data1.txt file as general table containing also approval or denial as 1 or 0 respectively
    exam_data = pd.read_csv("ex2data1.txt", sep=",", header=None)
    # name the columns
    exam_data.columns = ["Exam 1 score", "Exam 2 score", "Approval"]

    return exam_data


def grad_descent(x_, y_, theta_, alpha_=0.1, epsilon_=0.1):
    """
    Gradient descent algorithm of the cost function to find its minimum and the thetas that satisfy it. The function
    takes arbitrary theta nd learning rate and following the GD algorithm calculates the most optimal theta that would
    give the best fit of the model to the given data x_ (features) and y_ (labels). When the change between iterations
    of the algorithm is smaller than epsilon_, declare convergence.

    :param y_: m x 1 vector
    :param x_: m x n matrix
    :param theta_: n x 1 vector
    :param alpha_: scalar, default 0.1
    :param epsilon_: scalar, default 0.1
    :return: n x 1 vector
    """

    # initialize cost history and list of iterations
    cost_history_ = np.array([])
    gd_iterations_ = np.array([])
    j = 1

    for j in range(10000):
        cost_grad_ = cost_gradient(theta_, x_, y_)
        theta_ = theta_update(theta_, alpha_, cost_grad_)

        cost_history_ = np.append(cost_history_, cost_function(y_, x_, theta_))
        gd_iterations_ = np.append(gd_iterations_, j)

    return theta_, cost_history_, gd_iterations_


def decision_boundary(min_, max_, theta_):
    """
    :param min_: lower boundary for x1_range_
    :param max_: upper boundary for x1_range_
    :param theta_: weights for the decision boundary
    :return: two matrices containing 'x' and 'y' coordinates for plotting the decision boundary
    """
    x1_range_ = np.arange(min_, max_, 1)
    return x1_range_, (-theta_[1] * x1_range_ - theta_[0]) / theta_[2]


def predictor_avg(data_, theta_):
    """
    based on the given data and weights theta calculates probability that given features would result in y = 1 then
    compares it with the existing labels.

    :param data_: DataFrame containing normalized features and labels
    :param theta_: 3 by 1 vector of theta used to create the model
    :return: average success rate in %
    """

    # initialize an array to store success count
    is_success = np.array([])

    # iterate over rows of data_
    for index, row in data_.iterrows():
        x_i = np.array([
            [1],
            [row["Std. Exam 1 score"]],
            [row["Std. Exam 1 score"]]
        ])

        # probability that the exam score set would get approval
        probability_ = sigmoid(theta.T @ x_i)

        # set to 1 if the probability is above/equal the 0.5 threshold - else, set to 0
        pred_approval_ = 1 if probability_ >= 0.5 else 0

        # append the is_success array with 1 if the prediction matches the label, else append 0
        if pred_approval_ == row["Approval"]:
            is_success = np.append(is_success, 1)
        else:
            is_success = np.append(is_success, 0)

    # return average of all successes
    return np.sum(is_success) / len(data_)


# fetch the data
data = fetch_data()

# split data into features and labels
exam_scores = data[["Exam 1 score", "Exam 2 score"]]
labels = data["Approval"]

# initialize StandardScaler
scaler = StandardScaler()
scaler.fit(exam_scores)

# optimize data
std_exam_scores = scaler.transform(exam_scores)  # std_exam_scores is 'numpy.ndarray'

# recreate data DataFrame with normalized data for further plotting
std_data_dict = {
    "Std. Exam 1 score": std_exam_scores[:, 0],
    "Std. Exam 2 score": std_exam_scores[:, 1],
    "Approval": labels.to_numpy()
}
std_data = pd.DataFrame(std_data_dict)

# initialize theta with some arbitrary values
theta = np.array([
    [1],
    [1],
    [1]
])

# add extra column of ones to the left of the data set (x_0^(i) = 1 for all i)
features = np.hstack((
    np.ones((len(labels), 1)),
    std_exam_scores
))
# convert labels to numpy array and change shape (m,) -> (m,1)
labels = np.reshape(
    labels.to_numpy(),
    (len(labels), 1)
)

# # ex. 2 calculate cost function and its gradient
cost = cost_function(theta_=theta, x_=features, y_=labels)
# grad = cost_gradient(theta, features, labels)

# find the most optimal theta
theta, cost_history, gd_iterations = grad_descent(features, labels, theta)

# calculate the decision boundary for the data plot
exam_1_score_range, exam_2_score_range = decision_boundary(
    std_data["Std. Exam 1 score"].min(),
    std_data["Std. Exam 1 score"].max(),
    theta
)

# calculate cost function 'history' for 0.01 to 0.99 range
cost_func_history_range = np.arange(0.01, 0.99, 0.01)
cost_func_history_y0 = np.array([])
cost_func_history_y1 = np.array([])
for i in cost_func_history_range:
    cost_func_history_y0 = np.append(
        cost_func_history_y0,
        cost_func_scalar(
            sig_=i,
            y_=0
        )
    )
    cost_func_history_y1 = np.append(
        cost_func_history_y1,
        cost_func_scalar(
            sig_=i,
            y_=1
        )
    )

# initialize fig and axs for multiple plots
fig, axs = plt.subplots(2, 2, figsize=(10, 10), dpi=200)
fig.suptitle("Exam approval data analysis.")

# use seaborn to generate a scatter plot of data and decision boundary
sns.scatterplot(
    data=std_data,
    x="Std. Exam 1 score", y="Std. Exam 2 score",
    hue="Approval", style="Approval",
    ax=axs[0, 0]
)
sns.lineplot(
    x=exam_1_score_range, y=exam_2_score_range,
    ax=axs[0, 0],
    linewidth=2,
    color="green"
)
axs[0, 0].set_title("Data and decision boundary")

# plot cost function
sns.lineplot(x=cost_func_history_range, y=cost_func_history_y0, ax=axs[0, 1])
sns.lineplot(x=cost_func_history_range, y=cost_func_history_y1, ax=axs[0, 1])
axs[0, 1].set_title("Cost function for different hypothesis values")
axs[0, 1].set_xlabel("hypothesis")
axs[0, 1].set_ylabel("cost")
axs[0, 1].legend(labels=["y=0", "y=1"])

# plot gradient descent
axs[1, 0].plot(gd_iterations, cost_history, linestyle="--", color="red")
axs[1, 0].set_title("Cost value vs GD iterations")
axs[1, 0].set_xlabel("iterations")
axs[1, 0].set_ylabel("cost value")

# plot sigmoid and data sorted by the label
sns.scatterplot(x=(features @ theta)[:, 0], y=data["Approval"], hue=data["Approval"], ax=axs[1, 1])
sns.lineplot(x=(features @ theta)[:, 0], y=sigmoid(features @ theta)[:, 0], ax=axs[1, 1], color="green")
axs[1, 1].set_title("Approval probability")
axs[1, 1].set_xlabel("probability")
axs[1, 1].set_ylabel("x * theta")

# display the plot using matplotlib
plt.show()

print("model accuracy: " + str(predictor_avg(std_data, theta) * 100) + "%")
