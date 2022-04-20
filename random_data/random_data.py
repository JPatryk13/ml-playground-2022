import numpy as np
import matplotlib.pyplot as plt


def normal_eqn(x_, y_):
    # normal equation
    theta_ = np.linalg.inv(X.T @ X) @ X.T @ y
    return theta_


def gradient_descent(x_, y_, theta_=np.random.randn(2, 1), no_of_iterations=100, learning_rate=0.01):
    # number of samples
    m = y_.shape[0]

    cost_func_history = np.zeros(no_of_iterations)
    theta_history = np.zeros((no_of_iterations, 2))
    # gradient descent loop
    for i in range(no_of_iterations):
        theta_history[i] = theta_.T
        hypothesis = x_ @ theta_
        cost_func_history[i] = (1 / (2 * m)) * (hypothesis - y).T @ (hypothesis - y)
        cost_func_derivative = (1 / m) * (X.T @ (hypothesis - y_))
        theta_ = theta_ - learning_rate * cost_func_derivative

    return theta_, cost_func_history, theta_history


def cost(x_, y_, th0, th1):
    # total number of elements in th0/th1
    no_of_elements = th0.size

    # extruding x_ and y_ into 3rd dimension by copying entries
    x_ = np.stack([x_ for _ in range(no_of_elements)], axis=2)
    y_ = np.stack([y_ for _ in range(no_of_elements)], axis=2)

    # reshaping th0 and th1 so that they are 1 x 1 x no_of_elements 3d arrays
    th0 = th0.reshape((1, 1, no_of_elements))
    th1 = th1.reshape((1, 1, no_of_elements))

    # stacking theta vertically
    theta_ = np.vstack((th0, th1))

    # number of samples
    m = y_.shape[0]

    # multiplied matrices using Einstein summation convention. Had to treat each slice along the depth axis as a
    # separate 2D array to multiply a batch of arrays - faster than for loop. Any dot product works in a different
    # manner so had to express dimensions conversion explicitly.
    hypothesis = np.einsum('mnk,nlk->mlk', x_, theta_)

    # using the transpose function with explicit axis permutation for the reason above.
    j_func = (1 / (2 * m)) * np.einsum('nmk,mlk->nlk', np.transpose((hypothesis - y_), (1, 0, 2)), (hypothesis - y_))

    # converting 1x1xk matrix into sqrt(k) x sqrt(k) matrix
    return j_func.reshape(int(np.sqrt(no_of_elements)), int(np.sqrt(no_of_elements)))


# generating some random data
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# adding extra column of ones to get the right shape of the matrix
X = np.hstack((np.ones((X.shape[0], 1)), X))

# analytical way
# theta = normal_eqn(X, y)

# numerical way
gd_iters = 50
theta, cost_f, theta_h = gradient_descent(
    X,
    y,
    theta_=np.array([[1], [1]]),
    no_of_iterations=gd_iters,
    learning_rate=0.05
)

x_range = np.array([
    [1.0, 0.0],
    [1.0, 2.0]
])

# hypothesis
h = theta[0] + x_range[:, 1] * theta[1]

plt.style.use(["ggplot"])

# plotting
fig = plt.figure(figsize=(10, 10), dpi=200)
plt.subplots_adjust(wspace=0.5, hspace=0.5)

# plot 1
_ = fig.add_subplot(2, 2, 1).set_title("Data + model")
plt.plot(X[:, 1], y, marker='x', linestyle='')
plt.plot(x_range[:, 1], h)

plt.xlabel("feature value")
plt.ylabel("label")

# plot 2
_ = fig.add_subplot(2, 2, 2).set_title("Cost reduction")
plt.plot(range(gd_iters), cost_f, marker='o', linestyle='', markersize=4)

plt.xlabel("no. of iterations")
plt.ylabel("cost function")

plt.style.use(["seaborn-whitegrid"])

# plot 3
theta0_range = theta1_range = np.arange(0.0, 7.0, 0.1)
theta0_mesh, theta1_mesh = np.meshgrid(theta0_range, theta1_range)
J = cost(X, y, np.ravel(theta0_mesh), np.ravel(theta1_mesh))

ax = fig.add_subplot(2, 2, 3, projection='3d')
ax.plot_surface(theta0_mesh, theta1_mesh, J, cmap='binary')
ax.plot(theta_h[:, 0], theta_h[:, 1], cost_f, color='red')

ax.set_title("Cost function (3D)")
ax.view_init(20, -30)
ax.set_xlabel("theta0")
ax.set_ylabel("theta1")

# plot 4
_ = fig.add_subplot(2, 2, 4).set_title("Cost function (contour)")
plt.contour(theta0_mesh, theta1_mesh, J)
plt.plot(theta_h[:, 0], theta_h[:, 1], marker='o', linestyle='', markersize=4, color='red')

plt.xlabel("theta0")
plt.ylabel("theta1")


plt.tight_layout()
plt.show()
