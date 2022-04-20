# That code loads some example data relating size of a house and number of bedrooms with its price. It is a basic
# exercise for me to use machine learning and find the best model fitting the data using just numpy and matplotlib.
# Now, after writing this script I am completely unsure if it works properly even though it gives me results close to
# what was expected. As I started learning theory of machine learning quite recently it was a challenge for me to put it
# together, and I possibly will come back to it in future to see what could've been done differently.

import numpy as np
from matplotlib import pyplot as plt

x = []
y = []

# open the house file with feature data
with open("house_sizes.txt", "r") as hs:
    hs_lines = hs.readlines()

# clean it
for line in hs_lines:
    if "\n" in line:
        line = line.replace("\n", "")
    x.append([int(line.split(',')[0]), int(line.split(',')[1])])

# open the file with labels
with open("house_prices.txt", "r") as hp:
    hp_lines = hp.readlines()

# clean it
for line in hp_lines:
    if "\n" in line:
        line = line.replace("\n", "")
    y.append([int(line)])

# convert lists to numpy arrays
x = np.array(x)
y = np.array(y)

# hypothesis = theta_0 + theta_1*house_price + theta_2*no_of_bedrooms
# initialize with some arbitrary values
theta = np.array([
    [1],
    [1],
    [1]
])

# m = number of examples
m = x.shape[0]  # 'height' of the matrix

# alpha = learning rate / step
alpha = 0.0001

# insert a column of ones to the x matrix as it needs to match dimensions with theta
x_0 = np.ones((m, 1))
x_ = np.hstack((x_0, x))

# mean normalization
avg_x = np.average(x_[:, 1])
std_dev_x = np.amax(x_[:, 1]) - np.amin(x_[:, 1])
x_[:, 1] = (x_[:, 1] - avg_x) / std_dev_x

avg_y = np.average(y)
std_dev_y = np.amax(y) - np.amin(y)
y = (y - avg_y) / std_dev_y


# hypothesis
def hypothesis(th): return x_ @ th


# derivative of the cost function
def d_j(hyp): return (1 / m) * x_.T @ (hyp - y)


h = hypothesis(theta)
d_j = d_j(h)


# calculating vector of weights
theta = theta - alpha * d_j

# calculating new hypothesis. There was an error here - the line of best fit was shifted to the right. I don't want to
# take into account the second feature as it makes it more difficult later to plot. Quick and visually acceptable fix:
# subtract 1 from the equation. I am leaving it like that as I would rather move on to another project currently and
# explore some more convenient way of dealing with ML problems in python
h = lambda x_1: theta[0] + theta[1] * x_1
x1_range = np.linspace(float(np.amin(x_[:, 1])), float(np.amax(x_[:, 1])), num=100)

# plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle("House price")

ax1.set(xlabel="Size (normalized)", ylabel="Price (normalized)")
ax1.plot(x_[:, 1], y, marker='x', linestyle='')
ax1.plot(h(x1_range), x1_range, linestyle='-')

# that one just to play with two plots side by side
ax2.set(xlabel="# of bedrooms", ylabel="Price (normalized)")
ax2.plot(x_[:, 2], y, marker='x', linestyle='')

plt.show()
