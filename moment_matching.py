import numpy as np
import matplotlib.pyplot as plt

__author__ = 'John'


def gaussian(x, mu, sigma):
    return (1.0 / (sigma * np.sqrt(2*(np.pi)))) * np.exp(-((x - mu)*(x - mu)) / (2.0*sigma*sigma))

# Take a mixture of gaussians comprised of K conditional gaussians.  The resulting distribution
# is the sum of each conditional gaussian times the probability of the variable the CDF is conditioned on.
# Each conditional gaussian has its own mean and variance.  Here we look at a mixture of 9 1 dimensional gaussians.

I = 3
J = 3
joint_prob = np.array([[0.1, 0.1 ,0.1],
                       [0.1, 0.1, 0.2],
                       [0.1, 0.1, 0.1]])
mean = np.array([[1, 2.5, -2],
                 [-1.5, 0.5, 5],
                 [0, 3, -3.5]])
std = np.array([[0.2, 2, 1.5],
                [2, 3, 1],
                [2, 4, 2]])

x = np.linspace(-10,10,1000)
mixture = np.zeros(x.shape)

# Plot original mixture
for i in range(I):
    for j in range(J):
        mixture += gaussian(x, mean[i, j], std[i, j]) * joint_prob[i, j]

plt.figure(1)
plt.subplot(2,1,1)
plt.plot(x, mixture)

# Plot moment matched mixture
marg_prob = np.zeros(I)
cond_prob = np.zeros(joint_prob.shape)
mean_i = np.zeros(I)
std_i = np.zeros(I)
for i in range(I):
    marg_prob[i] = np.sum(joint_prob[i,:])
    for j in range(J):
        cond_prob[i,j] = joint_prob[i,j] / marg_prob[i]
        mean_i[i] = mean[i,j] * cond_prob[i,j]

    var_i = 0
    for j in range(J):
        var_i += std[i,j]**2 * cond_prob[i, j] + (mean[i, j] - mean_i[i])**2 * cond_prob[i, j]

    std_i[i] = np.sqrt(var_i)

matched_mixture = np.zeros(x.shape)
for i in range(I):
    matched_mixture += gaussian(x, mean_i[i], std_i[i]) * marg_prob[i]

print(std_i)
plt.subplot(2,1,2)
plt.plot(x, matched_mixture)
plt.show()