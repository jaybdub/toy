__author__ = 'John'

import numpy as np
import matplotlib.pyplot as plt

def gaussian(x, mu, sigma):
    return (1.0 / (sigma * np.sqrt(2*(np.pi)))) * np.exp(-((x - mu)*(x - mu)) / (2.0*sigma*sigma))

# Take a mixture of gaussians comprised of K conditional gaussians.  The resulting distribution
# is the sum of each conditional gaussian times the probability of the variable the CDF is conditioned on.
# Each conditional gaussian has its own mean and variance.  Here we look at a mixture of 5 1 dimensional gaussians.

mean = [1, 2.5, -2, -1.5, 0.5]
std = [0.5, 2, 1, 0.2, 3]

x = np.linspace(-10,10,1000)
mixture = np.zeros(x.shape)

for k in range(len(mean)):
    mixture += gaussian(x, mean[k], std[k])

plt.plot(x, mixture)
plt.show()