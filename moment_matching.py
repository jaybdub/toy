import numpy as np
import matplotlib.pyplot as plt
import random

__author__ = 'John'


def find_nearest(array, value):
    return (np.abs(array-value)).argmin()


def gaussian(x, mu, sigma):
    return (1.0 / (sigma * np.sqrt(2*(np.pi)))) * np.exp(-((x - mu)*(x - mu)) / (2.0*sigma*sigma))

# Take a mixture of gaussians comprised of K conditional gaussians.  The resulting distribution
# is the sum of each conditional gaussian times the probability of the variable the CDF is conditioned on.
# Each conditional gaussian has its own mean and variance.  Here we look at a mixture of 9 1 dimensional gaussians.

mixture_weight = np.array([0.1, 0.4, 0.1, 0.2, 0.2])
mean = np.array([1, 2.5, -2, -1.5, 4])
std = np.array([0.2, 2, 1.5, 2, 1])

min_x = -10.0
max_x = 10.0
count = 1000.0
dx = (max_x - min_x) / count
x = np.linspace(min_x, max_x, count)
mixture = np.zeros(x.shape)

# Plot original mixture
for k in range(len(mixture_weight)):
        mixture += gaussian(x, mean[k], std[k]) * mixture_weight[k]

plt.figure(1)
plt.subplot(3,1,1)
plt.plot(x, mixture)
plt.title("Mixture PDF")

# Plot cdf and inverse cdf
cdf = np.cumsum(mixture*dx)
plt.subplot(3,1,2)
plt.plot(x, cdf)
plt.title("Mixture CDF")

# Sample from distribution
num_samples = 1000
samples = np.zeros(num_samples)
for i in range(num_samples):
    y = random.random()
    samples[i] = min_x + (find_nearest(cdf,y) / count) * (max_x - min_x)

plt.subplot(3,1,3)
plt.hist(samples,100,normed=True, range=(-10,10))
plt.show()

print(random.random())