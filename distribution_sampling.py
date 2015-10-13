# Demonstrates a simple method of sampling from an arbitrary distribution using a lookup table
# CDF.

import numpy as np
import matplotlib.pyplot as plt
import random

__author__ = 'John'


def find_nearest(array, value):
    return (np.abs(array-value)).argmin()


def gaussian(x, mu, sigma):
    return (1.0 / (sigma * np.sqrt(2*(np.pi)))) * np.exp(-((x - mu)*(x - mu)) / (2.0*sigma*sigma))


def sample_cdf(cdf, domain):
    N = float(len(cdf))
    y = random.random()
    return min_x + (find_nearest(cdf, y) / N) * (domain[1] - domain[0])

# Define the PDF params.  The PDF is a mixture of two gaussians.
mean = [-2,2]
std = [1, 0.5]
weight = [0.5, 0.5]

# Create the domain
min_x = -5.0
max_x = 5.0
N = 1000.0
step = (max_x - min_x) / N
x = np.linspace(min_x, max_x, N)

# Generate the PDF
pdf = gaussian(x, mean[0], std[0])*weight[0] + gaussian(x, mean[1], std[1])*weight[1]

# Plot the PDF
plt.figure(1)
plt.subplot(2,1,1)
plt.plot(x, pdf)
plt.title("Probability Density Function (PDF)")

# Compute the CDF
cdf = np.cumsum(pdf*step)

# Plot the CDF
plt.subplot(2,1,2)
plt.plot(x,cdf)
plt.title("Cumulative Density Function (CDF)")
plt.figure(2)

# Sample from distribution for...

# 10^2 Samples
num_samples = 100
samples = np.zeros(num_samples)
for i in range(num_samples):
    y = random.random()
    samples[i] = min_x + (find_nearest(cdf, y) / N) * (max_x - min_x)
plt.subplot(3,1,1)
plt.hist(samples,100, normed=True, range=(-6,6))
plt.title("10^2 Samples")

# 10^3 Samples
num_samples = 1000
samples = np.zeros(num_samples)
for i in range(num_samples):
    samples[i] = sample_cdf(cdf, (min_x, max_x))
plt.subplot(3,1,2)
plt.hist(samples,100, normed=True, range=(-6,6))
plt.title("10^3 Samples")

# 10^4 Samples
num_samples = 10000
samples = np.zeros(num_samples)
for i in range(num_samples):
    samples[i] = sample_cdf(cdf, (min_x, max_x))
plt.subplot(3,1,3)
plt.hist(samples,100, normed=True, range=(-6,6))
plt.title("10^4 Samples")

plt.show()