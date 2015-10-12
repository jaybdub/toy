__author__ = 'John'

import numpy as np
import matplotlib.pyplot as plt

# Utility Functions
def gaussian(x, mu, sigma):
    return (1.0 / (sigma * np.sqrt(2*(np.pi)))) * np.exp(-((x - mu)*(x - mu)) / (2.0*sigma*sigma))

# Conditional Probabilities
def P_Yk_given_Xk(yk, xk, params):
    return gaussian(yk, xk, params['sigma_y'])

def P_Xk_given_Xkm_Sk(xk, xkm, Sk, params):
    return gaussian(xk, xkm + params['mu_sk'][Sk], params['sigma_sk'][Sk])

def P_Sk_given_Skm(sk, skm, params):
    return params['p_skm_sk'][skm,sk]

# Parameters
rainfall_rate = 4 # mm / minute
rainfall_rate_std = 0.06 # mm / minute
measurement_std = 2.0 # mm

params = {'sigma_y': measurement_std,
          'mu_sk': np.array([0.0, rainfall_rate]),
          'sigma_sk': np.array([0.001, rainfall_rate_std]),
          'p_skm_sk': np.array([[0.99, 0.01],
                                [0.05,0.95]])}

# Plot conditional densities

# prob observed water level given actual
actual_water_level = 5 #mm
observed_water_level = np.linspace(0.0, 10.0)
plt.figure(1)
plt.plot(observed_water_level, P_Yk_given_Xk(observed_water_level, actual_water_level, params))
plt.xlabel("Observed Water Level (mm)")
plt.ylabel("Probability given real = 5mm")

# plot water level in next step vs rate
plt.figure(2)

plt.subplot(2,1,1)
sk = 0 # not raining
xkm = 5.0 # 5 mm last minute
xk = np.linspace(4.5, 5.5, 1000)
plt.plot(xk, P_Xk_given_Xkm_Sk(xk, xkm, sk, params))
plt.xlabel("Xk")
plt.ylabel("P(Xk | Sk = 0, Xkm = 5)")

plt.subplot(2,1,2)
sk = 1 # raining
xkm = 5.0 # 5 mm last minute
xk = np.linspace(4.5, 5.5, 1000)
plt.plot(xk, P_Xk_given_Xkm_Sk(xk, xkm, sk, params))
plt.xlabel("Xk")
plt.ylabel("P(Xk | Sk = 1, Xkm = 5)")

# plot evolution of rain
plt.figure(3)
plt.imshow(params['p_skm_sk'],interpolation='nearest')
for x in range(params['p_skm_sk'].shape[0]):
    for y in range(params['p_skm_sk'].shape[1]):
        plt.annotate(str(params['p_skm_sk'][x,y]), xy=(y, x),
                    horizontalalignment='center',
                    verticalalignment='center')
plt.title("Rain transition probability")

# FORWARD PASS
# initial probs
mu_x0 = 1 # mm
sigma_x0 = 2# mm
p_s0_1 = 0.3 # prob raining

x1 = np.linspace(-10,10,1000)
px1 = np.zeros(x1.shape)
p_s1_1 = params['p_skm_sk'][0,1]*(1-p_s0_1) + params['p_skm_sk'][1,1]*p_s0_1
px1 = px1 + gaussian(x1, mu_x0 + params['mu_sk'][0], np.sqrt(sigma_x0+params['sigma_sk'][0]))*(1-p_s1_1) \
      + gaussian(x1, mu_x0 + params['mu_sk'][1], np.sqrt(sigma_x0+params['sigma_sk'][1]))*(p_s1_1)
print(np.sum(px1*20/1000))
plt.figure(4)
plt.plot(x1,px1)
plt.show()