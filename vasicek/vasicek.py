'''
This file is the entry point of the interest rate option pricing
based on the Vasicek model
'''

import numpy as np
import matplotlib.pyplot as plt

import util
import yield_curve

# dr(t) = (k - theta * r(t))dt + sigma dW(t)

# Parameters
T = 7 # In years
dt = 0.5 # Time step

k = 0.0045 # 0.45%
theta = 0.1 # 10%
sigma = 0.01 # 1%

# Starting point
r0 = 0.046 # 4.6%

# Strike
K = 0.047 # 4.7% 

# Early exercise parameters
tAlpha = 2
tBeta = 5

# Set the zero coupon yield curves
yieldCurve = yield_curve.zc # From the homework description

# Swithc between calculation modes
asCourseMaterial = False

# Sweeping sigma 
sweep = np.linspace(0.001, 0.05, 50)
prices = np.empty_like(sweep)

for idx, actSigma in enumerate(sweep):
    tree = util.Tree(T, dt, k, theta, actSigma, yieldCurve, asCourseMaterial)
    tree.setEarlyExercise(tAlpha, tBeta)
    #print(tree.pricing(K))  
    prices[idx] = tree.pricing(K)

plt.plot(sweep, prices)
plt.xlabel(r'$\sigma$')
plt.ylabel('Swaption price')
plt.savefig('vasicek_sigma_sweep.png')
plt.show()

# Sweeping theta
sweep = np.linspace(0.01, 0.5, 50)
prices = np.empty_like(sweep)

for idx, actTheta in enumerate(sweep):
    tree = util.Tree(T, dt, k, actTheta, sigma, yieldCurve, asCourseMaterial)
    tree.setEarlyExercise(tAlpha, tBeta)
    prices[idx] = tree.pricing(K)

plt.plot(sweep, prices)
plt.xlabel(r'$\theta$')
plt.ylabel('Swaption price')
plt.savefig('vasicek_theta_sweep.png')
plt.show()

# Sweeping 
sweep = np.linspace(0.0001, 0.0099, 50)
prices = np.empty_like(sweep)

for idx, actk in enumerate(sweep):
    tree = util.Tree(T, dt, actk, theta, sigma, yieldCurve, asCourseMaterial)
    tree.setEarlyExercise(tAlpha, tBeta)
    prices[idx] = tree.pricing(K)

plt.plot(sweep, prices)
plt.xlabel('k')
plt.ylabel('Swaption price')
plt.savefig('vasicek_k_sweep.png')
plt.show()
 



 

 


