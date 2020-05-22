import numpy as np
import matplotlib.pyplot as plt

import util
import yield_curve

# dr(t) = (k - theta * r(t))dt + sigma * dW(t) (Vasicek)
# dr(t) = (alpha - beta r(t))dt + sigma * sqrt(r(t)) * dW(t) (CIR)

# Parameters
T = 7 # In years
dt = 0.5 # Time step

# CIR model
alpha = 0.0045 #0.45%
beta = 0.1 # 10%
sigma = 0.0447 # 4.47%

# Starting point
r0 = 0.046 # 4.6%

# Strike
K = 0.047 # 4.7% 

# Early exercise parameters
tAlpha = 2
tBeta = 5

# Set the zero coupon yield curves
yieldCurve = yield_curve.zc # From the homework description

# Sweeping sigma 
trimTree = True
sweep = np.linspace(0.045, 0.3, 50)
prices = np.empty_like(sweep)

for idx, actSigma in enumerate(sweep):
    tree = util.Tree(T, dt, alpha, beta, actSigma, yieldCurve, trimTree)
    tree.setEarlyExercise(tAlpha, tBeta)
    #print(tree.pricing(K))  
    prices[idx] = tree.pricing(K)

plt.plot(sweep, prices)
plt.xlabel(r'$\sigma$')
plt.ylabel('Swaption price')
plt.savefig('cir_sigma_sweep.png')
plt.show()

# Sweeping alpha
trimTree = False
sweep = np.linspace(0.0045, 0.0099, 50)
prices = np.empty_like(sweep)

for idx, actAlpha in enumerate(sweep):
    tree = util.Tree(T, dt, actAlpha, beta, sigma, yieldCurve, trimTree)
    tree.setEarlyExercise(tAlpha, tBeta)
    #print(tree.pricing(K))  
    prices[idx] = tree.pricing(K)

plt.plot(sweep, prices)
plt.xlabel(r'$\alpha$')
plt.ylabel('Swaption price')
plt.savefig('cir_alpha_sweep.png')
plt.show()

# Sweeping alpha
trimTree = True
sweep = np.linspace(0.001, 0.3, 50)
prices = np.empty_like(sweep)

for idx, actBeta in enumerate(sweep):
    tree = util.Tree(T, dt, alpha, actBeta, sigma, yieldCurve, trimTree)
    tree.setEarlyExercise(tAlpha, tBeta)
    #print(tree.pricing(K))  
    prices[idx] = tree.pricing(K)

plt.plot(sweep, prices)
plt.xlabel(r'$\beta$')
plt.ylabel('Swaption price')
plt.savefig('cir_beta_sweep.png')
plt.show()
 

 
 



 

 


