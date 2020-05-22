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
#yieldCurve = yield_curve.genZc  # Hull book example
yieldCurve = yield_curve.zc # From the homework description

# Original example
tree = util.Tree(T, dt, k, theta, sigma, yieldCurve, False)
tree.setEarlyExercise(tAlpha, tBeta)
print(tree.pricing(K))
#tree.visualize('pdf')

 
