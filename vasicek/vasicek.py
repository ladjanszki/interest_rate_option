'''
This file is the entry point of the interest rate option pricing
based on the Vasicek model
'''

import numpy as np
import matplotlib.pyplot as plt

import util
import yield_curve


# Parameters
T = 7 # In years
#T = 2 # In years
dt = 0.5 # Time step
#nLevel = int(T / dt)

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


#plt.plot(yield_curve.genZc)
#plt.show()


tree = util.Tree(T, dt, k, theta, sigma, yield_curve.genZc)


#Settgin early exercise for all nodes
tree.setEarlyExercise(tAlpha, tBeta)

print(tree.pricing(K))

tree.visualize('pdf')


