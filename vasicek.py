'''
This file is the entry point of the interest rate option pricing
based on the Vasicek model
'''

import util


# Parameters
T = 5 # In years
dt = 0.5 # Time step
#nLevel = int(T / dt)

k = 0.0045 # 0.45%
theta = 0.1 # 10%
sigma = 0.01 # 1%

# Starting point
r0 = 0.046 # 4.6%

# Strike
K = 0.047 # 4.7% 


tree = util.Tree(T, dt, k, theta, sigma)

tree.visualize('pdf')


