'''
This file is the entry point of the interest rate option pricing
based on the Vasicek model
'''

import numpy as np

import util


# Parameters
#T = 7 # In years
T = 2 # In years
dt = 0.5 # Time step
#nLevel = int(T / dt)

k = 0.0045 # 0.45%
theta = 0.1 # 10%
sigma = 0.01 # 1%

# Starting point
r0 = 0.046 # 4.6%

# Strike
K = 0.047 # 4.7% 

# Forward rates
forward = np.zeros(15, dtype=float) 
forward[0] = 0.045
forward[1] = 0.050
forward[2] = 0.055
forward[3] = 0.045
forward[4] = 0.040
forward[5] = 0.045
forward[6] = 0.048
forward[7] = 0.050
forward[8] = 0.050
forward[9] = 0.044
forward[10] = 0.045 
forward[11] = 0.048
forward[12] = 0.045
forward[13] = 0.042
forward[14] = 0.043
 

# Zero coupon curve
#zeroCoupon = np.cumsum(forward).tolist()
zeroCoupon = np.cumsum(forward).tolist()

tree = util.Tree(T, dt, k, theta, sigma, zeroCoupon)

#print(tree.pricing(K))

tree.visualize('pdf')


