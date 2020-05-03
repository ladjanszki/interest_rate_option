import numpy as np

import util


# Parameters
T = 3 # In years
dt = 1# Time step

k = 0.0 
theta = 0.1 # 10%
sigma = 0.01 # 1%

# Starting point
#r0 = 0.046 # 4.6%

# Strike
#K = 0.047 # 4.7% 

nLevel = int(T / dt) + 1

# Zero coupon curve
zeroCoupon = np.zeros(nLevel + 1, dtype=float) 
zeroCoupon[0]  = 0.0
zeroCoupon[1]  = 0.03824
zeroCoupon[2]  = 0.04512
zeroCoupon[3]  = 0.05086
zeroCoupon[4]  = 0.05566

tree = util.Tree(T, dt, k, theta, sigma, zeroCoupon)


print(tree.pricing(0.04))

#df = tree.toPandas()
#print(df)


tree.visualize('pdf')

 
