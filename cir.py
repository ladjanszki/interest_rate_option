import util

# Original data
#T = 0, ..., 7 # There are 7 discrete time points
#print(T)
dt = 0.5 # Time step

# CIR model
alpha = 0.0045 #0.45%
beta = 0.1 # 10%
sigCir = 0.0447 # 4.47%

# Starting point
r0 = 0.046 # 4.6%

# Strike
K = 0.047 # 4.7% 


# Forward rates
# 0 = 4.5
# 0.5 = 5
# 1 = 5.5
# 1.5 = 4.5
# 2 = 4
# 2.5 = 4.5
# 3 = 4.8
# 3.5 = 5
# 4 = 5
# 4.5 = 4.4
# 5 = 4.5 
# 5.5 = 4.8
# 6 = 4.5
# 6.5 = 4.2
# 7 = 4.3


tree = util.Tree(3)

tree.addConnections()

tree.visualize()

 
