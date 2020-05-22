import numpy as np

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

# Calculate zero coupon curve
# TODO: is this right???

# Starting point
r0 = 0.046 # 4.6%

forward = forward / 2
#print(forward)

tmp = forward.tolist()
#print(tmp)

zc = np.zeros(16, dtype=float)
zc[0] = r0
for i in range(len(tmp)):
    #print(tmp[i])
    cumsum = 0
    for j in range(i + 1):
        cumsum += tmp[j]
    zc[i + 1] = cumsum / ((i + 1) / 2)

# Generate zero copupon curve according to Hull book
genZc = np.zeros(16, dtype=float)

for i in range(16):
    genZc[i] = 0.08 - 0.05 * np.exp(-0.18 * float((i + 1) /2))


