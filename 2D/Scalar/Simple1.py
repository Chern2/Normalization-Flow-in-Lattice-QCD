#import tensorflow as tf
import numpy as np
import numba as nb
import math
import matplotlib.pyplot as plt
#import seaborn as sns

#sns.set_style('whitegrid')

mb_size = 2**14

@nb.vectorize('float32(float32, float32)')
def partionZ1(u1, u2):
    return math.sqrt(-2*math.log(u1))*math.cos(2*math.pi*u2)

@nb.vectorize('float32(float32, float32)')
def partionZ2(u1, u2):
    return math.sqrt(-2*math.log(u1))*math.sin(2*math.pi*u2)


u = np.random.random(size=[mb_size, 2]).astype(np.float32)
z1 = partionZ1(u[:,0], u[:,1])
z2 = partionZ2(u[:,0], u[:,1])

fig, ax = plt.subplots(1,2, dpi=125, figsize=(4,2))
for a in ax:
    a.set_xticks([-2, 0, 2])
    a.set_yticks([-2, 0, 2])
    a.set_aspect('equal')
    ax[0].hist2d(u[:,0], u[:,1], bins=30, range=[[-3.0,3.0], [-3.0,3.0]])
    ax[0].set_xlabel(r"$U_1$")
    ax[0].set_ylabel(r"$U_2$", rotation=0, y=.46)
    ax[1].hist2d(z1, z2, bins=30, range=[[-3.0,3.0], [-3.0,3.0]])
    ax[1].set_yticklabels([])
    ax[1].set_xlabel(r"$Z_1$")
    ax[1].set_ylabel(r"$Z_2$", rotation=0, y=.53)
    ax[1].yaxis.set_label_position("right")
    ax[1].yaxis.tick_right()
plt.show()
