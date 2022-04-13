from SimpleNormal import SimpleNormal
from ScalarPhi4Action import ScalarPhi4Action
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

lattice_shape = (8, 8)
prior = SimpleNormal(tf.zeros(lattice_shape), tf.ones(lattice_shape))

sess = tf.InteractiveSession()
samples = prior.sample_n(1024)
z = samples.eval()
    
fig, ax = plt.subplots(4,4, dpi=125, figsize=(4,4))
for i in range(4):
    for j in range(4):
        ind = i*4 + j
        ax[i,j].imshow(np.tanh(z[ind]), vmin=-1, vmax=1, cmap='viridis')
        ax[i,j].axes.xaxis.set_visible(False)
        ax[i,j].axes.yaxis.set_visible(False)
plt.savefig('prior_Correlations.pdf', format='pdf')
print(f'z.shape = {z.shape}')


phi4_action = ScalarPhi4Action(M2=-4, lam=8.0)

S_eff = -prior.log_prob(samples).eval()
S = phi4_action(samples).eval()
fit_b = np.mean(S) - np.mean(S_eff)
print(f'slope 1 linear regression S = -logr + {fit_b:.4f}')
fig, ax = plt.subplots(1,1, dpi=125, figsize=(4,4))
ax.hist2d(S_eff, S, bins=20, range=[[-800, 800], [200,1800]])
xs = np.linspace(-800, 800, num=4, endpoint=True)
ax.plot(xs, xs + fit_b, ':', color='w', label='slope 1 fit')
ax.set_xlabel(r'$S_{\mathrm{eff}} \equiv -\log~r(z)$')
ax.set_ylabel(r'$S(z)$')
ax.set_aspect('equal')
plt.legend(prop={'size': 6})
plt.savefig('effective_action.pdf', format='pdf')
