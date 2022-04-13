import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
tf = tf.compat.v1

class SimpleCouplingLayer():
    def __init__(self, dtype):
        self.dtype = dtype

    def network(self, inputs):
        with tf.variable_scope('SimpleCouplingLayer', reuse=tf.AUTO_REUSE, dtype=self.dtype) as scope:
            layer1 = tf.layers.Dense(units=8,activation=tf.nn.relu)(inputs)
            layer2 = tf.layers.Dense(units=8,activation=tf.nn.relu)(layer1)
            output = tf.layers.Dense(units=1,activation=tf.nn.tanh)(layer2)
        return output

    def forward(self, x1, x2):
        s = self.network(x2)
        fx1 = tf.math.exp(s)*x1
        fx2 = x2
        logJ = s
        return fx1, fx2, logJ

    def reverse(self, fx1, fx2):
        x2 = fx2
        s = self.network(x2)
        logJ = -s
        x1 = tf.math.exp(-s)*fx1
        return x1, x2, logJ

if __name__ == '__main__':
    dtype = np.float64
    coupling_layer = SimpleCouplingLayer(dtype)

    batch_size = 1024
    x1 = np.random.uniform(size=[batch_size, 1], low=-1, high=1).astype(dtype)
    x2 = np.random.uniform(size=[batch_size, 1], low=-1, high=1).astype(dtype)

    gx1, gx2, fwd_logJ = coupling_layer.forward(x1, x2)
    xp1, xp2, bwd_logJ = coupling_layer.reverse(gx1, gx2)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    sess.graph.finalize()

    np_gx1 = gx1.eval()
    np_gx2 = gx2#.eval()
    np_xp1 = xp1.eval()
    np_xp2 = xp2#.eval()

    fig, ax = plt.subplots(1,3, dpi=125, figsize=(6,2.3), sharex=True, sharey=True)
    for a in ax:
        a.set_xlim(-1.1,1.1)
        a.set_ylim(-1.1,1.1)

    ax[0].scatter(np.squeeze(x1, -1), np.squeeze(x2,-1), marker='.')
    ax[0].set_title(r'$x$')
    ax[1].scatter(np.squeeze(np_gx1,-1), np.squeeze(np_gx2,-1), marker='.', color='tab:orange')
    ax[1].set_title(r'$g(x)$')
    ax[2].scatter(np.squeeze(np_xp1,-1), np.squeeze(np_xp2,-1), marker='.')
    ax[2].set_title(r"$g^{-1}(g(x))$")
    fig.set_tight_layout(True)
    plt.savefig('test.pdf',format='pdf')
