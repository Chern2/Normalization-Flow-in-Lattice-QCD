import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from ScalarPhi4Action import ScalarPhi4Action
from SimpleNormal import SimpleNormal

tf = tf.compat.v1

def make_checker_mask(shape, parity):
    checker = np.ones(shape) - parity
    checker[::2, ::2] = parity
    checker[1::2, 1::2] = parity
    return checker

#print("For example this is the mask for an 8x8 configuration:\n",
#make_checker_mask(lattice_shape, 0))



class AffineCoupling():
    def __init__(self, mask_shape, hidden_sizes, kernal_size, n_layers, dtype):
        self.hidden_sizes = hidden_sizes
        self.kernal_size = kernal_size
        self.mask_shape = mask_shape
        self.n_layers = n_layers
        self.dtype = dtype
        self.mask_zeros = make_checker_mask(self.mask_shape, 0)
        self.mask_ones = make_checker_mask(self.mask_shape, 1)

    def affine_layers(self, net, i_layers, fin_activation=tf.nn.tanh):
        with tf.variable_scope('AffineCouplingLayer/'+str(i_layers), reuse=tf.AUTO_REUSE, dtype=self.dtype) as scope:
            for i in range(len(self.hidden_sizes)):
                net = tf.layers.Conv2D(filters=self.hidden_sizes[i], kernel_size=self.kernal_size, strides=1, 
                                            #            activation=None, padding='same')(net)
                                                        activation=tf.nn.leaky_relu, padding='same')(net)

            s = tf.layers.Conv2D(filters=1, kernel_size=self.kernal_size, strides=1, activation=fin_activation,
                                                        padding='same')(net)
            t = tf.layers.Conv2D(filters=1, kernel_size=self.kernal_size, strides=1, activation=fin_activation,
                                                        padding='same')(net)
            return tf.squeeze(s, -1), tf.squeeze(t, -1)

    def forward(self, x, i_layers):
        if i_layers%2 == 0:
            mask = self.mask_zeros
        else:
            mask = self.mask_ones
       
        x_active = (1-mask)*x
        x_frozen = mask*x

        s, t = self.affine_layers(tf.expand_dims(x_frozen, -1), i_layers)
        fx = (1 - mask) * t + x_active * tf.math.exp(s) + x_frozen
        logJ = s*(1 - mask)

        for dim in range(1, len(self.mask_shape)+1):
            logJ = tf.reduce_sum(logJ, -1)
        return fx, logJ

    def reverse(self, fx, i_layers):
        if i_layers%2 == 0:
            mask = self.mask_zeros
        else:
            mask = self.mask_ones

        fx_frozen = mask*fx
        fx_active = (1 - mask) * fx

        s, t = self.affine_layers(tf.expand_dims(fx_frozen,-1), i_layers)
        logJ = -(1 - mask)*s
        x = (fx_active - (1 - mask) * t) * tf.math.exp(-s) + fx_frozen

        for im in range(1, len(self.mask_shape)+1):
            logJ = tf.reduce_sum(logJ, -1)
        return x, logJ

    def make_phi4_affine_layers(self, prior, batch_size, inputs=None):
        if inputs is None:
            inputs = prior.sample_n(batch_size)

        logq = prior.log_prob(inputs)
        for i_layers in range(self.n_layers):
            inputs, logJ = self.forward(inputs, i_layers)
            logq -=  logJ

        return inputs, logq

    def train(self, prior, batch_size, Nera, epoch, Action, sess):
        x, logq = self.make_phi4_affine_layers(prior, batch_size)
        logp = - Action(x)
        loss = tf.reduce_mean(logq - logp)
        solver = tf.train.AdamOptimizer(1e-3).minimize(loss, var_list=self.vars)

        sess.run(tf.global_variables_initializer())
        sess.graph.finalize()
        for i in range(Nera):
            for j in range(epoch):
                sess.run(solver)
                
            loss_v = sess.run(loss)
            print(i, loss_v)

    @property
    def vars(self):
        var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        return var

    def test(self, prior, batch_size):
        inputs = prior.sample_n(batch_size)
        diff = 0
        for i_layers in range(self.n_layers):
            inputs_f, logJ = self.forward(inputs, i_layers)
            inputs_g, logJ_inv = self.reverse(inputs_f, i_layers)
            diff = inputs - inputs_g

        return diff



if __name__ == '__main__':
    lattice_shape = [8,8]
    batch_size = 64
    Nera = 250
    epoch = 100

    Action = ScalarPhi4Action(M2=-4, lam=8.0)
    prior = SimpleNormal(loc=tf.zeros(lattice_shape), var = tf.ones(lattice_shape))
    Normalization_flow = AffineCoupling(mask_shape=lattice_shape, hidden_sizes=[8, 8], 
            kernal_size=3, n_layers=16, dtype=tf.float32)
    x_tf, logq_tf = Normalization_flow.make_phi4_affine_layers(prior, batch_size=batch_size)
    S_tf = -Action(x_tf)

    sess = tf.InteractiveSession()
    Normalization_flow.train(prior=prior, batch_size=batch_size, Nera=Nera, 
                                                    epoch=epoch, Action=Action, sess=sess)

    x, S_eff, S = sess.run([x_tf, logq_tf, S_tf])
    fig, ax = plt.subplots(4,4, dpi=125, figsize=(4,4))
    for i in range(4):
        for j in range(4):
            ind = i*4 + j
            ax[i,j].imshow(np.tanh(x[ind]), vmin=-1, vmax=1, cmap='viridis')
            ax[i,j].axes.xaxis.set_visible(False)
            ax[i,j].axes.yaxis.set_visible(False)
    plt.savefig('phi4.pdf', format='pdf')

    fit_b = np.mean(S) - np.mean(S_eff)
    print(f'slope 1 linear regression S = S_eff + {fit_b:.4f}')
    fig, ax = plt.subplots(1,1, dpi=125, figsize=(4,4))
    ax.hist2d(S_eff, S, bins=20, range=[[5, 35], [-5, 25]])
    ax.set_xlabel(r'$S_{\mathrm{eff}} = -\log~q(x)$')
    ax.set_ylabel(r'$S(x)$')
    ax.set_aspect('equal')
    xs = np.linspace(5, 35, num=4, endpoint=True)
    ax.plot(xs, xs + fit_b, ':', color='w', label='slope 1 fit')
    plt.legend(prop={'size': 6})
    plt.savefig('phi4_action.pdf',format='pdf')
