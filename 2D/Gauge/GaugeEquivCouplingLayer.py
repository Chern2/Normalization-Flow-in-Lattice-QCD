import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from Gauge_masks import *
from U1GaugeAction import U1GaugeAction
from NCPplaqCouplingLayer import NCPPlaqCouplingLayer
from MultivariateUniform import MultivariateUniform


class GaugeEquivCouplingLayer():
    def __init__(self, lattice_shape, Action, plaq_coupling, n_layers):
        self.links_shape = (len(lattice_shape), ) + lattice_shape
        self.lattice_shape = lattice_shape
        self.Action = Action
        self.plaq_coupling = plaq_coupling
        self.n_layers = n_layers


    def forward(self, x, i_layers):
        mask = make_2d_link_active_stripes(self.links_shape, i_layers%2, (i_layers//2)%4)
        plaq = self.Action.U1pladuette(x, mu=0, nv=1)
        plaq_masks = make_plaq_masks(lattice_shape, i_layers%2, (i_layers//2)%4)
        new_plaq, logJ = self.plaq_coupling.forward(plaq, plaq_masks, i_layers)
        delta_plaq = new_plaq - plaq
        delta_links = tf.stack((delta_plaq, -delta_plaq), 1)
        fx = mask * tf.math.mod(delta_links + x, 2*np.pi) + (1-mask) * x
        return fx, logJ

#    def reverse(self, fx, i_layers,  plaq_coupling):
#        mask = make_2d_link_active_stripes(self.links_shape, i_layers%2, (i_layers//2)%4)
#        new_plaq = self.Action.U1pladuette(fx, mu=0, nv=1)
#        plaq_masks = make_plaq_masks(self.lattice_shape, i_layers%2, (i_layers//2)%4)
#        plaq, logJ = plaq_coupling.reverse(new_plaq, plaq_mask, i_layers)
#        delta_plaq = plaq - new_plaq
#        delta_links = tf.stack((delta_plaq, -delta_plaq), 1) 
#        x = mask * tf.math.mod(delta_links + fx, 2*np.pi) + (1-mask) * fx
#        return x, logJ


    def Normalization_Flow(self, prior, batch_size, inputs=None):
        if inputs is None:
            inputs = prior.sample_n(batch_size)
        logq = prior.log_prob(inputs)
        for i_layers in range(self.n_layers):
            inputs, logJ = self.forward(inputs, i_layers)
            logq -=  logJ

        return inputs, logq

    def train(self, prior, batch_size, Nera, epoch, sess):
        x, logq = self.Normalization_Flow(prior, batch_size)
        logp = -self.Action.U1Action(x)

        #_, loss = tf.nn.moments(logq - logp, axes=0)
        loss = tf.reduce_mean((logq - logp))

        solver = tf.train.AdamOptimizer(1e-4).minimize(loss)
        sess.run(tf.global_variables_initializer())
        sess.graph.finalize()
        for i in range(Nera):
            for j in range(epoch):
                sess.run(solver)
                #pass
            loss_v, logp_np, logq_np = sess.run([loss, logp, logq])
            print(i, loss_v, np.mean(logp_np), np.mean(logq_np))



if __name__ == '__main__':
    L = 8
    lattice_shape = (L, L)
    link_shape = (len(lattice_shape), ) + lattice_shape
    Action = U1GaugeAction(2.)
    plaq_coupling = NCPPlaqCouplingLayer(hidden_sizes=[16,16], kernel_size=4)
    GaugeCoupling = GaugeEquivCouplingLayer(lattice_shape=lattice_shape, Action=Action, plaq_coupling=plaq_coupling, n_layers=32)
    prior = MultivariateUniform(tf.zeros(shape=link_shape), 2*np.pi*tf.ones(shape=link_shape))
    sess = tf.InteractiveSession()
    x_tf, log_tf = GaugeCoupling.Normalization_Flow(prior, batch_size=1024)
    S_tf = Action.U1Action(x_tf)
    s_tf = - log_tf
    GaugeCoupling.train(prior=prior, batch_size=64, Nera=100, epoch=200, sess=sess)


    x, S_eff, S = sess.run([x_tf, s_tf, S_tf])
    fit_b = np.mean(S) - np.mean(S_eff)
    print(f'slope 1 linear regression S = S_eff + {fit_b:.4f}')
    fig, ax = plt.subplots(1,1, dpi=125, figsize=(4,4))
    ax.hist2d(S_eff, S, bins=20, range=[[130, 230], [-250, -150]])
    #ax.hist2d(S_eff, S, bins=20, range=[[100, 180], [-180, -100]])
    ax.set_xlabel(r'$S_{\mathrm{eff}} = -\log~q(x)$')
    ax.set_ylabel(r'$S(x)$')
    ax.set_aspect('equal')
    xs = np.linspace(130, 230, num=4, endpoint=True)
    #xs = np.linspace(100, 180, num=4, endpoint=True)
    ax.plot(xs, xs + fit_b, ':', color='w', label='slope 1 fit')
    plt.legend(prop={'size': 6})
    plt.savefig('gauge_action.pdf', format='pdf')
