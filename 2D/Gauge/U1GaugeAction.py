import tensorflow as tf
import numpy as np
from MultivariateUniform import MultivariateUniform
tf = tf.compat.v1

class U1GaugeAction():
    def __init__(self, beta):
        self.beta = beta

    def U1Action(self, cfgs):
        action = 0
        Nd = cfgs.shape[1]
        for mu in range(Nd):
            for nv in range(mu+1, Nd):
                action = action + tf.math.cos(self.U1pladuette(cfgs, mu, nv))
        for dim in range(1, Nd+1):
             action = self.beta*tf.math.reduce_sum(action, 1)
        return -action

    def U1pladuette(self, links, mu, nv):
        pladu = links[:,mu] + tf.roll(links[:,nv], -1, mu+1) - tf.roll(links[:,mu], -1, nv+1) - links[:,nv]
        return pladu

    def gauge_transform(self, links, alpha):
        for mu in range(links.shape[1]):
            links[:,mu] = links[:,mu] + alpha - tf.roll(alpha, -1, mu+1)
        return links

    def random_gauge_transform(self, links):
        Nconf, Nd, VolShape = links.shape[0], links.shape[1], links.shape[2:]
        alpha = 2*np.pi*tf.random_uniform(shape=(Nconf,)+VolShape, dtype=links.dtype)
        Alpha = tf.stack([alpha - tf.roll(alpha, -1, mu+1) for mu in range(Nd)], axis=1)
        return links+Alpha

    def topo_charge(self, links):
        P01 = tf.math.mod(self.U1pladuette(links, mu=0, nv=1)+np.pi, 2*np.pi)
        P01 = P01 - np.pi
        for dim in range(1, len(P01.shape)):
            P01 = tf.math.reduce_sum(P01, 1)
        return P01/(2*np.pi)

if __name__ == '__main__':
    L = 8
    lattice_shape = (L,L)
    link_shape = (2,L,L)
    float_dtype = np.float64
    u1_ex1 = 2*np.pi*np.random.random(link_shape)#.astype(float_dtype)
    u1_ex2 = 2*np.pi*np.random.random(size=link_shape)#.astype(float_dtype)
    cfgs = np.stack([u1_ex1, u1_ex2], axis=0)


    print(cfgs[:,1].shape)
    #2*np.pi*tf.random_uniform(shape=(2,8,8), dtype=x.dtype)
    #print()
    U1Gauge = U1GaugeAction(1)

    cfgs_transformed = U1Gauge.random_gauge_transform(cfgs)
    action = U1Gauge.U1Action(cfgs)
    action_transformed = U1Gauge.U1Action(cfgs_transformed)
    Q = U1Gauge.topo_charge(cfgs)
    plad = U1Gauge.U1pladuette(cfgs, 0, 1)
    sess = tf.InteractiveSession()
    a1, a2 = sess.run([action, action_transformed])
    print(a1, 'vs', a2)
    print(Q.eval())

    print(plad.shape)
    #print(cfgs.shape)
#    print(U1Gauge.U1Action(cfgs).eval())
