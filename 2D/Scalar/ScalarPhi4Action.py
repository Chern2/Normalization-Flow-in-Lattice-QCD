import tensorflow as tf
import numpy as np


class ScalarPhi4Action():
    def __init__(self, M2, lam):
        self.M2 = M2
        self.lam = lam

    def __call__(self, cfgs):
        # potential term
        action_density = self.M2*cfgs**2 + self.lam*cfgs**4
        # kinetic term (discrete Laplacian)
        for mu in range(1, len(cfgs.shape)):
            action_density += 2*cfgs**2
            action_density -= cfgs*tf.roll(cfgs, shift=-1, axis=mu)
            action_density -= cfgs*tf.roll(cfgs, shift=1, axis=mu)
        
        for mu in range(1, len(cfgs.shape)):
            action_density = tf.math.reduce_sum(action_density, axis=1)
        return action_density


if __name__ == '__main__':
    L = 8
    lattice_shape = (L,L)
    float_dtype = np.float64
    phi_ex1 = np.random.normal(size=lattice_shape).astype(float_dtype)
    phi_ex2 = np.random.normal(size=lattice_shape).astype(float_dtype)
    cfgs = np.stack((phi_ex1, phi_ex2), axis=0)
    sess = tf.InteractiveSession()
    Action = ScalarPhi4Action(M2=1.0, lam=1.0)(cfgs)
    print(Action.eval())
