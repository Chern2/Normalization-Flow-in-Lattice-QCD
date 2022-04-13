import tensorflow as tf 
import numpy as np

tfd = tfd = tf.distributions

class MultivariateUniform():
    def __init__(self, a, b):
        self.dist = tfd.Uniform(low=a, high=b)

    def log_prob(self, x):
        logp = self.dist.log_prob(x)
        for i in range(1, len(x.shape)):
            logp = tf.math.reduce_sum(logp, 1)
        return logp

    def sample_n(self, batch_size):
        return self.dist.sample(batch_size)


if __name__ == '__main__':
    L = 8
    lattice_shape = (L,L)
    link_shape = (2,L,L)

    prior = MultivariateUniform(tf.zeros(link_shape), 2*np.pi*tf.ones(link_shape))
    z = prior.sample_n(17)
    prob = prior.log_prob(z)
    sess = tf.InteractiveSession()
    print(z.shape)
    print(prob.eval())
