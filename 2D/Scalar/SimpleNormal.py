import tensorflow as tf
import numpy as np

tfd = tf.distributions
#contrib.distributions

class SimpleNormal():
    def __init__(self, loc, var, dtpye=tf.float32):
        self.dist = tfd.Normal(loc=loc, scale=var)
        self.shape = loc.shape

    def log_prob(self, x):
        logp = self.dist.log_prob(x)
        for i in range(1, len(x.shape)):
            logp = tf.math.reduce_sum(logp, 1)
        return logp 
	
    def sample_n(self, batch_size):
        x = self.dist.sample(batch_size)
        return x

if __name__ =='__main__':
    sess = tf.InteractiveSession()
    normal_prior = SimpleNormal(tf.zeros([3,4,5]), tf.ones([3,4,5]))
    z = normal_prior.sample_n(17).eval()
    z_prob = normal_prior.log_prob(z).eval()
    print(np.shape(z))
    print(z_prob)

    sess.close()
