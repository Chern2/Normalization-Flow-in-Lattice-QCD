import tensorflow as tf
import numpy as np

def tan_transform(x, s, t):
    return tf.mod(2*tf.math.atan(tf.math.exp(s)*tf.math.tan(x/2) + t), 2*np.pi)

def tan_transform_logJ(x, s, t, mask):
    a1 = tf.math.exp(-s/2)*tf.math.cos(x/2)
    a2 = tf.math.exp(s/2)*tf.math.sin(x/2)
    logJ = -mask*tf.math.log(a1**2+(a2+a1*t)**2)
    for dim in range(1, len(logJ.shape)):
        logJ = tf.math.reduce_sum(logJ, 1)
    return logJ


class NCPPlaqCouplingLayer():
    def __init__(self, hidden_sizes, kernel_size, dtype=tf.float64):
        self.hidden_sizes = hidden_sizes
        self.kernal_size = kernel_size
        self.dtype = dtype

    def network(self, net, i_layers, fin_activation=None):
        with tf.variable_scope('GaugeCouplingLayer/'+str(i_layers), reuse=tf.AUTO_REUSE, dtype=self.dtype) as scope:
            for i in range(len(self.hidden_sizes)):
                net = tf.layers.Conv2D(filters=self.hidden_sizes[i], kernel_size=self.kernal_size, strides=1,
                                                        activation=tf.nn.leaky_relu, padding='same')(net)

            s = tf.layers.Conv2D(filters=1, kernel_size=self.kernal_size, strides=1, activation=fin_activation,
                                                        padding='same')(net)
            t = tf.layers.Conv2D(filters=1, kernel_size=self.kernal_size, strides=1, activation=fin_activation,
                                                        padding='same')(net)
            return tf.squeeze(s, -1), tf.squeeze(t, -1)


    def forward(self, x, mask,  i_layers):
        x2 = mask['frozen']*x
        s, t = self.network(tf.stack([tf.math.sin(x2), tf.math.cos(x2)], -1), i_layers)
        #s, t = self.network(tf.expand_dims(tf.math.tan(x2), -1), i_layers)

        x1 = mask['active'] * x

        logJ = tan_transform_logJ(x1,  s, t, mask['active'])
        fx1 =  mask['active']*tan_transform(x1, s, t)
        fx =  fx1  + mask['passive'] * x + mask['frozen'] * x
        return fx, logJ

    @property
    def vars(self):
        var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        return var




