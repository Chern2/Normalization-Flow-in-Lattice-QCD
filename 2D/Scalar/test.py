import tensorflow as tf

sess = tf.InteractiveSession()
x = tf.exp(-1.)*tf.exp(1.) - 1.

print(x.eval())
