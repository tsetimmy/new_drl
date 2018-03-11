import numpy as np
import tensorflow as tf

class gaussian_process:
    def __init__(self, xshape):
        self.xshape = xshape

        #Placeholders
        self.x = tf.placeholder(shape=self.xshape, dtype=tf.float64)
        #self.batch_size = tf.cast(tf.shape(self.x)[0], tf.float64)

        #Kernel
        self.kernel = self.squared_exponential_kernel(self.x, self.x)

        #Cholesky decomposition
        self.noise_variance = 5e-5#to be optimized
        self.L = tf.cholesky(self.kernel + tf.diag(self.noise_variance * tf.ones_like(self.x)))

    def squared_exponential_kernel(self, a, b):
        self.kernel_param = .1#to be optimized
        sqdist = tf.reduce_sum(tf.square(a), axis=-1, keep_dims=True) +\
                 -2. * tf.matmul(a, tf.transpose(b)) +\
                 tf.transpose(tf.reduce_sum(tf.square(b), axis=-1, keep_dims=True))
        self.sqdist = sqdist
        return tf.exp(-.5 * (1./self.kernel_param) * sqdist)

def main():

    gp = gaussian_process([None, 2])
    exit()

    x = np.array([[1, 2], [3, 4]])

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        k = sess.run(gp.kernel, feed_dict={gp.x:x})
        print k
        #print sqdist

if __name__ == '__main__':
    main()

