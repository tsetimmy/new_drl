import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

from a import get_data, main0


class convolutional_autoencoder:
    def __init__(self, shape):
        self.x = tf.placeholder(shape=shape, dtype=tf.float32)
        self.conv1 = slim.conv2d(inputs=self.x, num_outputs=64, kernel_size=[8, 8], stride=[1, 1], activation_fn=None, biases_initializer=None, padding='VALID')
        self.conv2 = slim.conv2d(inputs=self.conv1, num_outputs=128, kernel_size=[13, 13], stride=[2, 2], activation_fn=tf.nn.sigmoid, padding='VALID')

        self.deconv1 = slim.conv2d_transpose(self.conv2, 64, 13, 2, padding='VALID', activation_fn=tf.nn.sigmoid)
        self.deconv2 = slim.conv2d_transpose(self.deconv1, shape[-1], 8, 1, activation_fn=None, biases_initializer=None, padding='VALID')

        self.recon_loss = tf.reduce_mean(tf.reduce_sum(slim.flatten(tf.square(self.x - self.deconv2)), axis=-1))
        self.update_model_recon = tf.train.AdamOptimizer(learning_rate=.001).minimize(self.recon_loss)

    def run(self, sess, batch):
        _, recon_loss, recon = sess.run([self.update_model_recon, self.recon_loss, self.deconv2], feed_dict={self.x:batch})
        return _, recon_loss, recon, recon

if __name__ == '__main__':
    model = 'conv'
    dtype = 'breakout'
    mode = 'transformation'
    x, y, dim, size = get_data(dtype, mode)
    ca = convolutional_autoencoder([None, dim, dim, size])
    main0(dtype, mode, model, ca, x, y, dim, size)

