import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

class multivariate_gaussian_process:
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.gps = [gaussian_process(self.input_shape) for i in range(self.output_shape[-1])]

    def build(self, x, y, xtest):
        assert x.shape.as_list() == self.input_shape
        assert xtest.shape.as_list() == self.input_shape
        assert y.shape.as_list() == self.output_shape

        y_splits = tf.split(y, self.output_shape[-1], axis=-1)
        assert len(y_splits) == len(self.gps)

        #Allocate the gps
        output = tf.concat([self.gps[i].build(x, y_splits[i], xtest) for i in range(len(y_splits))], axis=-1)
        #output = tf.concat([gaussian_process(self.input_shape).build(x, y_split, xtest) for y_split in y_splits], axis=-1)
        return output

class gaussian_process:
    def __init__(self, xshape):
        self.xshape = xshape

    def build(self, x, y, xtest):
        assert x.shape.as_list() == self.xshape
        assert xtest.shape.as_list() == self.xshape
        assert y.shape.as_list() == [None, 1]

        #Placeholders
        #self.x = tf.placeholder(shape=self.xshape, dtype=tf.float64)

        #Kernel
        kernel = self.squared_exponential_kernel(x, x)

        #Cholesky decomposition
        self.noise_variance = 5e-4#to be optimized
        L = tf.cholesky(kernel + tf.diag(self.noise_variance * tf.ones_like(tf.reduce_sum(x, axis=-1))))

        #Placeholders for test points
        #self.xtest = tf.placeholder(shape=self.xshape, dtype=tf.float64)
        #self.y = tf.placeholder(shape=[None, 1], dtype=tf.float64)

        #Compute the mean at the test points
        Lk = tf.linalg.solve(L, self.squared_exponential_kernel(x, xtest))
        mu = tf.matmul(tf.transpose(Lk), tf.linalg.solve(L, y))

        #Compute the variance at the test points
        K_ = self.squared_exponential_kernel(xtest, xtest)
        var = K_ - tf.matmul(tf.transpose(Lk), Lk)
        std = tf.sqrt(tf.diag_part(var))
        return mu

    def squared_exponential_kernel(self, a, b):
        self.kernel_param = .1#to be optimized
        sqdist = tf.reduce_sum(tf.square(a), axis=-1, keep_dims=True) +\
                 -2. * tf.matmul(a, tf.transpose(b)) +\
                 tf.transpose(tf.reduce_sum(tf.square(b), axis=-1, keep_dims=True))
        self.sqdist = sqdist
        return tf.exp(-.5 * (1./self.kernel_param) * sqdist)
