import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

from gated_convolution import gated_convolution

import sys
sys.path.append('../..')
from utils import get_random_string

class gated_convolution_multi_inputs(gated_convolution):
    def __init__(self,\
                 shape,\
                 nummap,\
                 numfactors,\
                 learning_rate,\
                 w,\
                 s,\
                 no_inputs):
        print 'in __init__ gated_convolution_multi_inputs'
        assert no_inputs > 0
        self.shape = shape
        self.nummap = nummap
        self.numfactors = numfactors
        self.learning_rate = learning_rate
        self.w = w
        self.s = s
        self.no_inputs = no_inputs
        ###
        self.start_vars = len(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
        self.xavier_init = tf.contrib.layers.xavier_initializer()

        #Initialize some low-level variables
        self.whf = tf.Variable(self.xavier_init([self.nummap, self.numfactors]))
        #self.whf = tf.Variable(tf.exp(tf.random_uniform([self.nummap, self.numfactors], -3., -2.)))
        self.bh = tf.Variable(tf.zeros([self.nummap]))

        params = []
        for i in range(no_inputs):
            dictionary = {}
            dictionary['scope'] = 'conv' + get_random_string()
            dictionary['input'] = tf.placeholder(shape=shape, dtype=tf.float32)
            dictionary['corrupted_input'] = self.corrupt_data(dictionary['input'], .5)
            dictionary['factors'] = self.get_factors_via_convolution(dictionary['corrupted_input'],
                                                                     self.numfactors,
                                                                     self.w,
                                                                     self.s,
                                                                     dictionary['scope'])
            dictionary['biases'] = tf.Variable(tf.zeros([self.w**2 * self.shape[-1]]))
            params.append(dictionary)
        self.params = params

        factors = [params[i]['factors'] for i in range(len(self.params))]
        hidden, factors_h = self.get_hidden_factors(factors)

        assert len(factors) == len(self.params)
        for i in range(len(self.params)):
            self.params[i]['recon_loss'], self.params[i]['recon'] =\
                self.get_recon_loss(self.params[i]['input'],
                                    factors[0:i] + factors[i+1:],
                                    factors_h,
                                    self.params[i]['scope'],
                                    self.params[i]['biases'],
                                    self.shape,
                                    self.w,
                                    self.s)

        self.recon_loss = reduce((lambda x, y: x + y),
                                 [params[i]['recon_loss'] for i in range(len(self.params))])
        self.update_model_recon = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.recon_loss)
        self.recons = [params[i]['recon'] for i in range(len(self.params))]

    def get_recon_loss(self, target, givens, hidden, scope, biases, shape, w, s):
        #Calculate the hiden factors
        factors = [hidden] + givens
        factors_tensor = tf.concat([tf.expand_dims(f, axis=-1) for f in factors], axis=-1)
        rows = int(factors_tensor.shape[1])
        cols = int(factors_tensor.shape[2])
        factors_mult = tf.reduce_prod(factors_tensor, axis=-1)
        factors_mult_flat = tf.reshape(factors_mult, (-1, self.numfactors))

        #Get (shared) variable
        assert len(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)) == 1
        weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)[0]
        weights = tf.reshape(weights, shape=(-1, self.numfactors))
        weights = tf.transpose(weights)

        #Get reconstruction
        recon = tf.matmul(factors_mult_flat, weights) + biases
        recon_reshaped = tf.reshape(recon, shape=(-1, w, w, shape[-1]))

        #Get recon loss
        patches = tf.extract_image_patches(target,
                                           [1, w, w, 1],
                                           [1, s, s, 1],
                                           [1, 1, 1, 1], 'VALID')
        assert int(patches.shape[-1]) == w**2 * shape[-1]
        patches_reshaped = tf.reshape(patches, shape=(-1, w**2 * shape[-1]))
        recon_loss = tf.reduce_mean(tf.reduce_sum(tf.square(recon - patches_reshaped), axis=-1))#Loss

        return recon_loss, recon_reshaped

    def get_hidden_factors(self, factors):
        factors_tensor = tf.concat([tf.expand_dims(f, axis=-1) for f in factors], axis=-1)
        rows = int(factors_tensor.shape[1])
        cols = int(factors_tensor.shape[2])
        factors_mult = tf.reduce_prod(factors_tensor, axis=-1)
        factors_mult_flat = tf.reshape(factors_mult, shape=(-1, self.numfactors))#Reshape
        hidden_flat = tf.nn.sigmoid(tf.matmul(factors_mult_flat, tf.transpose(self.whf)) + self.bh)#Get hidden
        hidden = tf.reshape(hidden_flat, shape=(-1, rows, cols, self.nummap))#Reshape
        factors_h_flat = tf.matmul(hidden_flat, self.whf)#Get hidden factors
        factors_h = tf.reshape(factors_h_flat, shape=(-1, rows, cols, self.numfactors))#Reshape
        return hidden, factors_h




def main():
    batch_size = 100
    epochs = 200

    import pickle
    import numpy as np
    data = pickle.load( open( "../prototype11/roland_tutorial/data_segmented_stacked.p" , "rb" ) )
    x = data[0].T.reshape(-1, 12, 12, 4)
    y = data[1].T.reshape(-1, 12, 12, 4)
    x = x.astype(np.float64) / 255.
    y = y.astype(np.float64) / 255.

    #gc = gated_convolution([None, 12, 12, 4], nummap=100, numfactors=200, w=12, s=1, learning_rate=.01)
    #gc = gated_convolution([None, 12, 12, 4], nummap=32, numfactors=64, w=12, s=1, learning_rate=.001)
    no_inputs = 2
    gcmi = gated_convolution_multi_inputs(shape=[None, 12, 12, 4],
                                          nummap=32,
                                          numfactors=64,
                                          learning_rate=.001,
                                          w=12,
                                          s=1,
                                          no_inputs=no_inputs)

    for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        print v

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            idx = 0
            while idx < len(x):
                batch = x[idx:idx+batch_size]

                feed_dict = {}
                for i in range(no_inputs):
                    feed_dict[gcmi.params[i]['input']] = batch

                _, recon_loss, recons = sess.run([gcmi.update_model_recon, gcmi.recon_loss, gcmi.recons], feed_dict=feed_dict)
                print 'epoch:', epoch, 'recon_loss:', recon_loss, 'main1'
                idx += batch_size

        data = sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
        data.append(recons)
        data.append(batch)
        pickle.dump(data, open("model_params_small_a.p", "wb" ))

if __name__ == '__main__':
    main()




