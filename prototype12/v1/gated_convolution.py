import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

import sys
sys.path.append('../..')
from utils import get_random_string

#100, 200
#512, 1000??
class gated_convolution:
    def __init__(self,\
                 shape,\
                 nummap,\
                 numfactors,\
                 learning_rate,\
                 w,\
                 s):
        print 'in __init__ gated_convolution'
        self.shape = shape
        self.nummap = nummap
        self.numfactors = numfactors
        self.w = w
        self.s = s
        self.scope1 = 'conv1' + get_random_string()
        self.scope2 = 'conv2' + get_random_string()

        #Xavier init
        self.xavier_init = tf.contrib.layers.xavier_initializer()

        self.declare_lowlvl_vars()

        #Declare input variables
        self.x = tf.placeholder(shape=shape, dtype=tf.float32)
        self.y = tf.placeholder(shape=shape, dtype=tf.float32)

        #Corrupt input data
        corrupted_x = self.corrupt_data(self.x, .5)
        corrupted_y = self.corrupt_data(self.y, .5)

        #Get conv factors
        factors_x = self.get_factors_via_convolution(corrupted_x,
                                                     self.numfactors,
                                                     self.w,
                                                     self.s,
                                                     self.scope1)
        factors_y = self.get_factors_via_convolution(corrupted_y,
                                                     self.numfactors,
                                                     self.w,
                                                     self.s,
                                                     self.scope2)

        #Get hidden factors
        self.assert_dims(factors_x, factors_y, shape, self.w, self.s)
        hidden, factors_h = self.get_hidden_factors(factors_x, factors_y)

        #Get the recon losses
        recon_x_loss, self.recon_x_ = self.get_recon_loss_x(self.x, self.shape, factors_y, factors_h, self.w, self.s)
        recon_y_loss, self.recon_y_ = self.get_recon_loss_y(self.y, self.shape, factors_x, factors_h, self.w, self.s)

        #Optimizer
        self.recon_loss = recon_x_loss + recon_y_loss
        self.update_model_recon = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.recon_loss)


    def get_recon_loss_x(self, x, shape, factors_y, factors_h, w, s):
        patches_x = tf.extract_image_patches(x,\
                                             [1, w, w, 1],\
                                             [1, s, s, 1],\
                                             [1, 1, 1, 1], 'VALID')
        assert int(patches_x.shape[-1]) == w**2 * shape[-1]
        patches_x_reshaped = tf.reshape(patches_x, shape=(-1, w**2 * shape[-1]))
        assert len(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope1)) == 1
        wxf = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope1)[0]
        wxf = tf.reshape(wxf, shape=(-1, self.numfactors))
        wxf = tf.transpose(wxf)
        factors_yh = tf.multiply(factors_y, factors_h)
        factors_yh_flat = tf.reshape(factors_yh, (-1, self.numfactors))
        recon_x = tf.matmul(factors_yh_flat, wxf) + self.bx
        recon_x_ = tf.reshape(recon_x, shape=[-1, w, w, shape[-1]])
        recon_x_loss = tf.reduce_mean(tf.reduce_sum(tf.square(recon_x - patches_x_reshaped), axis=-1))
        return recon_x_loss, recon_x_

    def get_recon_loss_y(self, y, shape, factors_x, factors_h, w, s):
        patches_y = tf.extract_image_patches(y,\
                                             [1, w, w, 1],\
                                             [1, s, s, 1],\
                                             [1, 1, 1, 1], 'VALID')
        assert int(patches_y.shape[-1]) == w**2 * shape[-1]
        patches_y_reshaped = tf.reshape(patches_y, shape=(-1, w**2 * shape[-1]))
        assert len(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope2)) == 1
        wyf = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope2)[0]
        wyf = tf.reshape(wyf, shape=(-1, self.numfactors))
        wyf = tf.transpose(wyf)
        factors_xh = tf.multiply(factors_x, factors_h)
        factors_xh_flat = tf.reshape(factors_xh, (-1, self.numfactors))
        recon_y = tf.matmul(factors_xh_flat, wyf) + self.by
        recon_y_ = tf.reshape(recon_y, shape=[-1, w, w, shape[-1]])
        recon_y_loss = tf.reduce_mean(tf.reduce_sum(tf.square(recon_y - patches_y_reshaped), axis=-1))
        return recon_y_loss, recon_y_

    def assert_dims(self, factors_x, factors_y, inp_shape, w, s):
        #Get shapes
        rows = int(factors_x.shape[1])
        cols = int(factors_x.shape[2])
        chans = int(factors_x.shape[3])
        assert chans == self.numfactors
        assert rows == cols
        assert rows == int(np.floor(float(inp_shape[1] - w) / float(s))) + 1

    def get_hidden_factors(self, factors_x, factors_y):
        rows = int(factors_x.shape[1])
        cols = int(factors_x.shape[2])
        #Compute hidden factors
        factors_mult = tf.multiply(factors_x, factors_y)#Elementwise multiply
        factors_mult_flat = tf.reshape(factors_mult, shape=(-1, self.numfactors))#Reshape
        hidden_flat = tf.nn.sigmoid(tf.matmul(factors_mult_flat, tf.transpose(self.whf)) + self.bh)#Get hidden
        hidden = tf.reshape(hidden_flat, shape=(-1, rows, cols, self.nummap))#Reshape
        factors_h_flat = tf.matmul(hidden_flat, self.whf)#Get hidden factors
        factors_h = tf.reshape(factors_h_flat, shape=(-1, rows, cols, self.numfactors))#Reshape
        return hidden, factors_h

    def declare_lowlvl_vars(self):
        self.whf = tf.Variable(self.xavier_init([self.nummap, self.numfactors]))
        #self.whf = tf.Variable(tf.exp(tf.random_uniform([self.nummap, self.numfactors], -3., -2.)))
        self.bh = tf.Variable(tf.zeros([self.nummap]))
        self.bx = tf.Variable(tf.zeros([self.w**2 * self.shape[-1]]))
        self.by = tf.Variable(tf.zeros([self.w**2 * self.shape[-1]]))

    def get_factors_via_convolution(self, data, o, k, s, scope, reuse=None):
        return slim.conv2d(inputs=data,\
                           num_outputs=o,\
                           kernel_size=[k, k],\
                           stride=[s, s],\
                           activation_fn=None,\
                           biases_initializer=None,\
                           padding='VALID',\
                           scope=scope,\
                           reuse=reuse)

    def corrupt_data(self, data, corruption_level):
        return tf.multiply(data,\
            tf.ceil(tf.random_uniform(tf.shape(data)) -\
                corruption_level))

def main0():
    batch_size = 4
    epochs = 200

    import pickle
    import numpy as np
    x = pickle.load( open( "../../custom_environments/cart_data.p" , "rb" ) )

    x = x.astype(np.float64) / 255.

    gc = gated_convolution(shape=[None, 36, 36, 4], nummap=100, numfactors=200, w=12, s=1, learning_rate=.01)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            idx = 0
            while idx < len(x):
                batch = x[idx:idx+batch_size]
                _, recon_loss, recon_x, recon_y = sess.run([gc.update_model_recon, gc.recon_loss, gc.recon_x_, gc.recon_y_], feed_dict={gc.x:batch, gc.y:batch})
                print 'epoch:', epoch, 'recon_loss:', recon_loss, 'main0'
                idx += batch_size

        data = sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
        data.append(recon_x)
        data.append(recon_y)
        data.append(batch)
        pickle.dump(data, open("model_params_cart.p", "wb" ))

def main1():
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
    gc = gated_convolution([None, 12, 12, 4], nummap=32, numfactors=64, w=12, s=1, learning_rate=.001)

    for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        print v

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            idx = 0
            while idx < len(x):
                batch = x[idx:idx+batch_size]
                _, recon_loss, recon_x, recon_y = sess.run([gc.update_model_recon, gc.recon_loss, gc.recon_x_, gc.recon_y_], feed_dict={gc.x:batch, gc.y:batch})
                print 'epoch:', epoch, 'recon_loss:', recon_loss, 'main1'
                idx += batch_size

        data = sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
        data.append(recon_x)
        data.append(recon_y)
        data.append(batch)
        pickle.dump(data, open("model_params_small.p", "wb" ))

def main2():
    batch_size = 2
    epochs = 200
    stack = 4

    import pickle
    import numpy as np
    data = pickle.load( open( "../prototype11/roland_tutorial/data.p" , "rb" ) )

    x = data[0].T.reshape(-1, 84, 84, 1)
    y = data[1].T.reshape(-1, 84, 84, 1)
    x = x.astype(np.float64) / 255.
    y = y.astype(np.float64) / 255.

    x_ = []
    for i in range(len(x) - stack + 1):
        tmp = []
        for s in range(i, i + stack):
            tmp.append(x[s])
        x_.append(np.concatenate(tmp, axis=-1)[np.newaxis, ...])
    x = np.concatenate(x_, axis=0)

    gc = gated_convolution([None, 84, 84, stack], nummap=100, numfactors=200, w=12, s=1, learning_rate=.01)

    for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        print v

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            idx = 0
            while idx < len(x):
                batch = x[idx:idx+batch_size]
                _, recon_loss, recon_x, recon_y = sess.run([gc.update_model_recon, gc.recon_loss, gc.recon_x_, gc.recon_y_], feed_dict={gc.x:batch, gc.y:batch})
                print 'epoch:', epoch, 'recon_loss:', recon_loss, 'main2'
                idx += batch_size

                data = sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
                data.append(recon_x)
                data.append(recon_y)
                data.append(batch)
                pickle.dump(data, open("model_params_large.p", "wb" ))

if __name__ == '__main__':
    main0()
    #main1()
    #main2()
