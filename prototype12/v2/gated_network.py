import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

import sys
#sys.path.append('../v1')
sys.path.append('../..')
from utils import get_random_string

#from gated_convolution import gated_convolution

class gated_convolution2:
    def __init__(self,
                 shape,
                 nummap,
                 numfactors,
                 learning_rate,
                 w,
                 s,
                 a_size):
        print 'in __init__ gated_convolution2'
        self.shape = shape
        self.nummap = nummap
        self.numfactors = numfactors
        self.w = w
        self.s = s
        self.a_size = a_size
        self.scope1 = 'conv1' + get_random_string()
        self.scope2 = 'conv2' + get_random_string()

        #Xavier init
        self.xavier_init = tf.contrib.layers.xavier_initializer()

        self.declare_lowlvl_vars()

        #Action input
        self.actions = tf.placeholder(shape=[None, self.shape[-1]], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, self.a_size, dtype=tf.float32)


        #Declare input variables
        self.x = tf.placeholder(shape=shape, dtype=tf.float32)
        self.y = tf.placeholder(shape=shape, dtype=tf.float32)

        #batch size
        batch_size = self.x.shape[0]

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

        #Get action factors
        self.factors_a, factors_a_vars_scope = self.get_hidden_action_factors(self.actions_onehot, factors_x.shape.as_list()[1:])

        #Get hidden factors
        self.assert_dims(factors_x, factors_y, shape, self.w, self.s)
        hidden, factors_h = self.get_hidden_factors(factors_x, factors_y, self.factors_a)


        self.recon_x = self.get_recon([factors_y, self.factors_a], factors_h, self.scope1, self.shape, self.w, self.s)
        self.recon_y = self.get_recon([factors_x, self.factors_a], factors_h, self.scope2, self.shape, self.w, self.s)
        actions_onehot_shape = self.actions_onehot.shape.as_list()[1:]
        self.recon_a = self.get_recon_action([factors_x, factors_y], factors_h, factors_a_vars_scope, actions_onehot_shape)

        
        self.recon_a_loss = tf.reduce_mean(tf.reduce_sum(slim.flatten(tf.multiply(-tf.log(tf.nn.softmax(self.recon_a,
                                                                                                        dim=-1) + 1e-10),
                                                                                                        self.actions_onehot)),
                                                                                                        axis=-1))

        self.recon_loss = tf.reduce_mean(tf.reduce_sum(slim.flatten(tf.square(self.recon_x - self.x)), axis=-1)) +\
                          tf.reduce_mean(tf.reduce_sum(slim.flatten(tf.square(self.recon_y - self.y)), axis=-1)) +\
                          self.recon_a_loss
        self.update_model_recon = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.recon_loss)

    def get_factors_mult(self, givens, factors_h):
        factors = [factors_h] + givens
        factors_tensor = tf.concat([tf.expand_dims(f, axis=-1) for f in factors], axis=-1)
        rows = int(factors_tensor.shape[1])
        cols = int(factors_tensor.shape[2])
        factors_mult = tf.reduce_prod(factors_tensor, axis=-1)
        return factors_mult

    def get_recon_action(self, givens, factors_h, scope, shape):
        #Calculate the hiden factors
        factors_mult = self.get_factors_mult(givens, factors_h)

        #Get (shared) variable
        assert len(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)) == 1
        weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)[0]

        num_outs = reduce(lambda x, y: x*y, shape)
        biases = tf.Variable(tf.zeros([num_outs]))
        recon_a = tf.reshape(tf.matmul(slim.flatten(factors_mult),
                                            tf.transpose(weights)) + biases,
                                  shape=[-1]+shape)
        return recon_a






    def get_recon(self, givens, factors_h, scope, shape, w, s):
        #Calculate the hiden factors
        factors_mult = self.get_factors_mult(givens, factors_h)

        recon = slim.conv2d_transpose(factors_mult, 
                                      shape[-1],
                                      w,
                                      s,
                                      padding='VALID',
                                      biases_initializer=None,
                                      activation_fn=None,
                                      reuse=True,
                                      scope=scope)

        bias = tf.Variable(tf.zeros(shape[1:]))
        return tf.add(recon, bias)

    def assert_dims(self, factors_x, factors_y, inp_shape, w, s):
        #Get shapes
        rows = int(factors_x.shape[1])
        cols = int(factors_x.shape[2])
        chans = int(factors_x.shape[3])
        assert chans == self.numfactors
        assert rows == cols
        assert rows == int(np.floor(float(inp_shape[1] - w) / float(s))) + 1

    def get_hidden_factors(self, factors_x, factors_y, factors_a):
        rows = int(factors_x.shape[1])
        cols = int(factors_x.shape[2])
        #Compute hidden factors
        factors_mult = tf.multiply(tf.multiply(factors_x, factors_y), factors_a)#Elementwise multiply
        factors_mult_flat = tf.reshape(factors_mult, shape=(-1, self.numfactors))#Reshape
        hidden_flat = tf.nn.sigmoid(tf.matmul(factors_mult_flat, tf.transpose(self.whf)) + self.bh)#Get hidden
        hidden = tf.reshape(hidden_flat, shape=(-1, rows, cols, self.nummap))#Reshape
        factors_h_flat = tf.matmul(hidden_flat, self.whf)#Get hidden factors
        factors_h = tf.reshape(factors_h_flat, shape=(-1, rows, cols, self.numfactors))#Reshape
        return hidden, factors_h

    def get_hidden_action_factors(self, actions_onehot, target_shape):
        num_outs = reduce(lambda x, y: x*y, target_shape)
        scope = get_random_string()
        action_factors = slim.fully_connected(slim.flatten(actions_onehot),
                                              num_outs,
                                              activation_fn=None,
                                              biases_initializer=None,
                                              scope=scope)
        action_factors = tf.reshape(action_factors, shape=[-1]+target_shape)
        return action_factors, scope

    def declare_lowlvl_vars(self):
        self.whf = tf.Variable(self.xavier_init([self.nummap, self.numfactors]))
        #self.whf = tf.Variable(tf.exp(tf.random_uniform([self.nummap, self.numfactors], -3., -2.)))
        self.bh = tf.Variable(tf.zeros([self.nummap]))
        #self.bx = tf.Variable(tf.zeros([self.w**2 * self.shape[-1]]))
        #self.by = tf.Variable(tf.zeros([self.w**2 * self.shape[-1]]))

    def get_factors_via_convolution(self, data, o, k, s, scope, reuse=None):
        return slim.conv2d(inputs=data,
                           num_outputs=o,
                           kernel_size=[k, k],
                           stride=[s, s],
                           activation_fn=None,
                           biases_initializer=None,
                           padding='VALID',
                           scope=scope,
                           reuse=reuse)

    def corrupt_data(self, data, corruption_level):
        return tf.multiply(data,
            tf.ceil(tf.random_uniform(tf.shape(data)) -\
                corruption_level))

    def run(self, sess, batch):
        _, recon_loss, recon_x, recon_y = sess.run([self.update_model_recon, self.recon_loss, self.recon_x, self.recon_y], feed_dict={self.x:batch, self.y:batch})
        return _, recon_loss, recon_x, recon_y

    def run2(self, sess, states, actions, states_):
        _, recon_loss, recon_action_loss, recon_x, recon_y = sess.run([self.update_model_recon, self.recon_loss, self.recon_a_loss, self.recon_x, self.recon_y], feed_dict={self.x:states, self.y:states_, self.actions:actions})

        return _, recon_loss, recon_x, recon_y, recon_action_loss


























































def get_data(dtype, mode):
    import pickle
    assert dtype in ['cart', 'cartpole', 'breakout']
    assert mode in ['cross_correlation', 'transformation']
    dim = 36
    if dtype == 'breakout':
        dim = 84

    if mode == 'cross_correlation':
        x = pickle.load( open( "../../pickles_data/"+dtype+"_data.p" , "rb" ) )
        x = x.astype(np.float64) / 255.
        return x, x, dim, 4
    elif mode == 'transformation':
        x = pickle.load( open( "../../pickles_data/"+dtype+"_data_trans.p" , "rb" ) )
        x = x.astype(np.float64) / 255.
        x_tmp = []
        y_tmp = []

        for i in range(len(x)):
            tmp = x[i, :, :, 0]
            tmp = tmp[..., np.newaxis]
            tmp = tmp[np.newaxis, ...]
            x_tmp.append(tmp)
            tmp = x[i, :, :, 1]
            tmp = tmp[..., np.newaxis]
            tmp = tmp[np.newaxis, ...]
            y_tmp.append(tmp)

        x_tmp = np.concatenate(x_tmp, axis=0)
        y_tmp = np.concatenate(y_tmp, axis=0)

        return x_tmp, y_tmp, dim, 1

def main0(dtype, mode, model, gc, x, y, dim, size):
    batch_size = 4
    epochs = 40

    import pickle

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            idx = 0
            while idx < len(x):
                batch = x[idx:idx+batch_size]


                _, recon_loss, recon_x, recon_y = gc.run(sess, batch)


                print 'epoch:', epoch, 'recon_loss:', recon_loss, 'main0'
                idx += batch_size

            if epoch % 10 == 0:
                for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
                    print v
                data = sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
                data.append(recon_x)
                data.append(recon_y)
                data.append(batch)
                pickle.dump(data, open("model_params_"+dtype+"_"+mode+"_"+model+".p", "wb" ))

def MAIN0():
    model = 'gated'
    dtype = 'breakout'
    mode = 'transformation'
    x, y, dim, size = get_data(dtype, mode)
    gc = gated_convolution2(shape=[None, dim, dim, size], nummap=128, numfactors=128, learning_rate=.001, w=8, s=1, a_size=3)
    main0(dtype, mode, model, gc, x, y, dim, size)

def main1():
    import pickle
    gc = gated_convolution2(shape=[None, 84, 84, 4], nummap=128, numfactors=128, learning_rate=.001, w=8, s=1, a_size=3)
    data = pickle.load( open( "../../pickles_data/breakout_test_action_recon.p" , "rb" ) )
    states = data[0].astype(np.float64) / 255.
    actions = data[1]
    states_ = data[2].astype(np.float64) / 255.

    batch_size = 4
    epochs = 40

    assert len(states) == len(actions)
    assert len(states_) == len(states)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            idx = 0
            while idx < len(states):
                bs = states[idx:idx+batch_size]
                bs_ = states_[idx:idx+batch_size]
                ba = actions[idx:idx+batch_size]

                _, recon_loss, recon_x, recon_y, recon_action_loss = gc.run2(sess, bs, ba, bs_)
                print 'epoch:', epoch, 'recon_loss:', recon_loss, 'recon_action_loss', recon_action_loss, 'main1'
                idx += batch_size

def main2():
    import gym
    import copy
    from utils import Memory
    from utils import process_frame2

    env = gym.make('BreakoutDeterministic-v4')
    gc = gated_convolution2(shape=[None, 84, 84, 4], nummap=128, numfactors=128, learning_rate=.001, w=8, s=1, a_size=env.action_space.n)
    mem = Memory(50000)
    batch_size = 4
    steps = 1
    length = 4
    action_space = env.action_space.n

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        while True:
            s = env.reset()
            s = process_frame2(s)

            state = [s[..., np.newaxis]] * length
            state_ = [s[..., np.newaxis]] * length
            action = [-1] * length


            done = False
            while done == False:
                #env.render()


                a = np.random.randint(env.action_space.n)
                s_, r, done, _ = env.step(a)
                s_ = process_frame2(s_)

                state_.pop(0)
                action.pop(0)

                state_.append(s_[..., np.newaxis])
                action.append(a)

                mem.add([np.concatenate(state, axis=-1)[np.newaxis, ...], np.array(action)[np.newaxis, ...], np.concatenate(state_, axis=-1)[np.newaxis, ...]])

                if len(mem.mem) >= batch_size:
                    batch = mem.sample(batch_size)
                    #Do stuff
                    states = []
                    actions = []
                    states_ = []
                    for i in range(len(batch)):
                        states.append(batch[i][0])
                        actions.append(batch[i][1])
                        states_.append(batch[i][2])
                    states = np.concatenate(states, axis=0).astype(np.float64) / 255.
                    actions = np.concatenate(actions, axis=0)
                    states_ = np.concatenate(states_, axis=0).astype(np.float64) / 255.

                    _, recon_loss, recon_x, recon_y, recon_action_loss = gc.run2(sess, states, actions, states_)
                    print 'steps:', steps, 'recon_loss:', recon_loss, 'recon_action_loss', recon_action_loss, 'main2'

                steps += 1
                if done == True:
                    break






 


if __name__ == '__main__':
    #MAIN0()
    #main1()
    main2()

