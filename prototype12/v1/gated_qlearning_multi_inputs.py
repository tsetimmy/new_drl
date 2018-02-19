import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

from gated_convolution_multi_inputs import gated_convolution_multi_inputs

import sys
sys.path.append('../..')
from utils import Memory


class gated_qlearning_multi_inputs(gated_convolution_multi_inputs):
    def __init__(self,
                 shape=[None, 36, 36, 2],
                 nummap=32,
                 numfactors=32,
                 learning_rate=.001,
                 no_inputs=3,
                 frame_shape=[None, 36, 36, 2],
                 a_size=3,
                 stop_gradient=False,
                 lamb=0.,
                 w=8,
                 s=1,
                 use_conv_after_fe=True,
                 mode='cross_correlation',
                 action_placement='in',
                 tiled=False,
                 use_close_to_ones_init=False):
                 #action_type='discrete'):
        gated_convolution_multi_inputs.__init__(self,
                                                shape,
                                                nummap,
                                                numfactors,
                                                learning_rate,
                                                w,
                                                s,
                                                no_inputs)

        assert action_placement in ['in', 'out']
        assert mode in ['cross_correlation', 'transformation']
        #assert action_type in ['discrete', 'continuous']
        self.frame_shape = frame_shape
        self.a_size = a_size
        self.stop_gradient = stop_gradient
        self.lamb = lamb
        self.use_conv_after_fe = use_conv_after_fe
        #self.action_type = action_type
        self.mode = mode
        self.action_placement = action_placement
        self.tiled = tiled
        self.use_close_to_ones_init = use_close_to_ones_init

        if self.action_placement == 'in':
            self.q_out_size = 1
        elif self.action_placement == 'out':
            self.q_out_size = self.a_size

        #Get factors for frame (state) inputs
        for i in range(self.no_inputs):
            self.params[i]['frame_input'] = tf.placeholder(shape=self.frame_shape, dtype=tf.float32)
            self.params[i]['frame_factors'] = self.get_factors_via_convolution(self.params[i]['frame_input'],
                                                                               self.numfactors,
                                                                               self.w,
                                                                               4,
                                                                               self.params[i]['scope'],
                                                                               reuse=True)

        frame_factors = [self.params[i]['frame_factors'] for i in range(len(self.params))]

        if self.action_placement == 'in':
        #Get factors for action inputs
#        if self.action_type == 'discrete':
            self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
            actions_onehot = tf.one_hot(self.actions, self.a_size, dtype=tf.float32)
            if self.tiled == True:
                if self.use_close_to_ones_init == False:
                    self.action_factors = slim.fully_connected(actions_onehot,
                                                               self.numfactors,
                                                               activation_fn=tf.nn.sigmoid)
                else:
                    self.action_factors = slim.fully_connected(actions_onehot,
                                                               self.numfactors,
                                                               activation_fn=tf.nn.sigmoid,
                                                               weights_initializer=tf.random_uniform_initializer(minval=3., maxval=6.))

#        elif self.action_type == 'continuous':
#            self.actions = tf.placeholder(shape=[None, self.a_size], dtype=tf.float32)
#            self.action_factors = slim.fully_connected(self.actions,
#                                                       self.numfactors,
#                                                       activation_fn=None,
#                                                       biases_initializer=None)

            #Preprocess action factors
                action_factors_reshaped = tf.reshape(self.action_factors, shape=(-1, 1, 1, self.numfactors))
                action_factors_tiled = tf.tile(action_factors_reshaped, [1,
                                                                     int(frame_factors[0].shape[1]),
                                                                     int(frame_factors[0].shape[2]),
                                                                     1])
            else:
                size = int(frame_factors[0].shape[1]) * int(frame_factors[0].shape[2]) * self.numfactors 
                if self.use_close_to_ones_init == False:
                    self.action_factors = slim.fully_connected(actions_onehot,
                                                               self.numfactors *\
                                                                   int(frame_factors[0].shape[1]) *\
                                                                   int(frame_factors[0].shape[2]),
                                                               activation_fn=tf.nn.sigmoid)
                else:
                    self.action_factors = slim.fully_connected(actions_onehot,
                                                               self.numfactors *\
                                                                   int(frame_factors[0].shape[1]) *\
                                                                   int(frame_factors[0].shape[2]),
                                                               activation_fn=tf.nn.sigmoid,
                                                               weights_initializer=tf.random_uniform_initializer(minval=3., maxval=6.))
                action_factors_tiled = tf.reshape(self.action_factors, shape=(-1, 
                                                                              int(frame_factors[0].shape[1]),
                                                                              int(frame_factors[0].shape[2]),
                                                                              self.numfactors))
            #Gather all factors
            frame_factors.append(action_factors_tiled)

        #Get hidden factors
        frame_hidden, factors_h = self.get_hidden_factors(frame_factors)

        #Stop gradients?
        if self.stop_gradient:
            self.frame_hidden = tf.stop_gradient(frame_hidden)
        else:
            self.frame_hidden = frame_hidden

        if self.use_conv_after_fe == True:
            self.conv2 = slim.conv2d(activation_fn=tf.nn.relu,\
                                     inputs=self.frame_hidden,\
                                     num_outputs=64,\
                                     kernel_size=[4, 4],\
                                     stride=[2, 2],\
                                     padding='VALID')
            print self.conv2.shape

            self.conv3 = slim.conv2d(activation_fn=tf.nn.relu,\
                                     inputs=self.conv2,\
                                     num_outputs=64,\
                                     kernel_size=[3, 3],\
                                     stride=[1, 1],\
                                     padding='VALID')

            print self.conv3.shape
            self.fc1 = slim.fully_connected(slim.flatten(self.conv3),\
                                            512,\
                                            activation_fn=tf.nn.relu)

            self.Q = slim.fully_connected(self.fc1,\
                                          self.q_out_size,\
                                          activation_fn=None)

        else:
            self.Q = slim.fully_connected(slim.flatten(self.frame_hidden),\
                                          self.q_out_size,\
                                          activation_fn=None)

        self.targetQ = tf.placeholder(shape=[None, 1],\
                                      dtype=tf.float32)

        if self.action_placement == 'out':
            self.actions = tf.placeholder(shape=[None],\
                                          dtype=tf.int32)
            self.actions_onehot = tf.one_hot(self.actions,\
                                             self.a_size,\
                                             dtype=tf.float32)
            self.responsible_output = tf.reduce_sum(tf.multiply(self.Q,\
                                                                self.actions_onehot),\
                                                    axis=1,\
                                                    keep_dims=True)


        if self.action_placement == 'in':
            difference = tf.abs(self.Q - self.targetQ)
        elif self.action_placement == 'out':
            difference = tf.abs(self.responsible_output - self.targetQ)
        quadratic_part = tf.clip_by_value(difference, 0.0, 1.0)
        linear_part = difference - quadratic_part
        errors = (0.5 * tf.square(quadratic_part)) + linear_part
        self.q_loss = tf.reduce_sum(errors) 

        self.update_model = tf.train.RMSPropOptimizer(learning_rate=.00025,\
                                                      decay=.95,\
                                                      epsilon=.01).minimize(self.q_loss)

    def process_states(self, states):
        if self.mode == 'cross_correlation':
            states = [states] * self.no_inputs
        elif self.mode == 'transformation':
            assert states.shape[-1] == self.no_inputs
            states = np.split(states, states.shape[-1], axis=-1)
        return states

    def get_action(self, sess, states):
        if self.action_placement == 'out':
            return self.get_action_aout(sess, states)
        assert states.shape[0] == 1
        states = self.process_states(states)
        states = [s.astype(np.float64) / 255. for s in states]
        states = [np.concatenate([s] * self.a_size, axis=0) for s in states]

        feed_dict = {}
        for i in range(self.no_inputs):
            feed_dict[self.params[i]['frame_input']] = states[i]
        feed_dict[self.actions] = np.arange(self.a_size).astype(np.int32)
        Q = sess.run(self.Q, feed_dict=feed_dict)
        return np.argmax(Q)

    def get_Q1(self, sess, states1, tnet):
        if self.action_placement == 'out':
            return self.get_Q1_aout(sess, states1, tnet)
        assert self.no_inputs == tnet.no_inputs
        states1 = self.process_states(states1)
        states1 = [s1.astype(np.float64) / 255. for s1 in states1]
        feed_dict = {}
        for i in range(tnet.no_inputs):
            feed_dict[tnet.params[i]['frame_input']] = states1[i]
        Q1s = []
        for i in range(tnet.a_size):
            feed_dict[tnet.actions] = np.ones(len(states1[0])).astype(np.int32) * i
            Q1 = sess.run(tnet.Q, feed_dict=feed_dict)
            Q1s.append(Q1)
        return np.concatenate(Q1s, axis=-1)

    def train(self, sess, states, actions, targetQ):
        if self.action_placement == 'out':
            return self.train_aout(sess, states, actions, targetQ)
        states = self.process_states(states)
        states = [s.astype(np.float64) / 255. for s in states]
        feed_dict = {}
        for i in range(self.no_inputs):
            feed_dict[self.params[i]['frame_input']] = states[i]
        feed_dict[self.actions] = actions
        feed_dict[self.targetQ] = targetQ
        loss, _ = sess.run([self.q_loss, self.update_model], feed_dict=feed_dict)
        return loss, _, _

    def train_feature_extractor(self, sess, replay_buffer, batch_size=100, iterations=1, iterations_left=-1):
        try:
            self.buff
        except:
            self.buff = Memory(10000)

        for it in range(iterations):
            while len(self.buff.mem) < batch_size:
                state = replay_buffer.sample(1)[0][0]

                patches = []
                for i in range(0, state.shape[1]-self.w+1, self.s):
                    for j in range(0, state.shape[2]-self.w+1, self.s):
                        patches.append(state[0, i:i+self.w, j:j+self.w, :])
                        assert patches[-1].shape[0] == patches[-1].shape[1]
                        assert patches[-1].shape[0] == self.w
                from random import shuffle
                shuffle(patches)
                self.buff.mem += patches

            batch = self.buff.mem[:batch_size]
            self.buff.mem = self.buff.mem[batch_size:]

            batch = np.concatenate([b[np.newaxis, ...] for b in batch], axis=0)
            batch = self.process_states(batch)
            batch = [b.astype(np.float64) / 255. for b in batch]
            feed_dict = {}
            for i in range(self.no_inputs):
                feed_dict[self.params[i]['input']] = batch[i]

            _, recon_loss, = sess.run([self.update_model_recon,
                                       self.recon_loss],
                                       feed_dict=feed_dict)
                                                  
            print "train_feature_extractor - recon_loss:", recon_loss

########################################################################################################################
            if iterations_left <= 10:
                variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                import pickle
                recon_x, recon_y, conv1, conv2 = sess.run([self.params[0]['recon'], self.params[1]['recon'], variables[2], variables[4]], feed_dict=feed_dict)
                pickle.dump( [conv1, conv2, batch, recon_x, recon_y], open( "recons.p", "wb" ) )
########################################################################################################################


        return iterations


    def get_action_aout(self, sess, image_in):
        image_in = image_in.astype(np.float64) / 255.
        Q = sess.run(self.Q,\
                     feed_dict={self.params[0]['frame_input']:image_in,
                                self.params[1]['frame_input']:image_in})
        action = np.argmax(Q)
        return action

    def get_Q1_aout(self, sess, states1, tnet):
        states1 = states1.astype(np.float64) / 255.
        return sess.run(tnet.Q,\
                        feed_dict={tnet.params[0]['frame_input']:states1,
                                   tnet.params[1]['frame_input']:states1})

    def train_aout(self, sess, states, actions, targetQ):
        states = states.astype(np.float64) / 255.
        _, q_loss = sess.run([self.update_model,\
                                                self.q_loss],
                                                feed_dict={self.params[0]['frame_input']:states,\
                                                           self.params[1]['frame_input']:states,\
                                                           self.actions:actions,\
                                                           self.targetQ:targetQ})
        return q_loss, _, _

def main():
    print 'in main'
    tmp = gated_qlearning_multi_inputs()

if __name__ == '__main__':
    main()
