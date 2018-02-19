import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

from gated_convolution import gated_convolution

import sys
sys.path.append('../..')
from utils import Memory

class gated_qlearning(gated_convolution):
    def __init__(self,\
                 shape=[None, 12, 12, 4],\
                 nummap=100,\
                 numfactors=200,\
                 learning_rate=.001,\
                 frame_shape=[None, 84, 84, 4],\
                 a_size=3,\
                 stop_gradient=False,\
                 lamb=0.,\
                 w=12,\
                 s=1,\
                 use_conv_after_fe=False):

        print 'in __init__ gated_qlearning'
        gated_convolution.__init__(self,\
                                   shape,\
                                   nummap,\
                                   numfactors,\
                                   learning_rate,\
                                   w,\
                                   s)

        self.frame_shape = frame_shape
        self.a_size = a_size
        self.stop_gradient = stop_gradient
        self.lamb = lamb
        self.use_conv_after_fe = use_conv_after_fe
        self.q_s = 4#Need a bigger stride for faster computation...

        #Declare input variables
        self.frame_x = tf.placeholder(shape=frame_shape, dtype=tf.float32)
        self.frame_y = tf.placeholder(shape=frame_shape, dtype=tf.float32)

        #Get conv factors
        factors_x = self.get_factors_via_convolution(self.frame_x,\
                                                     self.numfactors,\
                                                     self.w,\
                                                     self.q_s,\
                                                     self.scope1,\
                                                     reuse=True)

        factors_y = self.get_factors_via_convolution(self.frame_y,\
                                                     self.numfactors,\
                                                     self.w,\
                                                     self.q_s,\
                                                     self.scope2,\
                                                     reuse=True)

        print factors_x.shape, factors_y.shape
        #Get hidden factors
        self.assert_dims(factors_x, factors_y, frame_shape, self.w, self.q_s)
        frame_hidden, factors_h = self.get_hidden_factors(factors_x, factors_y)

        #Get the recon losses
        if self.lamb != 0.:
            recon_x_loss, _ = self.get_recon_loss_x(self.frame_x, self.frame_shape, factors_y, factors_h, self.w, self.q_s)
            recon_y_loss, _ = self.get_recon_loss_y(self.frame_y, self.frame_shape, factors_x, factors_h, self.w, self.q_s)

        #Stop gradients?
        if self.stop_gradient:
            self.frame_hidden = tf.stop_gradient(frame_hidden)
        else:
            self.frame_hidden = frame_hidden

        #Convolve the hidden layer
        print self.frame_hidden.shape

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
                                          a_size,\
                                          activation_fn=None)

        else:
            self.Q = slim.fully_connected(slim.flatten(self.frame_hidden),\
                                          a_size,\
                                          activation_fn=None)

        self.actions = tf.placeholder(shape=[None],\
                                      dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions,\
                                         self.a_size,\
                                         dtype=tf.float32)
        self.targetQ = tf.placeholder(shape=[None, 1],\
                                      dtype=tf.float32)

        self.responsible_output = tf.reduce_sum(tf.multiply(self.Q,\
                                                            self.actions_onehot),\
                                                axis=1,\
                                                keep_dims=True)

        difference = tf.abs(self.responsible_output - self.targetQ)
        quadratic_part = tf.clip_by_value(difference, 0.0, 1.0)
        linear_part = difference - quadratic_part
        errors = (0.5 * tf.square(quadratic_part)) + linear_part
        self.q_loss = tf.reduce_sum(errors) 

        if self.lamb != 0.:
            self.q_recon_loss = recon_x_loss + recon_y_loss
            self.loss = self.q_loss + self.lamb * self.q_recon_loss
        else:
            self.q_recon_loss = self.q_loss
            self.loss = self.q_loss

        self.update_model = tf.train.RMSPropOptimizer(learning_rate=.00025,\
                                                      decay=.95,\
                                                      epsilon=.01).minimize(self.loss)
    def get_action(self, sess, image_in):
        image_in = image_in.astype(np.float64) / 255.
        Q = sess.run(self.Q,\
                     feed_dict={self.frame_x:image_in,
                                self.frame_y:image_in})
        action = np.argmax(Q)
        return action

    def get_Q1(self, sess, states1, tnet):
        states1 = states1.astype(np.float64) / 255.
        return sess.run(tnet.Q,\
                        feed_dict={tnet.frame_x:states1,
                                   tnet.frame_y:states1})

    def train(self, sess, states, actions, targetQ):
        states = states.astype(np.float64) / 255.
        _, loss, q_loss, recon_loss = sess.run([self.update_model,\
                                                self.loss,\
                                                self.q_loss,\
                                                self.q_recon_loss],\
                                                feed_dict={self.frame_x:states,\
                                                           self.frame_y:states,\
                                                           self.actions:actions,\
                                                           self.targetQ:targetQ})
        #print 'loss:', loss, 'q_loss:', q_loss, 'recon_loss:', recon_loss
        return loss, q_loss, recon_loss

    def train_feature_extractor(self, sess, replay_buffer, batch_size=100, iterations=1):
        try:
            self.buff
        except:
            self.buff = Memory(10000)

        for it in range(iterations):
            while len(self.buff.mem) < batch_size:
                state = replay_buffer.sample(1)[0][0]
                state = state.astype(np.float64)
                state = state / 255.

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

            _, recon_loss, = sess.run([self.update_model_recon,
                                       self.recon_loss],
                                       feed_dict={self.x:batch,
                                                  self.y:batch})
            print "train_feature_extractor - recon_loss:", recon_loss

########################################################################################################################
        import pickle
        recon_x, recon_y = sess.run([self.recon_x_, self.recon_y_], feed_dict={self.x:batch, self.y:batch})
        pickle.dump( [batch, recon_x, recon_y], open( "recons.p", "wb" ) )
########################################################################################################################

        return iterations

def main0():
    print 'in main0'
    #gq = gated_qlearning([None, 84, 84, 4], 3, True)
    gq = gated_qlearning([None, 36, 36, 2], 32, 32, .001, [None, 36, 36, 2], a_size=3, stop_gradient=False, lamb=0., w=8, s=4, use_conv_after_fe=True)
    exit()

    for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        print v

if __name__ == '__main__':
    main0()
