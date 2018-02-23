import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

import sys
sys.path.append('..')
from qnetwork import qnetwork2
from gated_env_modeler import environment_modeler_gated

class gated_regularized_qnetwork_visual_input:
    def __init__(self, s_shape, a_size, learning_rate=.95):
        self.string = 'gated_regularized_qnetwork_visual_input'
        assert len(s_shape) == 4
        self.s_shape = s_shape
        self.a_size = a_size

        #Declare the placeholders
        self.states = tf.placeholder(shape=s_shape, dtype=tf.float32)
        self.states_ = tf.placeholder(shape=s_shape, dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, a_size, dtype=tf.float32)
        self.rewards = tf.placeholder(shape=[None], dtype=tf.float32)
        self.dones = tf.placeholder(shape=[None], dtype=tf.float32)

        #Preprocess the states with a convnet
        states_conv, sc_shape = self.get_convolution(self.states, 'states')
        states_conv_vars, varlen = self.get_vars(0)

        states_conv_, sc__shape = self.get_convolution(self.states_, 'states_')
        states_conv__vars, varlen = self.get_vars(varlen)#get vars

        #Get the reconstructions
        self.modeler = environment_modeler_gated(states_conv.shape.as_list(), a_size, states_conv_.shape.as_list(), 'discrete', 256)
        modeler_vars, varlen = self.get_vars(varlen)#get vars
        recon_s_tmp, recon_s__tmp, recon_a = self.modeler.build_computational_graph(states_conv, states_conv_, self.actions_onehot)
        recon_s = tf.nn.relu(recon_s_tmp)
        recon_s_ = tf.nn.relu(recon_s__tmp)

        #Get deconvolutions
        states_pixel_recon = self.get_deconvolution(recon_s, sc_shape, 'states')
        states_pixel_recon_ = self.get_deconvolution(recon_s_, sc__shape, 'states_')
        deconv_biases_vars, varlen = self.get_vars(varlen)#get vars

        #Build the q values
        self.V, self.A, self.Q = self.build_q(states_conv, scope='sourceQ', reuse=None)
        q_vars, varlen = self.get_vars(varlen)#get vars

        #Target network
        states_conv_target, _ = self.get_convolution(self.states, 'target')
        states_conv_target_vars, varlen = self.get_vars(varlen)#get vars
        _, _, self.targetQ = self.build_q(states_conv_target, scope='targetQ', reuse=None)
        q_target_vars, varlen = self.get_vars(varlen)#get vars

        #Sanity check for variables
        total_vars = states_conv_vars + states_conv__vars + modeler_vars + deconv_biases_vars + q_vars + states_conv_target_vars + q_target_vars
        assert total_vars == tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        #Source and target variables
        self.qnet_vars = states_conv_vars+q_vars
        self.tnet_vars = states_conv_target_vars+q_target_vars

        #Get the hat values
        self.Vhat, self.Ahat, self.Qhat = self.build_q(recon_s, scope='sourceQ', reuse=True)

        #Pixel loss functions
        self.sloss = tf.reduce_mean(tf.reduce_sum(slim.flatten(tf.square(states_pixel_recon - self.states)), axis=-1))
        self.sloss_ = tf.reduce_mean(tf.reduce_sum(slim.flatten(tf.square(states_pixel_recon_ - self.states_)), axis=-1))
        self.aloss = tf.reduce_mean(tf.reduce_sum(tf.multiply(-tf.log(tf.nn.softmax(recon_a, dim=-1) + 1e-10), self.actions_onehot), axis=-1))
        self.vloss = tf.reduce_mean(tf.reduce_sum(tf.square(self.Vhat - self.V), axis=-1))
        self.adloss = tf.reduce_mean(tf.reduce_sum(tf.square(self.Ahat - self.A), axis=-1))

        #Q loss
        responsible_output = tf.reduce_sum(tf.multiply(self.Q, self.actions_onehot), axis=-1)
        self.qloss = tf.reduce_mean(tf.square(self.rewards + (1. - self.dones) * learning_rate * tf.reduce_max(self.targetQ, axis=-1) - responsible_output))

        #Total losses
        self.auxloss = .0001*(self.sloss + self.sloss_ + self.aloss + self.vloss + self.adloss)
        self.loss = self.auxloss + self.qloss

        #Model vars
        self.modeler_vars = states_conv_vars+states_conv__vars+modeler_vars+deconv_biases_vars

        self.optaux = tf.train.AdamOptimizer().minimize(self.loss, var_list=self.modeler_vars)
        self.opt = tf.train.AdamOptimizer().minimize(self.loss, var_list=self.qnet_vars)

        #Internal counter
        self.counter = 1

    def get_action(self, sess, state):
        state = state.astype(np.float64) / 255.
        Q = sess.run(self.Q, feed_dict={self.states:state})
        action = np.argmax(Q)
        return action

    def train(self, sess, states, actions, rewards, states_, dones, learning_rate, tnet):
        states = states.astype(np.float64) / 255.
        states_ = states_.astype(np.float64) / 255.
        dones = dones.astype(np.float64)

        feed_dict = {self.states:states,
                     self.actions:actions,
                     self.rewards:rewards,
                     self.states_:states_,
                     self.dones:dones}
        self.counter = (self.counter + 1) % 2
        if self.counter == 0:
            _, auxloss1, qloss1 = sess.run([self.optaux, self.auxloss, self.qloss], feed_dict=feed_dict)
            return auxloss1+qloss1
        elif self.counter == 1:
            _, auxloss2, qloss2 = sess.run([self.opt, self.auxloss, self.qloss], feed_dict=feed_dict)
            return auxloss2+qloss2 

    def get_vars(self, varlen):
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[varlen:]
        return variables, len(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

    def build_q(self, states_enc, scope, reuse):
        streamA, streamV = tf.split(slim.flatten(states_enc), 2, 1)
        A = slim.fully_connected(streamA, self.a_size, activation_fn=None, scope=scope+'_Advantage', reuse=reuse)
        V = slim.fully_connected(streamV, 1, activation_fn=None, scope=scope+'_Value', reuse=reuse)
        Q = V + tf.subtract(A, tf.reduce_mean(A, axis=-1, keepdims=True))
        return V, A, Q

    def get_deconvolution(self, states_enc, shape, scope):
        conv3_bak = tf.reshape(states_enc, shape=[-1]+shape[1:])
        biases2_bak = tf.Variable(tf.zeros([64]))
        conv2_bak = tf.nn.relu(slim.conv2d_transpose(conv3_bak,
                                                     num_outputs=64,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding='VALID',
                                                     biases_initializer=None,
                                                     activation_fn=None,
                                                     reuse=True,
                                                     scope=scope+'_conv3') + biases2_bak)
        biases3_bak = tf.Variable(tf.zeros([32]))
        conv1_bak = tf.nn.relu(slim.conv2d_transpose(conv2_bak,
                                                     num_outputs=32,
                                                     kernel_size=4,
                                                     stride=2,
                                                     padding='VALID',
                                                     biases_initializer=None,
                                                     activation_fn=None,
                                                     reuse=True,
                                                     scope=scope+'_conv2') + biases3_bak)
        biases4_bak = tf.Variable(tf.zeros([self.s_shape[-1]]))
        recon = tf.nn.sigmoid(slim.conv2d_transpose(conv1_bak,
                                                 num_outputs=self.s_shape[-1],
                                                 kernel_size=8,
                                                 stride=4,
                                                 padding='VALID',
                                                 biases_initializer=None,
                                                 activation_fn=None,
                                                 reuse=True,
                                                 scope=scope+'_conv1') + biases4_bak)
        return recon

    def get_convolution(self, states, scope):
        biases1 = tf.Variable(tf.zeros([32]))
        conv1 = tf.nn.relu(slim.conv2d(activation_fn=None,
                                       inputs=states,
                                       num_outputs=32,
                                       kernel_size=[8, 8],
                                       stride=[4, 4],
                                       biases_initializer=None,
                                       padding='VALID',
                                       scope=scope+'_conv1') + biases1)
        biases2 = tf.Variable(tf.zeros([64]))
        conv2 = tf.nn.relu(slim.conv2d(activation_fn=None,
                                       inputs=conv1,
                                       num_outputs=64,
                                       kernel_size=[4, 4],
                                       stride=[2, 2],
                                       biases_initializer=None,
                                       padding='VALID',
                                       scope=scope+'_conv2') + biases2)
        biases3 = tf.Variable(tf.zeros([64]))
        conv3 = tf.nn.relu(slim.conv2d(activation_fn=None,
                                       inputs=conv2,
                                       num_outputs=64,
                                       kernel_size=[3, 3],
                                       stride=[1, 1],
                                       biases_initializer=None,
                                       padding='VALID',
                                       scope=scope+'_conv3') + biases3)
        return slim.flatten(conv3), conv3.shape.as_list()

class gated_regularized_qnetwork:
    def __init__(self, s_shape, a_size, numfactors, learning_rate=.95):
        self.s_shape = s_shape
        self.a_size = a_size

        #Declare the networks and get their variables
        self.qnet = qnetwork2(s_shape, a_size)
        self.qnet_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.tnet = qnetwork2(s_shape, a_size)
        self.tnet_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[len(self.qnet_vars):]
        self.modeler = environment_modeler_gated(s_shape, a_size, s_shape, 'discrete', numfactors)
        self.modeler_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[len(self.qnet_vars)+len(self.tnet_vars):]

        #Declare the placeholders
        self.states = tf.placeholder(shape=s_shape, dtype=tf.float32)
        self.states_ = tf.placeholder(shape=s_shape, dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, a_size, dtype=tf.float32)

        self.rewards = tf.placeholder(shape=[None], dtype=tf.float32)
        self.dones = tf.placeholder(shape=[None], dtype=tf.float32)

        #Reconstructions from the environment modeler
        recon_s, recon_s_, recon_a = self.modeler.build_computational_graph(self.states, self.states_, self.actions_onehot)

        #Q values and its regularizers
        V, A, Q = self.qnet.build_computational_graph2(self.states)
        Vhat, Ahat, Qhat = self.qnet.build_computational_graph2(recon_s)

        #Loss function
        self.sloss = tf.reduce_mean(tf.reduce_sum(tf.square(recon_s - self.states), axis=-1))
        self.sloss_ = tf.reduce_mean(tf.reduce_sum(tf.square(recon_s_ - self.states_), axis=-1))
        self.aloss = tf.reduce_mean(tf.reduce_sum(tf.multiply(-tf.log(tf.nn.softmax(recon_a, dim=-1) + 1e-10), self.actions_onehot), axis=-1))
        self.vloss = tf.reduce_mean(tf.reduce_sum(tf.square(Vhat - V), axis=-1))
        self.adloss = tf.reduce_mean(tf.reduce_sum(tf.square(Ahat - A), axis=-1))

        #Get Q loss
        Qtarget = self.tnet.build_computational_graph(self.states_)
        responsible_output = tf.reduce_sum(tf.multiply(Q, self.actions_onehot), axis=-1)
        self.qloss = tf.reduce_mean(tf.square(self.rewards + (1. - self.dones) * learning_rate * tf.reduce_max(Qtarget, axis=-1) - responsible_output))

        #Total loss
        self.auxloss = .5*(self.sloss + self.sloss_ + self.aloss + self.vloss + self.adloss)
        self.loss = self.auxloss + self.qloss

        #Optimizers
        self.optaux = tf.train.AdamOptimizer().minimize(self.loss, var_list=self.modeler_vars)
        self.opt = tf.train.AdamOptimizer().minimize(self.loss, var_list=self.qnet_vars)

    def get_action(self, sess, state):
        Q = sess.run(self.qnet.Q, feed_dict={self.qnet.states:state[np.newaxis, ...]})
        return np.argmax(Q)

    def updateQ(self, sess, states, actions, rewards, states_, dones, states2, actions2, batch_size, latent_size):
        feed_dict = {self.states:states,
                     self.actions:actions,
                     self.rewards:rewards,
                     self.states_:states_,
                     self.dones:dones}
        _, auxloss, qloss = sess.run([self.optaux, self.auxloss, self.qloss], feed_dict=feed_dict)
        #print auxloss, qloss
        _, auxloss, qloss = sess.run([self.opt, self.auxloss, self.qloss], feed_dict=feed_dict)
        #print auxloss, qloss

    def updateS(self, sess, states, actions, states_, states2, actions2, batch_size, latent_size):
        pass#function not used

    def updateR(self, sess, states, actions, rewards, states2, actions2, batch_size, latent_size):
        pass#function not used




def main():
    print 'inside main'
    grq = gated_regularized_qnetwork_visual_input([None, 84, 84, 4], 4, 256)

if __name__ == '__main__':
    main()
