import numpy as np
import tensorflow as tf

import sys
sys.path.append('../')
from qnetwork import qnetwork2
from gated_env_modeler import gated_env_modeler


class joint_dqn_gated:
    def __init__(self, input_shape, action_size, learning_rate):

        self.lamb = .5
        #Initalize the networks
        self.qnet = qnetwork2(input_shape=input_shape, action_size=action_size)
        self.qnet_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.tnet = qnetwork2(input_shape=input_shape, action_size=action_size)
        self.tnet_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[len(self.qnet_vars):]

        #State modeler
        self.smodel = gated_env_modeler(s_shape=input_shape, a_size=action_size, out_shape=input_shape, a_type='discrete', numfactors=256)
        self.smodel_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[len(self.qnet_vars)+len(self.tnet_vars):]

        #Reward modeler
        self.rmodel = gated_env_modeler(s_shape=input_shape, a_size=action_size, out_shape=[None, 1], a_type='discrete', numfactors=256)
        self.rmodel_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[len(self.qnet_vars)+len(self.tnet_vars)+len(self.smodel_vars):]

        #Placeholders
        self.states = tf.placeholder(shape=input_shape, dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, action_size, dtype=tf.float32)
        self.rewards = tf.placeholder(shape=[None], dtype=tf.float32)
        self.states_ = tf.placeholder(shape=input_shape, dtype=tf.float32)
        self.dones = tf.placeholder(shape=[None], dtype=tf.float32)

        self.states_joint = tf.placeholder(shape=input_shape, dtype=tf.float32)
        self.actions_joint = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_joint_onehot = tf.one_hot(self.actions_joint, action_size, dtype=tf.float32)

        #Joint loss
        f = self.rmodel.build_recon_s_(self.states_joint, self.actions_joint_onehot)
        m = self.smodel.build_recon_s_(self.states_joint, self.actions_joint_onehot)
        self.jloss = tf.reduce_mean(tf.reduce_sum(
            tf.square(
                f + learning_rate * tf.reduce_max(self.tnet.build_computational_graph(m), axis=-1, keep_dims=True) -\
                tf.reduce_sum(tf.multiply(self.actions_joint_onehot, self.qnet.build_computational_graph(self.states_joint)), axis=-1, keep_dims=True)),
        axis=-1, keep_dims=False))

        #Q loss
        self.Q = self.qnet.build_computational_graph(self.states)
        self.responsible_output = tf.reduce_sum(tf.multiply(self.Q, self.actions_onehot), axis=-1)
        self.targetQ = self.rewards + (1. - self.dones) * learning_rate * tf.reduce_max(self.tnet.build_computational_graph(self.states_), axis=-1)
        self.qloss = tf.reduce_mean(tf.square(self.responsible_output - self.targetQ))

        #smodel loss
        srecon_s, srecon_s_, srecon_a =  self.smodel.build_computational_graph(self.states, self.states_, self.actions_onehot)
        self.sloss = sum(self.smodel.get_recon_losses(srecon_s, srecon_s_, srecon_a, self.states, self.states_, self.actions_onehot))

        #rmodel loss
        rrecon_s, rrecon_s_, rrecon_a = self.rmodel.build_computational_graph(self.states, self.rewards, self.actions_onehot)
        self.rloss = sum(self.rmodel.get_recon_losses(rrecon_s, rrecon_s_, rrecon_a, self.states, self.rewards, self.actions_onehot))

        #Q optimizer
        self.qopt = tf.train.AdamOptimizer().minimize(self.qloss + self.lamb * self.jloss, var_list=self.qnet_vars)

        #smodel optimizer
        self.sopt = tf.train.AdamOptimizer().minimize(self.sloss + self.lamb * self.jloss, var_list=self.smodel_vars)

        #rmodel optimizer
        self.ropt = tf.train.AdamOptimizer().minimize(self.rloss + self.lamb * self.jloss, var_list=self.rmodel_vars)

    def get_action(self, sess, states):
        Q = sess.run(self.Q, feed_dict={self.states:states[np.newaxis, ...]})
        return np.argmax(Q)

    def updateQ(self, sess, states, actions, rewards, states_, dones, states2, actions2, batch_size, latent_size):
        #batch_size and latent_size are not used
        feed_dict = {self.states:states,
                     self.actions:actions,
                     self.rewards:rewards,
                     self.states_:states_,
                     self.dones:dones,
                     self.states_joint:states2,
                     self.actions_joint:actions2}
        _ = sess.run(self.qopt, feed_dict=feed_dict)

    def updateS(self, sess, states, actions, states_, states2, actions2, batch_size, latent_size):
        #batch_size and latent_size are not used
        feed_dict = {self.states:states,
                     self.actions:actions,
                     self.states_:states_,
                     self.states_joint:states2,
                     self.actions_joint:actions2}
        _ = sess.run(self.sopt, feed_dict=feed_dict)

    def updateR(self, sess, states, actions, rewards, states2, actions2, batch_size, latent_size):
        #batch_size and latent_size are not used

        feed_dict = {self.states:states,
                     self.actions:actions,
                     self.rewards:rewards,
                     self.states_joint:states2,
                     self.actions_joint:actions2}
        _ = sess.run(self.ropt, feed_dict=feed_dict)





def main():
    jdqng = joint_dqn_gated(input_shape=[None, 4], action_size=4, learning_rate=.99)

if __name__ == '__main__':
    main()
