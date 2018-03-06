import numpy as np
import tensorflow as tf

import sys
sys.path.append('../')
from continuous_action.ddpg import actor, critic
from gated_env_modeler import gated_env_modeler

class joint_ddpg_gated:
    def __init__(self, input_shape, action_size, learning_rate, action_bound_low, action_bound_high):
        self.lamb = .1
        self.learning_rate = learning_rate

        #Initialize the networks
        self.actor_source = actor(state_shape=input_shape, action_shape=[None, action_size], output_bound_low=action_bound_low, output_bound_high=action_bound_high, scope='actor_source')
        self.critic_source = critic(state_shape=input_shape, action_shape=[None, action_size], scope='critic_source')
        self.actor_target = actor(state_shape=input_shape, action_shape=[None, action_size], output_bound_low=action_bound_low, output_bound_high=action_bound_high, scope='actor_target')
        self.critic_target = critic(state_shape=input_shape, action_shape=[None, action_size], scope='critic_target')

        var_len = 0
        #State modeler
        self.smodel = gated_env_modeler(s_shape=input_shape, a_size=action_size, out_shape=input_shape, a_type='continuous', numfactors=128)
        self.smodel_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        var_len += len(self.smodel_vars)

        #Reward modeler
        self.rmodel = gated_env_modeler(s_shape=input_shape, a_size=action_size, out_shape=[None, 1], a_type='continuous', numfactors=128)
        self.rmodel_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[var_len:]
        var_len += len(self.rmodel_vars)

        #Placeholders
        self.states = tf.placeholder(shape=input_shape, dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None, action_size], dtype=tf.float32)
        self.rewards = tf.placeholder(shape=[None], dtype=tf.float32)
        self.states_ = tf.placeholder(shape=input_shape, dtype=tf.float32)
        self.dones = tf.placeholder(shape=[None], dtype=tf.float32)

        self.states_joint = tf.placeholder(shape=input_shape, dtype=tf.float32)
        self.actions_joint = tf.placeholder(shape=[None, action_size], dtype=tf.float32)

        #Define the joint loss
        f = self.rmodel.build_recon_s_(self.states_joint, self.actions_joint)
        m = self.smodel.build_recon_s_(self.states_joint, self.actions_joint)
        mu_joint = self.actor_target.build(m)
        Q_joint = self.critic_target.build(m, mu_joint)
        Q_source_joint = self.critic_source.build(self.states_joint, self.actions_joint)
        self.jloss = tf.reduce_mean(tf.reduce_sum(tf.square(f + self.learning_rate * Q_joint - Q_source_joint), axis=-1))

        #Critic loss
        mu = self.actor_target.build(self.states_)
        Q_ = tf.reduce_sum(self.critic_target.build(self.states_, mu), axis=-1)
        Q_source = self.critic_source.build(self.states, self.actions)
        self.closs = tf.reduce_mean(tf.square(self.rewards + (1. - self.dones) * self.learning_rate * Q_ - tf.reduce_sum(Q_source, axis=-1)))

        #smodel loss
        srecon_s, srecon_s_, srecon_a =  self.smodel.build_computational_graph(self.states, self.states_, self.actions)
        self.sloss = sum(self.smodel.get_recon_losses(srecon_s, srecon_s_, srecon_a, self.states, self.states_, self.actions))

        #rmodel loss
        rrecon_s, rrecon_s_, rrecon_a = self.rmodel.build_computational_graph(self.states, self.rewards, self.actions)
        self.rloss = sum(self.rmodel.get_recon_losses(rrecon_s, rrecon_s_, rrecon_a, self.states, self.rewards, self.actions))

        #Critic optimizer
        self.copt = tf.train.AdamOptimizer().minimize(self.closs + self.lamb * self.jloss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic_source'))

        #smodel optimizer
        self.sopt = tf.train.AdamOptimizer().minimize(self.sloss + self.lamb * self.jloss, var_list=self.smodel_vars)

        #rmodel optimizer
        self.ropt = tf.train.AdamOptimizer().minimize(self.rloss + self.lamb * self.jloss, var_list=self.rmodel_vars)

        #Actor optimizer
        self.make_actions = self.actor_source.build(self.states)
        self.policy_q = self.critic_source.build(self.states, self.make_actions)
        self.loss_policy_q = -tf.reduce_mean(tf.reduce_sum(self.policy_q, axis=-1))
        self.aopt = tf.train.AdamOptimizer(1e-4).minimize(self.loss_policy_q, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'actor_source'))

        #Params assert
        params = self.smodel_vars + self.rmodel_vars + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor_target')+ tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic_target')+ tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic_source')+ tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor_source') 
        assert len(params) == len(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
        assert params == tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    
    def action(self, sess, state):
        return sess.run(self.make_actions, feed_dict={self.states:state})[0]

    def train(self, sess, states, actions, rewards, states_, dones, states_joint, actions_joint, batch_size, latent_size):
        dones = dones.astype(np.float64)
        #Update the critic
        feed_dict = {self.states:states,
                     self.actions:actions,
                     self.states_:states_,
                     self.rewards:rewards,
                     self.dones:dones,
                     self.states_joint:states_joint,
                     self.actions_joint:actions_joint}
        sess.run(self.copt, feed_dict=feed_dict)

        #Update the actor
        feed_dict = {self.states:states}
        sess.run(self.aopt, feed_dict=feed_dict)

        #Update the state model
        feed_dict = {self.states:states,
                     self.actions:actions,
                     self.states_:states_,
                     self.states_joint:states_joint,
                     self.actions_joint:actions_joint}
        sess.run(self.sopt, feed_dict=feed_dict)

        #Update the rewards model
        feed_dict = {self.states:states,
                     self.actions:actions,
                     self.rewards:rewards,
                     self.states_joint:states_joint,
                     self.actions_joint:actions_joint}
        sess.run(self.ropt, feed_dict=feed_dict)

def main():
    jdg = joint_ddpg_gated([None, 4], 1, .99, [1.])

if __name__ == '__main__':
    main()

