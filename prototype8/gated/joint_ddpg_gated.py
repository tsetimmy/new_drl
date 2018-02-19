import numpy as np
import tensorflow as tf

import sys
sys.path.append('../')
from continuous_action.ddpg import actor, critic
from gated_env_modeler import environment_modeler_gated



class joint_ddpg_gated:
    def __init__(self, input_shape, action_size, learning_rate, action_bound):
        self.lamb = 0.5

        var_len = 0
        #Initialize the networks
        self.actor_source = actor(state_shape=input_shape, action_shape=[None, action_size], output_bound=action_bound[0], scope='actor_source')
        self.actor_source_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        var_len += len(self.actor_source_vars)

        self.critic_source = critic(state_shape=input_shape, action_shape=[None, action_size], scope='critic_source')
        self.critic_source_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[var_len:]
        var_len += len(self.critic_source_vars)

        self.actor_target = actor(state_shape=input_shape, action_shape=[None, action_size], output_bound=action_bound[0], scope='actor_target')
        self.actor_target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[var_len:]
        var_len += len(self.actor_target_vars)

        self.critic_target = critic(state_shape=input_shape, action_shape=[None, action_size], scope='critic_target')
        self.critic_target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[var_len:]
        var_len += len(self.critic_target_vars)

        #State modeler
        self.smodel = environment_modeler_gated(s_shape=input_shape, a_size=action_size, out_shape=input_shape, a_type='continuous', numfactors=256)
        self.smodel_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[var_len:]
        var_len += len(self.smodel_vars)

        #Reward modeler
        self.rmodel = environment_modeler_gated(s_shape=input_shape, a_size=action_size, out_shape=[None, 1], a_type='continuous', numfactors=256)
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

        self.batch_size = tf.cast(tf.shape(self.states)[0], tf.float32)
        self.batch_size_joint = tf.cast(tf.shape(self.states_joint)[0], tf.float32)

        #Define the joint loss
        f = self.rmodel.build_recon_s_(self.states_joint, self.actions_joint)
        m = self.smodel.build_recon_s_(self.states_joint, self.actions_joint)
        mu_joint = self.actor_target.get_action(m)
        Q_joint = self.critic_target.get_Q(m, mu_joint)
        Q_source_joint = self.critic_source.get_Q(self.states_joint, self.actions_joint)
        self.jloss = tf.reduce_mean(tf.reduce_sum(f + learning_rate * Q_joint - Q_source_joint, axis=-1))

        #Critic loss
        mu = self.actor_target.get_action(self.states_)
        Q_ = tf.reduce_sum(self.critic_target.get_Q(self.states_, mu), axis=-1)
        Q_source = self.critic_source.get_Q(self.states, self.actions)
        self.closs = tf.reduce_mean(tf.square(self.rewards + (1. - self.dones) * learning_rate * Q_ - tf.reduce_sum(Q_source, axis=-1)))

        #smodel loss
        srecon_s, srecon_s_, srecon_a =  self.smodel.build_computational_graph(self.states, self.states_, self.actions)
        self.sloss = sum(self.smodel.get_recon_losses(srecon_s, srecon_s_, srecon_a, self.states, self.states_, self.actions))

        #rmodel loss
        rrecon_s, rrecon_s_, rrecon_a = self.rmodel.build_computational_graph(self.states, self.rewards, self.actions)
        self.rloss = sum(self.rmodel.get_recon_losses(rrecon_s, rrecon_s_, rrecon_a, self.states, self.rewards, self.actions))

        #Critic optimizer
        self.copt = tf.train.AdamOptimizer().minimize(self.closs + self.lamb * self.jloss, var_list=self.critic_source_vars)

        #smodel optimizer
        self.sopt = tf.train.AdamOptimizer().minimize(self.sloss + self.lamb * self.jloss, var_list=self.smodel_vars)

        #rmodel optimizer
        self.ropt = tf.train.AdamOptimizer().minimize(self.rloss + self.lamb * self.jloss, var_list=self.rmodel_vars)

        #Actor optimizer

        #Gradients from critic
        self.get_grads_joint = tf.gradients(Q_source_joint, self.actions_joint)
        self.act_joint = self.actor_source.get_action(self.states_joint)

        self.get_grads = tf.gradients(Q_source, self.actions)
        self.act = self.actor_source.get_action(self.states)

        #Actor gradients
        self.critic_grads_joint = tf.placeholder(shape=[None, action_size], dtype=tf.float32)
        self.actor_grads_joint = tf.gradients(self.act_joint, self.actor_source_vars, -self.critic_grads_joint)
        self.actor_grads_joint_normalized = list(map(lambda x: tf.div(x, self.batch_size_joint), self.actor_grads_joint))
        assert len(self.actor_grads_joint_normalized) == len(self.actor_source_vars)

        self.critic_grads = tf.placeholder(shape=[None, action_size], dtype=tf.float32)
        self.actor_grads = tf.gradients(self.act, self.actor_source_vars, -self.critic_grads)
        self.actor_grads_normalized = list(map(lambda x: tf.div(x, self.batch_size), self.actor_grads))
        assert len(self.actor_grads_normalized) == len(self.actor_source_vars)

        self.grads = list(map(lambda a, b: a + b, self.actor_grads_joint_normalized, self.actor_grads_normalized))
        assert len(self.grads) == len(self.actor_source_vars)

        self.aopt = tf.train.AdamOptimizer(1e-4).apply_gradients(zip(self.grads, self.actor_source_vars))
    
    def get_action(self, sess, state):
        return sess.run(self.act, feed_dict={self.states:state[np.newaxis, ...]})[0]

    def train(self, sess, states, actions, rewards, states_, dones, states_joint, actions_joint, batch_size, latent_size):
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
        feed_dict = {self.states_joint:states_joint,
                     self.actions_joint:actions_joint}
        critic_grads_joint = sess.run(self.get_grads_joint, feed_dict=feed_dict)[0]

        feed_dict = {self.states:states,
                     self.actions:actions}
        critic_grads = sess.run(self.get_grads, feed_dict=feed_dict)[0]

        feed_dict = {self.critic_grads_joint:critic_grads_joint,
                     self.critic_grads:critic_grads,
                     self.states_joint:states_joint,
                     self.states:states}
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



