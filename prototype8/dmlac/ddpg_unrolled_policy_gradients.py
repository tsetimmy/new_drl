import numpy as np
import tensorflow as tf

import sys
sys.path.append('..')
from continuous_action.ddpg import actor, critic

sys.path.append('../..')
from utils import update_target_graph2

class ddpg_unrolled_policy_gradients:
    def __init__(self, state_shape=[None, 2], action_shape=[None, 1], output_bound_low=[-1.],
                 output_bound_high=[1.], learning_rate=.99, tau=.001):
        assert (-np.array(output_bound_low) == np.array(output_bound_high)).all()
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.output_bound_high = output_bound_high
        self.learning_rate = learning_rate
        self.tau = tau

        #Placeholders (for tuples [s, a, r, s', d])
        self.states = tf.placeholder(shape=state_shape, dtype=tf.float32)
        self.actions = tf.placeholder(shape=action_shape, dtype=tf.float32)
        self.next_states = tf.placeholder(shape=state_shape, dtype=tf.float32)
        self.rewards = tf.placeholder(shape=[None], dtype=tf.float32)
        self.dones = tf.placeholder(shape=[None], dtype=tf.float32)

        #state and reward models
        self.smodel, self.rmodel = self.init_models()

        #Actor and critic networks
        self.actor_src = actor(state_shape, action_shape, output_bound_low, output_bound_high, 'actor_source')
        self.actor_tar = actor(state_shape, action_shape, output_bound_low, output_bound_high, 'actor_target')
        self.critic_src = critic(state_shape, action_shape, 'critic_source')
        self.critic_tar = critic(state_shape, action_shape, 'critic_target')

        '''
        #Vanilla policy gradients
        self.mu = self.actor_src.build(self.states)
        self.actor_loss = -tf.reduce_mean(tf.reduce_sum(self.critic_src.build(self.states, self.mu), axis=-1))
        self.actor_opt = tf.train.AdamOptimizer(1e-4).minimize(self.actor_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'actor_source'))
        '''
        #Unrolled policy gradients
        self.mu = self.actor_src.build(self.states)
        self.reward_predicted = self.rmodel.build(self.states, self.mu)
        self.next_state_predicted = self.smodel.build(self.states, self.mu)
        self.next_q = self.critic_src.build(self.next_state_predicted, self.actor_src.build(self.next_state_predicted))
        self.value = self.reward_predicted + self.learning_rate * self.next_q
        self.actor_loss = -tf.reduce_mean(tf.reduce_sum(self.value, axis=-1))
        self.actor_opt = tf.train.AdamOptimizer(1e-4).minimize(self.actor_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'actor_source'))

        #Value function squared loss
        self.q_tar = tf.reduce_sum(self.critic_tar.build(self.next_states, self.actor_tar.build(self.next_states)), axis=-1)
        self.q_src = tf.reduce_sum(self.critic_src.build(self.states, self.actions), axis=-1)
        self.critic_loss = tf.reduce_mean(tf.square(self.rewards + (1. - self.dones) * self.learning_rate * self.q_tar - self.q_src))
        self.critic_opt = tf.train.AdamOptimizer(1e-3).minimize(self.critic_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'critic_source'))

        #Reward model squared loss
        self.reward_predict = self.rmodel.build(self.states, self.actions)
        self.rmodel_loss, self.rmodel_vars = self.rmodel.get_losses(self.states, tf.expand_dims(self.rewards, axis=-1), self.actions)
        self.rmodel_opt = tf.train.AdamOptimizer().minimize(self.rmodel_loss, var_list=self.rmodel_vars)

        #State model squared loss
        self.state_predict = self.smodel.build(self.states, self.actions)
        self.smodel_loss, self.smodel_vars = self.smodel.get_losses(self.states, self.next_states, self.actions)
        self.smodel_opt = tf.train.AdamOptimizer().minimize(self.smodel_loss, var_list=self.smodel_vars)

        # Update and copy operators
        self.update_target_actor = update_target_graph2('actor_source', 'actor_target', tau)
        self.update_target_critic = update_target_graph2('critic_source', 'critic_target', tau)

        self.copy_target_actor = update_target_graph2('actor_source', 'actor_target', 1.)
        self.copy_target_critic = update_target_graph2('critic_source', 'critic_target', 1.)

    def train(self, sess, states, actions, rewards, next_states, dones, *_):
        dones = dones.astype(np.float64)

        #Train state model
        sess.run(self.smodel_opt, feed_dict={self.states:states, self.actions:actions, self.next_states:next_states})#Update state model

        #Train reward model
        sess.run(self.rmodel_opt, feed_dict={self.states:states, self.actions:actions, self.rewards:rewards})#Update reward model

        #Train policy function
        feed_dict = {self.states:states}
        sess.run(self.actor_opt, feed_dict=feed_dict)

        #Train value function
        feed_dict = {self.states:states,
                     self.actions:actions,
                     self.rewards:rewards,
                     self.next_states:next_states,
                     self.dones:dones}
        sess.run(self.critic_opt, feed_dict=feed_dict)

    def action(self, sess, state):
        return sess.run(self.mu, feed_dict={self.states:state})[0]

    def init_models(self):
        from mlp_env_modeler import mlp_env_modeler
        smodel = mlp_env_modeler(self.state_shape[-1], True)
        rmodel = mlp_env_modeler(1, True)
        return smodel, rmodel

def main():
    ddpg = ddpg_unrolled_policy_gradients()

if __name__ == '__main__':
    main()
