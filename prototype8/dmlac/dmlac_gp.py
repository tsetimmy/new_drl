import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tflearn

from gp2 import multivariate_gaussian_process

import sys
sys.path.append('..')
from continuous_action.ddpg import actor

sys.path.append('../..')
from utils import update_target_graph2

class dmlac_gp:
    def __init__(self, state_shape=[None, 2], action_shape=[None, 1], output_bound_low=[-1.],
                 output_bound_high=[1.], learning_rate=.9, tau=.01, forward_steps=1, trace_decay=.9, hist_len=10):
        assert (-np.array(output_bound_low) == np.array(output_bound_high)).all()
        assert forward_steps >= 1
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.output_bound_high = output_bound_high
        self.learning_rate = learning_rate
        self.tau = tau
        self.forward_steps = forward_steps
        self.trace_decay = trace_decay

        self.update_value_with_model = True
        self.use_gp = True

        self.hist_len = hist_len

        #Placeholders (for tuples [s, a, r, s', d])
        self.states = tf.placeholder(shape=state_shape, dtype=tf.float32)
        self.actions = tf.placeholder(shape=action_shape, dtype=tf.float32)
        self.next_states = tf.placeholder(shape=state_shape, dtype=tf.float32)
        self.rewards = tf.placeholder(shape=[None], dtype=tf.float32)
        self.dones = tf.placeholder(shape=[None], dtype=tf.float32)
        self.state_action = tf.placeholder(shape=[None, self.state_shape[-1]+self.action_shape[-1]], dtype=tf.float32)

        #state and reward models
        self.smodel, self.rmodel = self.init_models()

        #Actor and critic (value) networks
        self.actor_src = actor(state_shape, action_shape, output_bound_low, output_bound_high, 'actor_source')
        self.actor_tar = actor(state_shape, action_shape, output_bound_low, output_bound_high, 'actor_target')
        self.critic_src = critic_value(state_shape, True, 'critic_source')
        self.critic_tar = critic_value(state_shape, True, 'critic_target')

        #mu and value targets
        self.mu_tar = self.actor_tar.build(self.states)
        self.value_tar = self.critic_tar.build(self.states)

        #Policy gradient
        self.mu = self.actor_src.build(self.states)
        self.reward_predict_mu = self.rmodel_build()
        self.state_predict_mu = self.smodel_build()

        self.value = self.reward_predict_mu + self.learning_rate * self.critic_src.build(self.state_predict_mu)
        self.actor_loss = -tf.reduce_mean(tf.reduce_sum(self.value, axis=-1))
        self.actor_opt = tf.train.AdamOptimizer(1e-4).minimize(self.actor_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'actor_source'))

        #Value function squared loss
        self.value_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        self.value_predict = self.critic_src.build(self.states)
        self.value_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.value_target - self.value_predict), axis=-1))
        self.value_opt = tf.train.AdamOptimizer(1e-3).minimize(self.value_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'critic_source'))

        #Reward model squared loss
        self.reward_predict, self.rmodel_opt = self.get_rmodel_predict_and_opt()

        #State model squared loss
        self.state_predict, self.smodel_opt = self.get_smodel_predict_and_opt()

        # Update and copy operators
        self.update_target_actor = update_target_graph2('actor_source', 'actor_target', tau)
        self.update_target_critic = update_target_graph2('critic_source', 'critic_target', tau)

        self.copy_target_actor = update_target_graph2('actor_source', 'actor_target', 1.)
        self.copy_target_critic = update_target_graph2('critic_source', 'critic_target', 1.)

    def get_rmodel_predict_and_opt(self):
        if self.use_gp == True:
            reward_predict, rmodel_opt = self.rmodel.build_and_get_opt(self.state_action, tf.expand_dims(self.rewards, axis=-1), tf.concat([self.states, self.actions], axis=-1))
        else:
            reward_predict = self.rmodel.build(self.states, self.actions)
            rmodel_loss, rmodel_vars = self.rmodel.get_losses(self.states, tf.expand_dims(self.rewards, axis=-1), self.actions)
            rmodel_opt = tf.train.AdamOptimizer().minimize(rmodel_loss, var_list=rmodel_vars)
        return reward_predict, rmodel_opt

    def get_smodel_predict_and_opt(self):
        if self.use_gp == True:
            state_predict, smodel_opt = self.smodel.build_and_get_opt(self.state_action, self.next_states, tf.concat([self.states, self.actions], axis=-1))
        else:
            state_predict = self.smodel.build(self.states, self.actions)
            smodel_loss, smodel_vars = self.smodel.get_losses(self.states, self.next_states, self.actions)
            smodel_opt = tf.train.AdamOptimizer().minimize(smodel_loss, var_list=smodel_vars)
        return state_predict, smodel_opt

    def rmodel_build(self):
        if self.use_gp == True:
            reward_predict_mu = self.rmodel.build(self.state_action, tf.expand_dims(self.rewards, axis=-1), tf.concat([self.states, self.mu], axis=-1))
        else:
            reward_predict_mu = self.rmodel.build(self.states, self.mu)
        return reward_predict_mu

    def smodel_build(self):
        if self.use_gp == True:
            state_predict_mu = self.smodel.build(self.state_action, self.next_states, tf.concat([self.states, self.mu], axis=-1))
        else:
            state_predict_mu = self.smodel.build(self.states, self.mu)
        return state_predict_mu

    def update_hist(self, mem):
        hist = np.array(mem.mem[-self.hist_len:])
        self.hist_states = np.concatenate(hist[:, 0], axis=0)
        self.hist_actions = np.concatenate(hist[:, 1], axis=0)
        self.hist_rewards = hist[:, 2]
        self.hist_next_states = np.concatenate(hist[:, 3], axis=0)
        self.hist_state_action = np.concatenate([self.hist_states, self.hist_actions], axis=-1)

    def train(self, sess, states, actions, rewards, next_states, dones, *_):
        dones = dones.astype(np.float32)

        #Train state and reward model
        if self.use_gp == True:
            epochs = 20
            batch_size = len(self.hist_state_action)
            assert len(self.hist_state_action) > 0
            assert len(self.hist_state_action) == len(self.hist_rewards)
            assert len(self.hist_state_action) == len(self.hist_next_states)
            for i in xrange(epochs):
                idx = np.random.choice(np.arange(len(self.hist_state_action)), batch_size, replace=False)
                X_mini = self.hist_state_action[idx]
                next_states_mini = self.hist_next_states[idx]
                rewards_mini = self.hist_rewards[idx]
                for s_opt in self.smodel_opt:
                    sess.run(s_opt, feed_dict={self.state_action:X_mini, self.next_states:next_states_mini})
                for r_opt in self.rmodel_opt:
                    sess.run(r_opt, feed_dict={self.state_action:X_mini, self.rewards:rewards_mini})
        else:
            sess.run(self.rmodel_opt, feed_dict={self.states:states, self.actions:actions, self.rewards:rewards})#Update reward model
            sess.run(self.smodel_opt, feed_dict={self.states:states, self.actions:actions, self.next_states:next_states})#Update state model

        #Train policy function
        if self.use_gp == True:
            sess.run(self.actor_opt, feed_dict={self.state_action:self.hist_state_action,
                                                self.next_states:self.hist_next_states,
                                                self.rewards:self.hist_rewards,
                                                self.states:states})#Update policy
        else:
            sess.run(self.actor_opt, feed_dict={self.states:states})#Update policy

        #Train value function
        if self.update_value_with_model == True:
            states_iter = np.copy(states)
            value_estimates = []
            predicted_rewards = []
            for j in range(1, self.forward_steps+1):#for j = [1,...,n]
                action_predict = sess.run(self.mu_tar, feed_dict={self.states:states_iter})
                if self.use_gp == True:
                    states_iter, reward_predict = sess.run([self.state_predict, self.reward_predict], feed_dict={self.state_action:self.hist_state_action,
                                                                                                                 self.next_states:self.hist_next_states,
                                                                                                                 self.rewards:self.hist_rewards,
                                                                                                                 self.states:states_iter,
                                                                                                                 self.actions:action_predict})
                else:
                    states_iter, reward_predict = sess.run([self.state_predict, self.reward_predict], feed_dict={self.states:states_iter, self.actions:action_predict})

                predicted_rewards.append(reward_predict)
                assert len(predicted_rewards) == j
                value_estimate = np.zeros_like(reward_predict)
                for k in range(1, j+1):#for k = [1,...,j]
                    value_estimate += (self.learning_rate ** (k-1)) * predicted_rewards[k-1]
                value_estimate += (self.learning_rate ** j) * sess.run(self.value_tar, feed_dict={self.states:states_iter})        
                value_estimates.append(value_estimate)

            assert len(value_estimates) == self.forward_steps
            value_estimates_averaged = np.zeros_like(value_estimate)
            for j in range(len(value_estimates)):
                value_estimates_averaged += (self.trace_decay ** j) * value_estimates[j]
            value_estimates_averaged *= ((1. - self.trace_decay) / (1. - self.trace_decay ** self.forward_steps))
            sess.run(self.value_opt, feed_dict={self.states:states, self.value_target:value_estimates_averaged})#Update values
        else:
            value_tar = sess.run(self.value_tar, feed_dict={self.states:next_states})
            value_tar = rewards[..., np.newaxis] + (1. - dones[..., np.newaxis]) * self.learning_rate * value_tar
            sess.run(self.value_opt, feed_dict={self.states:states, self.value_target:value_tar})#Update values

    def action(self, sess, state):
        return sess.run(self.mu, feed_dict={self.states:state})[0]

    def init_models(self):
        if self.use_gp == True:
            smodel = multivariate_gaussian_process([None, self.state_shape[-1]+self.action_shape[-1]], self.state_shape)
            rmodel = multivariate_gaussian_process([None, self.state_shape[-1]+self.action_shape[-1]], [None, 1])
        else:
            from mlp_env_modeler import mlp_env_modeler
            smodel = mlp_env_modeler(self.state_shape[-1], True)
            rmodel = mlp_env_modeler(1, True)
        return smodel, rmodel

class critic_value:
    def __init__(self, input_shape=[None, 3], batch_norm=True, scope=None):
        self.input_shape = input_shape
        self.scope = scope
        self.batch_norm = batch_norm
        self.reuse = None

    def build(self, states):
        fc1 = slim.fully_connected(states, 256, activation_fn=(None if self.batch_norm else tf.nn.relu), scope=self.scope+'/fc1', reuse=self.reuse)
        if self.batch_norm:
            fc1 = tflearn.layers.normalization.batch_normalization(fc1, scope=self.scope+'/fc1_bn', reuse=self.reuse)
            fc1 = tf.nn.relu(fc1)
        fc2 = slim.fully_connected(fc1, 128, activation_fn=(None if self.batch_norm else tf.nn.relu), scope=self.scope+'/fc2', reuse=self.reuse)
        if self.batch_norm:
            fc2 = tflearn.layers.normalization.batch_normalization(fc2, scope=self.scope+'/fc2_bn', reuse=self.reuse)
            fc2 = tf.nn.relu(fc2)
        output = slim.fully_connected(fc2, 1, 
                                      weights_initializer=tf.initializers.random_uniform(-3e-3, 3e-3),
                                      biases_initializer=tf.initializers.random_uniform(-3e-3, 3e-3),
                                      activation_fn=None, scope=self.scope+'/output', reuse=self.reuse)
        self.reuse = True
        return output
