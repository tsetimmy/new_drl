import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tflearn
import uuid

import sys
sys.path.append('..')
from continuous_action.ddpg import actor

sys.path.append('../..')
from utils import update_target_graph2

class mlp_env_modeler:
    def __init__(self, output_size, batch_norm=True):
        self.output_size = output_size
        self.batch_norm = batch_norm
        self.scope = str(uuid.uuid4())
        self.reuse = None

    def build(self, states, actions):
        states_embed = self.fully_connected(states, self.scope+'/states_embed')
        actions_embed = self.fully_connected(actions, self.scope+'/actions_embed')
        concat = tf.concat([states_embed, actions_embed], axis=-1)
        hidden = self.fully_connected(concat, self.scope+'/hidden')
        output = slim.fully_connected(hidden, self.output_size, activation_fn=None, scope=self.scope+'/output', reuse=self.reuse)
        self.reuse = True
        return output

    def fully_connected(self, inputs, scope):
        outputs = slim.fully_connected(inputs, 128, activation_fn=(None if self.batch_norm else tf.nn.relu), scope=scope, reuse=self.reuse)
        if self.batch_norm:
            outputs = tflearn.layers.normalization.batch_normalization(outputs, scope=scope+'_bn', reuse=self.reuse)
            outputs = tf.nn.relu(outputs)
        return outputs

    def get_losses(self, states, targets, actions):
        predictions = self.build(states, actions)
        return tf.reduce_mean(tf.reduce_sum(tf.square(predictions - targets), axis=-1)),\
               tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

class dmlac:
    def __init__(self, state_shape=[None, 2], action_shape=[None, 1], output_bound_low=[-1.],
                 output_bound_high=[1.], learning_rate=.9, tau=.01, forward_steps=1, trace_decay=.9, model='dmlac_mlp'):
        assert (-np.array(output_bound_low) == np.array(output_bound_high)).all()
        assert forward_steps >= 1
        assert model in ['dmlac_mlp', 'dmlac_gated', 'dmlac_gan']
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.output_bound_high = output_bound_high
        self.learning_rate = learning_rate
        self.tau = tau
        self.forward_steps = forward_steps
        self.trace_decay = trace_decay
        self.model = model

        #Placeholders (for tuples [s, a, r, s', d])
        self.states = tf.placeholder(shape=state_shape, dtype=tf.float32)
        self.actions = tf.placeholder(shape=action_shape, dtype=tf.float32)
        self.next_states = tf.placeholder(shape=state_shape, dtype=tf.float32)
        self.rewards = tf.placeholder(shape=[None], dtype=tf.float32)
        self.dones = tf.placeholder(shape=[None], dtype=tf.float32)

        #state and reward models
        self.smodel, self.rmodel = self.declare_models()

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
        self.reward_predict_mu = self.rmodel.build(self.states, self.mu)
        self.state_predict_mu = self.smodel.build(self.states, self.mu)
        self.value = self.reward_predict_mu + self.learning_rate * self.critic_src.build(self.state_predict_mu)
        self.actor_loss = -tf.reduce_mean(tf.reduce_sum(self.value, axis=-1))
        self.actor_opt = tf.train.AdamOptimizer(1e-4).minimize(self.actor_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'actor_source'))

        #Value function squared loss
        self.value_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        self.value_predict = self.critic_src.build(self.states)
        self.value_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.value_target - self.value_predict), axis=-1))
        self.value_opt = tf.train.AdamOptimizer(1e-3).minimize(self.value_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'critic_source'))

        #Reward model squared loss
        self.reward_predict = self.rmodel.build(self.states, self.actions)
        self.rmodel_loss, self.rmodel_vars = self.rmodel.get_losses(self.states, self.rewards, self.actions)
        self.rmodel_opt = tf.train.AdamOptimizer().minimize(self.rmodel_loss, var_list=self.rmodel_vars)

        #State model squared loss
        self.state_predict = self.smodel.build(self.states, self.actions)
        self.smodel_loss, self.smodel_vars = self.smodel.get_losses(self.states, self.state_predict, self.actions)
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
        sess.run(self.actor_opt, feed_dict={self.states:states})#Update policy

        #Train value function
        states_iter = np.copy(states)
        value_estimates = []
        predicted_rewards = []
        for j in range(1, self.forward_steps+1):#for j = [1,...,n]
            action_predict = sess.run(self.mu_tar, feed_dict={self.states:states_iter})
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

    def action(self, sess, state):
        return sess.run(self.mu, feed_dict={self.states:state})[0]

    def declare_models(self):
        if self.model == 'dmlac_mlp':
            smodel = mlp_env_modeler(self.state_shape[-1], True)
            rmodel = mlp_env_modeler(1, True)
        elif self.model == 'dmlac_gated':
            from gated.gated_env_modeler import gated_env_modeler
            smodel = gated_env_modeler(s_shape=self.state_shape, a_size=self.action_shape[-1],
                                       out_shape=self.state_shape, a_type='continuous', numfactors=128)
            rmodel = gated_env_modeler(s_shape=self.state_shape, a_size=self.action_shape[-1],
                                       out_shape=[None, 1], a_type='continuous', numfactors=128)
        elif self.model == 'dmlac_gan':
            pass
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

def main():
    mlp = dmlac()

if __name__ == '__main__':
    main()
