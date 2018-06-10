import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.distributions as tfd

import gym

from gp_tf import gaussian_process

import matplotlib.pyplot as plt

import sys
sys.path.append('..')
from prototype8.dmlac.real_env_pendulum import real_env_pendulum_reward

class gp_model:
    def __init__(self, x_dim, y_dim, x_train_data=None, y_train_data=None):
        self.state_dim = 3
        self.action_dim = 1
        self.discount_factor = .9
        self.action_bound_low = np.array([-2.])
        self.action_bound_high = np.array([2.])
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.x_train_data = x_train_data
        self.y_train_data = y_train_data

        self.policy_scope = 'policy_scope'
        self.policy_reuse_vars = None

        self.models = [gaussian_process(self.x_dim, self.x_train_data, self.y_train_data)\
                       for i in range(self.y_dim)]

        self.outputs = [[model.mu, model.var] for model in self.models]

        '''
        self.states = tf.placeholder(shape=[None, self.state_dim], dtype=tf.float64)
        self.actions = self.build_policy(self.states)
        self.unroll(self.states)
        '''

    def unroll(self, seed_states):
        assert seed_states.shape.as_list() == [None, self.state_dim]
        no_samples = 20
        unroll_steps = 25
        self.reward_model = real_env_pendulum_reward()#Use true model.

        states = tf.expand_dims(seed_states, axis=1)
        states = tf.tile(states, [1, no_samples, 1])
        states = tf.reshape(states, shape=[-1, self.state_dim])

        costs = []
        for unroll_step in range(unroll_steps):
            actions = self.build_policy(states)

            rewards = (self.discount_factor ** unroll_step) * self.reward_model.build(states, actions)
            rewards = tf.reshape(tf.squeeze(rewards, axis=-1), shape=[-1, no_samples])
            costs.append(-rewards)

            states_actions = tf.concat([states, actions], axis=-1)

            next_states = self.get_next_states(states_actions)
            states = next_states

        costs = tf.stack(costs, axis=-1)
        loss = tf.reduce_mean(tf.reduce_sum(tf.reduce_mean(costs, axis=1), axis=-1))

        self.opt = tf.train.AdamOptimizer().minimize(loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'policy_scope'))

    def get_next_states(self, states_actions):
        outputs = [model.get_prediction(states_actions) for model in self.models]

        mu = tf.concat([output[0] for output in outputs], axis=-1)
        sd = tf.concat([output[1] for output in outputs], axis=-1)

        next_state = tfd.MultivariateNormalDiag(loc=mu, scale_diag=sd).sample()

        return next_state

    def set_training_data(self, x_train_data, y_train_data):
        assert x_train_data.shape[0] == y_train_data.shape[0]
        assert len(x_train_data.shape) == 2
        assert len(y_train_data.shape) == 2
        assert x_train_data.shape[-1] == self.x_dim
        assert y_train_data.shape[-1] == self.y_dim

        self.x_train_data = np.copy(x_train_data)
        self.y_train_data = np.copy(y_train_data)

        for i in range(len(self.models)):
            self.models[i].x_train_data = self.x_train_data
            self.models[i].y_train_data = self.y_train_data[:, i:i+1]

    def predict(self, sess, x_test):
        assert len(x_test.shape) == 2
        assert x_test.shape[-1] == self.x_train_data.shape[-1]

        feed_dict = {}
        for i in range(len(self.models)):
            feed_dict[self.models[i].x_train] = self.models[i].x_train_data
            feed_dict[self.models[i].y_train] = self.models[i].y_train_data
            feed_dict[self.models[i].x_test] = x_test

        outputs = sess.run(self.outputs, feed_dict=feed_dict)

        means = np.concatenate([output[0] for output in outputs], axis=-1)
        sds = np.stack([np.sqrt(np.diag(output[1])) for output in outputs], axis=-1)

        return means, sds

    def train_hyperparamters(self, sess, iterations=50000):
        total_size = len(self.x_train_data)
        batch_size = 64
        for it in range(iterations):
            idx = np.random.randint(total_size, size=batch_size)
            for i in range(len(self.models)):
                feed_dict = {self.models[i].x_train:self.models[i].x_train_data[idx, ...],
                             self.models[i].y_train:self.models[i].y_train_data[idx, ...]}
                try:
                    _, loss = sess.run([self.models[i].opt, self.models[i].log_marginal_likelihood], feed_dict=feed_dict)
                    print 'i:', i, 'iterations:', it, 'loss:', -loss, '|',
                except:
                    print 'Cholesky decomposition failed.',
            print ''

        '''
        for i in range(len(self.models)):
            feed_dict = {self.models[i].x_train:self.models[i].x_train_data,
                         self.models[i].y_train:self.models[i].y_train_data}
            for it in range(iterations):
                _, loss = sess.run([self.models[i].opt, self.models[i].log_marginal_likelihood], feed_dict=feed_dict)
                print 'i:', i, 'iterations:', it, 'loss:', -loss
        '''

    def build_policy(self, states):
        assert states.shape.as_list() == [None, 3]

        #Fully connected layer 1
        fc1 = slim.fully_connected(states, 256, activation_fn=tf.nn.relu, scope=self.policy_scope+'/fc1', reuse=self.policy_reuse_vars)

        #Output layer
        output = slim.fully_connected(fc1, self.action_dim, activation_fn=tf.nn.tanh, scope=self.policy_scope+'/output', reuse=self.policy_reuse_vars)

        #Apply action bounds
        action_bound = tf.constant(self.action_bound_high, dtype=tf.float64)
        policy = tf.multiply(output, action_bound)

        #Change flag
        self.policy_reuse_vars = True

        return policy

def plotting_experiment():
    gpm = gp_model(x_dim=4, y_dim=3)

    env = gym.make('Pendulum-v0')

    epochs = 3
    train_size = (epochs - 1) * 200
    policy = []

    data = []
    for epoch in range(epochs):
        state = env.reset()
        while True:
            action = np.random.uniform(env.action_space.low, env.action_space.high, 1)
            policy.append(action)
            next_state, reward, done, _ = env.step(action)
            data.append([state, action, next_state])
            state = np.copy(next_state)
            if done:
                break

    states = np.stack([d[0] for d in data], axis=0)
    actions = np.stack([d[1] for d in data], axis=0)
    next_states = np.stack([d[2] for d in data], axis=0)

    states_actions = np.concatenate([states, actions], axis=-1)

    x_train = states_actions[:train_size, ...]
    y_train = next_states[:train_size, ...]
    x_test = states_actions[train_size:, ...]
    y_test = next_states[train_size:, ...]

    gpm.set_training_data(x_train, y_train)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        gpm.train_hyperparamters(sess)
        means, sds = gpm.predict(sess, x_test)

        # ----- First plotting experiment. -----
        plt.figure(1)
        plt.clf()
        for i in range(3):
            plt.subplot(2, 3, i+1)
            plt.grid()
            plt.plot(np.arange(len(y_test)), y_test[:, i])
            plt.errorbar(np.arange(len(means)), means[:, i], yerr=sds[:, i], color='m', ecolor='g')

        # ----- Second plotting experiment. -----
        #plt.figure(2)
        #plt.clf()
        no_lines = 50
        policy = actions[-200:, ...]
        seed_state = x_test[0, :3]

        for line in range(no_lines):
            print 'At line:', line
            states = []
            state = np.copy(seed_state)
            states.append(np.copy(state))
            for action in policy:
                state_action = np.concatenate([state, action], axis=0)[np.newaxis, ...]
                means, sds = gpm.predict(sess, state_action)
                means = np.squeeze(means, axis=0)
                sds = np.squeeze(sds, axis=0)
                state = np.random.multivariate_normal(means, (sds**2)*np.eye(len(sds)))
                states.append(np.copy(state))
            states = np.stack(states, axis=0)

            for i in range(3):
                plt.subplot(2, 3, 3+i+1)
                plt.plot(np.arange(len(states[:, i])), states[:, i], color='r')

        for i in range(3):
            plt.subplot(2, 3, 3+i+1)
            plt.plot(np.arange(len(y_test)), y_test[:, i])
            plt.grid()
        plt.show()

def main():
    plotting_experiment()

if __name__ == '__main__':
    main()
