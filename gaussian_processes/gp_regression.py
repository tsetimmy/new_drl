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
    def __init__(self, x_dim, y_dim, state_dim, action_dim, observation_space_low,
                 observation_space_high, action_bound_low, action_bound_high, unroll_steps,
                no_samples, discount_factor, train_policy_batch_size, train_policy_iterations,
                x_train_data=None, y_train_data=None):

        assert x_dim == state_dim + action_dim
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.observation_space_low = observation_space_low
        self.observation_space_high = observation_space_high
        self.action_bound_low = action_bound_low
        self.action_bound_high = action_bound_high
        self.x_train_data = x_train_data
        self.y_train_data = y_train_data

        self.unroll_steps = unroll_steps
        self.no_samples = no_samples
        self.discount_factor = discount_factor

        self.train_policy_batch_size = train_policy_batch_size
        self.train_policy_iterations = train_policy_iterations

        self.policy_scope = 'policy_scope'
        self.policy_reuse_vars = None

        self.models = [gaussian_process(self.x_dim, self.x_train_data, self.y_train_data)\
                       for i in range(self.y_dim)]

        self.outputs = [[model.mu, model.var] for model in self.models]

        self.states = tf.placeholder(shape=[None, self.state_dim], dtype=tf.float64)
        self.actions = self.build_policy(self.states)
        self.unroll(self.states)

    def train(self, sess, memory):
        feed_dict = {}
        for model in self.models:
            feed_dict[model.x_train] = model.x_train_data
            feed_dict[model.y_train] = model.y_train_data

        for it in range(self.train_policy_iterations):
            batch = memory.sample(self.train_policy_batch_size)
            states = np.concatenate([b[0] for b in batch], axis=0)
            feed_dict[self.states] = states

            try:
                loss, _ = sess.run([self.loss, self.opt], feed_dict=feed_dict)
                print 'iteration:', it, 'loss:', loss
            except:
                print 'Cholesky decomposition failed.'
                for model in self.models:
                    print model.xtrain
                    print '------------------'
                    print model.squared_exponential_kernel(model.x_train, model.x_train)
                    print '*******************'
                    print sess.run(model.noise_sd)
                    print ')))))))))))))))))))'
                    print sess.run(model.signal_sd)
                    print '+++++++++++++++++++++'
                    print sess.run(model.length_scale)
                    print '!!!!!!!!!!!!!!!!!!!!!!!!'

    def act(self, sess, states):
        states = np.atleast_2d(states)
        assert states.shape == (1, self.state_dim)
        return sess.run(self.actions, feed_dict={self.states:states})[0]

    def unroll(self, seed_states):
        assert seed_states.shape.as_list() == [None, self.state_dim]
        no_samples = self.no_samples
        unroll_steps = self.unroll_steps
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
        self.loss = tf.reduce_mean(tf.reduce_sum(tf.reduce_mean(costs, axis=1), axis=-1))
        self.opt = tf.train.AdamOptimizer().minimize(self.loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'policy_scope'))

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

    def train_hyperparameters(self, sess, iterations=50000, verbose=True):
        total_size = len(self.x_train_data)
        batch_size = 64
        for it in range(iterations):
            idx = np.random.randint(total_size, size=batch_size)
            for i in range(len(self.models)):
                feed_dict = {self.models[i].x_train:self.models[i].x_train_data[idx, ...],
                             self.models[i].y_train:self.models[i].y_train_data[idx, ...]}
                try:
                    _, loss = sess.run([self.models[i].opt, self.models[i].log_marginal_likelihood], feed_dict=feed_dict)
                    if verbose:
                        print 'i:', i, 'iterations:', it, 'loss:', -loss, '|',
                except:
                    if verbose:
                        print 'Cholesky decomposition failed.',
            if verbose:
                print ''

        '''
        for i in range(len(self.models)):
            feed_dict = {self.models[i].x_train:self.models[i].x_train_data,
                         self.models[i].y_train:self.models[i].y_train_data}
            for it in range(iterations):
                _, loss = sess.run([self.models[i].opt, self.models[i].log_marginal_likelihood], feed_dict=feed_dict)
                print 'i:', i, 'iterations:', it, 'loss:', -loss
        '''

    def set_and_train_hyperparameters(self, sess, memory, batch_size, iterations, verbose=True):
        batch = memory.sample(batch_size)

        states = np.concatenate([b[0] for b in batch], axis=0)
        actions = np.concatenate([b[1] for b in batch], axis=0)
        next_states = np.concatenate([b[3] for b in batch], axis=0)

        states_actions = np.concatenate([states, actions], axis=-1)

        self.set_training_data(states_actions, next_states)
        self.train_hyperparameters(sess, iterations=iterations, verbose=verbose)
    
    def build_policy(self, states):
        assert states.shape.as_list() == [None, self.state_dim]

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

def policy_gradient_experiment():
    import argparse
    from utils import Memory
    parser = argparse.ArgumentParser()
    parser.add_argument("--unroll-steps", type=int, default=25)
    parser.add_argument("--no-samples", type=int, default=20)
    parser.add_argument("--discount-factor", type=float, default=.9)
    parser.add_argument("--replay-mem-size", type=int, default=1000000)
    parser.add_argument("--train-policy-batch-size", type=int, default=32)
    parser.add_argument("--train-policy-iterations", type=int, default=30)
    parser.add_argument("--replay-start-size-epochs", type=int, default=2)
    args = parser.parse_args()
    
    print args

    env = gym.make('Pendulum-v0')

    # Initialize the agent
    gpm = gp_model(x_dim=env.observation_space.shape[0]+env.action_space.shape[0],
                   y_dim=env.observation_space.shape[0],
                   state_dim=env.observation_space.shape[0],
                   action_dim=env.action_space.shape[0],
                   observation_space_low=env.observation_space.low,
                   observation_space_high=env.observation_space.high,
                   action_bound_low=env.action_space.low,
                   action_bound_high=env.action_space.high,
                   unroll_steps=args.unroll_steps,
                   no_samples=args.no_samples,
                   discount_factor=args.discount_factor,
                   train_policy_batch_size=args.train_policy_batch_size,
                   train_policy_iterations=args.train_policy_iterations)

    # Initialize the memory
    memory = Memory(args.replay_mem_size)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        state = env.reset()
        total_rewards = 0.0
        epoch = 1
        for time_steps in range(30000):
            if epoch <= args.replay_start_size_epochs:
                action = np.random.uniform(env.action_space.low, env.action_space.high, 1)
            else:
                action = gpm.act(sess, state)
            next_state, reward, done, _ = env.step(action)
            total_rewards += float(reward)

            # Append to the batch
            memory.add([np.atleast_2d(state), np.atleast_2d(action), reward, np.atleast_2d(next_state), done])

            # s <- s'
            state = np.copy(next_state)

            if done == True:
                print 'time steps', time_steps, 'epoch', epoch, 'total rewards', total_rewards

                if epoch == args.replay_start_size_epochs:
                    gpm.set_and_train_hyperparameters(sess, memory, memory.max_size, 50000, True)

                if epoch >= args.replay_start_size_epochs:
                    gpm.train(sess, memory)

                if epoch >= args.replay_start_size_epochs + 1:
                    gpm.set_and_train_hyperparameters(sess, memory, 500, 10000, False)

                epoch += 1
                total_rewards = 0.
                state = env.reset()

def main():
    #plotting_experiment()
    policy_gradient_experiment()

if __name__ == '__main__':
    main()
