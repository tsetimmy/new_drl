import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.kernel_methods as km

from matplotlib import cm
from scipy.stats import multivariate_normal
from scipy.stats import norm
import matplotlib.pyplot as plt

import uuid

def process(X, y, dim):
    X = np.atleast_1d(X)
    y = np.atleast_1d(y)
    if X.ndim == 1:
        X = np.reshape(X, [-1, dim])
    if y.ndim == 1:
        y = np.reshape(y, [-1, 1])
    assert len(X) == len(y)
    assert X.shape[-1] == dim
    assert y.shape[-1] == 1
    return X, y

def squared_exponential_kernel(a, b, signal_sd, length_scale):
    sqdist = tf.reduce_sum(tf.square(a), axis=-1, keep_dims=True) +\
             -2. * tf.matmul(a, tf.transpose(b)) +\
             tf.transpose(tf.reduce_sum(tf.square(b), axis=-1, keep_dims=True))
    kernel = tf.square(signal_sd) * tf.exp(-.5 * (1. / tf.square(length_scale)) * sqdist)
    return kernel

class hyperparameter_search:
    def __init__(self, dim):
        self.dim = dim

        # Placeholders.
        self.X = tf.placeholder(shape=[None, self.dim], dtype=tf.float64)
        self.y = tf.placeholder(shape=[None, 1], dtype=tf.float64)

        # UUID.
        self.uuid = str(uuid.uuid4())

        # Batch size
        self.n = tf.shape(self.X)[0]

        # Variables.
        self.length_scale = tf.get_variable(name='length_scale'+self.uuid, shape=[], dtype=tf.float64,
                                            initializer=tf.constant_initializer(.25))#.25
        self.signal_sd = tf.get_variable(name='signal_sd'+self.uuid, shape=[], dtype=tf.float64,
                                               initializer=tf.constant_initializer(1.))#1.
        self.noise_sd = tf.get_variable(name='noise_sd'+self.uuid, shape=[], dtype=tf.float64,
                                              initializer=tf.constant_initializer(.2))#.2

        # Get predictive distribution and log marginal likelihood (Algorithm 2.1 in the GP book).
        L = tf.cholesky(squared_exponential_kernel(self.X, self.X, self.signal_sd, self.length_scale) +\
                        tf.multiply(tf.square(self.noise_sd), tf.eye(self.n, dtype=tf.float64)))
        alpha = tf.linalg.solve(tf.transpose(L), tf.linalg.solve(L, self.y))
        self.log_marginal_likelihood = -.5 * tf.matmul(tf.transpose(self.y), alpha)[0, 0] +\
                                       -.5 * tf.reduce_sum(tf.log(tf.diag_part(L))) +\
                                       -.5 * tf.cast(self.n, dtype=tf.float64) * np.log(2. * np.pi)
        self.opt = tf.train.AdamOptimizer().minimize(-self.log_marginal_likelihood, var_list=[self.length_scale, self.signal_sd, self.noise_sd])

    # TODO: idxs require memory; make it memory-free
    def train_hyperparameters(self, sess, xtrain, ytrain, idxs):
        xtrain, ytrain = process(xtrain, ytrain, self.dim)
        for idx, it in zip(idxs, range(len(idxs))):
            try:
                log_marginal_likelihood, _ = sess.run([self.log_marginal_likelihood, self.opt], feed_dict={self.X:xtrain[idx, ...], self.y:ytrain[idx, ...]})
                print 'iteration:', it, 'loss:', -log_marginal_likelihood
            except:
                print 'Cholesky decomposition failed.'

class bayesian_model:
    def __init__(self, dim, observation_space_low, observation_space_high, action_space_low,
                 action_space_high, no_basis, length_scale=.25, signal_sd=1., noise_sd=.2):
        self.dim = dim
        self.observation_space_high = observation_space_high
        self.observation_space_low = observation_space_low

        self.action_space_low = action_space_low
        self.action_space_high = action_space_high

        self.no_basis = no_basis

        self.length_scale_np = length_scale
        self.signal_sd_np = signal_sd
        self.noise_sd_np = noise_sd

        if self.noise_sd_np < 0:
            print 'Warning: noise_sd is negative. Setting to 1e-5.'
            self.noise_sd_np = 1e-5
        # Assertions.
        assert self.length_scale_np > 0.
        assert self.signal_sd_np > 0.
        np.testing.assert_array_equal(-self.observation_space_low, self.observation_space_high)
        np.testing.assert_array_equal(-self.action_space_low, self.action_space_high)
        assert self.dim == len(self.observation_space_high) + len(self.action_space_high)

        # UUID.
        self.uuid = str(uuid.uuid4())

        # Prior noise
        self.s = 1.

        # Values to keep track
        self.cum_xx = np.zeros([self.no_basis, self.no_basis])
        self.cum_xy = np.zeros([self.no_basis, 1])

        self.mu_prior = np.zeros([self.no_basis, 1])
        self.sigma_prior = np.eye(self.no_basis) * (self.s**2)

        self.mu = np.copy(self.mu_prior)
        self.sigma = np.copy(self.sigma_prior)

        # Initialize basis function hyperparameters.
        self.length_scale = tf.get_variable(name='length_scale'+self.uuid, shape=[], dtype=tf.float64,
                                            initializer=tf.constant_initializer(self.length_scale_np))#.25
        self.signal_sd = tf.get_variable(name='signal_sd'+self.uuid, shape=[], dtype=tf.float64,
                                               initializer=tf.constant_initializer(self.signal_sd_np))#1.
        self.noise_sd = tf.get_variable(name='noise_sd'+self.uuid, shape=[], dtype=tf.float64,
                                              initializer=tf.constant_initializer(self.noise_sd_np))#.2

        # Placeholders.
        self.X = tf.placeholder(shape=[None, self.dim], dtype=tf.float64)
        self.y = tf.placeholder(shape=[None, 1], dtype=tf.float64)
        self.X_basis = self.approx_rbf_kern_basis(self.X)

        # Mean and variance priors placeholders
        self.mu_prior_pl = tf.placeholder(shape=[self.no_basis, 1], dtype=tf.float64)
        self.sigma_prior_pl = tf.placeholder(shape=[self.no_basis, self.no_basis], dtype=tf.float64)

        # Mean and variance placeholders
        self.mu_placeholder = tf.placeholder(shape=[self.no_basis, 1], dtype=tf.float64)
        self.sigma_placeholder = tf.placeholder(shape=[self.no_basis, self.no_basis], dtype=tf.float64)

        # Cumulative XX and cumulative Xy placeholders.
        self.cum_xx_pl = tf.placeholder(shape=[self.no_basis, self.no_basis], dtype=tf.float64)
        self.cum_xy_pl = tf.placeholder(shape=[self.no_basis, 1], dtype=tf.float64)

    # Given test points, return the posterior predictive distributions.
    def posterior_predictive_distribution(self, states_actions, _):
        assert states_actions.shape.as_list() == [None, self.dim]

        bases = self.approx_rbf_kern_basis(states_actions)

        posterior_predictive_mu = tf.matmul(bases, self.mu_placeholder)
        posterior_predictive_sigma = tf.square(self.noise_sd) + tf.reduce_sum(tf.multiply(tf.matmul(bases, self.sigma_placeholder), bases), axis=-1, keep_dims=True)

        return posterior_predictive_mu, posterior_predictive_sigma
        #return tf.concat([posterior_predictive_mu, posterior_predictive_sigma], axis=-1)

    def approx_rbf_kern_basis(self, X):
        try:
            self.rffm
        except:
            self.rffm_seed = np.random.randint(2**32-1)
            self.rffm = km.RandomFourierFeatureMapper(self.dim, self.no_basis, stddev=self.length_scale_np, seed=self.rffm_seed)

        basis_phi = tf.multiply(self.signal_sd, self.rffm.map(X))
        sqrt_sigma_inv = tf.matrix_inverse(self.s * tf.eye(self.no_basis, dtype=tf.float64))
        basis_psi = tf.transpose(tf.matmul(sqrt_sigma_inv, tf.transpose(basis_phi)))

        return basis_psi

    def update(self, sess, X, y):
        X, y = process(X, y, self.dim)
        X_basis = sess.run(self.X_basis, feed_dict={self.X:X})

        self.cum_xx += np.matmul(X_basis.T, X_basis)
        self.cum_xy += np.matmul(X_basis.T, y)

        self.sigma = self.noise_sd_np**2 * np.linalg.inv(self.noise_sd_np**2 * np.linalg.inv(self.sigma_prior) + self.cum_xx)
        self.mu = np.matmul(np.matmul(self.sigma, np.linalg.inv(self.sigma_prior)), self.mu_prior) + self.noise_sd_np**-2 * np.matmul(self.sigma, self.cum_xy)

    def mu_sigma(self, X, y):
        assert X.shape.as_list() == [self.no_basis, self.no_basis]
        assert y.shape.as_list() == [self.no_basis, 1]

        sigma = tf.multiply(tf.square(self.noise_sd), tf.matrix_inverse(tf.multiply(tf.square(self.noise_sd),
                tf.matrix_inverse(self.sigma_prior_pl)) + self.cum_xx_pl))
        mu = tf.matmul(tf.matmul(sigma, tf.matrix_inverse(self.sigma_prior_pl)), self.mu_prior_pl) + \
             tf.multiply(tf.reciprocal(tf.square(self.noise_sd)), tf.matmul(sigma, self.cum_xy_pl))
        return mu, sigma

    def post_pred2(self, states_actions, mu, sigma):
        #assert states_actions.shape.as_list() == [None, self.dim]
        assert mu.shape.as_list() == [self.no_basis, 1]
        assert sigma.shape.as_list() == [self.no_basis, self.no_basis]

        bases = self.approx_rbf_kern_basis(states_actions)

        post_pred_mu = tf.matmul(bases, mu)
        post_pred_sigma = tf.square(self.noise_sd) + tf.reduce_sum(tf.multiply(tf.matmul(bases, sigma), bases), axis=-1, keep_dims=True)

        return post_pred_mu, post_pred_sigma
        #return tf.concat([post_pred_mu, post_pred_sigma], axis=-1)

def plotting_experiment():
    import argparse
    import gym
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-data", type=int, default=1)
    args = parser.parse_args()

    uid = str(uuid.uuid4())

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

    if args.dump_data >= 1:
        pickle.dump(data, open('data_'+uid+'.p', 'wb'))

    states = np.stack([d[0] for d in data], axis=0)
    actions = np.stack([d[1] for d in data], axis=0)
    next_states = np.stack([d[2] for d in data], axis=0)

    states_actions = np.concatenate([states, actions], axis=-1)

    x_train = states_actions[:train_size, ...]
    y_train = next_states[:train_size, ...]
    x_test = states_actions[train_size:, ...]
    y_test = next_states[train_size:, ...]

    # Train the hyperparameters
    hs = [hyperparameter_search(dim=env.observation_space.shape[0]+env.action_space.shape[0])
          for _ in range(env.observation_space.shape[0])]
    hyperparameters = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        batch_size = 32
        iterations = 50000
        idxs = [np.random.randint(len(x_train), size=batch_size) for _ in range(iterations)]
        for i in range(len(hs)):
            hs[i].train_hyperparameters(sess, x_train, y_train[:, i], idxs)
            hyperparameters.append(sess.run([hs[i].length_scale, hs[i].signal_sd, hs[i].noise_sd]))

    if args.dump_data >= 1:
        pickle.dump(hyperparameters, open('hyperparameters_'+uid+'.p', 'wb'))

    # Prepare the models
    models = [bayesian_model(4, np.array([-1., -1., -8.]), np.array([1., 1., 8.]), np.array([-2.]), np.array([2.]), 256, *hyperparameters[i])
              for i in range(env.observation_space.shape[0])]
    states_actions_placeholder = tf.placeholder(shape=[None, env.observation_space.shape[0]+env.action_space.shape[0]], dtype=tf.float64)
    ppd = tf.stack([model.posterior_predictive_distribution(states_actions_placeholder, None) for model in models], axis=0)

    if args.dump_data >= 1:
        pickle.dump([model.rffm_seed for model in models], open('rffm_seed_'+uid+'.p', 'wb'))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #chunks = len(x_train)
        chunks = 10
        for c in range(0, len(x_train), chunks):
            print 'chunk:', c
            for i in range(len(models)):
                models[i].update(sess, x_train[c:c+chunks, ...], y_train[c:c+chunks, i])

        # ----- First plotting experiment. -----
        feed_dict = {}
        feed_dict[states_actions_placeholder] = x_test
        for model in models:
            feed_dict[model.mu_placeholder] = model.mu
            feed_dict[model.sigma_placeholder] = model.sigma

        mu_sigma = sess.run(ppd, feed_dict=feed_dict)

        means = mu_sigma[:, :, 0].T
        sds = np.sqrt(mu_sigma[:, :, 1].T)

        plt.figure(1)
        plt.clf()
        for i in range(3):
            plt.subplot(2, 3, i+1)
            plt.grid()
            plt.plot(np.arange(len(y_test)), y_test[:, i])
            plt.errorbar(np.arange(len(means)), means[:, i], yerr=sds[:, i], color='m', ecolor='g')

        # ----- Second plotting experiment. -----
        no_lines = 50
        policy = actions[-200:, ...]
        seed_state = x_test[0, :3]

        feed_dict = {}
        for model in models:
            feed_dict[model.mu_placeholder] = model.mu
            feed_dict[model.sigma_placeholder] = model.sigma

        for line in range(no_lines):
            print 'At line:', line
            states = []
            state = np.copy(seed_state)
            states.append(np.copy(state))
            for action in policy:
                state_action = np.concatenate([state, action], axis=0)[np.newaxis, ...]

                feed_dict[states_actions_placeholder] = state_action
                mu_sigma = sess.run(ppd, feed_dict=feed_dict)

                means = mu_sigma[:, :, 0].T
                variances = mu_sigma[:, :, 1].T

                means = np.squeeze(means, axis=0)
                variances = np.squeeze(variances, axis=0)

                state = np.random.multivariate_normal(means, variances*np.eye(len(variances)))
                states.append(np.copy(state))
            states = np.stack(states, axis=0)

            for i in range(3):
                plt.subplot(2, 3, 3+i+1)
                plt.plot(np.arange(len(states[:, i])), states[:, i], color='r')

        y_test = np.concatenate([y_test, seed_state[np.newaxis, ...]], axis=0)
        for i in range(3):
            plt.subplot(2, 3, 3+i+1)
            plt.plot(np.arange(len(y_test)), y_test[:, i])
            plt.grid()

        if args.dump_data >= 1:
            plt.savefig('plot_'+uid+'.pdf')
        else:
            plt.show()

if __name__ == '__main__':
    plotting_experiment()
