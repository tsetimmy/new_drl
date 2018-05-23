import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from matplotlib import cm
from scipy.stats import multivariate_normal
from scipy.stats import norm
import matplotlib.pyplot as plt

class bayesian_model:
    def __init__(self, dim, observation_space_low, observation_space_high, no_basis):
        self.dim = dim
        self.observation_space_high = observation_space_high
        self.observation_space_low = observation_space_low
        self.no_basis = no_basis

        # Assertions
        np.testing.assert_array_equal(-self.observation_space_low, self.observation_space_high)
        assert len(self.observation_space_high) == dim

        # Placeholders
        self.X = tf.placeholder(shape=[None, self.dim], dtype=tf.float32)
        self.y = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        self.X_basis = self.basis_functions(self.X)

        # Mean and variance of prior
        self.prior_precision = 2.#Assume known beforehand
        self.prior_mu = tf.Variable(np.zeros([self.no_basis, 1]), dtype=tf.float32)
        self.prior_sigma = tf.Variable(np.eye(self.no_basis) / self.prior_precision, dtype=tf.float32)

        self.likelihood_sd = .2#Assume known beforehand

        # Mean and variance of posterior
        self.posterior_sigma = tf.matrix_inverse(tf.matrix_inverse(self.prior_sigma) + \
                                                 pow(self.likelihood_sd, -2) * \
                                                 tf.matmul(tf.transpose(self.X_basis), self.X_basis))

        self.posterior_mu = tf.matmul(tf.matmul(self.posterior_sigma, tf.matrix_inverse(self.prior_sigma)), self.prior_mu) + \
                            pow(self.likelihood_sd, -2) * tf.matmul(tf.matmul(self.posterior_sigma, tf.transpose(self.X_basis)), self.y)

        # Operation for assigning prior = posterior
        self.posterior_mu_in = tf.placeholder(shape=[self.no_basis, 1], dtype=tf.float32)
        self.posterior_sigma_in = tf.placeholder(shape=[self.no_basis, self.no_basis], dtype=tf.float32)

        self.op_pos2prior_assign = [self.prior_mu.assign(self.posterior_mu_in), self.prior_sigma.assign(self.posterior_sigma_in)]

    # Basis functions using RBFs (to model nonlinear data)
    def basis_functions(self, X, sigma=.96):
        assert self.no_basis > 1
        means = np.stack([np.linspace(self.observation_space_low[i], self.observation_space_high[i], self.no_basis - 1) \
                         for i in range(len(self.observation_space_high))], axis=0)
        tf_means = tf.Variable(means, dtype=tf.float32)
        norm_of_difference = tf.square(tf.norm(X, axis=-1, keep_dims=True)) + (-2. * tf.matmul(X, tf_means)) + tf.square(tf.norm(tf_means, axis=0, keep_dims=True))
        bases = tf.exp(-norm_of_difference / 2. * pow(sigma, 2))
        bases = tf.concat([tf.ones_like(bases[:, 0:1]), bases], axis=-1)
        return bases

    def process(self, X, y):
        X = np.atleast_1d(X)
        y = np.atleast_1d(y)
        if X.ndim == 1:
            X = X[..., np.newaxis]
        if y.ndim == 1:
            y = y[..., np.newaxis]
        return X, y

    def update(self, sess, X, y):
        X, y = self.process(X, y)
        posterior_mu, posterior_sigma = sess.run([self.posterior_mu, self.posterior_sigma], feed_dict={self.X:X, self.y:y})
        sess.run(self.op_pos2prior_assign, feed_dict={self.posterior_mu_in:posterior_mu, self.posterior_sigma_in:posterior_sigma})
        return posterior_mu, posterior_sigma

def plot_sample_lines(mu, sigma, number_of_lines, data_points, number_of_basis, sess, model):
    X, y = data_points
    X = np.atleast_1d(X)
    y = np.atleast_1d(y)
    mu = np.squeeze(mu, axis=-1)

    x = np.linspace(-1., 1., 50)
    xbasis = sess.run(model.X_basis, feed_dict={model.X:x[..., np.newaxis]})

    lines = np.random.multivariate_normal(mu, sigma, number_of_lines)
    for line in lines:
        plt.plot(x, np.matmul(xbasis, line))
    plt.scatter(X, y)
    plt.grid()
    plt.show()

def sinusoid_experiment():
    trainingPoints = 100
    noiseSD = .2
    priorPrecision = 2.
    likelihoodSD = noiseSD

    number_of_basis = 100

    #Generate the training points
    xtrain = np.random.uniform(-1., 1., size=trainingPoints)
    ytrain = np.sin(10.*xtrain) + np.random.normal(loc=0., scale=noiseSD, size=trainingPoints)

    #Plot prior
    priorMean = np.zeros(number_of_basis)
    priorSigma = np.eye(number_of_basis) / priorPrecision

    model = bayesian_model(1, np.array([-1.]), np.array([1.]), number_of_basis)
    iterations = 100
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        '''
        for i in range(iterations):
            mu, sigma = model.update(sess, xtrain[i], ytrain[i])
            plot_sample_lines(mu, sigma, 6, [xtrain[0 : i + 1], ytrain[0 : i + 1]], number_of_basis, sess, model)
        '''
        mu, sigma = model.update(sess, xtrain, ytrain)
        plot_sample_lines(mu, sigma, 6, [xtrain, ytrain], number_of_basis, sess, model)

def get_training_data(training_points):
    import sys
    sys.path.append('..')
    from prototype8.dmlac.real_env_pendulum import get_next
    u = np.random.uniform(-2., 2., training_points)
    thdot = np.random.uniform(-8., 8., training_points)
    th = np.random.uniform(-np.pi, np.pi, training_points)

    costh = []
    sinth = []
    newthdot = []
    x = 0.
    counter = 0

    for i in range(training_points):
        a, b, c = get_next(th[i], thdot[i], u[i])
        costh.append(a)
        sinth.append(b)
        newthdot.append(c)

    return np.stack([np.cos(th), np.sin(th), thdot, u], axis=-1), costh

def pendulum_experiment():
    import sys
    sys.path.append('..')
    from prototype8.dmlac.real_env_pendulum import get_next
    def plot_truth_data():
        u = np.linspace(-2., 2., 10)
        thdot = np.linspace(-8., 8., 20)
        th = np.linspace(-np.pi, np.pi, 10)

        costh = []
        sinth = []
        newthdot = []
        X = []
        x = 0.
        counter = 0
        for i in range(len(u)):
            for j in range(len(thdot)):
                for k in range(len(th)):
                    counter += 1
                    a, b, c = get_next(th[k], thdot[j], u[i])
                    costh.append(a)
                    sinth.append(b)
                    newthdot.append(c)
                    X.append(x)
                    x += .1

        plt.scatter(X, costh)
        #plt.scatter(X, sinth)
        #plt.scatter(X, newthdot)
        plt.grid()
        ##plt.show()

    def plot_model_data(mu, sigma, sess, model):
        number_of_lines = 6
        lines = np.random.multivariate_normal(np.squeeze(mu, axis=-1), sigma, number_of_lines)

        u = np.linspace(-2., 2., 10)
        thdot = np.linspace(-8., 8., 20)
        th = np.linspace(-np.pi, np.pi, 10)

        states = []
        for i in range(len(u)):
            for j in range(len(thdot)):
                for k in range(len(th)):
                    states.append([np.cos(th[k]), np.sin(th[k]), thdot[j], u[i]])
        states_basis = sess.run(model.X_basis, feed_dict={model.X:np.stack(states, axis=0)})

        for line in lines:
            y = np.matmul(states_basis, line)
            plt.scatter(np.arange(len(y)) / 10., y)
        plt.show()

    training_points = 400*3
    noise_sd = .2
    prior_precision = 2.
    likelihood_sd = noise_sd
    no_basis = 20

    xtrain, ytrain = get_training_data(training_points)
    model = bayesian_model(dim=4, observation_space_low=np.array([-1., -1., -8., -2.]),
                           observation_space_high=np.array([1., 1., 8., 2.]), no_basis=no_basis)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        mu, sigma = model.update(sess, xtrain, ytrain)

        plot_truth_data()
        plot_model_data(mu, sigma, sess, model)

if __name__ == '__main__':
    #sinusoid_experiment()
    pendulum_experiment()
