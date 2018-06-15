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

    def train_hyperparameters(self, sess, xtrain, ytrain, iterations, batched=0):
        xtrain, ytrain = process(xtrain, ytrain, self.dim)
        for it in range(iterations):
            if batched > 0:
                idx = np.random.randint(len(xtrain), size=batched)
                try:
                    log_marginal_likelihood, _ = sess.run([self.log_marginal_likelihood, self.opt], feed_dict={self.X:xtrain[idx, ...], self.y:ytrain[idx, ...]})
                except:
                    print 'Cholesky decomposition failed.'
            else:
                log_marginal_likelihood, _ = sess.run([self.log_marginal_likelihood, self.opt], feed_dict={self.X:xtrain, self.y:ytrain})
            print 'iteration:', it, 'loss:', -log_marginal_likelihood

class bayesian_model:
    def __init__(self, dim, observation_space_low, observation_space_high, no_basis,
                 length_scale=.25, signal_sd=1., noise_sd=.2):
        self.dim = dim
        self.observation_space_high = observation_space_high
        self.observation_space_low = observation_space_low
        self.no_basis = no_basis

        self.length_scale_np = length_scale
        self.signal_sd_np = signal_sd
        self.noise_sd_np = noise_sd

        # Assertions.
        np.testing.assert_array_equal(-self.observation_space_low, self.observation_space_high)
        assert len(self.observation_space_high) == self.dim

        # UUID.
        self.uuid = str(uuid.uuid4())

        # Keep track of mu and sigma.
        self.s = 2**-.5
        #self.prior_precision = tf.get_variable(name='prior_precision'+self.uuid, shape=[], dtype=tf.float64, initializer=tf.constant_initializer(self.s))
        self.mu = np.zeros([self.no_basis, 1])
        self.sigma = np.eye(self.no_basis) * (self.s**2)

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

        # Mean and variance of prior.
        self.prior_mu = tf.placeholder(shape=[self.no_basis, 1], dtype=tf.float64)
        self.prior_sigma = tf.placeholder(shape=[self.no_basis, self.no_basis], dtype=tf.float64)

        # Mean and variance of posterior.
        self.posterior_sigma = tf.matrix_inverse(tf.matrix_inverse(self.prior_sigma) + \
                                                 (1. / tf.square(self.noise_sd)) * \
                                                 tf.matmul(tf.transpose(self.X_basis), self.X_basis))

        self.posterior_mu = tf.matmul(tf.matmul(self.posterior_sigma, tf.matrix_inverse(self.prior_sigma)), self.prior_mu) + \
                            (1. / tf.square(self.noise_sd)) * tf.matmul(tf.matmul(self.posterior_sigma, tf.transpose(self.X_basis)), self.y)

        # Loss function for training basis function hyperparameters
        #self.init_log_marginal_likelihood_loss2()
        #self.init_log_marginal_likelihood_loss()

    # Given test points, return the posterior predictive distributions.
    def posterior_predictive_distribution(self, states_actions, _):
        assert states_actions.shape.as_list() == [None, self.dim]

        bases = self.approx_rbf_kern_basis(states_actions)

        posterior_predictive_mu = tf.matmul(bases, self.prior_mu)
        posterior_predictive_sigma = tf.square(self.noise_sd) + tf.reduce_sum(tf.multiply(tf.matmul(bases, self.prior_sigma), bases), axis=-1, keep_dims=True)

        return tf.concat([posterior_predictive_mu, posterior_predictive_sigma], axis=-1)

    # For testing purposes
    def posterior_predictive_distribution2(self, states_actions, idx=0):
        import sys
        sys.path.append('..')
        from prototype8.dmlac.real_env_pendulum import real_env_pendulum_state
        assert states_actions.shape.as_list() == [None, self.dim]
        state_model = real_env_pendulum_state()

        posterior_predictive_mu = state_model.build(states_actions[:, 0:3], states_actions[:, 3:4])[:, idx:idx + 1]
        posterior_predictive_sigma = tf.reduce_sum((states_actions * 0.), axis=-1, keep_dims=True) + 0.

        return tf.concat([posterior_predictive_mu, posterior_predictive_sigma], axis=-1)

    '''
    def squared_exponential_kernel(self, a, b):
        sqdist = tf.reduce_sum(tf.square(a), axis=-1, keep_dims=True) +\
                 -2. * tf.matmul(a, tf.transpose(b)) +\
                 tf.transpose(tf.reduce_sum(tf.square(b), axis=-1, keep_dims=True))
        kernel = tf.square(self.signal_sd) * tf.exp(-.5 * (1. / tf.square(self.length_scale)) * sqdist)
        return kernel
    '''

    '''
    def init_log_marginal_likelihood_loss2(self):
        # Batch size
        self.n = tf.shape(self.X)[0]
        # Get predictive distribution and log marginal likelihood (Algorithm 2.1 in the GP book).
        L = tf.cholesky(self.squared_exponential_kernel(self.X, self.X) +\
                        tf.multiply(tf.square(self.noise_sd), tf.eye(self.n, dtype=tf.float64)))
        alpha = tf.linalg.solve(tf.transpose(L), tf.linalg.solve(L, self.y))
        self.log_marginal_likelihood = -.5 * tf.matmul(tf.transpose(self.y), alpha)[0, 0] +\
                                       -.5 * tf.reduce_sum(tf.log(tf.diag_part(L))) +\
                                       -.5 * tf.cast(self.n, dtype=tf.float64) * np.log(2. * np.pi)
        self.opt = tf.train.AdamOptimizer().minimize(-self.log_marginal_likelihood, var_list=[self.length_scale, self.signal_sd, self.noise_sd])
    '''

    '''
    def train_hyperparameters(self, sess, xtrain, ytrain, iterations, batched=0):
        xtrain, ytrain = self.process(xtrain, ytrain)
        for it in range(iterations):
            if batched > 0:
                idx = np.random.randint(len(xtrain), size=batched)
                log_marginal_likelihood, _ = sess.run([self.log_marginal_likelihood, self.opt], feed_dict={self.X:xtrain[idx, ...], self.y:ytrain[idx, ...]})
            else:
                log_marginal_likelihood, _ = sess.run([self.log_marginal_likelihood, self.opt], feed_dict={self.X:xtrain, self.y:ytrain})
            print 'iteration:', it, 'loss:', -log_marginal_likelihood
        #self.sigma = (sess.run(self.prior_precision) ** 2) * np.eye(self.no_basis)
    '''


    '''
    def init_basis_function_hyperparameters_assignment_ops(self):
        self.length_scale_placeholder = tf.placeholder(shape=[], dtype=tf.float64)
        self.signal_sd_placeholder = tf.placeholder(shape=[], dtype=tf.float64)
        self.noise_sd_placeholder = tf.placeholder(shape=[], dtype=tf.float64)

        self.hyperparameter_assign_op = [self.length_scale.assign(self.length_scale_placeholder),
                                         self.signal_sd.assign(self.signal_sd_placeholder),
                                         self.noise_sd.assign(self.noise_sd_placeholder)]
    '''

    def approx_rbf_kern_basis(self, X):
        try:
            self.rffm
        except:
            self.rffm = km.RandomFourierFeatureMapper(self.dim, self.no_basis, stddev=self.length_scale_np)

        basis_phi = tf.multiply(self.signal_sd, tf.cast(self.rffm.map(tf.cast(X, dtype=tf.float32)), dtype=tf.float64))
        sqrt_sigma_inv = tf.matrix_inverse(self.s * tf.eye(self.no_basis, dtype=tf.float64))
        basis_psi = tf.transpose(tf.matmul(sqrt_sigma_inv, tf.transpose(basis_phi)))

        return basis_psi

    '''
    def init_log_marginal_likelihood_loss(self):
        K = tf.matmul(tf.matmul(self.X_basis, tf.square(self.prior_precision) * tf.eye(self.no_basis, dtype=tf.float64)), tf.transpose(self.X_basis)) + \
            tf.square(self.noise_sd) * tf.eye(tf.shape(self.X_basis)[0], dtype=tf.float64)

        self.log_marginal_likelihood = -.5 * tf.matmul(tf.matmul(tf.transpose(self.y), tf.matrix_inverse(K)), self.y)[0, 0] + \
                                       -.5 * tf.linalg.logdet(K) + \
                                       -.5 * tf.cast(tf.shape(self.X_basis)[0], dtype=tf.float64) * np.log(2. * np.pi)
        self.opt = tf.train.AdamOptimizer().minimize(-self.log_marginal_likelihood, var_list=[self.length_scale, self.signal_sd, self.noise_sd])
        #self.opt = tf.train.GradientDescentOptimizer(.1).minimize(-self.log_marginal_likelihood, var_list=[self.length_scale, self.signal_sd, self.noise_sd])

    # Basis functions using RBFs (to model nonlinear data).
    def rbf_basis_functions(self, X):
        assert self.no_basis > 1

        no_basis = self.no_basis
        no_basis -= 1
        no_basis_original = no_basis
        grid_intervals = int(np.ceil(no_basis ** (1. / len(self.observation_space_low))))
        no_basis = grid_intervals ** len(self.observation_space_low)
        if no_basis != no_basis_original:
            print 'Warning, number of basis is', no_basis

        grid = [np.linspace(self.observation_space_low[i], self.observation_space_high[i], grid_intervals)
                for i in range(len(self.observation_space_low))]
        means = np.meshgrid(*grid)
        means = np.stack([m.flatten() for m in means], axis=-1)
        assert len(means) == no_basis

        tf_means = tf.Variable(means.T, dtype=tf.float64, trainable=False)
        norm_of_difference = tf.square(tf.norm(X, axis=-1, keep_dims=True)) + (-2. * tf.matmul(X, tf_means)) + tf.square(tf.norm(tf_means, axis=0, keep_dims=True))
        bases = tf.square(self.signal_sd) * tf.exp(-norm_of_difference / 2. * tf.square(self.length_scale))
        bases = tf.concat([tf.ones_like(bases[:, 0:1]), bases], axis=-1)
        return bases

    # Basis functions using RBFs (to model nonlinear data).
    def basis_functions2(self, X, sigma=.25):
        assert self.no_basis > 1
        print sigma

        no_basis = self.no_basis
        no_basis -= 1
        no_basis_original = no_basis
        grid_intervals = int(np.ceil(no_basis ** (1. / len(self.observation_space_low))))
        no_basis = grid_intervals ** len(self.observation_space_low)
        if no_basis != no_basis_original:
            print 'Warning, number of basis is', no_basis

        grid = [np.linspace(self.observation_space_low[i], self.observation_space_high[i], grid_intervals)
                for i in range(len(self.observation_space_low))]
        means = np.meshgrid(*grid)
        means = np.stack([m.flatten() for m in means], axis=-1)
        assert len(means) == no_basis

        means = means.T
        norm_of_difference = np.square(np.linalg.norm(X, axis=-1, keepdims=True)) + (-2. * np.matmul(X, means)) + np.square(np.linalg.norm(means, axis=0, keepdims=True))
        bases = np.exp(-norm_of_difference / 2. * pow(sigma, 2))
        bases = np.concatenate([np.ones_like(bases[:, 0:1]), bases], axis=-1)
        return bases

    def basisFunctions(self, xtrain, numberOfBasis=(5**4)+1, low=np.array([-1., -1., -8., -2.]), high=np.array([1., 1., 8., 2.]), sigma=.09):
        xtrain = np.atleast_1d(xtrain)
        if xtrain.ndim == 1:
            xtrain = xtrain[..., np.newaxis]

        assert numberOfBasis > 1
        assert xtrain.shape[-1] == len(low)
        np.testing.assert_array_equal(-low, high)

        numberOfBasis -= 1
        numberOfBasisOriginal = numberOfBasis
        grid_intervals = int(np.ceil(numberOfBasis ** (1. / len(low))))
        numberOfBasis = grid_intervals ** len(low)

        if numberOfBasis != numberOfBasisOriginal:
            print 'Warning, number of basis is', numberOfBasis

        grid = [np.linspace(low[i], high[i], grid_intervals) for i in range(len(low))]
        means = np.meshgrid(*grid)
        means = np.stack([m.flatten() for m in means], axis=-1)
        assert len(means) == numberOfBasis

        means = means.T
        norm_of_difference = np.square(np.linalg.norm(xtrain, axis=-1, keepdims=True)) + (-2. * np.matmul(xtrain, means)) +\
                             np.square(np.linalg.norm(means, axis=0, keepdims=True))
        bases = np.exp(-norm_of_difference / 2. * pow(sigma, 2))
        bases = np.concatenate([np.ones((len(xtrain), 1)), bases], axis=-1)
        return bases
    '''

    def update(self, sess, X, y):
        X, y = process(X, y, self.dim)

        posterior_mu, posterior_sigma = sess.run([self.posterior_mu, self.posterior_sigma], feed_dict={self.X:X, self.y:y, self.prior_mu:self.mu, self.prior_sigma:self.sigma})
        self.mu = np.copy(posterior_mu)
        self.sigma = np.copy(posterior_sigma)
        #print self.mu
        #sess.run(self.op_pos2prior_assign, feed_dict={self.posterior_mu_in:posterior_mu, self.posterior_sigma_in:posterior_sigma})
        return posterior_mu, posterior_sigma

def plot_sample_lines(mu, sigma, number_of_lines, data_points, number_of_basis, sess, model):
    X, y = data_points
    X = np.atleast_1d(X)
    y = np.atleast_1d(y)
    mu = np.squeeze(mu, axis=-1)

    x = np.linspace(-5., 5., 50)
    xbasis = sess.run(model.X_basis, feed_dict={model.X:x[..., np.newaxis]})

    lines = np.random.multivariate_normal(mu, sigma, number_of_lines)
    for line in lines:
        plt.plot(x, np.matmul(xbasis, line))
    plt.scatter(X, y)
    plt.grid()
    plt.show()

def sinusoid_experiment():
    trainingPoints = 10
    noiseSD = .2
    priorPrecision = 2.
    likelihoodSD = noiseSD

    number_of_basis = 100

    #Generate the training points
    xtrain = np.random.uniform(-5., 5., size=trainingPoints)
    #ytrain = np.sin(.9*xtrain) + np.random.normal(loc=0., scale=noiseSD, size=trainingPoints)
    ytrain = np.sin(.9*xtrain) + 0.00005 * np.random.randn(trainingPoints)

    #Plot prior
    priorMean = np.zeros(number_of_basis)
    priorSigma = np.eye(number_of_basis) / priorPrecision

    hs = hyperparameter_search(dim=1)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        hs.train_hyperparameters(sess, xtrain, ytrain, iterations=10000, batched=32)
        hyperparameters = sess.run([hs.length_scale, hs.signal_sd, hs.noise_sd])

    iterations = 100
    model = bayesian_model(1, np.array([-1.]), np.array([1.]), number_of_basis, *hyperparameters)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        mu, sigma = model.update(sess, xtrain, ytrain)
        plot_sample_lines(mu, sigma, 6, [xtrain, ytrain], number_of_basis, sess, model)
        '''
        for i in range(iterations):
            mu, sigma = model.update(sess, xtrain[i], ytrain[i])
            plot_sample_lines(mu, sigma, 6, [xtrain[0 : i + 1], ytrain[0 : i + 1]], number_of_basis, sess, model)
        '''


def get_training_data(training_points):
    import pickle
    data = pickle.load(open("save.p", "rb" ))
    return data[0], data[1]
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

def get_training_data2(training_points):
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

    costh = np.array(costh)
    sinth = np.array(sinth)
    newthdot = np.array(newthdot)
    return np.stack([np.cos(th), np.sin(th), thdot, u], axis=-1), np.stack([costh, sinth, newthdot], axis=-1)

def get_training_data3(training_points):
    import gym
    env = gym.make('Pendulum-v0')
    xtrain = []
    ytrain = []

    state = env.reset()

    while True:
        action = np.random.uniform(-2., 2., 1)
        next_state, reward, done, _ = env.step(action)

        xtrain.append(np.append(state, action))
        ytrain.append(next_state)
        assert len(xtrain) == len(ytrain)

        if len(xtrain) == training_points:
            return np.stack(xtrain, axis=0), np.stack(ytrain, axis=0)

        if done:
            state = env.reset()
        else:
            state = np.copy(next_state)

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
        for i in range(len(u)):
            for j in range(len(thdot)):
                for k in range(len(th)):
                    a, b, c = get_next(th[k], thdot[j], u[i])
                    costh.append(a)
                    sinth.append(b)
                    newthdot.append(c)

        plt.scatter(np.arange(len(costh)) / 10., costh)
        #plt.scatter(np.arange(len(sinth)) / 10., sinth)
        #plt.scatter(np.arange(len(newthdot)) / 10., newthdot)
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
        #states_basis = model.basis_functions2(np.stack(states, axis=0))

        for line in lines:
            y = np.matmul(states_basis, line)
            plt.scatter(np.arange(len(y)) / 10., y)
        plt.show()

    training_points = 200*10
    interval = 200*10
    noise_sd = .2
    prior_precision = 2.
    likelihood_sd = noise_sd
    no_basis = (5**4)+1

    xtrain, ytrain = get_training_data(training_points)
    hs = hyperparameter_search(dim=4)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        hs.train_hyperparameters(sess, xtrain, ytrain, iterations=10000, batched=32)
        hyperparameters = sess.run([hs.length_scale, hs.signal_sd, hs.noise_sd])

    model = bayesian_model(4, np.array([-1., -1., -8., -2.]), np.array([1., 1., 8., 2.]), no_basis, *hyperparameters)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #model.train_hyperparameters(sess, xtrain, ytrain, iterations=10000, batched=32)
        for i in range(0, training_points, interval):
            #print i
            mu, sigma = model.update(sess, xtrain[i:i+interval], ytrain[i:i+interval])
            plot_truth_data()
            plot_model_data(mu, sigma, sess, model)
            print mu

def random_seed_state():
    theta = np.random.uniform(-np.pi, np.pi)
    thetadot = np.random.uniform(-8., 8.)

    return np.array([np.cos(theta), np.sin(theta), thetadot])

def future_state_prediction_experiment():
    import sys
    sys.path.append('..')
    from prototype8.dmlac.real_env_pendulum import get_next_state

    training_points = 200*20
    noise_sd = .2
    prior_precision = 2.
    likelihood_sd = noise_sd
    no_basis = 256

    xtrain, ytrain = get_training_data3(training_points)

    hs0 = hyperparameter_search(dim=4)
    hs1 = hyperparameter_search(dim=4)
    hs2 = hyperparameter_search(dim=4)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        hs0.train_hyperparameters(sess, xtrain, ytrain[:, 0], iterations=2000, batched=32)
        hyperparameters0 = sess.run([hs0.length_scale, hs0.signal_sd, hs0.noise_sd])
        hs1.train_hyperparameters(sess, xtrain, ytrain[:, 1], iterations=2000, batched=32)
        hyperparameters1 = sess.run([hs1.length_scale, hs1.signal_sd, hs1.noise_sd])
        hs2.train_hyperparameters(sess, xtrain, ytrain[:, 2], iterations=2000, batched=32)
        hyperparameters2 = sess.run([hs2.length_scale, hs2.signal_sd, hs2.noise_sd])

    model0 = bayesian_model(4, np.array([-1., -1., -8., -2.]), np.array([1., 1., 8., 2.]), no_basis, *hyperparameters0)
    model1 = bayesian_model(4, np.array([-1., -1., -8., -2.]), np.array([1., 1., 8., 2.]), no_basis, *hyperparameters1)
    model2 = bayesian_model(4, np.array([-1., -1., -8., -2.]), np.array([1., 1., 8., 2.]), no_basis, *hyperparameters2)

    T = 100#Time horizon
    #import pickle
    #seed_state = pickle.load(open("random_state.p", "rb"))
    #policy = pickle.load(open("random_policy.p", "rb"))
    policy = np.random.uniform(-2., 2., T)
    seed_state = random_seed_state()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        #model0.train_hyperparameters(sess, xtrain, ytrain[:, 0], iterations=1000, batched=32)
        #model1.train_hyperparameters(sess, xtrain, ytrain[:, 1], iterations=1000, batched=32)
        #model2.train_hyperparameters(sess, xtrain, ytrain[:, 2], iterations=1000, batched=32)

        mu0, sigma0 = model0.update(sess, xtrain, ytrain[:, 0])
        mu1, sigma1 = model1.update(sess, xtrain, ytrain[:, 1])
        mu2, sigma2 = model2.update(sess, xtrain, ytrain[:, 2])

        # 2) Plot trajectories from model
        lines0 = np.random.multivariate_normal(np.squeeze(mu0, axis=-1), sigma0, 50)
        lines1 = np.random.multivariate_normal(np.squeeze(mu1, axis=-1), sigma1, 50)
        lines2 = np.random.multivariate_normal(np.squeeze(mu2, axis=-1), sigma2, 50)
        for i in range(len(lines0)):
            state = np.copy(seed_state)
            Y = [seed_state[0]]
            for action in policy:
                sa_concat = np.atleast_2d(np.append(state, action)).astype(np.float64)
                #states_basis = model0.basis_functions2(sa_concat)
                states_basis = sess.run(model0.X_basis, feed_dict={model0.X:sa_concat})

                next_states0 = np.matmul(states_basis, lines0[i])
                next_states1 = np.matmul(states_basis, lines1[i])
                next_states2 = np.matmul(states_basis, lines2[i])

                next_state = np.concatenate([next_states0, next_states1, next_states2], axis=0)
                state = np.copy(next_state)
                Y.append(state[0])
            plt.plot(np.arange(len(Y)), Y, color='r')

        # 1) Plot real dynamics
        Y = [seed_state[0]]
        state = np.copy(seed_state)
        for action in policy:
            state = get_next_state(state, action)
            Y.append(state[0, 0])
        plt.plot(np.arange(len(Y)), Y)

        plt.grid()
        plt.show()

def future_state_prediction_experiment_with_every_visit_sampling():
    import sys
    sys.path.append('..')
    from prototype8.dmlac.real_env_pendulum import get_next_state

    training_points = 200*2
    noise_sd = .2
    prior_precision = 2.
    likelihood_sd = noise_sd
    no_basis = 256

    xtrain, ytrain = get_training_data3(training_points)

    hs = [hyperparameter_search(dim=4) for _ in range(3)]
    hyperparameters = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(len(hs)):
            hs[i].train_hyperparameters(sess, xtrain, ytrain[:, i], iterations=50000, batched=32)
            hyperparameters.append(sess.run([hs[i].length_scale, hs[i].signal_sd, hs[i].noise_sd]))


    models = [bayesian_model(4, np.array([-1., -1., -8., -2.]), np.array([1., 1., 8., 2.]), no_basis, *hyperparameters[i]) for i in range(3)]

    states_actions = tf.placeholder(shape=[None, 4], dtype=tf.float64)
    ppd = tf.stack([model.posterior_predictive_distribution(states_actions, None) for model in models], axis=0)

    T = 100#Time horizon
    no_lines = 50*2*2
    policy = np.random.uniform(-2., 2., T)
    seed_state = random_seed_state()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(len(models)):
            models[i].update(sess, xtrain, ytrain[:, i])

        trajectories = []
        for _ in range(no_lines):
            trajectory = []
            state = np.copy(seed_state)
            trajectory.append(state)
            for action in policy:
                sa_concat = np.atleast_2d(np.append(state, action)).astype(np.float64)

                feed_dict = {}
                feed_dict[states_actions] = sa_concat
                for model in models:
                    feed_dict[model.prior_mu] = model.mu
                    feed_dict[model.prior_sigma] = model.sigma

                mu_sigma = np.squeeze(sess.run(ppd, feed_dict=feed_dict), axis=1)
                state = np.random.multivariate_normal(mu_sigma[:, 0], np.diag(mu_sigma[:, 1]))
                trajectory.append(state)

            trajectory = np.stack(trajectory, axis=0)
            trajectories.append(trajectory)

        trajectories = np.stack(trajectories, axis=0)

        state = np.copy(seed_state[np.newaxis, ...])
        Y = [state]
        for action in policy:
            state = get_next_state(state, action)
            Y.append(state)
        Y = np.concatenate(Y, axis=0)

        for i in range(3):
            plt.subplot(1, 3, i + 1)

            for l in range(no_lines):
                plt.plot(np.arange(len(trajectories[l, :, i])), trajectories[l, :, i], color='r')
            plt.plot(np.arange(len(Y[:, i])), Y[:, i])
            plt.grid()
        plt.show()

def regression_experiment():
    import gym

    env = gym.make('Pendulum-v0')

    epochs = 3

    data = []
    for epoch in range(epochs):
        state = env.reset()
        while True:
            action = np.random.uniform(env.action_space.low, env.action_space.high, 1)
            next_state, reward, done, _ = env.step(action)
            data.append([state, action, next_state])
            state = np.copy(next_state)
            if done:
                break

    states = np.stack([d[0] for d in data], axis=0)
    actions = np.stack([d[1] for d in data], axis=0)
    next_states = np.stack([d[2] for d in data], axis=0)

    states_actions = np.concatenate([states, actions], axis=-1)

    x_train = states_actions[:400, ...]
    y_train = next_states[:400, ...]
    x_test = states_actions[400:, ...]
    y_test = next_states[400:, ...]


    hs = [hyperparameter_search(dim=4) for _ in range(3)]
    hyperparameters = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(len(hs)):
            hs[i].train_hyperparameters(sess, x_train, y_train[:, i], iterations=50000, batched=32)
            hyperparameters.append(sess.run([hs[i].length_scale, hs[i].signal_sd, hs[i].noise_sd]))


    no_basis = 256
    models = [bayesian_model(4, np.array([-1., -1., -8., -2.]), np.array([1., 1., 8., 2.]), no_basis, *hyperparameters[i]) for i in range(3)]

    states_actions_placeholder = tf.placeholder(shape=[None, 4], dtype=tf.float64)
    ppd = tf.stack([model.posterior_predictive_distribution(states_actions_placeholder, None) for model in models], axis=0)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(len(models)):
            models[i].update(sess, x_train, y_train[:, i])

        feed_dict = {}
        feed_dict[states_actions_placeholder] = x_test
        for model in models:
            feed_dict[model.prior_mu] = model.mu
            feed_dict[model.prior_sigma] = model.sigma

        mu_sigma = sess.run(ppd, feed_dict=feed_dict)
        #mu_sigma = np.squeeze(sess.run(ppd, feed_dict=feed_dict), axis=1)

    means = mu_sigma[:, :, 0].T
    sds = np.sqrt(mu_sigma[:, :, 1].T)

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.grid()
        plt.plot(np.arange(len(y_test)), y_test[:, i])
        plt.errorbar(np.arange(len(means)), means[:, i], yerr=sds[:, i])
    plt.show()

if __name__ == '__main__':
    #sinusoid_experiment()
    #pendulum_experiment()
    #future_state_prediction_experiment()
    future_state_prediction_experiment_with_every_visit_sampling()
    regression_experiment()
