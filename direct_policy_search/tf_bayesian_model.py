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
        assert len(self.observation_space_high) == self.dim

        # Keep track of mu and sigma
        self.prior_precision = 2.#Assume known beforehand
        self.mu = np.zeros([self.no_basis, 1])
        self.sigma = np.eye(self.no_basis) / self.prior_precision

        # Placeholders
        self.X = tf.placeholder(shape=[None, self.no_basis], dtype=tf.float64)
        self.y = tf.placeholder(shape=[None, 1], dtype=tf.float64)
        #self.X_basis = self.basis_functions(self.X)

        # Mean and variance of prior
        self.prior_mu = tf.placeholder(shape=[self.no_basis, 1], dtype=tf.float64)
        self.prior_sigma = tf.placeholder(shape=[self.no_basis, self.no_basis], dtype=tf.float64)

        self.likelihood_sd = .2#Assume known beforehand

        # Mean and variance of posterior
        self.posterior_sigma = tf.matrix_inverse(tf.matrix_inverse(self.prior_sigma) + \
                                                 pow(self.likelihood_sd, -2) * \
                                                 tf.matmul(tf.transpose(self.X), self.X))

        self.posterior_mu = tf.matmul(tf.matmul(self.posterior_sigma, tf.matrix_inverse(self.prior_sigma)), self.prior_mu) + \
                            pow(self.likelihood_sd, -2) * tf.matmul(tf.matmul(self.posterior_sigma, tf.transpose(self.X)), self.y)

        '''
        # Operation for assigning prior = posterior
        self.posterior_mu_in = tf.placeholder(shape=[self.no_basis, 1], dtype=tf.float64)
        self.posterior_sigma_in = tf.placeholder(shape=[self.no_basis, self.no_basis], dtype=tf.float64)

        self.op_pos2prior_assign = [self.prior_mu.assign(self.posterior_mu_in), self.prior_sigma.assign(self.posterior_sigma_in)]
        '''

    # Basis functions using RBFs (to model nonlinear data)
    def basis_functions(self, X, sigma=.09):
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
        bases = tf.exp(-norm_of_difference / 2. * pow(sigma, 2))
        bases = tf.concat([tf.ones_like(bases[:, 0:1]), bases], axis=-1)
        return bases

    # Basis functions using RBFs (to model nonlinear data)
    def basis_functions2(self, X, sigma=.09):
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

        posterior_mu, posterior_sigma = sess.run([self.posterior_mu, self.posterior_sigma], feed_dict={self.X:self.basis_functions2(X), self.y:y, self.prior_mu:self.mu, self.prior_sigma:self.sigma})
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
    '''
    import pickle
    data = pickle.load(open("save.p", "rb" ))
    return data[0], data[1]
    '''
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
        #states_basis = sess.run(model.X_basis, feed_dict={model.X:np.stack(states, axis=0)})
        states_basis = model.basis_functions2(np.stack(states, axis=0))

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
    model = bayesian_model(dim=4, observation_space_low=np.array([-1., -1., -8., -2.]),
                           observation_space_high=np.array([1., 1., 8., 2.]), no_basis=no_basis)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(0, training_points, interval):
            #print i
            mu, sigma = model.update(sess, xtrain[i:i+interval], ytrain[i:i+interval])
            plot_truth_data()
            plot_model_data(mu, sigma, sess, model)

def random_seed_state():
    theta = np.random.uniform(-np.pi, np.pi)
    thetadot = np.random.uniform(-8., 8.)

    return np.array([np.cos(theta), np.sin(theta), thetadot])

def future_state_prediciton_experiment():
    import sys
    sys.path.append('..')
    from prototype8.dmlac.real_env_pendulum import get_next_state

    training_points = 200*20
    noise_sd = .2
    prior_precision = 2.
    likelihood_sd = noise_sd
    no_basis = (5**4)+1

    xtrain, ytrain = get_training_data3(training_points)
    model0 = bayesian_model(dim=4, observation_space_low=np.array([-1., -1., -8., -2.]),
                            observation_space_high=np.array([1., 1., 8., 2.]), no_basis=no_basis)
    model1 = bayesian_model(dim=4, observation_space_low=np.array([-1., -1., -8., -2.]),
                            observation_space_high=np.array([1., 1., 8., 2.]), no_basis=no_basis)
    model2 = bayesian_model(dim=4, observation_space_low=np.array([-1., -1., -8., -2.]),
                            observation_space_high=np.array([1., 1., 8., 2.]), no_basis=no_basis)

    T = 100#Time horizon
    policy = np.random.uniform(-2., 2., T)
    seed_state = random_seed_state()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
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
                states_basis = model0.basis_functions2(sa_concat)

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

if __name__ == '__main__':
    #sinusoid_experiment()
    pendulum_experiment()
    #future_state_prediciton_experiment()
