#Potential TODO:
#1) A regressor for the reward function.
#2) Propagate the posterior.
#3) Optimize for the prior hyperparamter (tau).
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.distributions as tfd

import gym
import pickle

from tf_bayesian_model import bayesian_model, hyperparameter_search

import sys
sys.path.append('..')
#from prototype8.dmlac.real_env_pendulum import real_env_pendulum_reward
from custom_environments.generateANN_env import ANN

from utils import Memory

class blr_model:
    def __init__(self, x_dim, y_dim, state_dim, action_dim, observation_space_low,
                 observation_space_high, action_bound_low, action_bound_high, unroll_steps,
                 no_samples, no_basis, discount_factor, train_policy_batch_size, train_policy_iterations,
                 hyperparameters, debugging_plot):
        
        assert x_dim == state_dim + action_dim
        assert len(hyperparameters) == y_dim
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.observation_space_low = observation_space_low
        self.observation_space_high = observation_space_high
        self.action_bound_low = action_bound_low
        self.action_bound_high = action_bound_high

        self.unroll_steps = unroll_steps
        self.no_samples = no_samples
        self.no_basis = no_basis
        self.discount_factor = discount_factor

        self.train_policy_batch_size = train_policy_batch_size
        self.train_policy_iterations = train_policy_iterations

        self.hyperparameters = hyperparameters
        self.debugging_plot = debugging_plot

        self.policy_scope = 'policy_scope'
        self.policy_reuse_vars = None

        self.models = [bayesian_model(self.x_dim, self.observation_space_low, self.observation_space_high,
                                      self.action_bound_low, self.action_bound_high, self.no_basis,
                                      *self.hyperparameters[i]) for i in range(self.y_dim)]

        self.states = tf.placeholder(shape=[None, self.state_dim], dtype=tf.float64)
        self.batch_size = tf.shape(self.states)[0]
        #self.batch_size = 3
        self.actions = self.build_policy(self.states)

        self.cum_xx = [tf.tile(tf.expand_dims(model.cum_xx_pl, axis=0),
                       [self.batch_size * self.no_samples, 1, 1]) for model in self.models]
        self.cum_xy = [tf.tile(tf.expand_dims(model.cum_xy_pl, axis=0),
                       [self.batch_size * self.no_samples, 1, 1]) for model in self.models]
        self.unroll(self.states)
        #self.unroll2(self.states)

    #TODO: for debugging purposes
    def unroll2(self, seed_states):
        assert seed_states.shape.as_list() == [None, self.state_dim]
        no_samples = self.no_samples
        unroll_steps = self.unroll_steps
        #self.reward_model = real_env_pendulum_reward()#Use true model.
        self.reward_model = ANN(self.state_dim+self.action_dim, 1)
        self.placeholders_reward = [tf.placeholder(shape=v.shape, dtype=tf.float64)
                                    for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.reward_model.scope)]
        self.assign_ops = [v.assign(pl) for v, pl in zip(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.reward_model.scope),
                           self.placeholders_reward)]

        states = tf.expand_dims(seed_states, axis=1)
        states = tf.tile(states, [1, no_samples, 1])
        states = tf.reshape(states, shape=[-1, self.state_dim])

        costs = []
        self.next_states = []
        for unroll_step in range(unroll_steps):
            actions = self.build_policy(states)

            rewards = (self.discount_factor ** unroll_step) * self.reward_model.build(states, actions)
            rewards = tf.reshape(tf.squeeze(rewards, axis=-1), shape=[-1, no_samples])
            costs.append(-rewards)

            states_actions = tf.concat([states, actions], axis=-1)

            next_states = self.get_next_states2(states_actions)
            self.next_states.append(next_states)
            states = next_states

        costs = tf.stack(costs, axis=-1)
        self.loss = tf.reduce_mean(tf.reduce_sum(tf.reduce_mean(costs, axis=1), axis=-1))
        self.opt = tf.train.AdamOptimizer().minimize(self.loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'policy_scope'))

    #TODO: for debugging purposes
    def get_next_states(self, states_actions):
        self.string = 'unroll2_gns'
        mu, sigma = [tf.concat(e, axis=-1) for e in zip(*[model.posterior_predictive_distribution(states_actions, None) for model in self.models])]
        self.mus1.append(mu)
        self.sigmas1.append(sigma)
        #print mu.shape
        #print sigma.shape
        next_state = tfd.MultivariateNormalDiag(loc=mu, scale_diag=tf.sqrt(sigma)).sample()
        return next_state

    #TODO: for debugging purposes
    def get_next_states2(self, states_actions):
        self.string = 'unroll2_gns2'
        mus = []
        sigmas = []
        for model in self.models:
            mu, sigma = model.mu_sigma(model.cum_xx_pl, model.cum_xy_pl)
            post_pred_mu, post_pred_sigma = model.post_pred2(states_actions, mu, sigma)

            mus.append(post_pred_mu)
            sigmas.append(post_pred_sigma)
        mus = tf.concat(mus, axis=-1)
        sigmas = tf.concat(sigmas, axis=-1)
        self.mus2.append(mus)
        self.sigmas2.append(sigmas)
        #print mus.shape
        #print sigmas.shape
        next_state = tfd.MultivariateNormalDiag(loc=mus, scale_diag=tf.sqrt(sigmas)).sample()
        return next_state

    def unroll(self, seed_states):
        assert seed_states.shape.as_list() == [None, self.state_dim]
        no_samples = self.no_samples
        unroll_steps = self.unroll_steps
        #self.reward_model = real_env_pendulum_reward()#Use true model.
        self.reward_model = ANN(self.state_dim+self.action_dim, 1)
        self.placeholders_reward = [tf.placeholder(shape=v.shape, dtype=tf.float64)
                                    for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.reward_model.scope)]
        self.assign_ops = [v.assign(pl) for v, pl in zip(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.reward_model.scope),
                           self.placeholders_reward)]

        states = tf.expand_dims(seed_states, axis=1)
        states = tf.tile(states, [1, no_samples, 1])
        states = tf.reshape(states, shape=[-1, self.state_dim])

        self.mus0 = []
        self.sigmas0 = []
        self.mus1 = []
        self.sigmas1 = []
        self.mus2 = []
        self.sigmas2 = []

        costs = []
        self.next_states = []
        #ns = []
        #bs = []
        for unroll_step in range(unroll_steps):
            print 'unrolling:', unroll_step
            if self.debugging_plot == True:
                actions = self.build_policy2(states)
            else:
                actions = self.build_policy(states)

            # Reward
            rewards = (self.discount_factor ** unroll_step) * self.reward_model.build(states, actions)
            rewards = tf.reshape(tf.squeeze(rewards, axis=-1), shape=[-1, no_samples])
            costs.append(-rewards)

            states_actions = tf.concat([states, actions], axis=-1)
            mus, sigmas = zip(*[self.mu_sigma(self.cum_xx[y], self.cum_xy[y], self.models[y].s, self.models[y].noise_sd) for y in range(self.y_dim)])

            bases = [model.approx_rbf_kern_basis(states_actions) for model in self.models]
            #bs.append(bases)
            mu_pred, sigma_pred = [tf.concat(e, axis=-1) for e in zip(*[self.prediction(mu, sigma, basis, model.noise_sd)
                                                                      for mu, sigma, basis, model in zip(mus, sigmas, bases, self.models)])]

            self.mus0.append(mu_pred)
            self.sigmas0.append(sigma_pred)
            self.get_next_states(states_actions)
            self.get_next_states2(states_actions)

            next_states = tfd.MultivariateNormalDiag(loc=mu_pred, scale_diag=tf.sqrt(sigma_pred)).sample()
            #ns.append(tf.split(next_states, self.y_dim, axis=-1))

            self.next_states.append(tf.reshape(next_states, shape=[-1, no_samples, self.state_dim]))

            for y in range(self.y_dim):
                self.update_posterior(bases[y], next_states[..., y:y+1], y)

            states = next_states

        if self.debugging_plot == False:
            print 'here1'
            costs = tf.stack(costs, axis=-1)
            print 'here2'
            self.loss = tf.reduce_mean(tf.reduce_sum(tf.reduce_mean(costs, axis=1), axis=-1))
            print 'here3'
            self.opt = tf.train.AdamOptimizer().minimize(self.loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'policy_scope'))
            print 'here4'
        self.string = 'unroll'

    def update_posterior(self, X, y, i):
        X_expanded_dims = tf.expand_dims(X, axis=-1)
        y_expanded_dims = tf.expand_dims(y, axis=-1)
        self.cum_xx[i] += tf.matmul(X_expanded_dims, tf.transpose(X_expanded_dims, perm=[0, 2, 1]))
        self.cum_xy[i] += tf.matmul(X_expanded_dims, y_expanded_dims)

    def prediction(self, mu, sigma, basis, noise_sd):
        basis_expanded_dims = tf.expand_dims(basis, axis=-1)
        mu_pred = tf.matmul(tf.transpose(mu, perm=[0, 2, 1]), basis_expanded_dims)
        sigma_pred = tf.square(noise_sd) + tf.matmul(tf.matmul(tf.transpose(basis_expanded_dims, perm=[0, 2, 1]), sigma), basis_expanded_dims)

        return tf.squeeze(mu_pred, axis=-1), tf.squeeze(sigma_pred, axis=-1)

    def mu_sigma(self, xx, xy, s, noise_sd):
        noise_sd_sq = tf.square(noise_sd)
        prior_sigma_inv = tf.matrix_inverse(tf.tile(tf.expand_dims(s*tf.eye(self.no_basis, dtype=tf.float64), axis=0),
                                            [self.batch_size * self.no_samples, 1, 1]))
        A = tf.matrix_inverse(tf.multiply(noise_sd_sq, prior_sigma_inv) + xx)
        sigma = tf.multiply(noise_sd_sq, A)
        # Assuming that prior mean is zero vector
        mu = tf.matmul(A, xy)
        return mu, sigma

    def mu_sigma2(self, xx, xy, s, noise_sd, bs, ns, idx):
        if bs and ns:
            assert len(zip(*bs)) == self.y_dim
            assert len(zip(*ns)) == self.y_dim
            X = zip(*bs)[idx]
            y = zip(*ns)[idx]

            X = tf.expand_dims(tf.stack(X, axis=0), axis=-1)
            XX = tf.matmul(X, tf.transpose(X, perm=[0, 1, 3, 2]))

            y = tf.expand_dims(tf.stack(y, axis=0), axis=-1)
            Xy = tf.matmul(X, y)

            XX_ = tf.reduce_sum(XX, axis=0)
            Xy_ = tf.reduce_sum(Xy, axis=0)

        else:
            XX_ = 0.
            Xy_ = 0.

        noise_sd_sq = tf.square(noise_sd)
        prior_sigma_inv = tf.matrix_inverse(tf.tile(tf.expand_dims(s*tf.eye(self.no_basis, dtype=tf.float64), axis=0),
                                            [self.batch_size * self.no_samples, 1, 1]))
        A = tf.matrix_inverse(tf.multiply(noise_sd_sq, prior_sigma_inv) + xx + XX_)
        sigma = tf.multiply(noise_sd_sq, A)
        # Assuming that prior mean is zero vector
        mu = tf.matmul(A, xy + Xy_)
        return mu, sigma

    def update(self, sess, X=None, y=None, memory=None):
        if memory is not None:
            states = np.stack([e[0] for e in memory], axis=0)
            actions = np.stack([e[1] for e in memory], axis=0)
            y = np.stack([e[3] for e in memory], axis=0)
            X = np.concatenate([states, actions], axis=-1)

        for i in range(self.y_dim):
            self.models[i].update(sess, X, y[..., i])

    def act(self, sess, state):
        state = np.atleast_2d(state)
        action = sess.run(self.actions, feed_dict={self.states:state})
        return action[0]

    def train(self, sess, memory):
        feed_dict = {}
        #TODO: for debugging purposes
        if self.string == 'unroll':
            for model in self.models:
                feed_dict[model.cum_xx_pl] = model.cum_xx
                feed_dict[model.cum_xy_pl] = model.cum_xy
                feed_dict[model.mu_placeholder] = model.mu#for testing
                feed_dict[model.sigma_placeholder] = model.sigma#for testing
                feed_dict[model.sigma_prior_pl] = model.sigma_prior#for testing
                feed_dict[model.mu_prior_pl] = model.mu_prior#for testing
        elif self.string == 'unroll2_gns':
            for model in self.models:
                feed_dict[model.mu_placeholder] = model.mu
                feed_dict[model.sigma_placeholder] = model.sigma
        elif self.string == 'unroll2_gns2':
            for model in self.models:
                feed_dict[model.cum_xx_pl] = model.cum_xx
                feed_dict[model.cum_xy_pl] = model.cum_xy
                feed_dict[model.sigma_prior_pl] = model.sigma_prior
                feed_dict[model.mu_prior_pl] = model.mu_prior

        for it in range(self.train_policy_iterations):
            batch = memory.sample(self.train_policy_batch_size)
            states = np.stack([b[0] for b in batch], axis=0)
            feed_dict[self.states] = states

            mus0, sigmas0, mus1, sigmas1, mus2, sigmas2, next_states, loss, _ = sess.run([self.mus0, self.sigmas0, self.mus1, self.sigmas1, self.mus2, self.sigmas2, self.next_states, self.loss, self.opt], feed_dict=feed_dict)
            if loss > 1000.:
                print next_states
            '''
            assert len(mus0) == len(sigmas0)
            assert len(mus0) == len(mus1)
            assert len(mus0) == len(sigmas1)
            assert len(mus0) == len(mus2)
            assert len(mus0) == len(sigmas2)
            '''
            '''
            for mu0, sigma0, mu1, sigma1, mu2, sigma2, ii in zip(mus0, sigmas0, mus1, sigmas1, mus2, sigmas2, range(len(mus0))):
                try:
                    np.testing.assert_almost_equal(sigma1, sigma2, decimal=4)
                except:
                    print ii, 'here0'
                    for i in range(len(sigma1)):
                        for j in range(len(sigma1[i])):
                            print sigma1[i, j], sigma2[i, j]
                    exit()
                try:
                    np.testing.assert_almost_equal(mu1, mu2, decimal=4)
                except:
                    print ii, 'here3',
                    for i in range(len(mu1)):
                        print mu1[i], mu2[i]
                    exit()
                try:
                    np.testing.assert_almost_equal(mu0, mu1, decimal=4)
                except:
                    print ii, 'here1',
                    for i in range(len(mu0)):
                        print mu0[i], mu1[i]
                    exit()
                try:
                    np.testing.assert_almost_equal(mu0, mu2, decimal=4)
                except:
                    print ii, 'here2',
                    for i in range(len(m0)):
                        print m0[i], m2[i]
                    exit()
                try:
                    np.testing.assert_almost_equal(sigma0, sigma1, decimal=4)
                except:
                    print ii, 'here4',
                    for i in range(len(sigma0)):
                        for j in range(len(sigma0[i])):
                            print sigma0[i, j], sigma1[i, j]
                    exit()
                try:
                    np.testing.assert_almost_equal(sigma0, sigma2, decimal=4)
                except:
                    print ii, 'here5',
                    for i in range(len(sigma0)):
                        for j in range(len(sigma0[i])):
                            print sigma0[i, j], sigma2[i, j]
                    exit()
            '''
            print 'iteration:', it, 'loss:', loss, self.string, len(mus0)
            '''
            try:
                mus0, sigmas0, mus1, sigmas1, mus2, sigmas2, next_states, loss, _ = sess.run([self.mus0, self.sigmas0, self.mus1, self.sigmas1, self.mus2, self.sigmas2, self.next_states, self.loss, self.opt], feed_dict=feed_dict)
                assert len(mus0) == len(sigmas0)
                assert len(mus0) == len(mus1)
                assert len(mus0) == len(sigmas1)
                assert len(mus0) == len(mus2)
                assert len(mus0) == len(sigmas2)
                for mu0, sigma0, mu1, sigma1, mu2, sigma2 in zip(mus0, sigmas0, mus1, sigmas1, mus2, sigmas2):
                    np.testing.assert_almost_equal(mu0, mu1)
                    np.testing.assert_almost_equal(mu0, mu2)
                    np.testing.assert_almost_equal(mu1, mu2)
                    np.testing.assert_almost_equal(sigma0, sigma1)
                    np.testing.assert_almost_equal(sigma0, sigma2)
                    np.testing.assert_almost_equal(sigma1, sigma2)
                if loss > 1000.:
                    print next_states
                print 'iteration:', it, 'loss:', loss, self.string
            except:
                print 'training step failed.'
            '''

    def build_policy(self, states):
        assert states.shape.as_list() == [None, self.state_dim]

        #Fully connected layer 1
        fc1 = slim.fully_connected(states, 256, activation_fn=tf.nn.relu, scope=self.policy_scope+'/fc1', reuse=self.policy_reuse_vars)

        #Fully connected layer 2
        fc2 = slim.fully_connected(fc1, 256, activation_fn=tf.nn.relu, scope=self.policy_scope+'/fc2', reuse=self.policy_reuse_vars)

        #Output layer
        output = slim.fully_connected(fc2, self.action_dim, activation_fn=tf.nn.tanh, scope=self.policy_scope+'/output', reuse=self.policy_reuse_vars)

        #Apply action bounds
        np.testing.assert_array_equal(-self.action_bound_low, self.action_bound_high)
        action_bound = tf.constant(self.action_bound_high, dtype=tf.float64)
        policy = tf.multiply(output, action_bound)

        #Change flag
        self.policy_reuse_vars = True

        return policy

    def build_policy2(self, states):
        try:
            self.policy
        except:
            self.idx = 0
            self.policy = tf.placeholder(shape=[self.unroll_steps, 1], dtype=tf.float64)

        action = self.policy[self.idx:self.idx+1, ...]
        tile_size = tf.shape(states)[0]

        action_tiled = tf.tile(action, [tile_size, 1])
        self.idx += 1

        return action_tiled

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default='Pendulum-v0')
    parser.add_argument("--unroll-steps", type=int, default=25)
    parser.add_argument("--no-samples", type=int, default=20)
    parser.add_argument("--no-basis", type=int, default=256)
    parser.add_argument("--discount-factor", type=float, default=.9)
    parser.add_argument("--replay-mem-size", type=int, default=1000000)
    parser.add_argument("--train-policy-batch-size", type=int, default=32)
    parser.add_argument("--train-policy-iterations", type=int, default=30)
    parser.add_argument("--replay-start-size-epochs", type=int, default=2)
    parser.add_argument("--train-hyperparameters-iterations", type=int, default=50000)
    parser.add_argument("--goal-position", type=float, default=.45)
    args = parser.parse_args()
    
    print args

    #env = gym.make(args.env, goal_position=args.goal_position)
    env = gym.make(args.env)
    env.seed(seed=args.goal_position)

    # Gather data to train hyperparameters
    data = []
    rewards = []
    dones = []
    for _ in range(2):
        state = env.reset()
        while True:
            action = np.random.uniform(env.action_space.low, env.action_space.high, 1)
            next_state, reward, done, _ = env.step(action)
            data.append([state, action, next_state])
            rewards.append(reward)
            dones.append(done)
            state = np.copy(next_state)
            if done:
                break

    states, actions, next_states = [np.stack(d, axis=0) for d in zip(*data)]

    permutation = np.random.permutation(len(data))
    states_actions = np.concatenate([states, actions], axis=-1)[permutation]
    next_states = next_states[permutation]

    # Train the hyperparameters
    hs = [hyperparameter_search(dim=env.observation_space.shape[0]+env.action_space.shape[0])
          for _ in range(env.observation_space.shape[0])]
    hyperparameters = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        batch_size = 32
        iterations = args.train_hyperparameters_iterations
        #idxs = [np.random.randint(len(states_actions), size=batch_size) for _ in range(iterations)]
        for i in range(len(hs)):
            hs[i].train_hyperparameters(sess, states_actions, next_states[:, i], iterations, batch_size)
            hyperparameters.append(sess.run([hs[i].length_scale, hs[i].signal_sd, hs[i].noise_sd]))

    blr = blr_model(x_dim=env.observation_space.shape[0]+env.action_space.shape[0],
                    y_dim=env.observation_space.shape[0],
                    state_dim=env.observation_space.shape[0],
                    action_dim=env.action_space.shape[0],
                    observation_space_low=env.observation_space.low,
                    observation_space_high=env.observation_space.high,
                    action_bound_low=env.action_space.low,
                    action_bound_high=env.action_space.high,
                    unroll_steps=args.unroll_steps,
                    no_samples=args.no_samples,
                    no_basis=args.no_basis,
                    discount_factor=args.discount_factor,
                    train_policy_batch_size=args.train_policy_batch_size,
                    train_policy_iterations=args.train_policy_iterations,
                    hyperparameters=hyperparameters,
                    debugging_plot=False)

    # Initialize the memory
    memory = Memory(args.replay_mem_size)
    assert len(data) == len(rewards)
    assert len(data) == len(dones)
    for dat, reward, done in zip(data, rewards, dones):
        memory.add([dat[0], dat[1], reward, dat[2], done])
    memory2 = []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        weights = pickle.load(open('../custom_environments/weights/mountain_car_continuous_reward'+str(args.goal_position)+'.p', 'rb'))
        sess.run(blr.assign_ops, feed_dict=dict(zip(blr.placeholders_reward, weights)))
        # Update the model with data used from training hyperparameters
        blr.update(sess, states_actions, next_states)
        blr.train(sess, memory)
        state = env.reset()
        total_rewards = 0.0
        epoch = 1
        for time_steps in range(30000):
            action = blr.act(sess, state)
            next_state, reward, done, _ = env.step(action)
            total_rewards += float(reward)

            # Append to the batch
            memory.add([state, action, reward, next_state, done])
            memory2.append([state, action, reward, next_state, done])

            # s <- s'
            state = np.copy(next_state)

            if done == True:
                print 'time steps', time_steps, 'epoch', epoch, 'total rewards', total_rewards

                # Update the memory
                blr.update(sess, memory=memory2)

                # Train the policy
                blr.train(sess, memory)

                epoch += 1
                total_rewards = 0.
                state = env.reset()
                memory2 = []

def plotting_experiment():
    import matplotlib.pyplot as plt
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--unroll-steps", type=int, default=25)
    parser.add_argument("--no-samples", type=int, default=20)
    parser.add_argument("--no-basis", type=int, default=256)
    parser.add_argument("--discount-factor", type=float, default=.9)
    parser.add_argument("--replay-mem-size", type=int, default=1000000)
    parser.add_argument("--train-policy-batch-size", type=int, default=32)
    parser.add_argument("--train-policy-iterations", type=int, default=30)
    parser.add_argument("--replay-start-size-epochs", type=int, default=2)
    args = parser.parse_args()
    
    print args

    #env = gym.make('Pendulum-v0')
    env = gym.make('MountainCarContinuous-v0')

    epochs = 3
    train_size = (epochs - 1) * 200
    policy = []

    # Gather data to train hyperparameters
    data = []
    for _ in range(epochs):
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

    blr = blr_model(x_dim=env.observation_space.shape[0]+env.action_space.shape[0],
                    y_dim=env.observation_space.shape[0],
                    state_dim=env.observation_space.shape[0],
                    action_dim=env.action_space.shape[0],
                    observation_space_low=env.observation_space.low,
                    observation_space_high=env.observation_space.high,
                    action_bound_low=env.action_space.low,
                    action_bound_high=env.action_space.high,
                    unroll_steps=args.unroll_steps,
                    no_samples=args.no_samples,
                    no_basis=args.no_basis,
                    discount_factor=args.discount_factor,
                    train_policy_batch_size=args.train_policy_batch_size,
                    train_policy_iterations=args.train_policy_iterations,
                    hyperparameters=hyperparameters,
                    debugging_plot=True)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # Update the model with data used from training hyperparameters
        blr.update(sess, states_actions, next_states)

        # Plotting experiment
        policy = actions[-200:-200+blr.unroll_steps, ...]
        seed_state = x_test[0, :3]

        feed_dict = {}
        for model in blr.models:
            feed_dict[model.cum_xx_pl] = model.cum_xx
            feed_dict[model.cum_xy_pl] = model.cum_xy
        feed_dict[blr.states] = seed_state[np.newaxis, ...]
        feed_dict[blr.policy] = policy

        next_states = sess.run(blr.next_states, feed_dict=feed_dict)

        seed_state = seed_state[np.newaxis, ...][np.newaxis, ...]
        seed_state = np.tile(seed_state, [1, blr.no_samples, 1])

        next_states = np.concatenate(next_states, axis=0)
        next_states = np.concatenate([seed_state, next_states], axis=0)

        for i in range(3):
            plt.subplot(1, 3, i+1)
            for j in range(blr.no_samples):
                print next_states[:, j, i]
                plt.plot(np.arange(len(next_states[:, j, i])), next_states[:, j, i], color='r')
            plt.plot(np.arange(len(x_test[:1+blr.unroll_steps, i])), x_test[:1+blr.unroll_steps, i])
            plt.grid()

        plt.show()

if __name__ == '__main__':
    main()
    #plotting_experiment()
