import numpy as np
import scipy
from scipy.optimize import minimize
import argparse

import sys
sys.path.append('..')

import tensorflow as tf
from custom_environments.generateANN_env import ANN
#from custom_environments.environment_state_functions import mountain_car_continuous_state_function
#from custom_environments.environment_reward_functions import mountain_car_continuous_reward_function
from prototype8.dmlac.real_env_pendulum import real_env_pendulum_reward

from utils import gather_data
from other_utils import RegressionWrapper, _basis

import gym
import pybullet_envs
import pickle
import warnings

from choldate import cholupdate

class RegressionWrapperReward(RegressionWrapper):
    def __init__(self, environment, input_dim, basis_dim, length_scale=1., signal_sd=1., noise_sd=5e-4, prior_sd=1., rffm_seed=1, train_hp_iterations=2000, matern_param=np.inf):
        #self.input_dim2 = input_dim
        #self.feature_dim0 = 512
        #self.feature_dim1 = 512
        #self.environment = environment
        #self.random_projection_matrix0 = np.random.normal(loc=0., scale=1./np.sqrt(self.feature_dim0), size=[self.input_dim2 + 1, self.feature_dim0])
        #self.random_projection_matrix1 = np.random.normal(loc=0., scale=1./np.sqrt(self.feature_dim1), size=[self.feature_dim0 + 1, self.feature_dim1])
        #RegressionWrapper.__init__(self, self.feature_dim1, basis_dim, length_scale, signal_sd, noise_sd, prior_sd, rffm_seed, train_hp_iterations, matern_param)

        self.environment = environment
        RegressionWrapper.__init__(self, input_dim, basis_dim, length_scale, signal_sd, noise_sd, prior_sd, rffm_seed, train_hp_iterations, matern_param)

    def _train_hyperparameters(self, X, y):
        if self.environment == 'MountainCarContinuous-v0':
            '''
            self.length_scale = 1.
            self.signal_sd = 10.
            self.noise_sd = 1.
            self.prior_sd = 1000.
            self.hyperparameters = np.array([self.length_scale, self.signal_sd, self.noise_sd, self.prior_sd])
            '''
            self.length_scale = 1.
            self.signal_sd = 2.3
            self.noise_sd = 1.
            self.prior_sd = 500.
            self.hyperparameters = np.array([self.length_scale, self.signal_sd, self.noise_sd, self.prior_sd])

        else:
            RegressionWrapper._train_hyperparameters(self, X, y)

    '''
    def _train_hyperparameters(self, X, y):
        X_features = np.matmul(add_bias(np.matmul(add_bias(X), self.random_projection_matrix0)), self.random_projection_matrix1)
        RegressionWrapper._train_hyperparameters(self, X_features, y)
    '''

    '''
    def _reset_statistics(self, X, y):
        X_features = np.matmul(add_bias(np.matmul(add_bias(X), self.random_projection_matrix0)), self.random_projection_matrix1)
        RegressionWrapper._reset_statistics(self, X_features, y)

    def _predict(self, X):
        X_features = np.matmul(add_bias(np.matmul(add_bias(X), self.random_projection_matrix0)), self.random_projection_matrix1)
        return RegressionWrapper._predict(self, X_features)
    '''

class Agent:
    def __init__(self, environment, x_dim, y_dim, state_dim, action_dim, observation_space_low, observation_space_high,
                 action_space_low, action_space_high, unroll_steps, no_samples, discount_factor, random_matrices, biases, basis_dims,
                 hidden_dim=32, learn_reward=0, use_mean_reward=0, update_hyperstate=1, policy_use_hyperstate=1, learn_diff=0):
        #assert environment in ['Pendulum-v0', 'MountainCarContinuous-v0']
        assert x_dim == state_dim + action_dim
        assert len(action_space_low.shape) == 1
        np.testing.assert_equal(-action_space_low, action_space_high)
        self.environment = environment
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.observation_space_low = observation_space_low
        self.observation_space_high = observation_space_high
        self.action_space_low = action_space_low
        self.action_space_high = action_space_high

        self.unroll_steps = unroll_steps
        self.no_samples = no_samples
        self.discount_factor = discount_factor
        self.random_matrices = random_matrices
        self.biases = biases
        self.basis_dims = basis_dims
        self.hidden_dim = hidden_dim
        self.learn_reward = learn_reward
        self.use_mean_reward = use_mean_reward
        self.update_hyperstate = update_hyperstate
        self.policy_use_hyperstate = policy_use_hyperstate
        self.learn_diff = learn_diff

        if self.environment == 'Pendulum-v0' and self.learn_reward == 0:
            #self.reward_function = real_env_pendulum_reward()
            self.reward_function = ANN(self.state_dim+self.action_dim, 1)
            self.placeholders_reward = [tf.placeholder(shape=v.shape, dtype=tf.float64)
                                        for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.reward_function.scope)]
            self.assign_ops0 = [v.assign(pl) for v, pl in zip(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.reward_function.scope),
                                self.placeholders_reward)]
        elif self.environment == 'MountainCarContinuous-v0' and self.learn_reward == 0:
            self.reward_function = mountain_car_continuous_reward_function()

        #self.hyperstate_dim = sum([(basis_dim*(basis_dim+1))/2 + basis_dim for basis_dim in self.basis_dims])
        self.hyperstate_dim = sum([basis_dim*(basis_dim+1) for basis_dim in self.basis_dims])

        self.random_projection_matrix = np.random.normal(loc=0., scale=1./np.sqrt(self.state_dim), size=[self.hyperstate_dim, self.state_dim])

        input_dim = self.state_dim
        if self.policy_use_hyperstate == 1:
            input_dim *= 2

        self.w1 = np.concatenate([np.random.normal(size=[input_dim, self.hidden_dim]), np.random.uniform(-3e-3, 3e-3, size=[1, self.hidden_dim])], axis=0)
        self.w2 = np.concatenate([np.random.normal(size=[self.hidden_dim, self.hidden_dim]), np.random.uniform(-3e-3, 3e-3, size=[1, self.hidden_dim])], axis=0)
        self.w3 = np.concatenate([np.random.normal(size=[self.hidden_dim, self.action_dim]), np.random.uniform(-3e-3, 3e-3, size=[1, self.action_dim])], axis=0)

        self.thetas = self._pack([self.w1, self.w2, self.w3])

        self.sizes = [[input_dim + 1, self.hidden_dim],
                      [self.hidden_dim + 1, self.hidden_dim],
                      [self.hidden_dim + 1, self.action_dim]]

        w1, w2, w3 = self._unpack(self.thetas, self.sizes)
        np.testing.assert_equal(w1, self.w1)
        np.testing.assert_equal(w2, self.w2)
        np.testing.assert_equal(w3, self.w3)

    def _pack(self, thetas):
        return np.concatenate([theta.flatten() for theta in thetas])

    def _unpack(self, thetas, sizes):
        sidx = 0
        weights = []
        for size in sizes:
            i, j = size
            w = thetas[sidx:sidx+i*j].reshape([i, j])
            sidx += i*j
            weights.append(w)
        return weights

    def _forward(self, thetas, X, hyperstate):
        #"Old" method of including hyperstate into policy network.
        '''
        w0, w1, w2, w3 = self._unpack(thetas, self.sizes)
        XXtr, Xytr = hyperstate

        A = [xx + noise for xx, noise in zip(XXtr, self.noises)]
        wn = [solve(a, xy) for a, xy in zip(A, Xytr)]

        indices = [np.triu_indices(basis_dim, 1) for basis_dim in self.basis_dims]
        hyperstate = []
        for i in range(len(X)):
            tmp0 = []
            for j in range(len(A)):
                A[j][i][indices[j]] = np.nan
                tmp1 = A[j][i]
                tmp0.append(tmp1[~np.isnan(tmp1)])
                tmp0.append(np.squeeze(wn[j][i]))
            tmp0 = np.concatenate(tmp0)
            hyperstate.append(tmp0)
        hyperstate = np.stack(hyperstate, axis=0)

        hyperstate = self._add_bias(hyperstate)
        hyperstate_embedding = np.tanh(np.matmul(hyperstate, w0))
        '''

        w1, w2, w3 = self._unpack(thetas, self.sizes)

        #Perform a simple random projection on the hyperstate.
        if self.policy_use_hyperstate == 1:
            hyperstate = np.concatenate([np.concatenate([np.reshape(XXtr, [len(XXtr), -1]), np.reshape(Xytr, [len(Xytr), -1])], axis=-1) for XXtr, Xytr in zip(*hyperstate)], axis=-1)
            hyperstate = np.tanh(hyperstate/50000.)
            hyperstate_embedding = np.matmul(hyperstate, self.random_projection_matrix)
            hyperstate_embedding = np.tanh(hyperstate_embedding)

            state_hyperstate = np.concatenate([X, hyperstate_embedding], axis=-1)
            policy_net_input = self._add_bias(state_hyperstate)
        else:
            policy_net_input = self._add_bias(X)

        h1 = np.tanh(np.matmul(policy_net_input, w1))
        h1 = self._add_bias(h1)

        h2 = np.tanh(np.matmul(h1, w2))
        h2 = self._add_bias(h2)

        out = np.tanh(np.matmul(h2, w3))
        out = out * self.action_space_high#action bounds.

        return out

    def _add_bias(self, X):
        assert len(X.shape) == 2
        return np.concatenate([X, np.ones([len(X), 1])], axis=-1)

    def _relu(self, X):
        return np.maximum(X, 0.)

    def _fit(self, cma_maxiter, X, XXtr, Xytr, hyperparameters, sess):
        warnings.filterwarnings('ignore', message='.*scipy.linalg.solve\nIll-conditioned matrix detected. Result is not guaranteed to be accurate.\nReciprocal.*')
        assert len(XXtr) == self.state_dim + self.learn_reward
        assert len(Xytr) == self.state_dim + self.learn_reward
        assert len(hyperparameters) == self.state_dim + self.learn_reward

        if self.use_mean_reward == 1: print('Warning: use_mean_reward is set to True but this flag is not used by this function.')

        X = np.copy(X)
        XXtr = [np.copy(ele) for ele in XXtr]
        Xytr = [np.copy(ele) for ele in Xytr]
        hyperparameters = [np.copy(ele) for ele in hyperparameters]

        X = np.expand_dims(X, axis=1)
        X = np.tile(X, [1, self.no_samples, 1])
        X = np.reshape(X, [-1, self.state_dim])

        Llowers = [scipy.linalg.cholesky((hp[-2]/hp[-1])**2*np.eye(basis_dim) + XX, lower=True) for hp, basis_dim, XX in zip(hyperparameters, self.basis_dims, XXtr)]
        Llowers = [np.tile(ele[np.newaxis, ...], [len(X), 1, 1]) for ele in Llowers]
        XXtr = [np.tile(ele[np.newaxis, ...], [len(X), 1, 1]) for ele in XXtr]
        Xytr = [np.tile(ele[np.newaxis, ...], [len(X), 1, 1]) for ele in Xytr]

        self.noises = [(hp[2]/hp[3])**2*np.eye(basis_dim) for hp, basis_dim in zip(hyperparameters, self.basis_dims)]

        import cma
        options = {'maxiter': cma_maxiter, 'verb_disp': 1, 'verb_log': 0}
        print('Before calling cma.fmin')
        res = cma.fmin(self._loss, self.thetas, 2., args=(np.copy(X), [np.copy(ele) for ele in Llowers], [np.copy(ele) for ele in XXtr], [np.copy(ele) for ele in Xytr], None, [np.copy(ele) for ele in hyperparameters], sess), options=options)
        self.thetas = np.copy(res[0])

    def _predict(self, Llower, Xytr, basis, noise_sd):
        '''
        Llower = Llower[0]
        Xytr = Xytr[0]
        basis = np.squeeze(basis, axis=1)
        LinvXT = scipy.linalg.solve_triangular(Llower, basis.T, lower=True)
        pred_sigma = np.sum(np.square(LinvXT), axis=0)*noise_sd**2+noise_sd**2
        pred_sigma = pred_sigma[..., np.newaxis]
        tmp0 = scipy.linalg.solve_triangular(Llower, basis.T, lower=True).T
        tmp1 = scipy.linalg.solve_triangular(Llower, Xytr, lower=True)
        pred_mu = np.matmul(tmp0, tmp1)
        return pred_mu, pred_sigma
        '''

        #TODO:fix this.
        LinvXT = solve_triangular(Llower, np.transpose(basis, [0, 2, 1]))
        pred_sigma = np.sum(np.square(LinvXT), axis=1)*noise_sd**2+noise_sd**2
        tmp0 = np.transpose(solve_triangular(Llower, np.transpose(basis, [0, 2, 1])), [0, 2, 1])
        tmp1 = solve_triangular(Llower, Xytr)
        pred_mu = np.matmul(tmp0, tmp1)
        pred_mu = np.squeeze(pred_mu, axis=-1)
        return pred_mu, pred_sigma

    def _loss(self, thetas, X, Llowers, XXtr, Xytr, A=[], hyperparameters=None, sess=None):
        rng_state = np.random.get_state()
        X = np.copy(X)
        Llowers = [np.copy(ele) for ele in Llowers]
        XXtr = [np.copy(ele) for ele in XXtr]
        Xytr = [np.copy(ele) for ele in Xytr]
        hyperparameters = [np.copy(ele) for ele in hyperparameters]
        try:
            np.random.seed(2)

            rewards = []
            state = X
            for unroll_step in xrange(self.unroll_steps):
                action = self._forward(thetas, state, hyperstate=[Llowers, Xytr])
                reward, basis_reward = self._reward(state, action, sess, Llowers[-1], Xytr[-1], hyperparameters[-1])
                rewards.append((self.discount_factor**unroll_step)*reward)
                state_action = np.concatenate([state, action], axis=-1)

                means = []
                covs = []
                bases = []
                for i in xrange(self.state_dim):
                    length_scale, signal_sd, noise_sd, prior_sd = hyperparameters[i]
                    basis = _basis(state_action, self.random_matrices[i], self.biases[i], self.basis_dims[i], length_scale, signal_sd)
                    basis = np.expand_dims(basis, axis=1)
                    bases.append(basis)
                    pred_mu, pred_sigma = self._predict(Llowers[i], Xytr[i], basis, noise_sd)
                    means.append(pred_mu)
                    covs.append(pred_sigma)
                means = np.concatenate(means, axis=-1)
                covs = np.concatenate(covs, axis=-1)

                bases.append(basis_reward)

                state_ = np.stack([np.random.multivariate_normal(mean=mean, cov=np.diag(cov)) for mean, cov in zip(means, covs)], axis=0)
                state = state + state_ if self.learn_diff else state_
                if self.learn_diff == 0: state_ = np.clip(state_, self.observation_space_low, self.observation_space_high)
                state = np.clip(state, self.observation_space_low, self.observation_space_high)

#                #Removable
#                import copy
#                Llowers2 = copy.deepcopy(Llowers)
#                Xytr2 = copy.deepcopy(Xytr)
#                XXtr2 = copy.deepcopy(XXtr)
#                #Removable -END-

                if self.update_hyperstate == 1 or self.policy_use_hyperstate == 1:
                    y = np.concatenate([state_, reward], axis=-1)[..., :self.state_dim + self.learn_reward]
                    y = y[..., np.newaxis, np.newaxis]
                    for i in xrange(self.state_dim + self.learn_reward):
                        Llowers[i] = Llowers[i].transpose([0, 2, 1])
                    for i in xrange(self.state_dim + self.learn_reward):
                        for j in xrange(len(Llowers[i])):
                            cholupdate(Llowers[i][j], bases[i][j, 0].copy())
                        Xytr[i] += np.matmul(bases[i].transpose([0, 2, 1]), y[:, i, ...])

#                        #Removable
#                        _, _, noise_sd, prior_sd = hyperparameters[i]
#                        XXtr2[i], Xytr2[i], Llowers2[i] = self._update_hyperstate(XXtr2[i], XXtr2[i] + np.matmul(np.transpose(bases[i], [0, 2, 1]), bases[i]), Xytr2[i], Xytr2[i] + np.matmul(np.transpose(bases[i], [0, 2, 1]), y[:, i, ...]), Llowers2[i], (noise_sd/prior_sd)**2)
#                        print i
#                        print np.allclose(Llowers[i], Llowers2[i].transpose([0, 2, 1]))
#                        print np.allclose(Xytr[i], Xytr2[i])
#                        #Removable -END-

                    for i in xrange(self.state_dim + self.learn_reward):
                        Llowers[i] = Llowers[i].transpose([0, 2, 1])

            rewards = np.concatenate(rewards, axis=-1)
            rewards = np.sum(rewards, axis=-1)
            loss = -np.mean(rewards)
            np.random.set_state(rng_state)
            return loss
        except Exception as e:
            np.random.set_state(rng_state)
            print(e, 'Returning 10e100')
            return 10e100

    def _update_hyperstate(self, XXold, XXnew, Xyold, Xynew, Llowerold, var_ratio):
        var_diag = var_ratio*np.eye(XXnew.shape[-1])
        XX = []
        Xy = []
        Llower = []
        for i in range(len(XXnew)):
            try:
                tmp = scipy.linalg.cholesky(XXnew[i] + var_diag, lower=True)
                XX.append(XXnew[i].copy())
                Xy.append(Xynew[i].copy())
                Llower.append(tmp.copy())
            except Exception as e:
                XX.append(XXold[i].copy())
                Xy.append(Xyold[i].copy())
                Llower.append(Llowerold[i].copy())
        XX = np.stack(XX, axis=0)
        Xy = np.stack(Xy, axis=0)
        Llower = np.stack(Llower, axis=0)
        return XX, Xy, Llower

    def _reward(self, state, action, sess, Llower, Xy, hyperparameters):
        basis = None
        if self.environment == 'Pendulum-v0' and self.learn_reward == 0:
            reward = self.reward_function.build_np(sess, state, action)
        elif self.environment == 'MountainCarContinuous-v0' and self.learn_reward == 0:
            reward = self.reward_function.build_np(state, action)
        else:
            state_action = np.concatenate([state, action], axis=-1)
            length_scale, signal_sd, noise_sd, prior_sd = hyperparameters
            basis = _basis(state_action, self.random_matrices[-1], self.biases[-1], self.basis_dims[-1], length_scale, signal_sd)
            basis = np.expand_dims(basis, axis=1)
            pred_mu, pred_sigma = self._predict(Llower, Xy, basis, noise_sd)
            if self.use_mean_reward == 1: pred_sigma = np.zeros_like(pred_sigma)
            reward = np.stack([np.random.normal(loc=loc, scale=scale) for loc, scale in zip(pred_mu, pred_sigma)], axis=0)
        return reward, basis

def solve(A, b):
    assert len(A.shape) == len(b.shape)
    assert len(A.shape) >= 3
    assert A.shape[:-1] == b.shape[:-1]
    A = np.copy(A)
    b = np.copy(b)

    bs = list(A.shape[:-2])
    dimA = list(A.shape[-2:])
    dimb = list(b.shape[-2:])

    A = np.reshape(A, [-1]+dimA)
    b = np.reshape(b, [-1]+dimb)

    results = [scipy.linalg.solve(_A, _b, sym_pos=True) for _A, _b in zip(A, b)]
    results = np.stack(results, axis=0)
    results = np.reshape(results, bs+dimb)

    return results

def solve_triangular(A, b):
    assert len(A.shape) == len(b.shape)
    assert len(A.shape) >= 3
    assert A.shape[:-1] == b.shape[:-1]
    A = np.copy(A)
    b = np.copy(b)

    bs = list(A.shape[:-2])
    dimA = list(A.shape[-2:])
    dimb = list(b.shape[-2:])

    A = np.reshape(A, [-1]+dimA)
    b = np.reshape(b, [-1]+dimb)

    results = [scipy.linalg.solve_triangular(_A, _b, lower=True) for _A, _b in zip(A, b)]
    results = np.stack(results, axis=0)
    results = np.reshape(results, bs+dimb)

    return results

def update_hyperstate(agent, hyperstate, hyperparameters, datum, dim, learn_diff):
    state, action, reward, next_state, _ = [np.atleast_2d(np.copy(dat)) for dat in datum]
    Llowers, Xy = [list(ele) for ele in hyperstate]
    assert len(Llowers) == len(hyperparameters)
    assert len(Xy) == len(hyperparameters)
    assert len(hyperparameters) == dim
    state_action = np.concatenate([state, action], axis=-1)
    y = np.concatenate([next_state - state if learn_diff else next_state, reward], axis=-1)[..., :dim]

    for i in range(len(Llowers)):
        Llowers[i] = Llowers[i].transpose([0, 2, 1])
    for i, hp in zip(range(dim), hyperparameters):
        length_scale, signal_sd, noise_sd, prior_sd = hp
        basis = _basis(state_action, agent.random_matrices[i], agent.biases[i], agent.basis_dims[i], length_scale, signal_sd)
        cholupdate(Llowers[i][0], basis[0].copy())
        Xy[i] += np.matmul(basis[:, None, :].transpose([0, 2, 1]), y[:, None, :][..., i:i+1])
    for i in range(len(Llowers)):
        Llowers[i] = Llowers[i].transpose([0, 2, 1])

    return [Llowers, Xy]

'''
def update_hyperstate_old(agent, XX, hyperstate, hyperparameters, datum, dim, learn_diff):
    state, action, reward, next_state, _ = [np.atleast_2d(np.copy(dat)) for dat in datum]
    Llowers, Xy = [list(ele) for ele in hyperstate]
    assert len(XX) == len(hyperparameters)
    assert len(Llowers) == len(hyperparameters)
    assert len(Xy) == len(hyperparameters)
    assert len(hyperparameters) == dim
    state_action = np.concatenate([state, action], axis=-1)
    y = np.concatenate([next_state - state if learn_diff else next_state, reward], axis=-1)[..., :dim]
    XX = list(XX)

    for i, hp in zip(range(dim), hyperparameters):
        length_scale, signal_sd, noise_sd, prior_sd = hp
        basis = _basis(state_action, agent.random_matrices[i], agent.biases[i], agent.basis_dims[i], length_scale, signal_sd)
        try:
            tmp = scipy.linalg.cholesky(XX[i] + np.matmul(basis.T, basis) + (noise_sd/prior_sd)**2*np.eye(agent.basis_dims[i]), lower=True)
            XX[i] += np.matmul(basis.T, basis)
            Xy[i] += np.matmul(basis.T, y[..., i:i+1])
            Llowers[i][:, :, :] = np.copy(tmp)
        except Exception as e:
            pass
        #Llowers[i] = np.transpose(cholupdate2(np.transpose(Llowers[i], [0, 2, 1]), basis), [0, 2, 1,])

    return XX, [Llowers, Xy]
'''

def unpack(data_buffer):
    states, actions, rewards, next_states = [np.stack(ele, axis=0) for ele in zip(*data_buffer)[:-1]]
    states_actions = np.concatenate([states, actions], axis=-1)
    return states_actions, states, rewards[..., np.newaxis], next_states

def scrub_data(environment, data_buffer, warn):
    if environment == 'MountainCarContinuous-v0':
        states, actions, rewards, next_states, dones = [np.stack(ele, axis=0) for ele in zip(*data_buffer)]
        for i in range(len(next_states)):
            if next_states[i, 0] == -1.2 and next_states[i, 1] == 0.:
                states = states[:i, ...]
                actions = actions[:i, ...]
                rewards = rewards[:i, ...]
                next_states = next_states[:i, ...]
                dones = dones[:i, ...]
                if warn: print('Warning: training data is cut short because the cart hit the left wall!')
                break
        data_buffer = zip(states, actions, rewards, next_states, dones)
    return data_buffer

def main_loop():
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default='Pendulum-v0')
    parser.add_argument("--unroll_steps", type=int, default=200)
    parser.add_argument("--discount_factor", type=float, default=.995)
    parser.add_argument("--gather_data_epochs", type=int, default=3, help='Epochs for initial data gather.')
    parser.add_argument("--train_hp_iterations", type=int, default=2000*10)
    parser.add_argument("--train_policy_batch_size", type=int, default=30)
    parser.add_argument("--no_samples", type=int, default=1)
    parser.add_argument("--basis_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--rffm_seed", type=int, default=1)
    parser.add_argument("--Agent", type=str, choices=['', '2', '3'], default='')
    parser.add_argument("--learn_reward", type=int, choices=[0, 1], default=1)
    parser.add_argument("--max_train_hp_datapoints", type=int, default=20000)
    parser.add_argument("--matern_param_reward", type=float, default=np.inf)
    parser.add_argument("--basis_dim_reward", type=int, default=600)
    parser.add_argument("--use_mean_reward", type=int, default=0)
    parser.add_argument("--update_hyperstate", type=int, default=1)
    parser.add_argument("--policy_use_hyperstate", type=int, default=1)
    parser.add_argument("--cma_maxiter", type=int, default=1000)
    parser.add_argument("--learn_diff", type=int, choices=[0, 1], default=0)
    args = parser.parse_args()

    print(sys.argv)
    print(args)
    from blr_regression2_sans_hyperstate import Agent2
    from blr_regression2_tf import Agent3

    env = gym.make(args.environment)

    regression_wrappers = [RegressionWrapper(input_dim=env.observation_space.shape[0]+env.action_space.shape[0],
                                             basis_dim=args.basis_dim,
                                             length_scale=1.,
                                             signal_sd=1.,
                                             noise_sd=5e-4,
                                             prior_sd=1.,
                                             rffm_seed=args.rffm_seed,
                                             train_hp_iterations=args.train_hp_iterations)
                           for _ in range(env.observation_space.shape[0])]
    if args.learn_reward == 1:
        regression_wrappers.append(RegressionWrapperReward(environment=args.environment,
                                                           input_dim=env.observation_space.shape[0]+env.action_space.shape[0],
                                                           basis_dim=args.basis_dim_reward,
                                                           length_scale=1.,
                                                           signal_sd=1.,
                                                           noise_sd=5e-4,
                                                           prior_sd=1.,
                                                           rffm_seed=args.rffm_seed,
                                                           train_hp_iterations=args.train_hp_iterations,
                                                           matern_param=args.matern_param_reward))
    agent = eval('Agent'+args.Agent)(environment=env.spec.id,
                                     x_dim=env.observation_space.shape[0]+env.action_space.shape[0],
                                     y_dim=env.observation_space.shape[0],
                                     state_dim=env.observation_space.shape[0],
                                     action_dim=env.action_space.shape[0],
                                     observation_space_low=env.observation_space.low,
                                     observation_space_high=env.observation_space.high,
                                     action_space_low=env.action_space.low,
                                     action_space_high=env.action_space.high,
                                     unroll_steps=args.unroll_steps,
                                     no_samples=args.no_samples,
                                     discount_factor=args.discount_factor,
                                     random_matrices=[rw.random_matrix for rw in regression_wrappers],
                                     biases=[rw.bias for rw in regression_wrappers],
                                     basis_dims=[rw.basis_dim for rw in regression_wrappers],
                                     hidden_dim=args.hidden_dim,
                                     learn_reward=args.learn_reward,
                                     use_mean_reward=args.use_mean_reward,
                                     update_hyperstate=args.update_hyperstate,
                                     policy_use_hyperstate=args.policy_use_hyperstate,
                                     learn_diff=args.learn_diff)

    flag = False
    data_buffer = gather_data(env, args.gather_data_epochs)
    data_buffer = scrub_data(args.environment, data_buffer, True)

    init_states = np.stack([env.reset() for _ in range(args.train_policy_batch_size)], axis=0)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if args.environment == 'Pendulum-v0' and args.learn_reward == 0:
            weights = pickle.load(open('../custom_environments/weights/pendulum_reward.p', 'rb'))
            sess.run(agent.assign_ops0, feed_dict=dict(zip(agent.placeholders_reward, weights)))
        for epoch in range(1000):
            #Train hyperparameters and update systems model.
            states_actions, states, rewards, next_states = unpack(data_buffer)
            targets = np.concatenate([next_states - states if args.learn_diff else next_states, rewards], axis=-1) 
            for i in range(env.observation_space.shape[0]+args.learn_reward):
                if flag == False:
                    regression_wrappers[i]._train_hyperparameters(states_actions, targets[:, i:i+1])
                    regression_wrappers[i]._reset_statistics(states_actions, targets[:, i:i+1])
                else:
                    regression_wrappers[i]._update(states_actions, targets[:, i:i+1])
            if len(data_buffer) >= args.max_train_hp_datapoints: flag = True
            if flag: data_buffer = []
            tmp_data_buffer = []

            #Fit policy network.
            XX, Xy, hyperparameters = zip(*[[rw.XX, rw.Xy, rw.hyperparameters] for rw in regression_wrappers])
            agent._fit(args.cma_maxiter, np.copy(init_states), [np.copy(ele) for ele in XX], [np.copy(ele) for ele in Xy], [np.copy(ele) for ele in hyperparameters], sess)

            #Get hyperstate & hyperparameters
            hyperstate = zip(*[[scipy.linalg.cholesky(np.copy(rw.XX)+(rw.noise_sd/rw.prior_sd)**2*np.eye(rw.basis_dim), lower=True)[np.newaxis, ...], np.copy(rw.Xy)[np.newaxis, ...]] for rw in regression_wrappers])

            total_rewards = 0.
            state = env.reset()
            while True:
                #env.render()
                action = agent._forward(agent.thetas, state[np.newaxis, ...], hyperstate)[0]
                next_state, reward, done, _ = env.step(action)

                #hyperstate = update_hyperstate_old(agent, XX, hyperstate, hyperparameters, [state, action, reward, next_state, done], agent.state_dim+agent.learn_reward, args.learn_diff)
                hyperstate = update_hyperstate(agent, hyperstate, hyperparameters, [state, action, reward, next_state, done], agent.state_dim+agent.learn_reward, args.learn_diff)

                tmp_data_buffer.append([state, action, reward, next_state, done])
                total_rewards += float(reward)
                state = np.copy(next_state)
                if done:
                    print('epoch:', epoch, 'total_rewards:', total_rewards)
                    data_buffer.extend(scrub_data(args.environment, tmp_data_buffer, False))
                    break

if __name__ == '__main__':
    main_loop()
