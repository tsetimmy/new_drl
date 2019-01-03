import numpy as np
import scipy
from scipy.optimize import minimize
import scipy.linalg as spla
import argparse

import sys
sys.path.append('..')

import tensorflow as tf
from custom_environments.generateANN_env import ANN
#from custom_environments.environment_state_functions import mountain_car_continuous_state_function
from custom_environments.environment_reward_functions import mountain_car_continuous_reward_function
from prototype8.dmlac.real_env_pendulum import real_env_pendulum_reward

from utils import gather_data

import gym
import pybullet_envs
import pickle
import warnings

from choldate import cholupdate

from blr_regression2 import _basis, RegressionWrapperReward

from morw import MultiOutputRegressionWrapper

class Agent:
    def __init__(self, environment, x_dim, y_dim, state_dim, action_dim, observation_space_low, observation_space_high,
                 action_space_low, action_space_high, unroll_steps, no_samples, discount_factor, random_matrix_state, bias_state,
                 basis_dim_state, random_matrix_reward, bias_reward, basis_dim_reward, hidden_dim=32, learn_reward=0,
                 use_mean_reward=0, update_hyperstate=1, policy_use_hyperstate=1, learn_diff=0):
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

        self.random_matrix_state = random_matrix_state
        self.bias_state = bias_state
        self.basis_dim_state = basis_dim_state
        self.random_matrix_reward = random_matrix_reward
        self.bias_reward = bias_reward
        self.basis_dim_reward = basis_dim_reward

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

        self.hyperstate_dim = self.basis_dim_state * (self.basis_dim_state + self.state_dim)
        if self.learn_reward == 1: self.hyperstate_dim += self.basis_dim_reward * (self.basis_dim_reward + 1)

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

    def _forward(self, thetas, X, hyperstate_params):
        w1, w2, w3 = self._unpack(thetas, self.sizes)

        #Perform a simple random projection on the hyperstate.
        if self.policy_use_hyperstate == 1:
            Llower_state, Xytr_state, Llower_reward, Xytr_reward = hyperstate_params

            print Llower_state.reshape([len(Llower_state), -1]).shape
            print Xytr_state.reshape([len(Xytr_state), -1]).shape
            print Llower_reward.reshape([len(Llower_reward), -1]).shape
            print Xytr_reward.reshape([len(Xytr_reward), -1]).shape
            exit()

            '''
            hyperstate = np.concatenate([np.concatenate([np.reshape(XXtr, [len(XXtr), -1]), np.reshape(Xytr, [len(Xytr), -1])], axis=-1) for XXtr, Xytr in zip(*hyperstate)], axis=-1)
            hyperstate = np.tanh(hyperstate/50000.)
            hyperstate_embedding = np.matmul(hyperstate, self.random_projection_matrix)
            hyperstate_embedding = np.tanh(hyperstate_embedding)

            state_hyperstate = np.concatenate([X, hyperstate_embedding], axis=-1)
            policy_net_input = self._add_bias(state_hyperstate)
            '''
        else:
            policy_net_input = self._add_bias(X)

        print 'here'
        exit()
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

    def _fit(self, cma_maxiter, X, XXtr_state, Xytr_state, hyperparameters_state, XXtr_reward, Xytr_reward, hyperparameters_reward, sess):
        warnings.filterwarnings('ignore', message='.*scipy.linalg.solve\nIll-conditioned matrix detected. Result is not guaranteed to be accurate.\nReciprocal.*')
        assert XXtr_state.shape == (self.basis_dim_state, self.basis_dim_state)
        assert Xytr_state.shape == (self.basis_dim_state, self.state_dim)
        assert XXtr_reward.shape == (self.basis_dim_reward, self.basis_dim_reward)
        assert Xytr_reward.shape == (self.basis_dim_reward, 1)
        assert hyperparameters_state.shape == hyperparameters_reward.shape

        if self.use_mean_reward == 1: print 'Warning: use_mean_reward is set to True but this flag is not used by this function.'

        #Copy the arrays (just to be safe no overwriting occurs).
        X = X.copy()
        XXtr_state = XXtr_state.copy()
        Xytr_state = Xytr_state.copy()
        hyperparameters_state = hyperparameters_state.copy()
        XXtr_reward = XXtr_reward.copy()
        Xytr_reward = Xytr_reward.copy()
        hyperparameters_reward = hyperparameters_reward.copy()

        X = np.expand_dims(X, axis=1)
        X = np.tile(X, [1, self.no_samples, 1])
        X = np.reshape(X, [-1, self.state_dim])

        #State
        Llower_state = spla.cholesky((hyperparameters_state[-2]/hyperparameters_state[-1])**2*np.eye(self.basis_dim_state) + XXtr_state, lower=True)
        Llower_state = np.tile(Llower_state, [len(X), 1, 1])

        XXtr_state = np.tile(XXtr_state, [len(X), 1, 1])
        Xytr_state = np.tile(Xytr_state, [len(X), 1, 1])

        #Reward
        Llower_reward = spla.cholesky((hyperparameters_reward[-2]/hyperparameters_reward[-1])**2*np.eye(self.basis_dim_reward) + XXtr_reward, lower=True)
        Llower_reward = np.tile(Llower_reward, [len(X), 1, 1])

        XXtr_reward = np.tile(XXtr_reward, [len(X), 1, 1])
        Xytr_reward = np.tile(Xytr_reward, [len(X), 1, 1])

        import cma
        options = {'maxiter': cma_maxiter, 'verb_disp': 1, 'verb_log': 0}
        print 'Before calling cma.fmin'
        res = cma.fmin(self._loss, self.thetas, 2., args=(X.copy(), Llower_state.copy(), XXtr_state.copy(), Xytr_state.copy(), hyperparameters_state, Llower_reward.copy(), XXtr_reward.copy(), Xytr_reward.copy(), hyperparameters_reward, sess), options=options)
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

        LinvXT = solve_triangular(Llower, np.transpose(basis, [0, 2, 1]))
        pred_sigma = np.sum(np.square(LinvXT), axis=1)*noise_sd**2+noise_sd**2
        tmp0 = np.transpose(solve_triangular(Llower, np.transpose(basis, [0, 2, 1])), [0, 2, 1])
        tmp1 = solve_triangular(Llower, Xytr)
        pred_mu = np.matmul(tmp0, tmp1)
        pred_mu = np.squeeze(pred_mu, axis=-1)
        return pred_mu, pred_sigma

    def _loss(self, thetas, X, Llower_state, XXtr_state, Xytr_state, hyperparameters_state, Llower_reward, XXtr_reward, Xytr_reward, hyperparameters_reward, sess=None):
        rng_state = np.random.get_state()
        X = X.copy()
        Llower_state = Llower_state.copy()
        XXtr_state = XXtr_state.copy()
        Xytr_state = Xytr_state.copy()
        hyperparameters_state = hyperparameters_state.copy()
        Llower_reward = Llower_reward.copy()
        XXtr_reward = XXtr_reward.copy()
        Xytr_reward = Xytr_reward.copy()
        hyperparameters_reward = hyperparameters_reward.copy()

        print Llower_state.shape
        print Xytr_state.shape
        print Llower_reward.shape
        print Xytr_reward.shape
        '''
        X = np.copy(X)
        Llowers = [np.copy(ele) for ele in Llowers]
        XXtr = [np.copy(ele) for ele in XXtr]
        Xytr = [np.copy(ele) for ele in Xytr]
        hyperparameters = [np.copy(ele) for ele in hyperparameters]
        '''
        #try:
        np.random.seed(2)

        rewards = []
        state = X
        for unroll_step in xrange(self.unroll_steps):
            #action = self._forward(thetas, state, hyperstate=[Llowers, Xytr])
            action = self._forward(thetas, state, hyperstate_params=[Llower_state, Xytr_state, Llower_reward, Xytr_reward])
            exit()
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

            if self.update_hyperstate == 1 and self.policy_use_hyperstate == 1:
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
        #except Exception as e:
            #np.random.set_state(rng_state)
            #print e, 'Returning 10e100'
            #return 10e100

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
                if warn: print 'Warning: training data is cut short because the cart hit the left wall!'
                break
        data_buffer = zip(states, actions, rewards, next_states, dones)
    return data_buffer

def main_loop():
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default='Pendulum-v0')
    parser.add_argument("--unroll-steps", type=int, default=200)
    parser.add_argument("--discount-factor", type=float, default=.995)
    parser.add_argument("--gather-data-epochs", type=int, default=3, help='Epochs for initial data gather.')
    parser.add_argument("--train-hp-iterations", type=int, default=2000*10)
    parser.add_argument("--train-policy-batch-size", type=int, default=30)
    parser.add_argument("--no-samples", type=int, default=1)
    parser.add_argument("--basis-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--rffm-seed", type=int, default=1)
    parser.add_argument("--Agent", type=str, choices=['', '2'], default='')
    parser.add_argument("--fit-function", type=str, choices=['_fit', '_fit_cma'], default='_fit')
    parser.add_argument("--learn-reward", type=int, choices=[0, 1], default=1)
    parser.add_argument("--max-train-hp-datapoints", type=int, default=20000)
    parser.add_argument("--matern-param-reward", type=float, default=np.inf)
    parser.add_argument("--basis-dim-reward", type=int, default=600)
    parser.add_argument("--use-mean-reward", type=int, default=0)
    parser.add_argument("--update-hyperstate", type=int, default=1)
    parser.add_argument("--policy-use-hyperstate", type=int, default=1)
    parser.add_argument("--cma-maxiter", type=int, default=1000)
    parser.add_argument("--learn-diff", type=int, choices=[0, 1], default=0)
    args = parser.parse_args()

    print sys.argv
    print args

    env = gym.make(args.environment)

    regression_wrapper_state = MultiOutputRegressionWrapper(input_dim=env.observation_space.shape[0]+env.action_space.shape[0],
                                                            output_dim=env.observation_space.shape[0],
                                                            basis_dim=args.basis_dim,
                                                            length_scale=1.,
                                                            signal_sd=1.,
                                                            noise_sd=5e-4,
                                                            prior_sd=1.,
                                                            rffm_seed=args.rffm_seed,
                                                            train_hp_iterations=args.train_hp_iterations)
    regression_wrapper_reward = RegressionWrapperReward(environment=args.environment,
                                                        input_dim=env.observation_space.shape[0]+env.action_space.shape[0],
                                                        basis_dim=args.basis_dim_reward,
                                                        length_scale=1.,
                                                        signal_sd=1.,
                                                        noise_sd=5e-4,
                                                        prior_sd=1.,
                                                        rffm_seed=args.rffm_seed,
                                                        train_hp_iterations=args.train_hp_iterations,
                                                        matern_param=args.matern_param_reward)
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

                                     random_matrix_state=regression_wrapper_state.random_matrix,
                                     bias_state=regression_wrapper_state.bias,
                                     basis_dim_state=regression_wrapper_state.basis_dim,


                                     random_matrix_reward=regression_wrapper_reward.random_matrix,
                                     bias_reward=regression_wrapper_reward.bias,
                                     basis_dim_reward=regression_wrapper_reward.basis_dim,

                                     #random_matrices=[rw.random_matrix for rw in regression_wrappers],
                                     #biases=[rw.bias for rw in regression_wrappers],
                                     #basis_dims=[rw.basis_dim for rw in regression_wrappers],

                                     hidden_dim=args.hidden_dim,
                                     learn_reward=args.learn_reward,
                                     use_mean_reward=args.use_mean_reward,
                                     update_hyperstate=args.update_hyperstate,
                                     policy_use_hyperstate=args.policy_use_hyperstate,
                                     learn_diff=args.learn_diff)


    #I have to work on the classes before working on the code below.
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

            next_states_train = next_states.copy() - states.copy() if args.learn_diff else next_states.copy()
            rewards_train = rewards.copy()

            if flag == False:
                #TODO: uncomment train hyperparameters
                #regression_wrapper_state._train_hyperparameters(states_actions, next_states_train)
                regression_wrapper_state._reset_statistics(states_actions, next_states_train)
                #regression_wrapper_reward._train_hyperparameters(states_actions, rewards_train)
                regression_wrapper_reward._reset_statistics(states_actions, rewards_train)
            else:
                regression_wrapper_state._update(states_actions, next_states_train)
                regression_wrapper_reward._update(states_actions, rewards_train)

            if len(data_buffer) >= args.max_train_hp_datapoints: flag = True
            if flag: data_buffer = []
            tmp_data_buffer = []

            #Fit policy network.
            #XX, Xy, hyperparameters = zip(*[[rw.XX, rw.Xy, rw.hyperparameters] for rw in regression_wrappers])
            #eval('agent.'+args.fit_function)(args.cma_maxiter, np.copy(init_states), [np.copy(ele) for ele in XX], [np.copy(ele) for ele in Xy], [np.copy(ele) for ele in hyperparameters], sess)
            eval('agent.'+args.fit_function)(args.cma_maxiter,
                                             init_states.copy(),
                                             regression_wrapper_state.XX.copy(),
                                             regression_wrapper_state.Xy.copy(),
                                             regression_wrapper_state.hyperparameters.copy(),
                                             regression_wrapper_reward.XX.copy(),
                                             regression_wrapper_reward.Xy.copy(),
                                             regression_wrapper_reward.hyperparameters.copy(),
                                             sess)
            exit()

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
                    print 'epoch:', epoch, 'total_rewards:', total_rewards
                    data_buffer.extend(scrub_data(args.environment, tmp_data_buffer, False))
                    break

if __name__ == '__main__':
    main_loop()
