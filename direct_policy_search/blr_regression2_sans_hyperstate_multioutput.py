import numpy as np
import scipy
from scipy.optimize import minimize
import scipy.linalg as spla
import warnings
from blr_regression2_multioutput import Agent, _basis
import uuid
import os
import pickle

class Agent2(Agent):
    def __init__(self, environment, x_dim, y_dim, state_dim, action_dim, observation_space_low, observation_space_high,
                 action_space_low, action_space_high, unroll_steps, no_samples, discount_factor, random_matrix_state,
                 bias_state, basis_dim_state, random_matrix_reward, bias_reward, basis_dim_reward, hidden_dim=32,
                 learn_reward=0, use_mean_reward=0, update_hyperstate=1, policy_use_hyperstate=1, learn_diff=0, dump_model=0):
        Agent.__init__(self, environment, x_dim, y_dim, state_dim, action_dim, observation_space_low, observation_space_high,
                       action_space_low, action_space_high, unroll_steps, no_samples, discount_factor, random_matrix_state,
                       bias_state, basis_dim_state, random_matrix_reward, bias_reward, basis_dim_reward, hidden_dim, learn_reward,
                       use_mean_reward, update_hyperstate, policy_use_hyperstate, learn_diff, dump_model)
        del self.update_hyperstate
        del self.policy_use_hyperstate
        self._init_thetas2()

    def _init_thetas2(self):
        del self.hyperstate_dim
        self.w1 = np.concatenate([np.random.normal(size=[self.state_dim, self.hidden_dim]), np.random.uniform(-3e-3, 3e-3, size=[1, self.hidden_dim])], axis=0)
        self.w2 = np.concatenate([np.random.normal(size=[self.hidden_dim, self.hidden_dim]), np.random.uniform(-3e-3, 3e-3, size=[1, self.hidden_dim])], axis=0)
        self.w3 = np.concatenate([np.random.normal(size=[self.hidden_dim, self.action_dim]), np.random.uniform(-3e-3, 3e-3, size=[1, self.action_dim])], axis=0)

        self.thetas = self._pack([self.w1, self.w2, self.w3])

        self.sizes = [[self.state_dim + 1, self.hidden_dim],
                      [self.hidden_dim + 1, self.hidden_dim],
                      [self.hidden_dim + 1, self.action_dim]]

        w1, w2, w3 = self._unpack(self.thetas, self.sizes)
        np.testing.assert_equal(w1, self.w1)
        np.testing.assert_equal(w2, self.w2)
        np.testing.assert_equal(w3, self.w3)

        if self.learn_reward == 0 and self.use_mean_reward == 1:
            print 'Warning: flags learn_reward is False but use_mean_reward is True.'

    def _loss(self, thetas, X, Llower_state, XXtr_state, Xytr_state, hyperparameters_state, Llower_reward, XXtr_reward, Xytr_reward, hyperparameters_reward, sess=None):
        X = X.copy()
        Llower_state = Llower_state.copy()
        XXtr_state = XXtr_state.copy()
        Xytr_state = Xytr_state.copy()
        hyperparameters_state = hyperparameters_state.copy()
        if self.learn_reward:
            Llower_reward = Llower_reward.copy()
            XXtr_reward = XXtr_reward.copy()
            Xytr_reward = Xytr_reward.copy()
            hyperparameters_reward = hyperparameters_reward.copy()

        rng_state = np.random.get_state()
        #try:
        np.random.seed(2)

        rewards = []
        state = X
        for unroll_step in xrange(self.unroll_steps):
            action = self._forward(thetas, state)
            state_action = np.concatenate([state, action], axis=-1)

            reward = self._reward(state, action, state_action, sess, Llower_reward, Xytr_reward, hyperparameters_reward)
            rewards.append((self.discount_factor**unroll_step)*reward)

            length_scale, signal_sd, noise_sd, prior_sd = hyperparameters_state
            basis = _basis(state_action, self.random_matrix_state, self.bias_state, self.basis_dim_state, length_scale, signal_sd)

            tmp0 = spla.solve_triangular(Llower_state, basis.T, lower=True).T
            sigma = np.sum(np.square(tmp0), axis=-1, keepdims=True)*noise_sd**2+noise_sd**2

            tmp1 = spla.solve_triangular(Llower_state, Xytr_state, lower=True)
            mu = np.matmul(tmp0, tmp1)

            state_ = mu + np.sqrt(sigma) * np.random.standard_normal(size=mu.shape)
            state = np.clip(state + state_ if self.learn_diff else state_, self.observation_space_low, self.observation_space_high)

        rewards = np.concatenate(rewards, axis=-1)
        rewards = np.sum(rewards, axis=-1)
        loss = -np.mean(rewards)
        np.random.set_state(rng_state)
        return loss
        #except Exception as e:
            #np.random.set_state(rng_state)
            #print e, 'Returning 10e100.'
            #return 10e100

    def _forward(self, thetas, X, *unused):
        w1, w2, w3 = self._unpack(thetas, self.sizes)

        X = self._add_bias(X)

        h1 = np.tanh(np.matmul(X, w1))
        h1 = self._add_bias(h1)

        h2 = np.tanh(np.matmul(h1, w2))
        h2 = self._add_bias(h2)

        out = np.tanh(np.matmul(h2, w3))
        out = out * self.action_space_high#action bounds.

        return out

    def _fit(self, cma_maxiter, X, XXtr_state, Xytr_state, hyperparameters_state, XXtr_reward, Xytr_reward, hyperparameters_reward, sess):
        warnings.filterwarnings('error')
        assert XXtr_state.shape == (self.basis_dim_state, self.basis_dim_state)
        assert Xytr_state.shape == (self.basis_dim_state, self.state_dim)
        assert XXtr_reward.shape == (self.basis_dim_reward, self.basis_dim_reward)
        assert Xytr_reward.shape == (self.basis_dim_reward, 1)
        assert hyperparameters_state.shape == hyperparameters_reward.shape

        X = X.copy()
        XXtr_state = XXtr_state.copy()
        Xytr_state = Xytr_state.copy()
        hyperparameters_state = hyperparameters_state.copy()
        XXtr_reward = XXtr_reward.copy()
        Xytr_reward = Xytr_reward.copy()
        hyperparameters_reward = hyperparameters_reward.copy()

        Llower_state = spla.cholesky((hyperparameters_state[-2]/hyperparameters_state[-1])**2*np.eye(self.basis_dim_state) + XXtr_state, lower=True)
        Llower_reward = spla.cholesky((hyperparameters_reward[-2]/hyperparameters_reward[-1])**2*np.eye(self.basis_dim_reward) + XXtr_reward, lower=True)

        import cma
        options = {'maxiter': cma_maxiter, 'verb_disp': 1, 'verb_log': 0}
        res = cma.fmin(self._loss, self.thetas, 2., args=(X.copy(),
                                                          Llower_state.copy(),
                                                          XXtr_state.copy(),
                                                          Xytr_state.copy(),
                                                          hyperparameters_state.copy(),
                                                          Llower_reward.copy() if self.learn_reward else None,
                                                          XXtr_reward.copy() if self.learn_reward else None,
                                                          Xytr_reward.copy() if self.learn_reward else None,
                                                          hyperparameters_reward.copy() if self.learn_reward else None,
                                                          sess), options=options)
        self.thetas = np.copy(res[0])
        if self.dump_model:
            print 'Unique identifier:', self.uid
            directory = './models/'
            if not os.path.exists(directory):
                os.makedirs(directory)
            with open(directory+self.uid+'_epoch:'+str(self.epoch)+'.p', 'wb') as fp:
                pickle.dump(self.thetas, fp)
            self.epoch += 1

    def _reward(self, state, action, state_action, sess, Llower, Xy, hyperparameters):
        if self.environment == 'Pendulum-v0' and self.learn_reward == 0:
            reward = self.reward_function.build_np(sess, state, action)
        elif self.environment == 'MountainCarContinuous-v0' and self.learn_reward == 0:
            reward = self.reward_function.build_np(state, action)
        else:
            length_scale, signal_sd, noise_sd, prior_sd = hyperparameters
            basis = _basis(state_action, self.random_matrix_reward, self.bias_reward, self.basis_dim_reward, length_scale, signal_sd)
            tmp0 = spla.solve_triangular(Llower, basis.T, lower=True).T
            sigma = np.zeros([len(basis), 1]) if self.use_mean_reward else np.sum(np.square(tmp0), axis=-1, keepdims=True)*noise_sd**2+noise_sd**2
            tmp1 = spla.solve_triangular(Llower, Xy, lower=True)
            mu = np.matmul(tmp0, tmp1)

            reward = mu + np.sqrt(sigma) * np.random.standard_normal(size=mu.shape)
        return reward
