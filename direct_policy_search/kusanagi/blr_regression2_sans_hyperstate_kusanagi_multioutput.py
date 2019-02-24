import numpy as np
import scipy
from scipy.optimize import minimize
import scipy.linalg as spla
import warnings
from blr_regression2_kusanagi_multioutput import Agent, _basis
import uuid
import os
import pickle

class Agent2(Agent):
    def __init__(self, env, x_dim, y_dim, state_dim, action_dim, observation_space_low, observation_space_high,
                 action_space_low, action_space_high, unroll_steps, no_samples, discount_factor, random_matrix_state,
                 bias_state, basis_dim_state, hidden_dim=32, update_hyperstate=1, policy_use_hyperstate=1, learn_diff=0, dump_model=0):
        Agent.__init__(self, env, x_dim, y_dim, state_dim, action_dim, observation_space_low, observation_space_high,
                       action_space_low, action_space_high, unroll_steps, no_samples, discount_factor, random_matrix_state,
                       bias_state, basis_dim_state, hidden_dim, update_hyperstate, policy_use_hyperstate, learn_diff, dump_model)
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

    def _loss(self, thetas, X, Llower_state, XXtr_state, Xytr_state, hyperparameters_state):
        X = X.copy()
        Llower_state = Llower_state.copy()
        XXtr_state = XXtr_state.copy()
        Xytr_state = Xytr_state.copy()
        hyperparameters_state = hyperparameters_state.copy()

        rng_state = np.random.get_state()
        #try:
        np.random.seed(2)

        rewards = []
        state = X
        for unroll_step in range(self.unroll_steps):
            action = self._forward(thetas, state)
            state_action = np.concatenate([state, action], axis=-1)

            length_scale, signal_sd, noise_sd, prior_sd = hyperparameters_state
            basis = _basis(state_action, self.random_matrix_state, self.bias_state, self.basis_dim_state, length_scale, signal_sd)

            tmp0 = spla.solve_triangular(Llower_state, basis.T, lower=True).T
            sigma = np.sum(np.square(tmp0), axis=-1, keepdims=True)*noise_sd**2+noise_sd**2

            tmp1 = spla.solve_triangular(Llower_state, Xytr_state, lower=True)
            mu = np.matmul(tmp0, tmp1)

            state_ = mu + np.sqrt(sigma) * np.random.standard_normal(size=mu.shape)
            state = np.clip(state + state_ if self.learn_diff else state_, self.observation_space_low, self.observation_space_high)

            reward = -self.env.loss_func(state)
            rewards.append((self.discount_factor**unroll_step)*reward)


        rewards = np.stack(rewards, axis=-1).sum(axis=-1)
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

    def _fit(self, cma_maxiter, X, XXtr_state, Xytr_state, hyperparameters_state):
        warnings.filterwarnings('error')
        assert XXtr_state.shape == (self.basis_dim_state, self.basis_dim_state)
        assert Xytr_state.shape == (self.basis_dim_state, self.state_dim)

        X = X.copy()
        XXtr_state = XXtr_state.copy()
        Xytr_state = Xytr_state.copy()
        hyperparameters_state = hyperparameters_state.copy()

        Llower_state = spla.cholesky((hyperparameters_state[-2]/hyperparameters_state[-1])**2*np.eye(self.basis_dim_state) + XXtr_state, lower=True)

        import cma
        options = {'maxiter': cma_maxiter, 'verb_disp': 1, 'verb_log': 0}
        res = cma.fmin(self._loss, self.thetas, 2., args=(X.copy(),
                                                          Llower_state.copy(),
                                                          XXtr_state.copy(),
                                                          Xytr_state.copy(),
                                                          hyperparameters_state.copy()), options=options)
        self.thetas = res[0].copy()
        if self.dump_model:
            print('Unique identifier:', self.uid)
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
