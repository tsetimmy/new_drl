import numpy as np
import scipy
from scipy.optimize import minimize
import warnings
from blr_regression2 import Agent, _basis

class Agent2(Agent):
    def __init__(self, environment, x_dim, y_dim, state_dim, action_dim, observation_space_low, observation_space_high,
                 action_space_low, action_space_high, unroll_steps, no_samples, discount_factor, random_matrices, biases, basis_dims,
                 hidden_dim=32, learn_reward=0, use_mean_reward=0, update_hyperstate=1):
        Agent.__init__(self, environment, x_dim, y_dim, state_dim, action_dim, observation_space_low, observation_space_high,
                       action_space_low, action_space_high, unroll_steps, no_samples, discount_factor, random_matrices, biases, basis_dims,
                       hidden_dim, learn_reward, use_mean_reward, update_hyperstate)
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

    def _loss(self, thetas, X, XX, Xy, hyperparameters, sess):
        rng_state = np.random.get_state()
        try:
            np.random.seed(2)

            rewards = []
            state = X
            for unroll_step in xrange(self.unroll_steps):
                action = self._forward(thetas, state)
                reward = self._reward(state, action, sess, XX[-1], Xy[-1], hyperparameters[-1])
                rewards.append((self.discount_factor**unroll_step)*reward)
                state_action = np.concatenate([state, action], axis=-1)

                means = []
                covs = []
                for i in range(self.state_dim):
                    length_scale, signal_sd, noise_sd, prior_sd = hyperparameters[i]
                    basis = _basis(state_action, self.random_matrices[i], self.biases[i], self.basis_dims[i], length_scale, signal_sd)

                    tmp = (noise_sd/prior_sd)**2*np.eye(self.basis_dims[i]) + XX[i]

                    pred_sigma = noise_sd**2 + np.sum(np.multiply(basis, noise_sd**2*scipy.linalg.solve(tmp, basis.T, sym_pos=True).T), axis=-1, keepdims=True)
                    pred_mu = np.matmul(basis, scipy.linalg.solve(tmp, Xy[i], sym_pos=True))

                    means.append(pred_mu)
                    covs.append(pred_sigma)
                means = np.concatenate(means, axis=-1)
                covs = np.concatenate(covs, axis=-1)

                state = np.stack([np.random.multivariate_normal(mean=mean, cov=np.diag(cov)) for mean, cov in zip(means, covs)], axis=0)
                state = np.clip(state, self.observation_space_low, self.observation_space_high)
            rewards = np.concatenate(rewards, axis=-1)
            rewards = np.sum(rewards, axis=-1)
            loss = -np.mean(rewards)
            np.random.set_state(rng_state)
            return loss
        except Exception as e:
            np.random.set_state(rng_state)
            print e, 'Returning 10e100.'
            return 10e100

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

    def _fit_cma(self, X, XXtr, Xytr, hyperparameters, sess):
        warnings.filterwarnings('error')
        assert len(XXtr) == self.state_dim + self.learn_reward
        assert len(Xytr) == self.state_dim + self.learn_reward
        assert len(hyperparameters) == self.state_dim + self.learn_reward

        X = np.copy(X)
        XXtr = [np.copy(ele) for ele in XXtr]
        Xytr = [np.copy(ele) for ele in Xytr]
        hyperparameters = [np.copy(ele) for ele in hyperparameters]

        import cma
        options = {'maxiter': 1000, 'verb_disp': 1, 'verb_log': 0}
        res = cma.fmin(self._loss, self.thetas, 2., args=(np.copy(X), [np.copy(ele) for ele in XXtr], [np.copy(ele) for ele in Xytr], [np.copy(ele) for ele in hyperparameters], sess), options=options)
        self.thetas = np.copy(res[0])

    def _reward(self, state, action, sess, XX, Xy, hyperparameters):
        if self.environment == 'Pendulum-v0' and self.learn_reward == 0:
            reward = self.reward_function.build_np(sess, state, action)
        elif self.environment == 'MountainCarContinuous-v0' and self.learn_reward == 0:
            reward = self.reward_function.build_np(state, action)
        else:
            state_action = np.concatenate([state, action], axis=-1)
            length_scale, signal_sd, noise_sd, prior_sd = hyperparameters
            basis = _basis(state_action, self.random_matrices[-1], self.biases[-1], self.basis_dims[-1], length_scale, signal_sd)
            tmp = (noise_sd/prior_sd)**2*np.eye(self.basis_dims[-1]) + XX
            if self.use_mean_reward == 1:
                predict_sigma = np.zeros([len(basis), 1])
            else:
                predict_sigma = noise_sd**2 + np.sum(np.multiply(basis, noise_sd**2*scipy.linalg.solve(tmp, basis.T, sym_pos=True).T), axis=-1, keepdims=True)
            predict_mu = np.matmul(basis, scipy.linalg.solve(tmp, Xy, sym_pos=True))
            reward = np.stack([np.random.normal(loc=loc, scale=scale) for loc, scale in zip(predict_mu, predict_sigma)], axis=0)
        return reward

