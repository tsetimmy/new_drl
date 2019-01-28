import numpy as np
import scipy
from scipy.optimize import minimize
import scipy.linalg as spla
import warnings
from blr_regression2_kusanagi import Agent, _basis

class Agent2(Agent):
    def __init__(self, env, x_dim, y_dim, state_dim, action_dim, observation_space_low, observation_space_high,
                 action_space_low, action_space_high, unroll_steps, no_samples, discount_factor, random_matrices, biases, basis_dims,
                 hidden_dim=32, update_hyperstate=1, policy_use_hyperstate=1, learn_diff=0):
        Agent.__init__(self, env, x_dim, y_dim, state_dim, action_dim, observation_space_low, observation_space_high,
                       action_space_low, action_space_high, unroll_steps, no_samples, discount_factor, random_matrices, biases, basis_dims,
                       hidden_dim, update_hyperstate, policy_use_hyperstate, learn_diff)
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

    def _loss(self, thetas, X, Llowers, Xy, hyperparameters):
        rng_state = np.random.get_state()
        try:
            np.random.seed(2)

            rewards = []
            state = X
            for unroll_step in range(self.unroll_steps):
                action = self._forward(thetas, state)
                state_action = np.concatenate([state, action], axis=-1)

                means = []
                covs = []
                for i in range(self.state_dim):
                    length_scale, signal_sd, noise_sd, prior_sd = hyperparameters[i]
                    basis = _basis(state_action, self.random_matrices[i], self.biases[i], self.basis_dims[i], length_scale, signal_sd)

                    tmp0 = spla.solve_triangular(Llowers[i], basis.T, lower=True).T
                    pred_sigma = np.square(tmp0).sum(axis=-1, keepdims=True)*noise_sd**2+noise_sd**2

                    tmp1 = spla.solve_triangular(Llowers[i], Xy[i], lower=True)
                    pred_mu = np.matmul(tmp0, tmp1)

                    means.append(pred_mu)
                    covs.append(pred_sigma)
                means = np.concatenate(means, axis=-1)
                covs = np.concatenate(covs, axis=-1)

                state_ = means + np.sqrt(covs) * np.random.standard_normal(size=covs.shape)
                state = state + state_ if self.learn_diff else state_.copy()

                reward = -self.env.loss_func(state)
                rewards.append((self.discount_factor**unroll_step)*reward)


            rewards = np.stack(rewards, axis=-1).sum(axis=-1)
            loss = -np.mean(rewards)
            np.random.set_state(rng_state)
            return loss
        except Exception as e:
            np.random.set_state(rng_state)
            print(e, 'Returning 10e100.')
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

    def _fit(self, cma_maxiter, X, XXtr, Xytr, hyperparameters):
        warnings.filterwarnings('error')
        assert len(XXtr) == self.state_dim
        assert len(Xytr) == self.state_dim
        assert len(hyperparameters) == self.state_dim

        X = X.copy()
        XXtr = [ele.copy() for ele in XXtr]
        Xytr = [ele.copy() for ele in Xytr]
        hyperparameters = [ele.copy() for ele in hyperparameters]

        Llowers = [spla.cholesky((hp[-2]/hp[-1])**2*np.eye(len(XX)) + XX, lower=True) for XX, hp in zip(XXtr, hyperparameters)]

        import cma
        options = {'maxiter': cma_maxiter, 'verb_disp': 1, 'verb_log': 0}
        res = cma.fmin(self._loss, self.thetas, 2., args=(X.copy(), [ele.copy() for ele in Llowers], [ele.copy() for ele in Xytr], [ele.copy() for ele in hyperparameters]), options=options)
        self.thetas = np.copy(res[0])
