import numpy as np
from scipy.optimize import minimize
import warnings
from blr_regression2 import Agent, _basis

class Agent2(Agent):
    def __init__(self, environment, x_dim, y_dim, state_dim, action_dim, observation_space_low, observation_space_high,
                 action_space_low, action_space_high, unroll_steps, no_samples, discount_factor, rffm_seed=1, basis_dim=256):
        Agent.__init__(self, environment, x_dim, y_dim, state_dim, action_dim, observation_space_low, observation_space_high,
                       action_space_low, action_space_high, unroll_steps, no_samples, discount_factor, rffm_seed, basis_dim)
        self._init_thetas2()
        self.flag = False

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

    def _loss(self, thetas, X, wn, Vn, hyperparameters, sess):
        rng_state = np.random.get_state()
        try:
            np.random.seed(2)

            rewards = []
            state = X
            for unroll_step in xrange(self.unroll_steps):
                action = self._forward(thetas, state)

                if self.environment == 'Pendulum-v0':
                    reward = self.reward_function.build_np(sess, state, action)
                elif self.environment == 'MountainCarContinuous-v0':
                    reward = self.reward_function.build_np(state, action)
                rewards.append((self.discount_factor**unroll_step)*reward)

                state_action = np.concatenate([state, action], axis=-1)

                means = []
                covs = []
                for i in range(self.state_dim):
                    length_scale, signal_sd, noise_sd, prior_sd = hyperparameters[i]
                    basis = _basis(state_action, self.random_matrix, self.bias, self.basis_dim, length_scale, signal_sd)
                    pred_mu = np.matmul(basis, wn[i])
                    pred_sigma = noise_sd**2 + np.sum(np.multiply(np.matmul(basis, Vn[i]), basis), axis=-1, keepdims=True)
                    means.append(pred_mu)
                    covs.append(pred_sigma)
                means = np.concatenate(means, axis=-1)
                covs = np.concatenate(covs, axis=-1)

                state = np.stack([np.random.multivariate_normal(mean=mean, cov=np.diag(cov)) for mean, cov in zip(means, covs)], axis=0)
            rewards = np.concatenate(rewards, axis=-1)
            rewards = np.sum(rewards, axis=-1)
            loss = -np.mean(rewards)
            np.random.set_state(rng_state)
            self.count += 1
            if self.flag:
                print 'count:', self.count, 'loss:', loss
            else:
                print 'count:', self.count, 'loss:', loss,
            return loss
        except Exception as e:
            np.random.set_state(rng_state)
            print e, 'Returning 10e100'
            return 10.**100

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

    def _fit_random_search(self, X, XXtr, Xytr, hyperparameters, sess):
        warnings.filterwarnings('error')
        assert len(XXtr) == self.state_dim
        assert len(Xytr) == self.state_dim
        assert len(hyperparameters) == self.state_dim

        Vn = []
        wn = []
        for xx, xy, hp in zip(XXtr, Xytr, hyperparameters):
            length_scale, signal_sd, noise_sd, prior_sd = hp
            tmp = np.linalg.inv((noise_sd/prior_sd)**2*np.eye(self.basis_dim) + xx)
            Vn.append(noise_sd**2*tmp)
            wn.append(np.matmul(tmp, xy))
        Vn = np.stack(Vn, axis=0)
        wn = np.stack(wn, axis=0)
        thetas = np.copy(self.thetas)#Remove later.

        '********'
        maxiter = 1000
        lowest = self._loss(self.thetas, np.copy(X), np.copy(wn), np.copy(Vn), np.copy(hyperparameters), sess)
        print 'Starting loss:', lowest
        for i in xrange(maxiter):
            scale = np.random.uniform(low=0., high=5.)
            perterbations = scale*np.random.normal(size=[len(self.thetas)])
            loss = self._loss(self.thetas + perterbations, np.copy(X), np.copy(wn), np.copy(Vn), np.copy(hyperparameters), sess)
            if loss < lowest:
                self.thetas += perterbations
                lowest = loss
            print 'lowest:', lowest

        '--------'
        self.thetas = np.copy(thetas)#Remove later.
        self.flag = True
        options = {'maxiter': 1, 'maxfev': maxiter, 'disp': True}
        _res = minimize(self._loss, self.thetas, method='powell', args=(np.copy(X), np.copy(wn), np.copy(Vn), np.copy(hyperparameters), sess), options=options)
        assert self.thetas.shape == _res.x.shape
        self.thetas = np.copy(_res.x)
