import numpy as np
from scipy.optimize import minimize
import argparse

import sys
sys.path.append('..')

#from custom_environments.environment_state_functions import mountain_car_continuous_state_function
from custom_environments.environment_reward_functions import mountain_car_continuous_reward_function
from prototype8.dmlac.real_env_pendulum import real_env_pendulum_reward

from more_gradfree_experiments import posterior
from utils import gather_data, gather_data2

import gym
import time

def _basis(X, random_matrix, bias, output_dim, length_scale, signal_sd):
    x_omega_plus_bias = np.matmul(X, (1./length_scale)*random_matrix) + bias
    z = signal_sd * np.sqrt(2./output_dim) * np.cos(x_omega_plus_bias)
    return z

class regression_wrapper:
    def __init__(self, input_dim, output_dim, length_scale=1., signal_sd=1., noise_sd=5e-4, prior_sd=1., rffm_seed=1, train_hp_iterations=2000, noise_sd_clip_threshold=5e-5):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.length_scale = length_scale
        self.signal_sd = signal_sd
        self.noise_sd = noise_sd
        self.prior_sd = prior_sd

        self.rffm_seed = rffm_seed
        self.train_hp_iterations = train_hp_iterations
        self.noise_sd_clip_threshold = noise_sd_clip_threshold

        self._init_statistics()

        rng_state = np.random.get_state()
        np.random.seed(self.rffm_seed)

        self.random_matrix = np.random.normal(size=[self.input_dim, self.output_dim])
        self.bias = np.random.uniform(low=0., high=2.*np.pi, size=[self.output_dim])

        np.random.set_state(rng_state)

    def _init_statistics(self):
        self.XX = np.zeros([self.output_dim, self.output_dim])
        self.Xy = np.zeros([self.output_dim, 1])

    '''
    def _basis(self, X, length_scale, signal_sd):
        x_omega_plus_bias = np.matmul(X, (1./length_scale)*self.random_matrix) + self.bias
        z = signal_sd * np.sqrt(2./self.output_dim) * np.cos(x_omega_plus_bias)
        return z
    '''

    def _update(self, X, y):
        assert len(X.shape) == 2
        assert len(y.shape) == 2
        assert X.shape[0] == y.shape[0]

        basis = _basis(X, self.random_matrix, self.bias, self.output_dim, self.length_scale, self.signal_sd)
        self.XX += np.matmul(basis.T, basis)
        self.Xy += np.matmul(basis.T, y)

    def _train_hyperparameters(self, X, y):
        thetas = np.array([self.length_scale, self.signal_sd, self.noise_sd, self.prior_sd])
        options = {'maxiter': self.train_hp_iterations, 'disp': True}
        _res = minimize(self._log_marginal_likelihood, thetas, method='nelder-mead', args=(X, y), options=options)
        self.length_scale, self.signal_sd, self.noise_sd, self.prior_sd = _res.x
        self.noise_sd = np.maximum(self.noise_sd, self.noise_sd_clip_threshold)
        print self.length_scale, self.signal_sd, self.noise_sd, self.prior_sd

    def _log_marginal_likelihood(self, thetas, X, y):
        try:
            length_scale, signal_sd, noise_sd, prior_sd = thetas
            noise_sd_clipped = np.maximum(noise_sd, self.noise_sd_clip_threshold)

            basis = _basis(X, self.random_matrix, self.bias, self.output_dim, length_scale, signal_sd)
            N = len(basis.T)
            XX = np.matmul(basis.T, basis)
            Xy = np.matmul(basis.T, y)

            wn, Vn, V0, tmp = posterior(XX, Xy, noise_sd_clipped, prior_sd)

            s1, logdet1 = np.linalg.slogdet(V0)
            s2, logdet2 = np.linalg.slogdet(Vn)
            assert s1 == 1 and s2 == 1

            lml = .5*(-N*np.log(noise_sd_clipped**2) - logdet1 + logdet2 - np.matmul(y.T, y)[0, 0]/noise_sd_clipped**2 + np.matmul(np.matmul(Xy.T, tmp.T), Xy)[0, 0]/noise_sd_clipped**2)
            loss = -lml
            return loss
        except:
            return np.inf

    def _reset_statistics(self, X, y):
        self._init_statistics()
        self._update(X, y)

    def _predict(self, X):
        basis = _basis(X, self.random_matrix, self.bias, self.output_dim, self.length_scale, self.signal_sd)
        mu, sigma, _, _ = posterior(self.XX, self.Xy, self.noise_sd, self.prior_sd)
        predict_mu = np.matmul(basis, mu)
        predict_sigma = self.noise_sd**2 + np.sum(np.multiply(np.matmul(basis, sigma), basis), axis=-1, keepdims=True)
        return predict_mu, predict_sigma

class agent:
    def __init__(self, environment, x_dim, y_dim, state_dim, action_dim, action_space_low, action_space_high,
                 unroll_steps, no_samples, discount_factor, rffm_seed=1, output_dim=256):
        assert environment in ['Pendulum-v0', 'MountainCarContinuous-v0']
        assert x_dim == state_dim + action_dim
        assert len(action_space_low.shape) == 1
        np.testing.assert_equal(-action_space_low, action_space_high)
        self.environment = environment
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_space_low = action_space_low
        self.action_space_high = action_space_high

        self.unroll_steps = unroll_steps
        self.no_samples = no_samples
        self.discount_factor = discount_factor
        self.rffm_seed = rffm_seed
        self.output_dim = output_dim

        #Initialize random matrix
        rng_state = np.random.get_state()
        np.random.seed(self.rffm_seed)
        self.random_matrix = np.random.normal(size=[self.x_dim, self.output_dim])
        self.bias = np.random.uniform(low=0., high=2.*np.pi, size=[self.output_dim])
        np.random.set_state(rng_state)

        #Use real reward function
        if self.environment == 'Pendulum-v0':
            self.reward_function = real_env_pendulum_reward()
        elif self.environment == 'MountainCarContinuous-v0':
            self.reward_function = mountain_car_continuous_reward_function()

        #TODO: initialize neural network
        self.thetas = None

    def _forward(self, X, hyperstate):
        wn, Vn = hyperstate

        batch_size, state_dim, _, _ = Vn.shape

        indices = np.triu_indices(self.output_dim, 1)
        for i in range(batch_size):
            for j in range(state_dim):
                Vn[i, j][indices] = np.nan

        Vn = Vn[~np.isnan(Vn)]
        Vn = np.reshape(Vn, [batch_size, state_dim, -1])
        print Vn.shape

        #wn = np.reshape(wn, [len(wn), -1])
        print wn.shape
        exit()
        #TODO: code this
        return np.random.uniform(size=[len(X), 1])

    def _fit(self, X, XXtr, Xytr, hyperparamters):
        assert len(XXtr) == self.state_dim
        assert len(Xytr) == self.state_dim

        A = []
        for i in xrange(self.state_dim):
            _, _, noise_sd, prior_sd = hyperparameters[i]
            V0 = prior_sd**2*np.eye(self.output_dim)
            noise = noise_sd**2*np.linalg.inv(V0)
            tmp = np.linalg.inv(noise + XXtr[i])
            A.append(tmp)
        A = np.stack(A, axis=0)

        X = np.expand_dims(X, axis=1)
        X = np.tile(X, [1, self.no_samples, 1])
        X = np.reshape(X, [-1, self.state_dim])

        XXtr = np.tile(XXtr[np.newaxis, ...], [len(X), 1, 1, 1])
        Xytr = np.tile(Xytr[np.newaxis, ...], [len(X), 1, 1, 1])
        A = np.tile(A[np.newaxis, ...], [len(X), 1, 1, 1])

        options = {'maxiter': 1, 'disp': True}
        _res = minimize(self._loss, self.thetas, method='powell', args=(X, XXtr, Xytr, A, hyperparameters), options=options)
        assert self.thetas.shape == _res.x.shape
        self.thetas = np.copy(_res.x)

    def _loss(self, thetas, X, XXtr, Xytr, A=[], hyperparameters=None):
        #---------------------------------------------------#
        assert len(XXtr) == self.state_dim
        assert len(Xytr) == self.state_dim

        A = []
        for i in xrange(self.state_dim):
            _, _, noise_sd, prior_sd = hyperparameters[i]
            V0 = prior_sd**2*np.eye(self.output_dim)
            noise = noise_sd**2*np.linalg.inv(V0)
            tmp = np.linalg.inv(noise + XXtr[i])
            A.append(tmp)
        A = np.stack(A, axis=0)

        X = np.expand_dims(X, axis=1)
        X = np.tile(X, [1, self.no_samples, 1])
        X = np.reshape(X, [-1, self.state_dim])

        XXtr = np.tile(XXtr[np.newaxis, ...], [len(X), 1, 1, 1])
        Xytr = np.tile(Xytr[np.newaxis, ...], [len(X), 1, 1, 1])
        A = np.tile(A[np.newaxis, ...], [len(X), 1, 1, 1])
        print XXtr.shape
        print '-------------------'
        #---------------------------------------------------#





        rng_state = np.random.get_state()
        np.random.seed(2)

        rewards = []
        state = X
        for unroll_step in xrange(self.unroll_steps):

            Vns = []
            wns = []
            for i in xrange(self.state_dim):
                length_scale, signal_sd, noise_sd, prior_sd = hyperparameters[i]
                Vn = noise_sd**2*A[:, i, ...]
                wn = np.matmul(A[:, i, ...], Xytr[:, i, ...])
                Vns.append(Vn)
                wns.append(wn)

            action = self._forward(state, hyperstate=[np.stack(wns, axis=1), np.stack(Vns, axis=1)])

            reward = self.reward_function.build_np(state, action)
            rewards.append((self.discount_factor**unroll_step)*reward)

            state_action = np.concatenate([state, action], axis=-1)

            means = []
            covs = []
            bases = []
            for i in xrange(self.state_dim):
                length_scale, signal_sd, noise_sd, prior_sd = hyperparameters[i]
                basis = _basis(state_action, self.random_matrix, self.bias, self.output_dim, length_scale, signal_sd)
                bases.append(basis)
                #Vn = noise_sd**2*A[:, i, ...]
                #wn = np.matmul(A[:, i, ...], Xytr[:, i, ...])
                basis = np.expand_dims(basis, axis=1)
                
                pred_mu = np.squeeze(np.matmul(basis, wns[i]))
                pred_sigma = noise_sd**2 + np.squeeze(np.matmul(np.matmul(basis, Vns[i]), np.transpose(basis, [0, 2, 1])))

                means.append(pred_mu)
                covs.append(pred_sigma)
            means = np.stack(means, axis=-1)
            covs = np.stack(covs, axis=-1)

            state = np.stack([np.random.multivariate_normal(mean=mean, cov=np.diag(cov)) for mean, cov in zip(means, covs)], axis=0)

            bases = np.stack(bases, axis=1)
            bases = np.expand_dims(bases, axis=2)
            bases_transpose = np.transpose(bases, [0, 1, 3, 2])

            XXtr += np.matmul(bases_transpose, bases)
            state_expand_dims = state[..., np.newaxis][..., np.newaxis]
            Xytr += np.matmul(bases_transpose, state_expand_dims)

            tmp = np.matmul(bases, A)
            A -= np.matmul(np.matmul(A, bases_transpose), tmp) /\
                 (1. + np.matmul(tmp, bases_transpose))

        rewards = np.concatenate(rewards, axis=-1)
        rewards = np.sum(rewards, axis=-1)
        loss = -np.mean(rewards)
        np.random.set_state(rng_state)
        return loss

def main2():
    env = gym.make('Pendulum-v0')
    output_dim = 5
    tmp = agent(environment=env.spec.id,
                x_dim=env.observation_space.shape[0]+env.action_space.shape[0],
                y_dim=env.observation_space.shape[0],
                state_dim=env.observation_space.shape[0],
                action_dim=env.action_space.shape[0],
                action_space_low=env.action_space.low,
                action_space_high=env.action_space.high,
                unroll_steps=200,
                no_samples=7,
                discount_factor=.999,
                rffm_seed=1,
                output_dim=output_dim)
    tmp._loss(thetas=np.random.uniform(size=[10, 3]),
              X=np.random.uniform(size=[5, 3]),
              XXtr=np.random.uniform(size=[3, output_dim, output_dim]),
              Xytr=np.random.uniform(size=[3, output_dim, 1]),
              hyperparameters=np.random.uniform(size=[3, 4]))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default='Pendulum-v0')
    parser.add_argument("--train-hp-iterations", type=int, default=2000)
    args = parser.parse_args()
    print args

    env = gym.make(args.environment)

    predictors = []
    for i in range(env.observation_space.shape[0]):
        predictors.append(regression_wrapper(input_dim=env.observation_space.shape[0]+env.action_space.shape[0], output_dim=128*2, length_scale=1.,
                                          signal_sd=1., noise_sd=5e-4, prior_sd=1., rffm_seed=1, train_hp_iterations=args.train_hp_iterations, noise_sd_clip_threshold=5e-5))

    states, actions, next_states = gather_data(env, 5, unpack=True)
    states_actions = np.concatenate([states, actions], axis=-1)

    # Quick plotting experiment (for sanity check).
    import matplotlib.pyplot as plt
    if args.environment == 'Pendulum-v0':
        states2, actions2, next_states2 = gather_data(env, 1, unpack=True)
    elif args.environment == 'MountainCarContinuous-v0':
        from utils import mcc_get_success_policy
        states2, actions2, next_states2 = mcc_get_success_policy(env)
    states_actions2 = np.concatenate([states2, actions2], axis=-1)

    plt.figure()
    for i in range(env.observation_space.shape[0]):
        plt.subplot(2, env.observation_space.shape[0], i+1)

        predictors[i]._train_hyperparameters(states_actions, next_states[:, i:i+1])
        predictors[i]._update(states_actions, next_states[:, i:i+1])
        predict_mu, predict_sigma = predictors[i]._predict(states_actions2)

        plt.plot(np.arange(len(next_states2[:, i:i+1])), next_states2[:, i:i+1])
        plt.errorbar(np.arange(len(predict_mu)), predict_mu, yerr=np.sqrt(predict_sigma), color='m', ecolor='g')
        plt.grid()

    traj = []
    no_lines = 50
    state = np.tile(np.copy(states2[0:1, ...]), [no_lines, 1])
    for a in actions2:
        action = np.tile(a[np.newaxis, ...], [no_lines, 1])
        state_action = np.concatenate([state, action], axis=-1)

        mu_vec = []
        sigma_vec = []
        for i in range(env.observation_space.shape[0]):
            predict_mu, predict_sigma = predictors[i]._predict(state_action)
            mu_vec.append(predict_mu)
            sigma_vec.append(predict_sigma)

        mu_vec = np.concatenate(mu_vec, axis=-1)
        sigma_vec = np.concatenate(sigma_vec, axis=-1)

        state = np.stack([np.random.multivariate_normal(mu, np.diag(sigma)) for mu, sigma in zip(mu_vec, sigma_vec)], axis=0)
        traj.append(np.copy(state))

    traj = np.stack(traj, axis=-1)

    for i in range(env.observation_space.shape[0]):
        plt.subplot(2, env.observation_space.shape[0], env.observation_space.shape[0]+i+1)
        for j in range(no_lines):
            y = traj[j, i, :]
            plt.plot(np.arange(len(y)), y, color='r')

        plt.plot(np.arange(len(next_states2[..., i])), next_states2[..., i])
        plt.grid()

    plt.show()

if __name__ == '__main__':
    #main()
    main2()
