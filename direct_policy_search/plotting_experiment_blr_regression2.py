import numpy as np
import scipy
import argparse
import gym

from blr_regression2 import RegressionWrapper, RegressionWrapperReward, _basis, solve_triangular

import sys
sys.path.append('..')

from utils import gather_data

class RegressionWrapper2(RegressionWrapper):
    def __init__(self, input_dim, basis_dim, length_scale=1., signal_sd=1., noise_sd=5e-4, prior_sd=1., rffm_seed=1, train_hp_iterations=2000, matern_param=np.inf):
        RegressionWrapper.__init__(self, input_dim, basis_dim, length_scale, signal_sd, noise_sd, prior_sd, rffm_seed, train_hp_iterations, matern_param)

    def _reset_statistics(self, X, y, update_hyperstate):
        self._init_statistics()
        self._update(X, y, update_hyperstate)

    def _update(self, X, y, update_hyperstate):
        RegressionWrapper._update(self, X, y)
        if update_hyperstate:
            self.XX_tiled = np.tile(self.XX[np.newaxis, ...], [50, 1, 1])
            self.Xy_tiled = np.tile(self.Xy[np.newaxis, ...], [50, 1, 1])
            self.Llower_tiled = np.tile(self.Llower[np.newaxis, ...], [50, 1, 1])

    def _predict(self, X, update_hyperstate):
        if update_hyperstate:
            basis = _basis(X, self.random_matrix, self.bias, self.basis_dim, self.length_scale, self.signal_sd)
            basis = np.expand_dims(basis, axis=1)

            LinvXT = solve_triangular(self.Llower_tiled, np.transpose(basis, [0, 2, 1]))
            pred_sigma = np.sum(np.square(LinvXT), axis=1)*self.noise_sd**2+self.noise_sd**2
            tmp0 = np.transpose(solve_triangular(self.Llower_tiled, np.transpose(basis, [0, 2, 1])), [0, 2, 1])
            tmp1 = solve_triangular(self.Llower_tiled, self.Xy_tiled)
            pred_mu = np.matmul(tmp0, tmp1)
            pred_mu = np.squeeze(pred_mu, axis=-1)
            return pred_mu, pred_sigma
        else:
            return RegressionWrapper._predict(self, X)

    def _update_hyperstate(self, X, y, update_hyperstate):
        if update_hyperstate:
            basis = _basis(X, self.random_matrix, self.bias, self.basis_dim, self.length_scale, self.signal_sd)
            basis = np.expand_dims(basis, axis=1)
            y = y[..., np.newaxis]
            XX_tiled_new = self.XX_tiled + np.matmul(np.transpose(basis, [0, 2, 1]), basis)
            Xy_tiled_new = self.Xy_tiled + np.matmul(np.transpose(basis, [0, 2, 1]), y)

            XX_tiled, Xy_tiled, Llower_tiled = [], [], []
            for XX_old, XX_new, Xy_old, Xy_new, Llower_old in zip(self.XX_tiled, XX_tiled_new, self.Xy_tiled, Xy_tiled_new, self.Llower_tiled):
                try:
                    tmp = scipy.linalg.cholesky(XX_new + (self.noise_sd/self.prior_sd)**2*np.eye(self.basis_dim), lower=True)
                    XX_tiled.append(XX_new.copy())
                    Xy_tiled.append(Xy_new.copy())
                    Llower_tiled.append(tmp.copy())
                except Exception as e:
                    print e
                    XX_tiled.append(XX_old.copy())
                    Xy_tiled.append(Xy_old.copy())
                    Llower_tiled.append(Llower_old.copy())
            self.XX_tiled = np.stack(XX_tiled, axis=0)
            self.Xy_tiled = np.stack(Xy_tiled, axis=0)
            self.Llower_tiled = np.stack(Llower_tiled, axis=0)

class RegressionWrapperReward2(RegressionWrapperReward):
    def __init__(self, environment, input_dim, basis_dim, length_scale=1., signal_sd=1., noise_sd=5e-4, prior_sd=1., rffm_seed=1, train_hp_iterations=2000, matern_param=np.inf):
        RegressionWrapperReward.__init__(self, environment, input_dim, basis_dim, length_scale, signal_sd, noise_sd, prior_sd, rffm_seed, train_hp_iterations, matern_param)

    def _reset_statistics(self, X, y, update_hyperstate):
        self._init_statistics()
        self._update(X, y, update_hyperstate)

    def _update(self, X, y, update_hyperstate):
        RegressionWrapperReward._update(self, X, y)
        if update_hyperstate:
            self.XX_tiled = np.tile(self.XX[np.newaxis, ...], [50, 1, 1])
            self.Xy_tiled = np.tile(self.Xy[np.newaxis, ...], [50, 1, 1])
            self.Llower_tiled = np.tile(self.Llower[np.newaxis, ...], [50, 1, 1])

    def _predict(self, X, update_hyperstate):
        if update_hyperstate:
            basis = _basis(X, self.random_matrix, self.bias, self.basis_dim, self.length_scale, self.signal_sd)
            basis = np.expand_dims(basis, axis=1)

            LinvXT = solve_triangular(self.Llower_tiled, np.transpose(basis, [0, 2, 1]))
            pred_sigma = np.sum(np.square(LinvXT), axis=1)*self.noise_sd**2+self.noise_sd**2
            tmp0 = np.transpose(solve_triangular(self.Llower_tiled, np.transpose(basis, [0, 2, 1])), [0, 2, 1])
            tmp1 = solve_triangular(self.Llower_tiled, self.Xy_tiled)
            pred_mu = np.matmul(tmp0, tmp1)
            pred_mu = np.squeeze(pred_mu, axis=-1)
            return pred_mu, pred_sigma
        else:
            return RegressionWrapperReward._predict(self, X)

    def _update_hyperstate(self, X, y, update_hyperstate):
        if update_hyperstate:
            basis = _basis(X, self.random_matrix, self.bias, self.basis_dim, self.length_scale, self.signal_sd)
            basis = np.expand_dims(basis, axis=1)
            y = y[..., np.newaxis]
            XX_tiled_new = self.XX_tiled + np.matmul(np.transpose(basis, [0, 2, 1]), basis)
            Xy_tiled_new = self.Xy_tiled + np.matmul(np.transpose(basis, [0, 2, 1]), y)

            XX_tiled, Xy_tiled, Llower_tiled = [], [], []
            for XX_old, XX_new, Xy_old, Xy_new, Llower_old in zip(self.XX_tiled, XX_tiled_new, self.Xy_tiled, Xy_tiled_new, self.Llower_tiled):
                try:
                    tmp = scipy.linalg.cholesky(XX_new + (self.noise_sd/self.prior_sd)**2*np.eye(self.basis_dim), lower=True)
                    XX_tiled.append(XX_new.copy())
                    Xy_tiled.append(Xy_new.copy())
                    Llower_tiled.append(tmp.copy())
                except Exception as e:
                    print e
                    XX_tiled.append(XX_old.copy())
                    Xy_tiled.append(Xy_old.copy())
                    Llower_tiled.append(Llower_old.copy())
            self.XX_tiled = np.stack(XX_tiled, axis=0)
            self.Xy_tiled = np.stack(Xy_tiled, axis=0)
            self.Llower_tiled = np.stack(Llower_tiled, axis=0)

def plotting_experiments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default='Pendulum-v0')
    parser.add_argument("--train-hp-iterations", type=int, default=2000)
    parser.add_argument("--basis-dim", type=int, default=256)
    parser.add_argument("--basis-dim-reward", type=int, default=600)
    parser.add_argument("--matern-param-reward", type=float, default=np.inf)

    parser.add_argument("--train-hit-wall", type=int, default=0)#Only used when --environment=MountainCarContinuous-v0
    parser.add_argument("--train-reach-goal", type=int, default=0)#Only used when --environment=MountainCarContinuous-v0
    parser.add_argument("--test-hit-wall", type=int, default=0)#Only used when --environment=MountainCarContinuous-v0
    parser.add_argument("--test-reach-goal", type=int, default=0)#Only used when --environment=MountainCarContinuous-v0

    parser.add_argument("--update-hyperstate", type=int, default=0)

    args = parser.parse_args()
    print args

    import matplotlib.pyplot as plt
    from utils import get_mcc_policy

    if args.environment == 'MountainCarContinuous-v0':
        train_set_size = 1
    else:
        train_set_size = 3

    env = gym.make(args.environment)

    predictors = []
    for i in range(env.observation_space.shape[0]):
        predictors.append(RegressionWrapper2(input_dim=env.observation_space.shape[0]+env.action_space.shape[0], basis_dim=args.basis_dim, length_scale=1.,
                                          signal_sd=1., noise_sd=5e-4, prior_sd=1., rffm_seed=1, train_hp_iterations=args.train_hp_iterations))
    predictors.append(RegressionWrapperReward2(args.environment, input_dim=env.observation_space.shape[0]+env.action_space.shape[0], basis_dim=args.basis_dim_reward, length_scale=1.,
                                              signal_sd=1., noise_sd=5e-4, prior_sd=1., rffm_seed=1, train_hp_iterations=args.train_hp_iterations, matern_param=args.matern_param_reward))

    if args.environment == 'MountainCarContinuous-v0':
        states, actions, rewards, next_states= get_mcc_policy(env, hit_wall=bool(args.train_hit_wall), reach_goal=bool(args.train_reach_goal), train=True)
    else:
        states, actions, rewards, next_states = gather_data(env, train_set_size, unpack=True)
    states_actions = np.concatenate([states, actions], axis=-1)

    for i in range(env.observation_space.shape[0]):
        predictors[i]._train_hyperparameters(states_actions, next_states[:, i:i+1])
        predictors[i]._reset_statistics(states_actions, next_states[:, i:i+1], bool(args.update_hyperstate))
    predictors[-1]._train_hyperparameters(states_actions, rewards)
    predictors[-1]._reset_statistics(states_actions, rewards, bool(args.update_hyperstate))

    while True:
        if args.environment == 'MountainCarContinuous-v0':
            states2, actions2, rewards2, next_states2 = get_mcc_policy(env, hit_wall=bool(args.test_hit_wall), reach_goal=bool(args.test_reach_goal), train=False)
        else:
            states2, actions2, rewards2, next_states2 = gather_data(env, 1, unpack=True)
        states_actions2 = np.concatenate([states2, actions2], axis=-1)

        plt.figure()
        for i in range(env.observation_space.shape[0]):
            plt.subplot(4, env.observation_space.shape[0], i+1)

            predict_mu, predict_sigma = predictors[i]._predict(states_actions2, False)

            plt.plot(np.arange(len(next_states2[:, i:i+1])), next_states2[:, i:i+1])
            plt.errorbar(np.arange(len(predict_mu)), predict_mu, yerr=np.sqrt(predict_sigma), color='m', ecolor='g')
            plt.grid()

        traj_reward = []
        traj = []
        no_lines = 50
        state = np.tile(np.copy(states2[0:1, ...]), [no_lines, 1])
        for a in actions2:
            action = np.tile(a[np.newaxis, ...], [no_lines, 1])
            state_action = np.concatenate([state, action], axis=-1)

            mu_reward, sigma_reward = predictors[-1]._predict(state_action, bool(args.update_hyperstate))
            reward = np.stack([np.random.normal(loc=mu, scale=sigma) for mu, sigma in zip(mu_reward, sigma_reward)], axis=0)
            traj_reward.append(reward)

            mu_vec = []
            sigma_vec = []
            for i in range(env.observation_space.shape[0]):
                predict_mu, predict_sigma = predictors[i]._predict(state_action, bool(args.update_hyperstate))
                mu_vec.append(predict_mu)
                sigma_vec.append(predict_sigma)

            mu_vec = np.concatenate(mu_vec, axis=-1)
            sigma_vec = np.concatenate(sigma_vec, axis=-1)

            state = np.stack([np.random.multivariate_normal(mu, np.diag(sigma)) for mu, sigma in zip(mu_vec, sigma_vec)], axis=0)
            state = np.clip(state, env.observation_space.low, env.observation_space.high)
            traj.append(np.copy(state))

        for i in range(env.observation_space.shape[0]):
            predictors[i]._update_hyperstate(state_action, state[:, i:i+1], bool(args.update_hyperstate))
        predictors[-1]._update_hyperstate(state_action, reward, bool(args.update_hyperstate))

        traj_reward = np.stack(traj_reward, axis=-1)
        traj = np.stack(traj, axis=-1)
        
        plt.subplot(4, 1, 4)
        for j in range(no_lines):
            y = traj_reward[j, 0, :]
            plt.plot(np.arange(len(y)), y, color='r')
        plt.plot(np.arange(len(rewards2)), rewards2)
        plt.grid()

        for i in range(env.observation_space.shape[0]):
            plt.subplot(4, env.observation_space.shape[0], env.observation_space.shape[0]+i+1)
            for j in range(no_lines):
                y = traj[j, i, :]
                plt.plot(np.arange(len(y)), y, color='r')

            plt.plot(np.arange(len(next_states2[..., i])), next_states2[..., i])
            plt.grid()

        plt.subplot(4, 1, 3)
        predict_mu, predict_sigma = predictors[-1]._predict(states_actions2, False)
        plt.plot(np.arange(len(rewards2)), rewards2)
        plt.errorbar(np.arange(len(predict_mu)), predict_mu, yerr=np.sqrt(predict_sigma), color='m', ecolor='g')
        plt.grid()

        plt.show(block=True)


if __name__ == '__main__':
    plotting_experiments()
