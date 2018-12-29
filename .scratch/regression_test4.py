#Multi-output global regression with individual outputs
import numpy as np
import scipy.linalg as spla
import gym
import pybullet_envs
import argparse
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})

#from morw import MultiOutputRegressionWrapper
import sys
sys.path.append('..')

from direct_policy_search.blr_regression2 import RegressionWrapper, _basis, solve_triangular
from choldate import cholupdate

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

            self.Llower_tiled = self.Llower_tiled.transpose([0, 2, 1])
            assert len(self.Llower_tiled) == len(basis)
            for i in range(len(self.Llower_tiled)):
                cholupdate(self.Llower_tiled[i], basis[i].copy())
            self.Llower_tiled = self.Llower_tiled.transpose([0, 2, 1])

            self.Xy_tiled += np.matmul(basis[:, None, :].transpose([0, 2, 1]), y[:, None, :])

    def _update_hyperstate2(self, X, y, update_hyperstate):
        if update_hyperstate:
            basis = _basis(X, self.random_matrix, self.bias, self.basis_dim, self.length_scale, self.signal_sd)
            basis = np.expand_dims(basis, axis=1)
            y = y[..., np.newaxis]
            XX_tiled_new = self.XX_tiled + np.matmul(np.transpose(basis, [0, 2, 1]), basis)
            Xy_tiled_new = self.Xy_tiled + np.matmul(np.transpose(basis, [0, 2, 1]), y)

            XX_tiled, Xy_tiled, Llower_tiled = [], [], []
            for XX_old, XX_new, Xy_old, Xy_new, Llower_old in zip(self.XX_tiled, XX_tiled_new, self.Xy_tiled, Xy_tiled_new, self.Llower_tiled):
                try:
                    tmp = spla.cholesky(XX_new + (self.noise_sd/self.prior_sd)**2*np.eye(self.basis_dim), lower=True)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default='Pendulum-v0')
    parser.add_argument("--update_hyperstate", type=int, default=0)
    parser.add_argument("--train_hp_iterations", type=int, default=2000)
    parser.add_argument("--basis_dim", type=int, default=256)
    parser.add_argument("--basis_dim_reward", type=int, default=256)
    args = parser.parse_args()

    print args

    no_data_points = 1000
    no_samples = 50
    env = gym.make(args.environment)
    #env.render(mode='human')

    states = []
    actions = []
    rewards = []
    next_states = []
    data_points = 0
    while no_data_points > data_points:
        state = env.reset()

        while True:
            action = np.random.uniform(low=env.action_space.low, high=env.action_space.high)
            next_state, reward, done, _ = env.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            data_points += 1

            state = next_state.copy()

            if done:
                break

    X_train = np.concatenate([np.stack(states, axis=0), np.stack(actions, axis=0)], axis=-1)
    y_train = np.stack(next_states, axis=0)
    r_train = np.array(rewards)[..., None]

    rws = [RegressionWrapper2(env.observation_space.shape[0]+env.action_space.shape[0], basis_dim=args.basis_dim, train_hp_iterations=args.train_hp_iterations) for _ in range(env.observation_space.shape[0])]

    rwr = RegressionWrapper2(env.observation_space.shape[0]+env.action_space.shape[0], basis_dim=args.basis_dim_reward, train_hp_iterations=args.train_hp_iterations)

    for i in range(len(rws)):
        rws[i]._train_hyperparameters(X_train, y_train[..., i:i+1])
    rwr._train_hyperparameters(X_train, r_train)

    while True:
        state = env.reset()
        states = []
        actions = []
        rewards = []
        next_states = []
        while True:
            action = np.random.uniform(low=env.action_space.low, high=env.action_space.high)
            next_state, reward, done, _ = env.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)

            state = next_state.copy()

            if done:
                break

        X_test = np.concatenate([np.stack(states, axis=0), np.stack(actions, axis=0)], axis=-1)
        y_test= np.stack(next_states, axis=0)
        r_test = np.array(rewards)[..., None]

        for i in range(len(rws)):
            rws[i]._reset_statistics(X_train, y_train[..., i:i+1], bool(args.update_hyperstate))
        rwr._reset_statistics(X_train, r_train, bool(args.update_hyperstate))

        mu0, sigma0 = [np.concatenate(ele, axis=-1) for ele in zip(*[rws[i]._predict(X_test, False) for i in range(env.observation_space.shape[0])])]

        for i in range(env.observation_space.shape[0]):
            plt.figure()
            plt.plot(np.arange(len(y_test[:, i])), y_test[:, i], 'b-')
            plt.errorbar(np.arange(len(mu0[:, i])), mu0[:, i], yerr=np.sqrt(sigma0[:, i]), color='m', ecolor='g')
            plt.grid()

        mu1, sigma1 = rwr._predict(X_test, False)
        plt.figure()
        plt.plot(np.arange(len(r_test)), r_test, 'b-')
        plt.errorbar(np.arange(len(mu1)), mu1, yerr=np.sqrt(sigma1), color='m', ecolor='g')
        plt.grid()
        plt.title('rewards')

        #Updating of hyperstate by rank one Cholesky updates
        sample_states = []
        sample_rewards = []
        sample_state = np.tile(states[0][None, ...], [no_samples, 1])
        for i in range(len(actions)):
            sample_state_action = np.concatenate([sample_state, np.tile(actions[i][None, ...], [no_samples, 1])], axis=-1)
            mu, sigma = [np.concatenate(ele, axis=-1) for ele in zip(*[rws[i]._predict(sample_state_action, bool(args.update_hyperstate)) for i in range(env.observation_space.shape[0])])]
            sample_state = mu + np.sqrt(sigma) * np.random.standard_normal(size=sigma.shape)
            sample_states.append(sample_state)

            for j in range(env.observation_space.shape[0]):
                rws[j]._update_hyperstate(sample_state_action, sample_state[..., j:j+1], bool(args.update_hyperstate))

            mu, sigma = rwr._predict(sample_state_action, bool(args.update_hyperstate))
            sample_reward = mu + np.sqrt(sigma) * np.random.standard_normal(size=sigma.shape)
            sample_rewards.append(sample_reward)

            rwr._update_hyperstate(sample_state_action, sample_reward, bool(args.update_hyperstate))
        sample_states = np.stack(sample_states, axis=1)
        sample_rewards = np.stack(sample_rewards, axis=1)

        for i in range(env.observation_space.shape[0]):
            plt.figure()
            for sample in sample_states[..., i]:
                plt.plot(np.arange(len(sample)), sample, 'r-')
            plt.plot(np.arange(len(y_test[:, i])), y_test[:, i], 'b-')
            plt.grid()

        plt.figure()
        for sample in sample_rewards.squeeze():
            plt.plot(np.arange(len(sample)), sample, 'r-')
        plt.plot(np.arange(len(rewards)), rewards, 'b-')
        plt.grid()
        plt.title('rewards')

        for i in range(len(rws)):
            rws[i]._reset_statistics(X_train, y_train[..., i:i+1], bool(args.update_hyperstate))
        rwr._reset_statistics(X_train, r_train, bool(args.update_hyperstate))

        #Updating of hyperstate by Cholesky decomposition
        sample_states = []
        sample_rewards = []
        sample_state = np.tile(states[0][None, ...], [no_samples, 1])
        for i in range(len(actions)):
            sample_state_action = np.concatenate([sample_state, np.tile(actions[i][None, ...], [no_samples, 1])], axis=-1)
            mu, sigma = [np.concatenate(ele, axis=-1) for ele in zip(*[rws[i]._predict(sample_state_action, bool(args.update_hyperstate)) for i in range(env.observation_space.shape[0])])]
            sample_state = mu + np.sqrt(sigma) * np.random.standard_normal(size=sigma.shape)
            sample_state = np.clip(sample_state, env.observation_space.low, env.observation_space.high)
            sample_states.append(sample_state)

            for j in range(env.observation_space.shape[0]):
                rws[j]._update_hyperstate2(sample_state_action, sample_state[..., j:j+1], bool(args.update_hyperstate))

            mu, sigma = rwr._predict(sample_state_action, bool(args.update_hyperstate))
            sample_reward = mu + np.sqrt(sigma) * np.random.standard_normal(size=sigma.shape)
            sample_rewards.append(sample_reward)

            rwr._update_hyperstate2(sample_state_action, sample_reward, bool(args.update_hyperstate))
        sample_states = np.stack(sample_states, axis=1)
        sample_rewards = np.stack(sample_rewards, axis=1)

        for i in range(env.observation_space.shape[0]):
            plt.figure()
            for sample in sample_states[..., i]:
                plt.plot(np.arange(len(sample)), sample, 'y-')
            plt.plot(np.arange(len(y_test[:, i])), y_test[:, i], 'b-')
            plt.grid()

        plt.figure()
        for sample in sample_rewards.squeeze():
            plt.plot(np.arange(len(sample)), sample, 'y-')
        plt.plot(np.arange(len(rewards)), rewards, 'b-')
        plt.grid()
        plt.title('rewards')


        plt.show()

if __name__ == '__main__':
    main()
