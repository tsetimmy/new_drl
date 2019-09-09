#Multi-output global regression
import numpy as np
import gym
import pybullet_envs
import argparse
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})

import sys
sys.path.append('..')
sys.path.append('../direct_policy_search')
from direct_policy_search.blr_regression2 import RegressionWrapper
from direct_policy_search.morw import MultiOutputRegressionWrapper
from regression_test4 import RegressionWrapper2, _basis, solve_triangular
from choldate import cholupdate

class MultiOutputRegressionWrapper2(MultiOutputRegressionWrapper):
    def __init__(self, input_dim, output_dim, basis_dim, length_scale=1., signal_sd=1., noise_sd=5e-4, prior_sd=1., rffm_seed=1, train_hp_iterations=2000, matern_param=np.inf):
        MultiOutputRegressionWrapper.__init__(self, input_dim, output_dim, basis_dim, length_scale, signal_sd, noise_sd, prior_sd, rffm_seed, train_hp_iterations, matern_param)

    def _reset_statistics(self, X, y, update_hyperstate):
        self._init_statistics()
        self._update(X, y, update_hyperstate)

    def _update(self, X, y, update_hyperstate):
        MultiOutputRegressionWrapper._update(self, X, y)
        if update_hyperstate:
            self.XX_tiled = np.tile(self.XX[np.newaxis, ...], [50, 1, 1])
            self.Xy_tiled = np.tile(self.Xy[np.newaxis, ...], [50, 1, 1])
            self.Llower_tiled = np.tile(self.Llower[np.newaxis, ...], [50, 1, 1])

    def _predict(self, X, update_hyperstate):
        if update_hyperstate:
            basis = _basis(X, self.random_matrix, self.bias, self.basis_dim, self.length_scale, self.signal_sd)
            basis = basis[:, None, :]#"batch_size" axis

            LinvXT = solve_triangular(self.Llower_tiled, basis.transpose([0, 2, 1]))
            pred_sigma = np.sum(np.square(LinvXT), axis=1)*self.noise_sd**2+self.noise_sd**2

            tmp0 = solve_triangular(self.Llower_tiled, basis.transpose([0, 2, 1])).transpose([0, 2, 1])
            tmp1 = solve_triangular(self.Llower_tiled, self.Xy_tiled)

            pred_mu = np.matmul(tmp0, tmp1).squeeze(axis=1)
            return pred_mu, pred_sigma
        else:
            return MultiOutputRegressionWrapper._predict(self, X)

    def _update_hyperstate(self, X, y, update_hyperstate):
        if update_hyperstate:
            basis = _basis(X, self.random_matrix, self.bias, self.basis_dim, self.length_scale, self.signal_sd)

            self.Llower_tiled = self.Llower_tiled.transpose([0, 2, 1])
            assert len(self.Llower_tiled) == len(basis)
            for i in range(len(self.Llower_tiled)):
                cholupdate(self.Llower_tiled[i], basis[i].copy())
            self.Llower_tiled = self.Llower_tiled.transpose([0, 2, 1])

            self.Xy_tiled += np.matmul(basis[:, None, :].transpose([0, 2, 1]), y[:, None, :])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default='Pendulum-v0')
    parser.add_argument("--update_hyperstate", type=int, default=0)
    parser.add_argument("--basis_dim", type=int, default=256)
    parser.add_argument("--basis_dim_reward", type=int, default=256)
    args = parser.parse_args()

    print(args)

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
            if env.spec.id == 'InvertedPendulumBulletEnv-v0':
                reward = next_state[2]
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            data_points += 1

            state = next_state.copy()

            if done:
                if args.environment == 'InvertedPendulumBulletEnv-v0':
                    for _ in range(10):
                        action = np.random.uniform(low=env.action_space.low, high=env.action_space.high)
                        next_state, _, done, _ = env.step(action)
                        states.append(state)
                        actions.append(action)
                        rewards.append(next_state[2])
                        next_states.append(next_state)
                        data_points += 1
                        state = next_state.copy()
                break

    X_train = np.concatenate([np.stack(states, axis=0), np.stack(actions, axis=0)], axis=-1)
    y_train = np.stack(next_states, axis=0)
    r_train = np.array(rewards)[..., None]

    morw = MultiOutputRegressionWrapper2(env.observation_space.shape[0]+env.action_space.shape[0], env.observation_space.shape[0], basis_dim=args.basis_dim)
    rw = RegressionWrapper2(env.observation_space.shape[0]+env.action_space.shape[0], basis_dim=args.basis_dim_reward)

    morw._train_hyperparameters(X_train, y_train)
    morw._reset_statistics(X_train, y_train, bool(args.update_hyperstate))

    rw._train_hyperparameters(X_train, r_train)
    rw._reset_statistics(X_train, r_train, bool(args.update_hyperstate))

    while True:
        state = env.reset()
        states = []
        actions = []
        rewards = []
        next_states = []
        while True:
            action = np.random.uniform(low=env.action_space.low, high=env.action_space.high)
            next_state, reward, done, _ = env.step(action)
            if env.spec.id == 'InvertedPendulumBulletEnv-v0':
                reward = next_state[2]
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)

            state = next_state.copy()

            if done:
                if args.environment == 'InvertedPendulumBulletEnv-v0':
                    for _ in range(10):
                        action = np.random.uniform(low=env.action_space.low, high=env.action_space.high)
                        next_state, _, done, _ = env.step(action)
                        states.append(state)
                        actions.append(action)
                        rewards.append(next_state[2])
                        next_states.append(next_state)
                        data_points += 1
                        state = next_state.copy()
                break

        X_test = np.concatenate([np.stack(states, axis=0), np.stack(actions, axis=0)], axis=-1)
        y_test = np.stack(next_states, axis=0)
        r_test = np.array(rewards)[..., None]

        mu0, sigma0 = morw._predict(X_test, False)
        for i in range(env.observation_space.shape[0]):
            plt.figure()
            plt.plot(np.arange(len(y_test[:, i])), y_test[:, i], 'b-')
            plt.errorbar(np.arange(len(mu0[:, i])), mu0[:, i], yerr=np.sqrt(sigma0.squeeze()), color='m', ecolor='g')
            plt.grid()
            plt.savefig('state'+str(i)+'.pdf')

        mu1, sigma1 = rw._predict(X_test, False)
        plt.figure()
        plt.plot(np.arange(len(rewards)), rewards, 'b-')
        plt.errorbar(np.arange(len(mu1)), mu1, yerr=np.sqrt(sigma1), color='m', ecolor='g')
        plt.grid()
        plt.title('rewards')
        plt.savefig('reward.pdf')

        sample_states = []
        sample_rewards = []
        sample_state = np.tile(states[0][None, ...], [no_samples, 1])
        for i in range(len(actions)):
            sample_state_action = np.concatenate([sample_state, np.tile(actions[i][None, ...], [no_samples, 1])], axis=-1)
            mu, sigma = morw._predict(sample_state_action, bool(args.update_hyperstate))
            sigma = np.tile(sigma, [1, mu.shape[-1]])
            #sample_state = np.random.normal(loc=mu, scale=sigma)
            sample_state = mu + np.sqrt(sigma) * np.random.standard_normal(size=sigma.shape)
            sample_states.append(sample_state)

            morw._update_hyperstate(sample_state_action, sample_state, bool(args.update_hyperstate))

            mu, sigma = rw._predict(sample_state_action, bool(args.update_hyperstate))
            #sample_reward = np.random.normal(loc=mu, scale=sigma)
            sample_reward = mu + np.sqrt(sigma) * np.random.standard_normal(size=sigma.shape)
            sample_rewards.append(sample_reward)

            rw._update_hyperstate(sample_state_action, sample_reward, bool(args.update_hyperstate))
        sample_states = np.stack(sample_states, axis=1)
        sample_rewards = np.stack(sample_rewards, axis=1)

        for i in range(env.observation_space.shape[0]):
            plt.figure()
            for sample in sample_states[..., i]:
                plt.plot(np.arange(len(sample)), sample, 'r-')
            plt.plot(np.arange(len(y_test[:, i])), y_test[:, i], 'b-')
            plt.grid()
            plt.savefig('state_traj'+str(i)+'.pdf')

        plt.figure()
        for sample in sample_rewards.squeeze():
            plt.plot(np.arange(len(sample)), sample, 'r-')
        plt.plot(np.arange(len(rewards)), rewards, 'b-')
        plt.grid()
        plt.title('rewards')
        plt.savefig('reward_traj.pdf')

        #plt.show()
        exit()

if __name__ == '__main__':
    main()
