import numpy as np
import argparse

from kusanagi.shell import experiment_utils, cartpole, double_cartpole, pendulum
from functools import partial

#from regression_test3 import MultiOutputRegressionWrapper
import matplotlib.pyplot as plt

from utils import RegressionWrapper2, RegressionWrapperReward2, get_data2

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--env", type=str, choices=['cartpole', 'double_cartpole', 'pendulum'], default='cartpole')
    parser.add_argument("--train_hp_iterations", type=int, default=2000)
    parser.add_argument("--basis_dim", type=int, default=256)
    parser.add_argument("--basis_dim_reward", type=int, default=600)
    parser.add_argument("--matern_param", type=float, default=np.inf)
    parser.add_argument("--matern_param_reward", type=float, default=np.inf)
    parser.add_argument("--update_hyperstate", type=int, default=0)

    parser.add_argument("--trials", type=int, default=1)

    args = parser.parse_args()
    print(args)

    if args.env == 'cartpole':
        params = cartpole.default_params()
        cost = partial(cartpole.cartpole_loss, **params['cost'])
        env = cartpole.Cartpole(loss_func=cost, **params['plant'])
        max_steps = 25
        maxA = 10.
    elif args.env == 'double_cartpole':
        params = double_cartpole.default_params()
        cost = partial(double_cartpole.double_cartpole_loss, **params['cost'])
        env = double_cartpole.DoubleCartpole(loss_func=cost, **params['plant'])
        max_steps = 30
        maxA = 20.
    elif args.env == 'pendulum':
        params = pendulum.default_params()
        cost = partial(pendulum.pendulum_loss, **params['cost'])
        env = pendulum.Pendulum(loss_func=cost, **params['plant'])
        max_steps = 40
        maxA = 2.5
    else:
        raise Exception('Unknown environment.')


    states, actions, rewards, next_states = get_data2(env, trials=args.trials, max_steps=max_steps, maxA=maxA)
    states_actions = np.concatenate([states, actions], axis=-1)

    predictors = []
    for i in range(env.observation_space.shape[0]):
        predictors.append(RegressionWrapper2(input_dim=env.observation_space.shape[0]+env.action_space.shape[0], basis_dim=args.basis_dim, length_scale=1.,
                                          signal_sd=1., noise_sd=5e-4, prior_sd=1., rffm_seed=1, train_hp_iterations=args.train_hp_iterations, matern_param=args.matern_param))
    #predictors.append(RegressionWrapperReward2('tmp', input_dim=env.observation_space.shape[0]+env.action_space.shape[0], basis_dim=args.basis_dim_reward, length_scale=1.,
                                              #signal_sd=1., noise_sd=5e-4, prior_sd=1., rffm_seed=1, train_hp_iterations=args.train_hp_iterations, matern_param=args.matern_param_reward))

    for i in range(env.observation_space.shape[0]):
        predictors[i]._train_hyperparameters(states_actions, next_states[:, i:i+1])
    #predictors[-1]._train_hyperparameters(states_actions, rewards)

    while True:
        for i in range(env.observation_space.shape[0]):
            predictors[i]._reset_statistics(states_actions, next_states[:, i:i+1], bool(args.update_hyperstate))
        #predictors[-1]._reset_statistics(states_actions, rewards, bool(args.update_hyperstate))

        states2, actions2, rewards2, next_states2 = get_data2(env, trials=1, max_steps=max_steps, maxA=maxA)
        states_actions2 = np.concatenate([states2, actions2], axis=-1)

        plt.figure()
        for i in range(env.observation_space.shape[0]):
            plt.subplot(3, env.observation_space.shape[0], i+1)

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

            #mu_reward, sigma_reward = predictors[-1]._predict(state_action, bool(args.update_hyperstate))
            #reward = np.stack([np.random.normal(loc=mu, scale=sigma) for mu, sigma in zip(mu_reward, sigma_reward)], axis=0)
            #traj_reward.append(reward)

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


            reward = -env.loss_func(state)
            traj_reward.append(reward)

            for i in range(env.observation_space.shape[0]):
                predictors[i]._update_hyperstate(state_action, state[:, i:i+1], bool(args.update_hyperstate))
            #predictors[-1]._update_hyperstate(state_action, reward, bool(args.update_hyperstate))

        traj_reward = np.stack(traj_reward, axis=-1)
        traj = np.stack(traj, axis=-1)
        
        plt.subplot(3, 1, 3)
        for j in range(no_lines):
            y = traj_reward[j, :]
            plt.plot(np.arange(len(y)), y, color='r')
        plt.plot(np.arange(len(rewards2)), rewards2)
        plt.grid()

        for i in range(env.observation_space.shape[0]):
            plt.subplot(3, env.observation_space.shape[0], env.observation_space.shape[0]+i+1)
            for j in range(no_lines):
                y = traj[j, i, :]
                plt.plot(np.arange(len(y)), y, color='r')

            plt.plot(np.arange(len(next_states2[..., i])), next_states2[..., i])
            plt.grid()

        #plt.subplot(4, 1, 3)
        #predict_mu, predict_sigma = predictors[-1]._predict(states_actions2, False)
        #plt.plot(np.arange(len(rewards2)), rewards2)
        #plt.errorbar(np.arange(len(predict_mu)), predict_mu, yerr=np.sqrt(predict_sigma), color='m', ecolor='g')
        #plt.grid()

        #plt.savefig('fig.pdf')
        #break
        plt.show(block=True)

if __name__ == '__main__':
    main()
