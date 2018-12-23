#Multi-output global regression with individual outputs
import numpy as np
import gym
import pybullet_envs
import argparse
import matplotlib.pyplot as plt

#from morw import MultiOutputRegressionWrapper
import sys
sys.path.append('..')

from direct_policy_search.blr_regression2 import RegressionWrapper

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default='Pendulum-v0')
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

    X = np.concatenate([np.stack(states, axis=0), np.stack(actions, axis=0)], axis=-1)
    y = np.stack(next_states, axis=0)
    r = np.array(rewards)[..., None]

    rws = [RegressionWrapper(env.observation_space.shape[0]+env.action_space.shape[0], basis_dim=128*2, train_hp_iterations=2000) for _ in range(env.observation_space.shape[0])]

    rwr = RegressionWrapper(env.observation_space.shape[0]+env.action_space.shape[0], 128*2, train_hp_iterations=2000)

    for i in range(len(rws)):
        rws[i]._train_hyperparameters(X, y[..., i:i+1])
        rws[i]._reset_statistics(X, y[..., i:i+1])

    rwr._train_hyperparameters(X, r)
    rwr._reset_statistics(X, r)
    #morw = MultiOutputRegressionWrapper(env.observation_space.shape[0]+env.action_space.shape[0], env.observation_space.shape[0], basis_dim=128*2)
    #morw._train_hyperparameters(X, y)
    #morw._reset_statistics(X, y)

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

        X = np.concatenate([np.stack(states, axis=0), np.stack(actions, axis=0)], axis=-1)
        y = np.stack(next_states, axis=0)
        r = np.array(rewards)[..., None]

        #mu0, sigma0 = morw._predict(X)
        mu0, sigma0 = [np.concatenate(ele, axis=-1) for ele in zip(*[rws[i]._predict(X) for i in range(env.observation_space.shape[0])])]

        for i in range(env.observation_space.shape[0]):
            plt.figure()
            plt.plot(np.arange(len(y[:, i])), y[:, i], 'b-')
            #plt.errorbar(np.arange(len(mu0[:, i])), mu0[:, i], yerr=np.sqrt(sigma0.squeeze()), color='m', ecolor='g')
            plt.errorbar(np.arange(len(mu0[:, i])), mu0[:, i], yerr=np.sqrt(sigma0[:, i]), color='m', ecolor='g')
            plt.grid()

        mu1, sigma1 = rwr._predict(X)
        plt.figure()
        plt.plot(np.arange(len(r)), r, 'b-')
        plt.errorbar(np.arange(len(mu1)), mu1, yerr=np.sqrt(sigma1), color='m', ecolor='g')
        plt.grid()
        plt.title('rewards')

        sample_states = []
        sample_rewards = []
        sample_state = np.tile(states[0][None, ...], [no_samples, 1])
        for i in range(len(actions)):
            sample_state_action = np.concatenate([sample_state, np.tile(actions[i][None, ...], [no_samples, 1])], axis=-1)
            mu, sigma = [np.concatenate(ele, axis=-1) for ele in zip(*[rws[i]._predict(sample_state_action) for i in range(env.observation_space.shape[0])])]
            #mu, sigma = morw._predict(sample_state_action)
            #sample_state = np.random.normal(loc=mu, scale=sigma)
            sample_state = mu + np.sqrt(sigma) * np.random.standard_normal(size=sigma.shape)
            sample_states.append(sample_state)

            mu, sigma = rwr._predict(sample_state_action)
            #sample_reward = np.random.normal(loc=mu, scale=sigma)
            sample_reward = mu + np.sqrt(sigma) * np.random.standard_normal(size=sigma.shape)
            sample_rewards.append(sample_reward)
        sample_states = np.stack(sample_states, axis=1)
        sample_rewards = np.stack(sample_rewards, axis=1)

        for i in range(env.observation_space.shape[0]):
            plt.figure()
            for sample in sample_states[..., i]:
                plt.plot(np.arange(len(sample)), sample, 'r-')
            plt.plot(np.arange(len(y[:, i])), y[:, i], 'b-')
            plt.grid()

        plt.figure()
        for sample in sample_rewards.squeeze():
            plt.plot(np.arange(len(sample)), sample, 'r-')
        plt.plot(np.arange(len(rewards)), rewards, 'b-')
        plt.grid()
        plt.title('rewards')

        plt.show()

if __name__ == '__main__':
    main()
