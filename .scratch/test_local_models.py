import numpy as np
import gym
import pybullet_envs
import argparse

import sys
sys.path.append('..')
from direct_policy_search.blr_regression2 import RegressionWrapper

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default='AntBulletEnv-v0')
    args = parser.parse_args()

    print(args)

    time_horizon = 100
    num_sample_trajectories_train = 5
    num_sample_trajectories_test = 1

    env = gym.make(args.environment)
    policy = np.random.uniform(low=env.action_space.low, high=env.action_space.high, size=[time_horizon, len(env.action_space.low)])

    trajectories = []
    policies_with_noise = []
    for __ in range(num_sample_trajectories_train + num_sample_trajectories_test):
        policy_with_noise = policy + np.random.normal(scale=.1, size=policy.shape)
        policies_with_noise.append(policy_with_noise)
        env.seed(1)
        state = env.reset()
        done = False

        trajectory = []
        for action in policy_with_noise:
            action_clipped = np.clip(action, env.action_space.low, env.action_space.high)
            next_state, reward, done, _ = env.step(action_clipped)
            trajectory.append(next_state)

            if done:
                print('Done. Break.')
                break
        
        trajectories.append(np.stack(trajectory, axis=0))

    #Assume same dimensions.
    trajectories = np.stack(trajectories, axis=1)
    policies_with_noise = np.stack(policies_with_noise, axis=1)
    X = np.concatenate([trajectories, policies_with_noise], axis=-1)

    trajectories_train = trajectories[:, :num_sample_trajectories_train, :]
    policies_with_noise_train = policies_with_noise[:, :num_sample_trajectories_train, :]
    X_train = X[:, :num_sample_trajectories_train, :]

    trajectories_test = trajectories[:, num_sample_trajectories_train:, :]
    policies_with_noise_test = policies_with_noise[:, num_sample_trajectories_train:, :]
    X_test = X[:, num_sample_trajectories_train:, :]

    basis_dim = 128

    x_train = X_train[0]
    y_train = trajectories_train[1]
    x_test = X_test[0]
    regression_wrappers = [RegressionWrapper(env.observation_space.shape[0]+env.action_space.shape[0], basis_dim, train_hp_iterations=2)
                           for _ in range(env.observation_space.shape[0])]

    for i, rw in zip(range(len(regression_wrappers)), regression_wrappers):
        rw._train_hyperparameters(x_train, y_train[:, i:i+1])
        rw._reset_statistics(x_train, y_train[:, i:i+1])

    for rw in regression_wrappers:
        mu, sigma = rw._predict(x_test)
        print (mu)














    

if __name__ == '__main__':
    main()
