import numpy as np
import argparse
import gym

import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor

from sklearn.gaussian_process.kernels import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared, Matern

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default='Pendulum-v0')
    args = parser.parse_args()

    print(args)

    no_data_points = 100
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
    print (X_train.shape)
    print (y_train.shape)



    '''
    # Kernel with optimized parameters
    k1 = 50.0**2 * RBF(length_scale=50.0)  # long term smooth rising trend
    k2 = 2.0**2 * RBF(length_scale=100.0) \
        * ExpSineSquared(length_scale=1.0, periodicity=1.0,
                         periodicity_bounds="fixed")  # seasonal component
    # medium term irregularities
    k3 = 0.5**2 * RationalQuadratic(length_scale=1.0, alpha=1.0)
    k4 = 0.1**2 * RBF(length_scale=0.1) \
        + WhiteKernel(noise_level=0.1**2,
                      noise_level_bounds=(1e-3, np.inf))  # noise terms
    #kernel = k1 + k2 + k3 + k4
    kernel = k1 + k3 + k4
    '''
    #kernel = Matern(nu=.5) + RBF()
    kernel =  RBF() + RationalQuadratic() + WhiteKernel()

    gp = GaussianProcessRegressor(kernel=kernel).fit(X_train, y_train)

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
        y_test = np.stack(next_states, axis=0)

        y_mean, y_cov = gp.predict(X_test, return_cov=True)


        for i in range(y_test.shape[-1]):
            plt.figure()
            plt.plot(np.arange(len(y_test[:, i])), y_test[:, i])

            y = y_mean[:, i]
            error = np.sqrt(np.diag(y_cov))

            plt.plot(np.arange(len(y)), y)
            plt.fill_between(np.arange(len(y)), y+error, y-error, alpha=.4, color='C1')

            plt.grid()
        plt.show()
        exit()



if __name__ == '__main__':
    main()
