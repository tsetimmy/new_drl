#Multi-output local regression per time step
import numpy as np
import argparse
import gym
import pybullet_envs
import matplotlib.pyplot as plt

from morw import MultiOutputRegressionWrapper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default='Pendulum-v0')
    parser.add_argument("--time_steps", type=int, default=100)
    parser.add_argument("--train_set_size", type=int, default=5)
    parser.add_argument("--test_set_size", type=int, default=1)
    args = parser.parse_args()

    print args

    env = gym.make(args.environment)
    
    X = []
    y = []
    #for epoch in range(args.train_set_size+args.test_set_size):
    while len(X) != args.train_set_size+args.test_set_size:
        state = env.reset()

        states = []
        actions = []
        next_states = []
        for t in range(args.time_steps):
            action = np.random.uniform(low=env.action_space.low, high=env.action_space.high)
            next_state, reward, done, _ = env.step(action)

            actions.append(action)
            states.append(state)
            next_states.append(next_state)

            state = next_state.copy()
            if done:
                break

        if len(states) != args.time_steps:
            continue

        states = np.stack(states, axis=0)
        actions = np.stack(actions, axis=0)
        next_states = np.stack(next_states, axis=0)
        X.append(np.concatenate([states, actions], axis=-1))
        y.append(next_states)
    X = np.stack(X, axis=0)
    y = np.stack(y, axis=0)

    X_train = X[:args.train_set_size, ...]
    y_train = y[:args.train_set_size, ...]
    X_test = X[-args.test_set_size:, ...]
    y_test = y[-args.test_set_size:, ...]

    morws = [MultiOutputRegressionWrapper(env.observation_space.shape[0]+env.action_space.shape[0], env.observation_space.shape[0], basis_dim=128) for _ in range(args.time_steps)]

    for t in range(args.time_steps):
        morws[t]._train_hyperparameters(X_train[:, t, :], y_train[:, t, :])
        morws[t]._reset_statistics(X_train[:, t, :], y_train[:, t, :])

    mu, sigma = [np.stack(ele, axis=1) for ele in zip(*[morws[t]._predict(X_test[:, t, :]) for t in range(args.time_steps)])]
    assert (sigma >= 0.).all()

    print y_test.shape
    print mu.shape
    print sigma.shape
    plt.figure()
    for i in range(args.test_set_size):
        for j in range(env.observation_space.shape[0]):
            plt.clf()
            plt.plot(np.arange(len(y_test[i, :, j])), y_test[i, :, j], 'b-')
            plt.errorbar(np.arange(len(mu[i, :, j])) , mu[i, :, j], yerr=np.sqrt(sigma[i, :].squeeze()), color='m', ecolor='g')
            plt.grid()
            plt.savefig(str(j)+'.pdf')
    #plt.show()








if __name__ == '__main__':
    main()
