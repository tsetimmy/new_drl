import numpy as np
import argparse
import gym
import matplotlib.pyplot as plt

import GPy

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
    rewards = np.array(rewards)[..., None]
    print (X_train.shape)
    print (y_train.shape)


    k1 = GPy.kern.RBF(X_train.shape[-1])
    k2 = GPy.kern.Exponential(X_train.shape[-1])
    k3 = GPy.kern.Matern32(X_train.shape[-1])
    #k3 = GPy.kern.Brownian(X_train.shape[-1])
    kernel = k1 + k2 + k3

    m = GPy.models.GPRegression(X_train.copy(), y_train.copy(), kernel.copy())
    n = GPy.models.GPRegression(X_train.copy(), rewards.copy(), kernel.copy())
    m.optimize(messages=True)
    n.optimize(messages=True)

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

        states = np.stack(states, axis=0)
        actions = np.stack(actions, axis=0)

        X_test = np.concatenate([np.stack(states, axis=0), np.stack(actions, axis=0)], axis=-1)
        y_test = np.stack(next_states, axis=0)
        rewards = np.array(rewards)[..., None]

        y_mean, y_cov = m.predict(X_test)

        for i in range(y_test.shape[-1]):
            plt.figure()
            plt.plot(np.arange(len(y_test[:, i])), y_test[:, i])

            y = y_mean[:, i]
            error = np.sqrt(y_cov).squeeze(-1)

            plt.plot(np.arange(len(y)), y)
            plt.fill_between(np.arange(len(y)), y+error, y-error, alpha=.4, color='C1')

            plt.grid()

            #plt.savefig('state'+str(i)+'.pdf')

        y_mean, y_cov = n.predict(X_test)
        plt.figure()
        plt.plot(np.arange(len(rewards[:, 0])), rewards[:, 0])

        y = y_mean[:, 0]
        error = np.sqrt(y_cov).squeeze(-1)

        plt.plot(np.arange(len(y)), y)
        plt.fill_between(np.arange(len(y)), y+error, y-error, alpha=.4, color='C1')

        plt.grid()
        plt.title('rewards')
        #plt.savefig('reward.pdf')

        S = []
        R = []
        n_samples = 50
        state = np.tile(states[0:1], [n_samples, 1])
        #S.append(state.copy())
        for i in range(len(actions)):
            action = np.tile(actions[i:i+1], [n_samples, 1])
            state_action = np.concatenate([state, action], axis=-1)

            mean, cov = m.predict(state_action)
            state = mean + np.sqrt(cov)*np.random.standard_normal(size=mean.shape)
            S.append(state.copy())

            mean, cov = n.predict(state_action)
            reward = mean + np.sqrt(cov)*np.random.standard_normal(size=mean.shape)
            R.append(reward.copy())
        S = np.stack(S, axis=0)
        R = np.stack(R, axis=0)

        for i in range(y_test.shape[-1]):
            plt.figure()
            for j in range(no_samples):
                plt.plot(np.arange(len(S[:, j, i])), S[:, j, i], color='r')
            plt.plot(np.arange(len(y_test[:, i])), y_test[:, i], color='C0')
            plt.grid()
            #plt.savefig('state_traj'+str(i)+'.pdf')

        plt.figure()
        for j in range(no_samples):
            plt.plot(np.arange(len(R[:, j, :])), R[:, j, :], color='r')
        plt.plot(np.arange(len(rewards[:, 0])), rewards[:, 0], color='C0')
        plt.grid()
        plt.title('rewards')
        #plt.savefig('reward_traj.pdf')



        plt.show()
        exit()


                
        
        






    
    

if __name__ == '__main__':
    main()
