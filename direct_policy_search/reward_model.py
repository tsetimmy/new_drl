import numpy as np
import gym

import argparse

from tf_bayesian_model import bayesian_model, hyperparameter_search

'''
class blr_reward_model:
    def __init__(self, x_dim, y_dim, state_dim, action_dim):
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
'''

def main():
    env = gym.make('Pendulum-v0')
    epochs = 2

    data = []
    for epoch in range(epochs):
        state = env.reset()
        while True:
            action = np.random.uniform(env.action_space.low, env.action_space.high, 1)
            next_state, reward, done, _ = env.step(action)
            data.append([state, action, reward, next_state, done])

            state = np.copy(next_state)
            if done:
                break

    states, _, rewards, _, _ = zip(*data)
    states = np.stack(states, axis=0)
    rewards = np.array(rewards)[..., np.newaxis]

if __name__ == '__main__':
    main()
