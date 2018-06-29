import numpy as np
import tensorflow as tf
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

    states, actions, rewards, _, _ = zip(*data)
    states = np.stack(states, axis=0)
    actions = np.stack(actions, axis=0)
    rewards = np.array(rewards)

    states_actions = np.concatenate([states, actions], axis=-1)

    # Train the hyperparameters.
    hs = hyperparameter_search(dim=env.observation_space.shape[0]+env.action_space.shape[0])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        batch_size = 32
        iterations = 1000*50
        idxs = [np.random.randint(len(states_actions), size=batch_size) for _ in range(iterations)]
        hs.train_hyperparameters(sess, states_actions, rewards, idxs)
        hyperparameters = sess.run([hs[i].length_scale, hs[i].signal_sd, hs[i].noise_sd])

    # Initialize the regressor.

    model = bayesian_model(env.observation_space.shape[0]+env.action_space.shape[0], 

if __name__ == '__main__':
    main()
