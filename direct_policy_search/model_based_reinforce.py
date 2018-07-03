import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import random
import gym

import sys
sys.path.append('..')
from prototype8.dmlac.real_env_pendulum import real_env_pendulum_state
from prototype8.dmlac.real_env_pendulum import real_env_pendulum_reward

class model_based_reinforce:
    def __init__(self, state_dim, action_dim, action_space_low, action_space_high, discount_factor):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_space_low = action_space_low
        self.action_space_high = action_space_high
        self.discount_factor = discount_factor

        self.states = tf.placeholder(shape=[None, self.state_dim], dtype=tf.float64)
        fc1 = slim.fully_connected(self.states, 128, activation_fn=tf.nn.relu)
        fc2 = slim.fully_connected(fc1, 128, activation_fn=tf.nn.relu)
        self.mu = slim.fully_connected(fc2, self.action_dim, activation_fn=None)
        self.sigma = tf.exp(tf.Variable(np.zeros([1, self.action_dim])))
        
        self.batch_size = tf.shape(self.states)[0]
        self.sigma_tiled = tf.tile(self.sigma, [self.batch_size, 1])
        self.actions_pl = tf.placeholder(shape=[None, self.action_dim], dtype=tf.float64)

        self.G = tf.placeholder(shape=[None, 1], dtype=tf.float64)
        self.pi = ((2*np.pi)**-.5)*tf.div(tf.exp(-.5 * tf.div(tf.square(self.actions_pl - self.mu), tf.square(self.sigma_tiled))), self.sigma_tiled)
        self.log_pi = tf.log(tf.maximum(self.pi, 1e-7))
        print self.log_pi.shape
        print self.log_pi
        exit()





    def train(self, sess, batch, no_samples, unroll_steps):
        states, actions, rewards, _, _ = zip(*batch)
        states = np.stack(states, axis=0)
        actions = np.stack(actions, axis=0)
        rewards = np.array(rewards, np.float64)

    def act(self, sess, states):
        return np.array([0.])

def main():
    #env = gym.make('Pendulum-v0')
    env = gym.make('LunarLanderContinuous-v2')
    no_samples = 20
    unroll_steps = 200
    batch_size = 10
    epoch_experience = []

    agent = model_based_reinforce(env.observation_space.shape[0], env.action_space.shape[0],
                                  env.action_space.low, env.action_space.high, .99)

    for epoch in range(1000):
        state = env.reset()
        total_rewards = 0.
        while True:
            action = agent.act(sess, state)
            next_state, reward, done, _ = env.step(action)
            epoch_experience.append([state, action, reward, next_state, done])
            total_rewards += float(reward)

            state = np.copy(next_state)

            if done:
                print 'epoch:', epoch, 'total_rewards:', total_rewards
                batch = random.sample(epoch_experience, batch_size)
                agent.train(sess, batch, no_samples, unroll_steps)
                epoch_experience = []
                break






if __name__ == '__main__':
    main()
