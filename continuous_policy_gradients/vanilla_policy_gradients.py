import numpy as np
import gym
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.distributions as tfd

import sys
sys.path.append('..')

import uuid

from custom_environments.environment_state_functions import mountain_car_continuous_state_function
from custom_environments.environment_reward_functions import mountain_car_continuous_reward_function

class gaussain_policy_network:
    def __init__(self, state_dim, action_dim, action_space_low, action_space_high, discount_factor):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_space_low = action_space_low
        self.action_space_high = action_space_high
        self.discount_factor = discount_factor

        self.scope = str(uuid.uuid4())
        self.reuse = None

        self.experience = []#Buffer to hold episode tuples.

        self.states = tf.placeholder(shape=[None, self.state_dim], dtype=tf.float64)

 
        self.mu, self.sigma = self.build(self.states)

        self.make_action = tfd.MultivariateNormalDiag(loc=self.mu, scale_diag=self.sigma).sample()

        self.actions = tf.placeholder(shape=[None, self.action_dim], dtype=tf.float64)

        self.log_pi = self.log_gaussian_pdf(self.actions, self.mu, self.sigma)
        self.G = tf.placeholder(shape=[None, 1], dtype=tf.float64)

        self.J = tf.multiply(-self.G, self.log_pi)

        self.loss = tf.reduce_sum(self.J)

        self.opt = tf.train.AdamOptimizer().minimize(self.loss)

        '''
        self.policy_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
        self.policy_gradients = tf.gradients(self.J, self.policy_params)

        self.opt = tf.train.AdamOptimizer().apply_gradients(zip(self.policy_gradients, self.policy_params))
        '''
        '''
        self.all_rewards = []
        self.max_reward_length = 1000000
        '''

    def build(self, states):
        assert states.shape.as_list() == [None, self.state_dim]

        fc1 = slim.fully_connected(states, 128, activation_fn=tf.nn.relu, scope=self.scope+'/fc1', reuse=self.reuse)
        fc2 = slim.fully_connected(fc1, 128, activation_fn=tf.nn.relu, scope=self.scope+'/fc2', reuse=self.reuse)
        output = slim.fully_connected(fc2, 2 * self.action_dim, activation_fn=None, scope=self.scope+'/output', reuse=self.reuse)

        output_reshaped = tf.reshape(output, shape=[-1, self.action_dim, 2])

        mu = output_reshaped[..., 0]
        sigma = tf.exp(output_reshaped[..., 1])

        self.reuse = True

        return mu, sigma

    def log_gaussian_pdf(self, a, mu, sigma):
        return tf.log(tf.maximum(tf.exp(-tf.square(a - mu) / (2. * tf.square(sigma))) / (np.sqrt(2. * np.pi) * sigma), 1e-7))

    def act(self, sess, states):
        actions = sess.run(self.make_action, feed_dict={self.states:np.atleast_2d(states)})[0, ...]
        return np.clip(actions, self.action_space_low, self.action_space_high)

    def train(self, sess):
        states, actions, next_states, rewards, dones = zip(*self.experience)
        states = np.stack(states, axis=0)
        actions = np.stack(actions, axis=0)
        next_states = np.stack(next_states, axis=0)
        rewards = np.array(rewards)

        G = np.copy(rewards)
        for i in reversed(range(len(G) - 1)):
            G[i] += self.discount_factor * G[i + 1]
        
        '''
        self.all_rewards += G.tolist()
        self.all_rewards = self.all_rewards[:self.max_reward_length]
        G -= np.mean(self.all_rewards)
        G /= np.std(self.all_rewards)
        '''

        assert len(states) == len(actions)
        assert len(states) == len(G)
        for _ in range(42):
            idx = np.random.randint(len(states), size=24)
            _, loss = sess.run([self.opt, self.loss], feed_dict={self.states:states[idx], self.actions:actions[idx], self.G:G[..., np.newaxis][idx]})
            #print 'loss:', loss

        self.experience = []

def main2():
    env = gym.make('Pendulum-v0')
    #env = gym.make('MountainCarContinuous-v0')
    #env = gym.make('LunarLanderContinuous-v2')

    '''
    print env.observation_space.low
    print env.observation_space.high
    print env.action_space.low
    print env.action_space.high
    '''
    agent = gaussain_policy_network(env.observation_space.shape[0], env.action_space.shape[0], env.action_space.low, env.action_space.high, .95)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(300):
            total_rewards = 0.
            state = env.reset()

            while True:
                #env.render()
                action = agent.act(sess, state)
                next_state, reward, done, _ = env.step(action)
                total_rewards += float(reward)
                agent.experience.append([state, action, next_state, reward, done])
                state = np.copy(next_state)

                if done == True:
                    print 'epoch:', epoch, 'total_rewards:', total_rewards
                    agent.train(sess)
                    break
if __name__ == '__main__':
    #main()
    main2()

