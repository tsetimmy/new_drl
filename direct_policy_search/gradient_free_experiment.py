import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

import argparse
import gym

import sys
sys.path.append('..')
from custom_environments.environment_reward_functions import mountain_car_continuous_reward_function
from custom_environments.environment_state_functions import mountain_car_continuous_state_function

from prototype8.dmlac.real_env_pendulum import real_env_pendulum_state, real_env_pendulum_reward

import pickle
import uuid

iterations = 0

class gradient_free_experiment:
    def __init__(self, env, state_dim, action_dim, action_space_high, action_space_low, batch_size, unroll_steps, discount_factor):

        #self.X = np.linspace(-2., 2., self.batch_size)
        #self.y = np.sin(self.X) + 5e-5 * np.random.randn(self.batch_size)

        #self.Xin = np.concatenate([self.X[..., np.newaxis], np.ones([self.batch_size, 1])], axis=-1)
        assert len(action_space_low.shape) == 1
        np.testing.assert_equal(-action_space_low, action_space_high)

        self.action_space_high = action_space_high
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = 32

        self.h1 = np.random.normal(size=[self.state_dim, self.hidden_dim])
        self.h2 = np.concatenate([np.random.normal(size=[self.hidden_dim, self.hidden_dim]), np.random.uniform(-3e-3, 3e-3, size=[1, self.hidden_dim])], axis=0)
        self.o = np.concatenate([np.random.normal(size=[self.hidden_dim, self.action_dim]), np.random.uniform(-3e-3, 3e-3, size=[1, self.action_dim])], axis=0)

        self.thetas = np.concatenate([self.h1.flatten(), self.h2.flatten(), self.o.flatten()])

        self.uuid = str(uuid.uuid4())
        self.batch_size = batch_size
        self.unroll_steps = unroll_steps
        self.discount_factor = discount_factor
        if self.env == 'MountainCarContinuous-v0':
            self.reward_function = mountain_car_continuous_reward_function(goal_position=.45)
            self.state_function = mountain_car_continuous_state_function()
        elif self.env == 'Pendulum-v0':
            self.reward_function = real_env_pendulum_reward()
            self.state_function = real_env_pendulum_state()
        self.it = 0

    def unpack(self, thetas):
        offset = 0
        h1 = thetas[:self.state_dim*self.hidden_dim].reshape([self.state_dim, self.hidden_dim])
        offset += self.state_dim*self.hidden_dim
        h2 = thetas[offset:offset+(self.hidden_dim+1)*self.hidden_dim].reshape([self.hidden_dim+1, self.hidden_dim])
        offset += (self.hidden_dim+1)*self.hidden_dim
        h3 = thetas[offset:offset+(self.hidden_dim+1)*self.action_dim].reshape([self.hidden_dim+1, self.action_dim])

        return [h1, h2, h3]

    def relu(self, x):
        return np.maximum(x, 0.)

    def forward(self, X, h1, h2, o):
        layer1 = self.relu(np.matmul(self.add_bias(X), h1))
        layer2 = self.relu(np.matmul(self.add_bias(layer1), h2))
        out = np.tanh(np.matmul(self.add_bias(layer2), o))
        out = out * self.action_space_high#action bounds.
        return out

    def add_bias(self, state):
        ones = np.ones([len(state), 1])
        return np.concatenate([state, ones], axis=-1)

    def loss(self, thetas):
        if self.env == 'MountainCarContinuous-v0':
            try:
                self.state = np.copy(state)
            except:
                state = np.stack([np.random.uniform(low=-0.6, high=-0.4, size=self.batch_size),
                                  np.zeros(self.batch_size)], axis=-1)
                self.state = np.copy(state)
        elif self.env == 'Pendulum-v0':
            try:
                state = np.copy(self.state)
            except:
                high = np.array([np.pi, 1])
                state = np.random.uniform(low=-high, high=high, size=[self.batch_size, len(high)])
                state = np.stack([np.cos(state[:, 0]), np.sin(state[:, 0]), state[:, 1]], axis=-1)

                self.state = np.copy(state)

        rewards = []
        for unroll_steps in range(self.unroll_steps):
            action = self.forward(state, *self.unpack(thetas))
            reward = self.reward_function.step_np(state, action)
            rewards.append((self.discount_factor**unroll_steps)*reward)
            state = self.state_function.step_np(state, action)

        self.it += 1
        rewards = np.concatenate(rewards, axis=-1)
        rewards = np.sum(rewards, axis=-1)
        loss = np.mean(-rewards)
        print self.it, loss
        return loss

    def callback(self, thetas):
        global iterations
        loss = self.loss(thetas)
        print 'loss:', loss, 'iterations:', iterations, 'self.it:', self.it, 'uuid:', self.uuid

    def fit(self):
        options = {'maxiter': 100000, 'disp': True}

        _res = minimize(self.loss, self.thetas, method='powell', options=options, callback=self.callback)
        self._res = np.copy(_res.x)
        print 'Dumping weights...'
        pickle.dump(self._res, open('weights_'+self.uuid+'.p', 'wb'))
        '''
        res = self.unpack(_res.x)
        self.h1 = np.copy(res[0])
        self.h2 = np.copy(res[1])
        self.o = np.copy(res[2])
        '''

    def act(self, state):
        action = self.forward(state[np.newaxis, ...], self.h1, self.h2, self.o)
        return action[0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default='MountainCarContinuous-v0')
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--unroll-steps", type=int, default=200)
    parser.add_argument("--discount-factor", type=float, default=.999)
    args = parser.parse_args()
    print args

    env = gym.make(args.environment)
    gfe = gradient_free_experiment(args.environment, env.observation_space.shape[0] + 1, env.action_space.shape[0], env.action_space.low, env.action_space.high, args.batch_size, args.unroll_steps, args.discount_factor)
    gfe.fit()

def main2():
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default='MountainCarContinuous-v0')
    parser.add_argument("--path", type=str, default='weights_2bf4ab45-c2c4-4184-8b9b-0e42bdc61c7a.p')
    args = parser.parse_args()
    print args

    env = gym.make(args.environment)
    gfe = gradient_free_experiment(args.environment, env.observation_space.shape[0] + 1, env.action_space.shape[0], env.action_space.low, env.action_space.high)

    weights = pickle.load(open(args.path, 'rb'))

    offset = 0
    gfe.h1 = np.copy(weights[:gfe.state_dim*gfe.hidden_dim].reshape([gfe.state_dim, gfe.hidden_dim]))

    offset += gfe.state_dim*gfe.hidden_dim
    gfe.h2 = np.copy(weights[offset:offset+(gfe.hidden_dim+1)*gfe.hidden_dim].reshape([gfe.hidden_dim+1, gfe.hidden_dim]))

    offset += (gfe.hidden_dim+1)*gfe.hidden_dim
    gfe.o = np.copy(weights[offset:offset+(gfe.hidden_dim+1)*gfe.action_dim].reshape([gfe.hidden_dim+1, gfe.action_dim]))

    '''
    gfe.h1 = np.copy(weights[:gfe.state_dim*32].reshape([gfe.state_dim, 32]))
    gfe.h2 = np.copy(weights[gfe.state_dim*32:gfe.state_dim*32+32*32].reshape([32, 32]))
    gfe.o = np.copy(weights[gfe.state_dim*32+32*32:gfe.state_dim*32+32*32+32].reshape([32, gfe.action_dim]))
    '''

    for epoch in range(1000):
        state = env.reset()
        #print state
        total_rewards = 0.
        while True:
            env.render()
            action = gfe.act(state)
            next_state, reward, done, _ = env.step(action)
            total_rewards += float(reward)

            state = np.copy(next_state)

            if done:
                print 'epoch:', epoch, 'total_rewards:', total_rewards
                break

if __name__ == '__main__':
    main()
    #main2()
