import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

import sys
sys.path.append('..')
from custom_environments.environment_reward_functions import mountain_car_continuous_reward_function
from custom_environments.environment_state_functions import mountain_car_continuous_state_function

import pickle
import uuid

iterations = 0

class nelder_mead_experiment:
    def __init__(self):

        #self.X = np.linspace(-2., 2., self.batch_size)
        #self.y = np.sin(self.X) + 5e-5 * np.random.randn(self.batch_size)

        #self.Xin = np.concatenate([self.X[..., np.newaxis], np.ones([self.batch_size, 1])], axis=-1)
        self.state_dim = 3
        self.action_dim = 1

        self.load = True

        if self.load == True:
            weights = pickle.load(open('weights_effbf081-2f52-457d-b261-6bbb262b4deb.p', 'rb'))
            self.h1 = np.copy(weights[:3*32].reshape([3, 32]))
            self.h2 = np.copy(weights[3*32:3*32+32*32].reshape([32, 32]))
            self.o = np.copy(weights[3*32+32*32:3*32+32*32+32].reshape([32, 1]))
        else:
            self.h1 = np.random.normal(size=[self.state_dim, 32])
            self.h2 = np.random.normal(size=[32, 32])
            self.o = np.random.normal(size=[32, self.action_dim])

        self.thetas = np.concatenate([self.h1.flatten(), self.h2.flatten(), self.o.flatten()])

        self.uuid = str(uuid.uuid4())
        self.batch_size = 3
        self.unroll_steps = 100*2
        self.discount_factor = .98
        self.reward_function = mountain_car_continuous_reward_function(goal_position=.45)
        self.state_function = mountain_car_continuous_state_function()
        self.it = 0

    def unpack(self, thetas):
        h1 = thetas[:self.state_dim*32].reshape([self.state_dim, 32])
        h2 = thetas[self.state_dim*32:self.state_dim*32+32*32].reshape([32, 32])
        h3 = thetas[self.state_dim*32+32*32:self.state_dim*32+32*32+32].reshape([32, self.action_dim])

        return [h1, h2, h3]

    def relu(self, x):
        return np.maximum(x, 0.)

    def forward(self, X, h1, h2, o):
        layer1 = self.relu(np.matmul(X, h1))
        layer2 = self.relu(np.matmul(layer1, h2))
        out = np.tanh(np.matmul(layer2, o))
        return out

    def add_bias(self, state):
        ones = np.ones([self.batch_size, 1])
        return np.concatenate([state, ones], axis=-1)

    def loss(self, thetas):
        state = np.stack([np.random.uniform(low=-0.6, high=-0.4, size=self.batch_size),
                          np.zeros(self.batch_size)], axis=-1)

        rewards = []
        for unroll_steps in range(self.unroll_steps):
            action = self.forward(self.add_bias(state), *self.unpack(thetas))
            reward = self.reward_function.step_np(state, action)
            rewards.append((self.discount_factor**unroll_steps)*reward)
            state = self.state_function.step_np(state, action)

        self.it += 1
        rewards = np.stack(rewards, axis=-1)
        rewards = np.sum(rewards, axis=-1)
        loss = np.mean(-rewards)
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
        state = np.concatenate([np.atleast_2d(state), np.ones([1, 1])], axis=-1)
        action = self.forward(state, self.h1, self.h2, self.o)
        return action[0]

def main():
    nme = nelder_mead_experiment()
    nme.fit()

def main2():
    import gym
    import pickle
    env = gym.make('MountainCarContinuous-v0')

    nme = nelder_mead_experiment()

    weights = pickle.load(open('weights_eaa034c0-4bb1-425a-82c0-0470a41944b2.p', 'rb'))
    nme.h1 = np.copy(weights[:3*32].reshape([3, 32]))
    nme.h2 = np.copy(weights[3*32:3*32+32*32].reshape([32, 32]))
    nme.o = np.copy(weights[3*32+32*32:3*32+32*32+32].reshape([32, 1]))

    for epoch in range(1000):
        state = env.reset()
        #print state
        total_rewards = 0.
        while True:
            #env.render()
            action = nme.act(state)
            next_state, reward, done, _ = env.step(action)
            total_rewards += float(reward)

            state = np.copy(next_state)

            if done:
                print 'epoch:', epoch, 'total_rewards:', total_rewards
                break

if __name__ == '__main__':
    #main()
    main2()
