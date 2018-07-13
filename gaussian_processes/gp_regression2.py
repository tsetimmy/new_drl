import numpy as np
from gp_np import gaussian_process

import sys
sys.path.append('..')
#from custom_environments.environment_state_functions import mountain_car_continuous_state_function
from custom_environments.environment_reward_functions import mountain_car_continuous_reward_function

class gp_model:
    def __init__(self, x_dim, y_dim, state_dim, action_dim, unroll_steps, no_samples, discount_factor):
        assert x_dim == state_dim + action_dim
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.unroll_steps = unroll_steps
        self.no_samples = no_samples
        self.discount_factor = discount_factor

        #TODO:implement the hyperparameter inputs
        #TODO:feed it real data
        self.models = [gaussian_process(self.x_dim, .5, .5, .5, x_train=np.random.uniform(-1., 1., size=[50, self.x_dim]), y_train=np.random.uniform(-1., 1., size=[50, 1])) for i in range(self.y_dim)]

        #Use real reward function
        #Note: mountain car for the time being
        self.reward_fucntion = mountain_car_continuous_reward_function()

        #Neural network initialization
        self.hidden_size = 32
        self.w1 = np.random.normal(size=[self.x_dim + 1, self.hidden_size])
        self.w2 = np.random.normal(size=[self.hidden_size + 1, self.hidden_size])
        self.w3 = np.random.normal(size=[self.hidden_size + 1, self.action_dim])

    def _loss(self, X):
        assert len(X.shape) == 2
        assert X.shape[-1] == self.x_dim
        X = np.copy(X)

        X = np.expand_dims(X, axis=1)
        X = np.tile(X, [1, self.no_samples, 1])
        X = np.reshape(X, [-1, self.x_dim])

        state = np.copy(X)
        for unroll_step in range(self.unroll_steps):
            action = self._forward(state)
            reward = self.reward_function.step_np(state, action)
            #TODO:next state

    def _forward(self, X):
        assert len(X.shape) == 2
        assert X.shape[-1] == self.x_dim
        X = np.copy(X)

        X = self._add_bias(X)

        h1 = self._relu(np.matmul(X, self.w1))
        h1 = self._add_bias(h1)

        h2 = self._relu(np.matmul(h1, self.w2))
        h2 = self._add_bias(h2)

        out = np.tanh(np.matmul(h2, self.w3))
        #TODO:bound the action here

        return out

    def _add_bias(self, X):
        assert len(X.shape) == 2
        return np.concatenate([X, np.ones([len(X), 1])], axis=-1)

    def _relu(self, X):
        return np.maximum(X, 0.)

        










def main():
    model = gp_model(3, 2, 2, 1, 100, 30, .95)
    model._forward(np.random.uniform(size=[4, 3]))

if __name__ == '__main__':
    main()

