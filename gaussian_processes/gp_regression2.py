import numpy as np
from scipy.optimize import minimize
import tensorflow as tf
from gp_np import gaussian_process
from gp_regression import gp_model as gp_model_tf

import sys
sys.path.append('..')
#from custom_environments.environment_state_functions import mountain_car_continuous_state_function
#from custom_environments.environment_reward_functions import mountain_car_continuous_reward_function
from prototype8.dmlac.real_env_pendulum import real_env_pendulum_reward

import argparse
import gym

class gp_model:
    def __init__(self, x_dim, y_dim, state_dim, action_dim, action_space_low, action_space_high,
                 unroll_steps, no_samples, discount_factor, hyperparameters, x_train, y_train):
        assert x_dim == state_dim + action_dim
        assert len(action_space_low.shape) == 1
        np.testing.assert_equal(-action_space_low, action_space_high)
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_space_low = action_space_low
        self.action_space_high = action_space_high

        self.unroll_steps = unroll_steps
        self.no_samples = no_samples
        self.discount_factor = discount_factor
        self.hyperparameters = hyperparameters
        self.x_train = x_train
        self.y_train = y_train

        self.models = [gaussian_process(self.x_dim, *self.hyperparameters[i], x_train=self.x_train, y_train=self.y_train[..., i:i+1]) for i in range(self.y_dim)]

        #Use real reward function
        #Note: mountain car for the time being
        #self.reward_function = mountain_car_continuous_reward_function()
        self.reward_function = real_env_pendulum_reward()

        #Neural network initialization
        self.hidden_size = 32
        self.w1 = np.random.normal(size=[self.state_dim + 1, self.hidden_size])
        self.w2 = np.random.normal(size=[self.hidden_size + 1, self.hidden_size])
        self.w3 = np.random.normal(size=[self.hidden_size + 1, self.action_dim])

        #print self.w3[-1, -1]
        self.thetas = self._pack([self.w1, self.w2, self.w3])
        #self.thetas[-1] = 0.
        #print self.w3[-1, -1]

        self.sizes = [[self.state_dim + 1, self.hidden_size],
                      [self.hidden_size + 1, self.hidden_size],
                      [self.hidden_size + 1, self.action_dim]]
        w1, w2, w3 = self._unpack(self.thetas, self.sizes)
        np.testing.assert_equal(w1, self.w1)
        np.testing.assert_equal(w2, self.w2)
        np.testing.assert_equal(w3, self.w3)

    def _pack(self, thetas):
        return np.concatenate([theta.flatten() for theta in thetas])

    def _unpack(self, thetas, sizes):
        sidx = 0
        weights = []
        for size in sizes:
            i, j = size
            w = thetas[sidx:sidx+i*j].reshape([i, j])
            sidx += i*j
            weights.append(w)
        return weights

    def _fit(self, X):
        options = {'maxiter': 1, 'disp': True}
        _res = minimize(self._loss, self.thetas, method='powell', args=(X), options=options)
        assert self.thetas.shape == _res.x.shape
        self.thetas = np.copy(_res.x)

    def _loss(self, thetas, X):
        assert len(X.shape) == 2
        assert X.shape[-1] == self.state_dim
        #np.random.seed(0)
        X = np.copy(X)

        X = np.expand_dims(X, axis=1)
        X = np.tile(X, [1, self.no_samples, 1])
        X = np.reshape(X, [-1, self.state_dim])

        rewards = []
        state = np.copy(X)
        for unroll_step in range(self.unroll_steps):
            action = self._forward(thetas, state)
            reward = self.reward_function.build_np(state, action)
            rewards.append((self.discount_factor**unroll_step)*reward)

            mu, sigma = zip(*[model.predict(np.concatenate([state, action], axis=-1)) for model in self.models])

            mu = np.concatenate(mu, axis=-1)
            sigma = np.stack([np.diag(s) for s in sigma], axis=-1)

            state = np.stack([np.random.multivariate_normal(mean=mean, cov=np.diag(cov)) for mean, cov in zip(mu, sigma)], axis=0)

        rewards = np.stack(rewards, axis=-1)
        rewards = np.sum(rewards, axis=-1)
        loss = -np.mean(rewards)
        print loss
        return loss

    def _forward(self, thetas, X):
        assert len(X.shape) == 2
        assert X.shape[-1] == self.state_dim
        X = np.copy(X)

        w1, w2, w3 = self._unpack(thetas, self.sizes)

        X = self._add_bias(X)

        h1 = self._relu(np.matmul(X, w1))
        h1 = self._add_bias(h1)

        h2 = self._relu(np.matmul(h1, w2))
        h2 = self._add_bias(h2)

        out = np.tanh(np.matmul(h2, w3))
        out = out * self.action_space_high#action bounds.

        return out

    def _add_bias(self, X):
        assert len(X.shape) == 2
        return np.concatenate([X, np.ones([len(X), 1])], axis=-1)

    def _relu(self, X):
        return np.maximum(X, 0.)

    def set_training_data(self, x_train, y_train):
        assert x_train.shape[0] == y_train.shape[0]
        assert len(x_train.shape) == 2
        assert len(y_train.shape) == 2
        assert x_train.shape[-1] == self.x_dim
        assert y_train.shape[-1] == self.y_dim

        self.x_train = np.copy(x_train)
        self.y_train = np.copy(y_train)

        for i in range(len(self.models)):
            self.models[i].set_training_data(np.copy(x_train), np.copy(y_train[..., i:i+1]))

    def predict(self, state, action):
        mu, sigma = zip(*[model.predict(np.concatenate([state, action], axis=-1)) for model in self.models])
        mu = np.concatenate(mu, axis=-1)
        sigma = np.stack([np.diag(s) for s in sigma], axis=-1)

        return mu, sigma
#-----------------------------------------------------------------------------------------------#

def gather_data(env, epochs):
    data = []
    for epoch in range(epochs):
        state = env.reset()
        while True:
            action = np.random.uniform(env.action_space.low, env.action_space.high, 1)
            next_state, reward, done, _ = env.step(action)
            data.append([state, action, next_state])
            state = np.copy(next_state)
            if done:
                break
    states, actions, next_states = [np.stack(e, axis=0) for e in zip(*data)]
    return states, actions, next_states

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default='Pendulum-v0')
    parser.add_argument("--unroll-steps", type=int, default=100)
    parser.add_argument("--no-samples", type=int, default=30)
    parser.add_argument("--discount-factor", type=float, default=.95)
    parser.add_argument("--gather-data-epochs", type=int, default=1, help='Epochs for initial data gather.')
    parser.add_argument("--train-hp-iterations", type=int, default=50000)
    parser.add_argument("--train-policy-batch-size", type=int, default=1)
    args = parser.parse_args()

    print args

    env = gym.make(args.environment)

    # Gather data.
    states, actions, next_states = gather_data(env, args.gather_data_epochs)
    states_actions = np.concatenate([states, actions], axis=-1)

    # Train hyperparameters.
    gpm_tf = gp_model_tf (x_dim=env.observation_space.shape[0]+env.action_space.shape[0],
                          y_dim=env.observation_space.shape[0],
                          state_dim=env.observation_space.shape[0],
                          action_dim=env.action_space.shape[0],
                          observation_space_low=env.observation_space.low,
                          observation_space_high=env.observation_space.high,
                          action_bound_low=env.action_space.low,
                          action_bound_high=env.action_space.high,
                          unroll_steps=2,#Not used
                          no_samples=2,#Not used
                          discount_factor=.95,#Not used
                          train_policy_batch_size=2,#Not used
                          train_policy_iterations=2)#Not used
    gpm_tf.set_training_data(np.copy(states_actions), np.copy(next_states))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        gpm_tf.train_hyperparameters(sess, iterations=args.train_hp_iterations)
        hyperparameters = [sess.run([model.length_scale, model.signal_sd, model.noise_sd]) for model in gpm_tf.models]
    del gpm_tf

    # Initialize the model.
    gpm = gp_model(x_dim=env.observation_space.shape[0]+env.action_space.shape[0],
                   y_dim=env.observation_space.shape[0],
                   state_dim=env.observation_space.shape[0],
                   action_dim=env.action_space.shape[0],
                   action_space_low=env.action_space.low,
                   action_space_high=env.action_space.high,
                   unroll_steps=args.unroll_steps,
                   no_samples=args.no_samples,
                   discount_factor=args.discount_factor,
                   hyperparameters=hyperparameters,
                   x_train=np.copy(states_actions),
                   y_train=np.copy(next_states))

    # Quick plotting experiment (for sanity check).
    import matplotlib.pyplot as plt
    no_lines = 50
    states, actions, next_states = gather_data(env, 1)

    seed_state = np.copy(states[0:1, ...])
    seed_state = np.tile(seed_state, [no_lines, 1])

    states = []
    state = np.copy(seed_state)
    for action, i in zip(actions, range(len(actions))):
        print i
        mu, sigma = gpm.predict(state, np.tile(action[np.newaxis, ...], [no_lines, 1]))
        state = np.stack([np.random.multivariate_normal(mean=mean, cov=np.diag(cov)) for mean, cov in zip(mu, sigma)], axis=0)
        states.append(state)
    states = np.stack(states, axis=-1)

    for i in range(env.observation_space.shape[0]):
        plt.subplot(1, env.observation_space.shape[0], i+1)
        for j in range(no_lines):
            plt.plot(np.arange(len(states[j, i, :])), states[j, i, :], color='r')
        plt.plot(np.arange(len(next_states[:, i])), next_states[:, i])
        plt.grid()
    plt.show()

    # Try fitting the model.
    init_states = np.stack([env.reset() for _ in range(args.train_policy_batch_size)], axis=0)
    #rng_state = np.random.get_state()
    gpm._fit(init_states)
    #np.random.set_state(rng_state)

    # Test the model on the environment.
    for _ in range(10):
        total_rewards = 0.
        state = env.reset()
        while True:
            env.render()
            action = gpm._forward(gpm.thetas, state[np.newaxis, ...])[0]
            next_state, reward, done, _ = env.step(action)
            total_rewards += float(reward)
            state = np.copy(next_state)
            if done:
                print total_rewards
                break

if __name__ == '__main__':
    main()

