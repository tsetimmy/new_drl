import gym
import numpy as np
import numpy.random as nr
import random

class IExplorationStrategy:
    def __init__(self, agent, environment, seed=None, **kwargs):
        self.agent = agent
        self.environment = environment
        self.seed = seed

    def action(self, sess, state, exploration_parameter):
        pass


class EpsilonGreedyStrategy(IExplorationStrategy):
    def __init__(self, agent, environment, seed=None):
        IExplorationStrategy.__init__(self, agent, environment, seed)
        random.seed(seed)

    def action(self, sess, state, exploration_probability):
        if random.uniform(0, 1) < exploration_probability:
            action = self.environment.action_space.sample()
        else:
            action = self.agent.action(sess, state)

        action = np.clip(action, self.environment.action_space.low, self.environment.action_space.high)
        return action

class OUStrategy(IExplorationStrategy):
    def __init__(self, agent, environment, seed=None, mu=0, theta=0.15, sigma=0.2):
        IExplorationStrategy.__init__(self, agent, environment, seed)
        self.ou_noise = OUNoise(
            environment.action_space.shape[0], mu=mu, theta=theta, sigma=sigma, seed=seed,
            bounds=self.environment.action_space)

    def action(self, sess, state, noise_scale):
        action = self.agent.action(sess, state)
        noise = noise_scale * self.ou_noise.noise()

        action = action + noise
        action = np.clip(action, self.environment.action_space.low, self.environment.action_space.high)
        return action

class OUStrategy2(IExplorationStrategy):
    def __init__(self, agent, environment, seed=None, mu=0, theta=0.15, sigma=0.3):
        IExplorationStrategy.__init__(self, agent, environment, seed)
        self.ou_noise = OUNoise2(mu=np.ones(environment.action_space.shape[0])*float(mu), theta=theta, sigma=sigma)

    def action(self, sess, state, noise_scale):
        action = self.agent.action(sess, state)
        noise_scale = 1.
        noise = noise_scale * self.ou_noise()

        action = action + noise
        action = np.clip(action, self.environment.action_space.low, self.environment.action_space.high)
        return action

class OUNoise:
    def __init__(self, action_dimension, mu=0, theta=0.15, sigma=0.2, seed=None, bounds=None):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.bounds = bounds
        self.reset()
        nr.seed(seed)

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * nr.randn(len(x))
        self.state = x + dx

        if self.bounds is not None:
            self.state = np.clip(self.state, self.bounds.low, self.bounds.high)

        return self.state

class OUNoise2:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

