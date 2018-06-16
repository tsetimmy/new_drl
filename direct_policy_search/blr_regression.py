import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.distributions as tfd

import gym

import sys
sys.path.append('..')
from prototype8.dmlac.real_env_pendulum import real_env_pendulum_reward

class blr_model:
    def __init__(self, x_dim, y_dim, state_dim, action_dim, observation_space_low,
                 obervation_space_high, action_bound_low, action_bound_high, unroll_steps,
                 no_samples, discount_factor, train_policy_batch_size, train_policy_iterations):
        
        assert x_dim == state_dim + action_dim
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.observation_space_low = observation_space_low
        self.observation_space_high = observation_space_high
        self.action_bound_low = action_bound_low
        self.action_bound_high = action_bound_high

        self.unroll_steps = unroll_steps
        self.no_samples = no_samples
        self.discount_factor = discount_factor

