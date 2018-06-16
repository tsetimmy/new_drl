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
                 obervation_space_high, action_bound_low, action_bound_high, unroll_steps

