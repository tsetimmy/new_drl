import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

import gym

from tf_bayesian_model import bayesian_model

class policy_search_bayesian:
    def __init__(self, state_dim, action_dim, observation_space_low, observation_space_high,
                 action_bound_low, action_bound_high, unroll_length, discount_factor):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.observation_space_low = observation_space_low
        self.observation_space_high = observation_space_high
        self.action_bound_low = action_bound_low
        self.action_bound_high = action_bound_high
        self.unroll_length = unroll_length
        self.discount_factor = discount_factor

        # Make sure bounds are same (assumption can be relaxed later).
        np.testing.assert_array_equal(-self.action_bound_low, self.action_bound_high)
        np.testing.assert_array_equal(-self.observation_space_low, self.observation_space_high)

        # Declare model for state dynamics.
        self.model = bayesian_model(dim=self.state_dim, observation_space_low=self.observation_space_low,
                                    observation_space_high=self.observation_space_high, no_basis=(5**4)+1)

        # Flags.
        self.policy_reuse_vars = None

        # Build computational graph (i.e., unroll policy).



def main():
    env = gym.make('Pendulum-v0')

    psb = policy_search_bayesian(state_dim=env.observation_space.shape[0],
                                 action_dim=env.action_space.shape[0],
                                 observation_space_low=env.observation_space.low,
                                 observation_space_high=env.observation_space.high,
                                 action_bound_low=env.action_space.low,
                                 action_bound_high=env.action_space.high,
                                 unroll_length=20, discount_factor=.9)







    





if __name__ == '__main__':
    main()
