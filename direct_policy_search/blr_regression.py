import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.distributions as tfd

import gym

from tf_bayesian_model import bayesian_model

import sys
sys.path.append('..')
from prototype8.dmlac.real_env_pendulum import real_env_pendulum_reward

#TODO: feed in the hyperparameters that will be trained by evidence maximization.
class blr_model:
    def __init__(self, x_dim, y_dim, state_dim, action_dim, observation_space_low,
                 observation_space_high, action_bound_low, action_bound_high, unroll_steps,
                 no_samples, no_basis, discount_factor, train_policy_batch_size, train_policy_iterations):
        
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
        self.no_basis = no_basis
        self.discount_factor = discount_factor

        self.train_policy_batch_size = train_policy_batch_size
        self.train_policy_iterations = train_policy_iterations

        self.policy_scope = 'policy_scope'
        self.policy_reuse_vars = None

        self.models = [bayesian_model(self.x_dim, self.observation_space_low, self.observation_space_high,
                                      self.action_bound_low, self.action_bound_high, self.no_basis) for i in range(self.y_dim)]

        self.states = tf.placeholder(shape=[None, self.state_dim], dtype=tf.float64)
        self.batch_size = tf.shape(self.states)[0]
        self.actions = self.build_policy(self.states)

        self.cum_xx = [tf.tile(tf.expand_dims(model.cum_xx_pl, axis=0), [self.batch_size * self.no_samples, 1, 1]) for model in self.models]
        self.cum_xy = [tf.tile(tf.expand_dims(model.cum_xy_pl, axis=0), [self.batch_size * self.no_samples, 1, 1]) for model in self.models]


        self.unroll(self.states)


    def unroll(self, seed_states):
        states = tf.expand_dims(seed_states, axis=1)
        states = tf.tile(states, [1, self.no_samples, 1])
        states = tf.reshape(states, [-1, self.state_dim])
        print states.shape
        exit()


    def build_policy(self, states):
        assert states.shape.as_list() == [None, self.state_dim]

        #Fully connected layer 1
        fc1 = slim.fully_connected(states, 256, activation_fn=tf.nn.relu, scope=self.policy_scope+'/fc1', reuse=self.policy_reuse_vars)

        #Output layer
        output = slim.fully_connected(fc1, self.action_dim, activation_fn=tf.nn.tanh, scope=self.policy_scope+'/output', reuse=self.policy_reuse_vars)

        #Apply action bounds
        action_bound = tf.constant(self.action_bound_high, dtype=tf.float64)
        policy = tf.multiply(output, action_bound)

        #Change flag
        self.policy_reuse_vars = True

        return policy

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--unroll-steps", type=int, default=25)
    parser.add_argument("--no-samples", type=int, default=20)
    parser.add_argument("--no-basis", type=int, default=256)
    parser.add_argument("--discount-factor", type=float, default=.9)
    parser.add_argument("--train-policy-batch-size", type=int, default=32)
    parser.add_argument("--train-policy-iterations", type=int, default=30)
    parser.add_argument("--replay-start-size-epochs", type=int, default=2)
    args = parser.parse_args()
    
    print args

    env = gym.make('Pendulum-v0')
    blr = blr_model(x_dim=env.observation_space.shape[0]+env.action_space.shape[0],
                    y_dim=env.observation_space.shape[0],
                    state_dim=env.observation_space.shape[0],
                    action_dim=env.action_space.shape[0],
                    observation_space_low=env.observation_space.low,
                    observation_space_high=env.observation_space.high,
                    action_bound_low=env.action_space.low,
                    action_bound_high=env.action_space.high,
                    unroll_steps=args.unroll_steps,
                    no_samples=args.no_samples,
                    no_basis=args.no_basis,
                    discount_factor=args.discount_factor,
                    train_policy_batch_size=args.train_policy_batch_size,
                    train_policy_iterations=args.train_policy_iterations)

if __name__ == '__main__':
    main()
