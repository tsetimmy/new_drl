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

        self.states = tf.placeholder(shape=[3, self.state_dim], dtype=tf.float64)#batch_size=3 for testing
        self.batch_size = tf.shape(self.states)[0]
        self.batch_size = 3

        self.actions = self.build_policy(self.states)

        self.cum_xx = [tf.tile(tf.expand_dims(model.cum_xx_pl, axis=0), [self.batch_size * self.no_samples, 1, 1]) for model in self.models]
        self.cum_xy = [tf.tile(tf.expand_dims(model.cum_xy_pl, axis=0), [self.batch_size * self.no_samples, 1, 1]) for model in self.models]

        # Testing here
        states_tiled = tf.tile(tf.expand_dims(self.states, axis=1), [1, self.no_samples, 1])
        states_tiled_reshape = tf.reshape(states_tiled, shape=[-1, self.state_dim])
        actions = self.build_policy(states_tiled_reshape)

        states_actions = tf.concat([states_tiled_reshape, actions], axis=-1)

        mus, sigmas = zip(*[self.mu_sigma(self.cum_xx[y], self.cum_xy[y], self.models[y].s, self.models[y].noise_sd) for y in range(self.y_dim)])
        bases = [model.approx_rbf_kern_basis(states_actions) for model in self.models]

        mu_pred, sigma_pred = [tf.concat(e, axis=-1) for e in zip(*[self.prediction(mu, sigma, basis, model.noise_sd)
                                                                  for mu, sigma, basis, model in zip(mus, sigmas, bases, self.models)])]

        next_states = tfd.MultivariateNormalDiag(loc=mu_pred, scale_diag=tf.sqrt(sigma_pred)).sample()

        for y in range(self.y_dim):
            self.update_posterior(bases[i], next_states[..., i:i+1], i)

        exit()
        #self.unroll(self.states)

    def unroll(self, seed_states):
        states = tf.expand_dims(seed_states, axis=1)
        states = tf.tile(states, [1, self.no_samples, 1])
        states = tf.reshape(states, [-1, self.state_dim])

    def update_posterior(self, X, y, i):
        X_expanded_dims = tf.expand_dims(X, axis=-1)
        y_expanded_dims = tf.expand_dims(y, axis=-1)
        self.cum_xx[i] += tf.matmul(X_expanded_dims, tf.transpose(X_expanded_dims, perm=[0, 2, 1]))
        self.cum_xy[i] += tf.matmul(X_expanded_dims, y_expanded_dims)

    def prediction(self, mu, sigma, basis, noise_sd):
        basis_expanded_dims = tf.expand_dims(basis, axis=-1)
        mu_pred = tf.matmul(tf.transpose(mu, perm=[0, 2, 1]), basis_expanded_dims)
        sigma_pred = tf.square(noise_sd) + tf.matmul(tf.matmul(tf.transpose(basis_expanded_dims, perm=[0, 2, 1]), sigma), basis_expanded_dims)

        return tf.squeeze(mu_pred, axis=-1), tf.squeeze(sigma_pred, axis=-1)

    def mu_sigma(self, xx, xy, s, noise_sd):

        prior_sigma_inv = tf.matrix_inverse(tf.tile(tf.expand_dims(s*tf.eye(self.no_basis, dtype=tf.float64), axis=0),
                                            [self.batch_size * self.no_samples, 1, 1]))
        sigma = tf.multiply(tf.square(noise_sd), tf.matrix_inverse(tf.multiply(tf.square(noise_sd), prior_sigma_inv) + xx))
        # Assuming that prior mean is zero vector
        mu = tf.multiply(tf.reciprocal(tf.square(noise_sd)), tf.matmul(sigma, xy))
        return mu, sigma

    def build_policy(self, states):
        #assert states.shape.as_list() == [None, self.state_dim]

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
