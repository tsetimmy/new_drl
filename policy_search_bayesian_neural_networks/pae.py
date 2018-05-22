import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.distributions as tfd

import sys
sys.path.append('..')
from prototype8.dmlac.real_env_pendulum import get_next, get_next_state
from bayesian_neural_network_edward.bayesian_neural_network\
     import bayesian_dynamics_model

class PAE:
    def __init__(self, environment, state_size, action_size, hidden_size, iterations,
                 K, T, action_bound_high, action_bound_low, moment_matching=True, scope='pae'):
        self.environment = environment
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size

        self.iterations = iterations

        self.K = K#Number of particles
        self.T = T#Time horizon

        self.action_bound_high = action_bound_high
        self.action_bound_low = action_bound_low

        self.moment_matching = moment_matching
        self.scope = scope

        self.policy_reuse_vars = None

        # Assertion
        np.testing.assert_array_equal(-self.action_bound_low, self.action_bound_high)

        # Initialize the Bayesian neural network.
        self.bnn = bayesian_dynamics_model(self.state_size + self.action_size, self.state_size)
        self.bnn.initialize_inference(n_iter=self.iterations, n_samples=10)

        # Declare variables and assignment operators for each W_k.
        self.assign_op = []
        for k in range(K):
            self.declare_vars_and_assign_op(scope='W_'+str(k))

        # Predict x_t for t = 1,...,T.
        self.particles = tf.placeholder(shape=[self.K, self.state_size], dtype=tf.float32)
        particles = self.particles
        for t in range(T):
            actions = self.build_policy(particles)
            states_actions = tf.concat([particles, actions], axis=-1)
            next_states = []
            for k in range(K):
                W_k = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'W_'+str(k))
                next_state = self.bnn.build(*([tf.expand_dims(states_actions[k, :], axis=0)] + W_k))
                next_states.append(next_state)
            next_states = tf.concat(next_states, axis=0)

            # Perform moment matching.
            mu, cov = self.mu_and_cov(next_states)
            particles = tfd.MultivariateNormalFullCovariance(loc=mu, covariance_matrix=cov).sample(self.K)
            print particles.shape
            exit()

    def mu_and_cov(self, x):
        assert len(x.shape.as_list()) == 2
        # Calculate mean
        mu = tf.reduce_mean(x, axis=0)

        # Calculate covariance
        offset = x - tf.expand_dims(mu, axis=0)
        cov = tf.matmul(tf.transpose(offset), offset) /\
              (tf.cast(tf.size(x[:, 0]), dtype=tf.float32) - 1.)

        return mu, cov

    def train(self, x, y):
        info_dict = self.bnn.inference.update({self.bnn.x:x, self.bnn.y_ph:y})
        self.bnn.inference.print_progress(info_dict)

    def rbf(self, x):
        return tf.exp(-tf.square(x))

    def declare_vars_and_assign_op(self, scope):
        with tf.variable_scope(scope):
            # Declare the variables
            W_0 = tf.Variable(tf.zeros([self.state_size + self.action_size, self.hidden_size]), trainable=False)
            W_1 = tf.Variable(tf.zeros([self.hidden_size, self.hidden_size]), trainable=False)
            W_2 = tf.Variable(tf.zeros([self.hidden_size, self.state_size]), trainable=False)

            b_0 = tf.Variable(tf.zeros([self.hidden_size]), trainable=False)
            b_1 = tf.Variable(tf.zeros([self.hidden_size]), trainable=False)
            b_2 = tf.Variable(tf.zeros([self.state_size]), trainable=False)

        # Variable assignments
        self.assign_op.append(W_0.assign(self.bnn.qW_0.sample()))
        self.assign_op.append(W_1.assign(self.bnn.qW_1.sample()))
        self.assign_op.append(W_2.assign(self.bnn.qW_2.sample()))

        self.assign_op.append(b_0.assign(self.bnn.qb_0.sample()))
        self.assign_op.append(b_1.assign(self.bnn.qb_1.sample()))
        self.assign_op.append(b_2.assign(self.bnn.qb_2.sample()))

    def build_policy(self, states):
        assert states.shape.as_list() == [self.K, self.state_size]

        # Fully connected layer 1
        fc1 = slim.fully_connected(states, 256, activation_fn=tf.nn.relu, scope=self.scope+'/fc1',
                                   reuse=self.policy_reuse_vars)

        # Fully conected layer 2
        fc2 = slim.fully_connected(fc1, 256, activation_fn=tf.nn.relu, scope=self.scope+'/fc2',
                                   reuse=self.policy_reuse_vars)

        # Output layer
        output = slim.fully_connected(fc2, self.action_size, activation_fn=tf.nn.tanh,
                                      scope=self.scope+'/output', reuse=self.policy_reuse_vars)

        #Apply action bounds
        action_bound = tf.constant(self.action_bound_high, dtype=tf.float32)
        policy = tf.multiply(output, action_bound)

        #Change flag
        self.policy_reuse_vars = True

        return policy




def main():
    pae = PAE(environment='Pendulum-v0', state_size=4, action_size=1, hidden_size=20, iterations=10, K=7, T=2,
              action_bound_low=np.array([-2.]), action_bound_high=np.array([2.]))
    
if __name__ == '__main__':
    main()
