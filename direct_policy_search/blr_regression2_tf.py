import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.distributions as tfd
from blr_regression2 import Agent

import warnings

class Agent3(Agent):
    def __init__(self, environment, x_dim, y_dim, state_dim, action_dim, observation_space_low, observation_space_high,
                 action_space_low, action_space_high, unroll_steps, no_samples, discount_factor, rffm_seed=1, basis_dim=256):
        Agent.__init__(self, environment, x_dim, y_dim, state_dim, action_dim, observation_space_low, observation_space_high,
                       action_space_low, action_space_high, unroll_steps, no_samples, discount_factor, rffm_seed, basis_dim)
        del self.hyperstate_dim
        del self.w0
        del self.w1
        del self.w2
        del self.w3
        del self.thetas
        del self.sizes

        self.policy_scope = 'policy_scope'
        self.policy_reuse_vars = None
        self.X = tf.placeholder(shape=[None, self.state_dim], dtype=tf.float64)
        self.random_matrix_tf = tf.constant(self.random_matrix)
        self.bias_tf = tf.constant(self.bias)

        self.hyperparameters = tf.placeholder(shape=[self.state_dim, 4], dtype=tf.float64)
        self.Vn = tf.placeholder(shape=[self.state_dim, self.basis_dim, self.basis_dim], dtype=tf.float64)
        self.wn = tf.placeholder(shape=[self.state_dim, self.basis_dim, 1], dtype=tf.float64)

        rewards = []
        state = self.X
        for unroll_steps in xrange(self.unroll_steps):
            action = self.build_policy(state)

            reward = self.reward_function.build(state, action)
            rewards.append((self.discount_factor**unroll_steps)*reward)

            state_action = tf.concat([state, action], axis=-1)
            
            loc = []
            scale_diag = []
            for i in range(self.state_dim):
                x_omega_plus_bias = tf.matmul(state_action, (1./self.hyperparameters[i, 0])*self.random_matrix) + self.bias
                z = self.hyperparameters[i, 1] * np.sqrt(2./self.basis_dim) * tf.cos(x_omega_plus_bias)
                pred_sigma = self.hyperparameters[i, 2]**2 + tf.reduce_sum(tf.multiply(tf.matmul(z, self.Vn[i]), z), axis=-1, keepdims=True)
                pred_mu = tf.matmul(z, self.wn[i])
                
                scale_diag.append(pred_sigma)
                loc.append(pred_mu)

            loc = tf.concat(loc, axis=-1)
            scale_diag = tf.concat(scale_diag, axis=-1)

            state = tfd.MultivariateNormalDiag(loc=loc, scale_diag=tf.sqrt(scale_diag)).sample()
        rewards = tf.concat(rewards, axis=-1)
        rewards = tf.reduce_sum(rewards, axis=-1)
        rewards = tf.reduce_mean(rewards)
        self.loss = -rewards
        self.opt = tf.train.AdamOptimizer().minimize(self.loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'policy_scope'))

    def _fit(self, X, XXtr, Xytr, hyperparameters, sess):
        warnings.filterwarnings('error')
        assert len(XXtr) == self.state_dim
        assert len(Xytr) == self.state_dim
        assert len(hyperparameters) == self.state_dim

        Vns = []
        wns = []
        for i in range(self.state_dim):
            assert XXtr[i].shape == (self.basis_dim, self.basis_dim)
            assert Xytr[i].shape == (self.basis_dim, 1)
            length_scale, signal_sd, noise_sd, prior_sd = hyperparameters[i]
            tmp = np.linalg.inv((noise_sd/prior_sd)**2*np.eye(self.basis_dim) + XXtr[i])
            Vn = noise_sd**2*tmp
            wn = np.matmul(tmp, Xytr[i])

            Vns.append(Vn)
            wns.append(wn)

        Vns = np.stack(Vns, axis=0)
        wns = np.stack(wns, axis=0)

        opt, loss = sess.run([self.opt, self.loss], feed_dict={self.X:X, self.Vn:Vns, self.wn:wns, self.hyperparameters:hyperparameters})
        print 'loss:', loss
        print 'opt:', opt
        exit()

    def build_policy(self, states):
        assert states.shape.as_list() == [None, self.state_dim]

        #Fully connected layer 1
        fc1 = slim.fully_connected(states, 32, activation_fn=tf.nn.relu, scope=self.policy_scope+'/fc1', reuse=self.policy_reuse_vars)

        #Fully connected layer 2
        fc2 = slim.fully_connected(fc1, 32, activation_fn=tf.nn.relu, scope=self.policy_scope+'/fc2', reuse=self.policy_reuse_vars)

        #Output layer
        output = slim.fully_connected(fc2, self.action_dim, activation_fn=tf.nn.tanh, scope=self.policy_scope+'/output', reuse=self.policy_reuse_vars)

        #Apply action bounds
        np.testing.assert_array_equal(-self.action_space_low, self.action_space_high)
        action_space = tf.constant(self.action_space_high, dtype=tf.float64)
        policy = tf.multiply(output, action_space)

        #Change flag
        self.policy_reuse_vars = True

        return policy




def main():
    import argparse
    import gym
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default='Pendulum-v0')
    parser.add_argument("--unroll-steps", type=int, default=200)
    parser.add_argument("--discount-factor", type=float, default=.995)
    parser.add_argument("--gather-data-epochs", type=int, default=3, help='Epochs for initial data gather.')
    parser.add_argument("--train-hp-iterations", type=int, default=2000*10)
    parser.add_argument("--train-policy-batch-size", type=int, default=30)
    parser.add_argument("--no-samples", type=int, default=1)
    parser.add_argument("--basis-dim", type=int, default=256)
    parser.add_argument("--rffm-seed", type=int, default=1)
    parser.add_argument("--Agent", type=str, default='')
    args = parser.parse_args()

    print args
    from blr_regression2_sans_hyperstate import Agent2

    env = gym.make(args.environment)

    agent = eval('Agent'+args.Agent)(environment=env.spec.id,
                                     x_dim=env.observation_space.shape[0]+env.action_space.shape[0],
                                     y_dim=env.observation_space.shape[0],
                                     state_dim=env.observation_space.shape[0],
                                     action_dim=env.action_space.shape[0],
                                     observation_space_low=env.observation_space.low,
                                     observation_space_high=env.observation_space.high,
                                     action_space_low=env.action_space.low,
                                     action_space_high=env.action_space.high,
                                     unroll_steps=args.unroll_steps,
                                     no_samples=args.no_samples,
                                     discount_factor=args.discount_factor,
                                     rffm_seed=args.rffm_seed,
                                     basis_dim=args.basis_dim)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

if __name__ == '__main__':
    main()

