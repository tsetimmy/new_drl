# To fit the dynamics model:
# 5000 optimization steps, each step
# using 100 particles (batch size).
#
# To fit the controller:
# 1000 steps, each step with a batch
# size of 10.
#
# Replay buffer:
# We use a "replay buffer" of finite size
# (the most recent 10 trials), discarding
# older trials of data.
#
# Algorithm 1's main loop
# ... then iteraate Algorithm 1's
# main loop for 100 times.
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.distributions as tfd

import sys
sys.path.append('..')
from bayesian_neural_network_edward.bayesian_neural_network\
     import bayesian_dynamics_model

from prototype8.dmlac.real_env_pendulum import real_env_pendulum_reward

from utils import Memory

class PAI:
    def __init__(self, environment, state_size, action_size, hidden_size, it_tloop, it_dyn,
                 bs_dyn, it_policy, bs_policy, K, T, action_bound_high, action_bound_low,
                 discount_factor, moment_matching=True, scope='pai'):
        self.environment = environment
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size

        self.it_tloop = it_tloop
        self.it_dyn = it_dyn
        self.bs_dyn = bs_dyn
        self.it_policy = it_policy
        self.bs_policy = bs_policy

        self.K = K#Number of particles
        assert self.bs_policy == self.K#Does this have to be true?
        self.T = T#Time horizon

        self.action_bound_high = action_bound_high
        self.action_bound_low = action_bound_low
        self.discount_factor = discount_factor

        self.moment_matching = moment_matching
        self.scope = scope

        self.policy_reuse_vars = None

        # Assertion
        np.testing.assert_array_equal(-self.action_bound_low, self.action_bound_high)

        # Initialize the Bayesian neural network.
        self.bnn = bayesian_dynamics_model(self.state_size + self.action_size, self.state_size)
        self.bnn.initialize_inference(n_iter=self.it_tloop*self.it_dyn*300, n_samples=10)

        # Declare variables and assignment operators for each W_k.
        self.assign_op = []
        for k in range(K):
            self.declare_vars_and_assign_op(scope='W_'+str(k)+'_')

        # True reward model
        self.reward_model = real_env_pendulum_reward()
        rewards = []

        # Predict x_t for t = 1,...,T.
        self.particles = tf.placeholder(shape=[self.K, self.state_size], dtype=tf.float32)
        self.action = self.build_policy(self.particles)
        particles = self.particles
        for t in range(T):
            actions = self.build_policy(particles)
            rewards.append((self.discount_factor ** t) * self.reward_model.build(particles, actions))
            states_actions = tf.concat([particles, actions], axis=-1)
            next_states = []
            for k in range(K):
                W_k = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'W_'+str(k)+'_')
                next_state = self.bnn.build(*([tf.expand_dims(states_actions[k, :], axis=0)] + W_k))
                next_states.append(next_state)
            next_states = tf.concat(next_states, axis=0)

            # Perform moment matching.
            mu, cov = self.mu_and_cov(next_states)
            cov = cov + 5e-5 * np.eye(self.state_size)# To prevent singular matrix
            particles = tfd.MultivariateNormalFullCovariance(loc=mu, covariance_matrix=cov).sample(self.K)

        # Maximize cumulative rewards in horizon T.
        rewards = tf.reduce_sum(tf.stack(rewards, axis=-1), axis=-1)
        self.loss = -tf.reduce_mean(tf.reduce_sum(rewards, axis=-1))
        self.opt = tf.train.AdamOptimizer().minimize(self.loss)

    def mu_and_cov(self, x):
        assert len(x.shape.as_list()) == 2
        # Calculate mean.
        mu = tf.reduce_mean(x, axis=0)

        # Calculate covariance.
        offset = x - tf.expand_dims(mu, axis=0)
        cov = tf.matmul(tf.transpose(offset), offset) /\
              (tf.cast(tf.size(x[:, 0]), dtype=tf.float32) - 1.)

        return mu, cov

    def act(self, sess, states):
        states = np.atleast_2d(states)
        states = np.tile(states, [self.K, 1])
        action = sess.run(self.action, feed_dict={self.particles:states})
        return action[0]

    def train(self, sess, memory):
        if len(memory.mem) < self.bs_dyn or len(memory.mem) < self.K:
            return

        for i in range(self.it_tloop):
            # Train model dynamics.
            print i, 'of', self.it_tloop, 'iterations'
            for _ in range(self.it_dyn):
                batch = np.array(memory.sample(self.bs_dyn))
                states = np.concatenate(batch[:, 0], axis=0)
                actions = np.concatenate(batch[:, 1], axis=0)
                next_states = np.concatenate(batch[:, 3], axis=0)

                info_dict = self.bnn.inference.update({self.bnn.x:np.concatenate([states, actions], axis=-1),
                                                       self.bnn.y_ph:next_states})
                #self.bnn.inference.print_progress(info_dict)

            # Train policy.
            for _ in range(self.it_policy):
                batch2 = np.array(memory.sample(self.K))
                states2 = np.concatenate(batch2[:, 0], axis=0)
                sess.run(self.opt, feed_dict={self.particles:states2})

    def rbf(self, x):
        return tf.exp(-tf.square(x))

    def declare_vars_and_assign_op(self, scope):
        with tf.variable_scope(scope):
            # Declare variables.
            W_0 = tf.Variable(tf.zeros([self.state_size + self.action_size, self.hidden_size]), trainable=False)
            W_1 = tf.Variable(tf.zeros([self.hidden_size, self.hidden_size]), trainable=False)
            W_2 = tf.Variable(tf.zeros([self.hidden_size, self.state_size]), trainable=False)

            b_0 = tf.Variable(tf.zeros([self.hidden_size]), trainable=False)
            b_1 = tf.Variable(tf.zeros([self.hidden_size]), trainable=False)
            b_2 = tf.Variable(tf.zeros([self.state_size]), trainable=False)

        # Variable assignments.
        self.assign_op.append(W_0.assign(self.bnn.qW_0.sample()))
        self.assign_op.append(W_1.assign(self.bnn.qW_1.sample()))
        self.assign_op.append(W_2.assign(self.bnn.qW_2.sample()))

        self.assign_op.append(b_0.assign(self.bnn.qb_0.sample()))
        self.assign_op.append(b_1.assign(self.bnn.qb_1.sample()))
        self.assign_op.append(b_2.assign(self.bnn.qb_2.sample()))

    def build_policy(self, states):
        assert states.shape.as_list() == [self.K, self.state_size]

        # Fully connected layer 1.
        fc1 = slim.fully_connected(states, 256, activation_fn=tf.nn.relu, scope=self.scope+'/fc1',
                                   reuse=self.policy_reuse_vars)

        # Fully conected layer 2.
        fc2 = slim.fully_connected(fc1, 256, activation_fn=tf.nn.relu, scope=self.scope+'/fc2',
                                   reuse=self.policy_reuse_vars)

        # Output layer.
        output = slim.fully_connected(fc2, self.action_size, activation_fn=tf.nn.tanh,
                                      scope=self.scope+'/output', reuse=self.policy_reuse_vars)

        # Apply action bounds.
        action_bound = tf.constant(self.action_bound_high, dtype=tf.float32)
        policy = tf.multiply(output, action_bound)

        # Change flag.
        self.policy_reuse_vars = True

        return policy

def main1():
    pai = PAI(environment='Pendulum-v0', state_size=3, action_size=1, hidden_size=20, it_tloop=100,
          it_dyn=5000, bs_dyn=100, it_policy=1000, bs_policy=10, K=50, T=25, action_bound_low=np.array([-2.]),
          action_bound_high=np.array([2.]), discount_factor=.9)

def main2():
    # Initialize environment.
    import gym
    env = gym.make('Pendulum-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    action_bound_high = env.action_space.high
    action_bound_low = env.action_space.low

    # Initialize agent.
    pai = PAI(environment='Pendulum-v0', state_size=state_size, action_size=action_size, hidden_size=20, it_tloop=100,
          it_dyn=5000, bs_dyn=100, it_policy=1000, bs_policy=50, K=50, T=25, action_bound_low=action_bound_low,
          action_bound_high=action_bound_high, discount_factor=.9)

    # Initialize replay memory
    memory = Memory(400*10)#Data from most recent 10 trials (each trial is 400 time steps long).

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(300):
            total_rewards = 0.
            state = env.reset()
            while True:
                action = pai.act(sess, state)
                next_state, reward, done, _ = env.step(action)
                total_rewards += float(reward)

                # Store tuple in replay memory
                memory.add([np.atleast_2d(state), np.atleast_2d(action), reward, np.atleast_2d(next_state), done])

                # s <- s'
                state = np.copy(next_state)

                if done == True:
                    print 'epoch', epoch, 'total rewards', total_rewards

                    # Train the agent
                    pai.train(sess, memory)
                    break

if __name__ == '__main__':
    main2()
