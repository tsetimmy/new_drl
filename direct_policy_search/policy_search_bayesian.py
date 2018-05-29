import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

import gym

from tf_bayesian_model import bayesian_model

class policy_search_bayesian:
    def __init__(self, state_dim, action_dim, observation_space_low, observation_space_high,
                 no_basis, action_bound_low, action_bound_high, unroll_length, discount_factor):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.observation_space_low = observation_space_low
        self.observation_space_high = observation_space_high
        self.no_basis = no_basis
        self.action_bound_low = action_bound_low
        self.action_bound_high = action_bound_high
        self.unroll_length = unroll_length
        self.discount_factor = discount_factor

        # Make sure bounds are same (assumption can be relaxed later).
        np.testing.assert_array_equal(-self.action_bound_low, self.action_bound_high)
        np.testing.assert_array_equal(-self.observation_space_low, self.observation_space_high)

        # Declare model for state dynamics (one for each state dimension).
        self.model = [bayesian_model(dim=self.state_dim+self.action_dim,
                                     observation_space_low=np.append(self.observation_space_low, self.action_bound_low),
                                     observation_space_high=np.append(self.observation_space_high, self.action_bound_high),
                                     no_basis=self.no_basis) for _ in range(self.state_dim)]

        # Flags.
        self.policy_reuse_vars = None

        # Declare placeholders.
        self.states = tf.placeholder(shape=[None, self.state_dim], dtype=tf.float64)

        # Build computational graph (i.e., unroll policy).
    
    def unroll_one_step(self, states):
        assert states.shape.as_list() == [None, 

    def act(self, sess, states):
        states = np.atleast_2d(states)
        # TODO: implement this.
        return np.random.uniform(-2., 2., 1)

    def train_dynamics(self, sess, states, actions, next_states):
        if states.ndim == 1:
            states = np.expand_dims(states, axis=0)
        if actions.ndim == 1:
            actions = np.expand_dims(actions, axis=0)
        if next_states.ndim == 1:
            next_states = np.expand_dims(next_states, axis=0)

        states_actions = np.concatenate([states, actions], axis=-1)
        for i in range(self.state_dim):
            self.model[i].update(sess, states_actions, next_states[:, i])

    def train_policy(self, sess):
        pass

    def build_policy(self, states_actions):
        assert states_actions.shape.as_list() == [None, self.state_dim + self.action_dim]

        #Fully connected layer 1
        fc1 = slim.fully_connected(states, 256, activation_fn=tf.nn.relu, scope=self.scope+'/fc1', reuse=self.policy_reuse_vars)

        #Output layer
        output = slim.fully_connected(fc1, self.action_dim, activation_fn=tf.nn.tanh, scope=self.scope+'/output', reuse=self.policy_reuse_vars)

        #Apply action bounds
        action_bound = tf.constant(self.action_bound_high, dtype=tf.float64)
        policy = tf.multiply(output, action_bound)

        #Change flag
        self.policy_reuse_vars = True

        return policy

    def visualize_trajectories(self, sess):
        import matplotlib.pyplot as plt
        import sys
        sys.path.append('..')
        from prototype8.dmlac.real_env_pendulum import get_next_state
        from tf_bayesian_model import random_seed_state

        T = 100#Time horizon
        no_lines = 50
        dims = len(self.model)

        policy = np.random.uniform(-2., 2., T)
        seed_state = random_seed_state()
        lines = [np.random.multivariate_normal(np.squeeze(model.mu, axis=-1), model.sigma, no_lines)
                 for model in self.model]

        # Real trajectory.
        state = np.copy(seed_state)
        real_trajectory = [np.expand_dims(np.copy(state), axis=0)]
        for action in policy:
            state = get_next_state(state, action)
            real_trajectory.append(state)
        real_trajectory = np.concatenate(real_trajectory, axis=0)

        # Model trajectory.
        state = np.concatenate([np.expand_dims(seed_state, axis=0)] * no_lines, axis=0)
        model_trajectory = [state]
        for action in policy:
            state_action = np.concatenate([state, np.expand_dims(np.array([action] * no_lines), axis=-1)], axis=-1)
            basis = sess.run(self.model[0].X_basis, feed_dict={self.model[0].X:state_action})
            state = np.concatenate([np.sum(basis * line, axis=-1, keepdims=True) for line in lines], axis=-1)
            model_trajectory.append(state)
        model_trajectory = np.stack(model_trajectory, axis=0)

        # Plot.
        for i in range(dims):
            plt.subplot(1, dims, i + 1)
            plt.grid()
            for j in range(no_lines):
                plt.plot(np.arange(len(model_trajectory)), model_trajectory[:, j, i], color='r')
            plt.plot(np.arange(len(real_trajectory)), real_trajectory[:, i])
        plt.show()

def main():
    env = gym.make('Pendulum-v0')

    psb = policy_search_bayesian(state_dim=env.observation_space.shape[0],
                                 action_dim=env.action_space.shape[0],
                                 observation_space_low=env.observation_space.low,
                                 observation_space_high=env.observation_space.high,
                                 no_basis=(6**4)+1,
                                 action_bound_low=env.action_space.low,
                                 action_bound_high=env.action_space.high,
                                 unroll_length=20, discount_factor=.9)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        state = env.reset()
        total_rewards = 0.0
        epoch = 1
        batch = []
        for time_steps in range(30000):
            # Get action and step in environment
            action = psb.act(sess, state)
            next_state, reward, done, _ = env.step(action)
            total_rewards += float(reward)

            # Append to the batch
            batch.append([state, action, reward, next_state, done])

            # s <- s'
            state = np.copy(next_state)

            if done == True:
                print 'time steps', time_steps, 'epoch', epoch, 'total rewards', total_rewards
                epoch += 1
                total_rewards = 0.

                states = np.stack([b[0] for b in batch], axis=0)
                actions = np.stack([b[1] for b in batch], axis=0)
                rewards = np.array([b[2] for b in batch])
                next_states = np.stack([b[3] for b in batch], axis=0)
                dones = np.array([float(b[4]) for b in batch])
                psb.train_dynamics(sess, states, actions, next_states)
                psb.visualize_trajectories(sess)

                state = env.reset()

if __name__ == '__main__':
    main()
