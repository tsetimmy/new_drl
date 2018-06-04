import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.distributions as tfd
import gym

import argparse

from tf_bayesian_model import bayesian_model

import sys
sys.path.append('..')
from prototype8.dmlac.real_env_pendulum import real_env_pendulum_reward

from utils import Memory

class policy_search_bayesian:
    def __init__(self, state_dim, action_dim, observation_space_low, observation_space_high,
                 no_basis, action_bound_low, action_bound_high, unroll_steps, no_samples, discount_factor):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.observation_space_low = observation_space_low
        self.observation_space_high = observation_space_high
        self.no_basis = no_basis
        self.action_bound_low = action_bound_low
        self.action_bound_high = action_bound_high
        self.unroll_steps = unroll_steps
        self.no_samples = no_samples
        self.discount_factor = discount_factor

        # Make sure bounds are same (assumption can be relaxed later).
        np.testing.assert_array_equal(-self.action_bound_low, self.action_bound_high)
        np.testing.assert_array_equal(-self.observation_space_low, self.observation_space_high)

        # Declare model for state dynamics (one for each state dimension).
        self.model = [bayesian_model(dim=self.state_dim+self.action_dim,
                                     observation_space_low=np.append(self.observation_space_low, self.action_bound_low),
                                     observation_space_high=np.append(self.observation_space_high, self.action_bound_high),
                                     no_basis=self.no_basis) for _ in range(self.state_dim)]

        # Scope and flags.
        self.policy_scope = 'policy_scope'
        self.policy_reuse_vars = None

        # Declare placeholders.
        self.states = tf.placeholder(shape=[None, self.state_dim], dtype=tf.float64)
        self.batch_size = tf.placeholder(tf.int32, shape=[])

        # -- For testing purposes --
        self.test_policy = tf.placeholder(shape=[self.unroll_steps, self.action_dim], dtype=tf.float64)
        self.test_policy_idx = 0

        # Unroll to get trajectories.
        #self.trajectories = self.unroll2(tf.concat([self.states, self.build_policy(self.states)], axis=-1))
        self.trajectories, self.loss = self.unroll2(self.states)

        # Optimizer
        #self.opt = tf.train.AdamOptimizer().minimize(self.loss)

        '''
        self.trajectories = self.unroll(tf.concat([self.states, self.build_policy(self.states)], axis=-1))

        # Loss.
        self.loss = self.build_loss(self.trajectories)

        # Optimizer
        self.opt = tf.train.AdamOptimizer().minimize(self.loss)
        '''

    def build_loss(self, trajectories):
        no_samples = 2
        self.reward_model = real_env_pendulum_reward()#Use true model.

        costs = []
        for i in range(len(trajectories)):
            samples_standard_normal = tf.random_normal(shape=([self.batch_size] + trajectories[i].shape.as_list()[1:-1] + [no_samples]), dtype=tf.float64)
            #samples_standard_normal = tf.random_normal(shape=tf.shape(tf.placeholder(shape=(trajectories[i].shape.as_list()[:-1] + [no_samples]), dtype=tf.float64)), dtype=tf.float64)

            samples = trajectories[i][..., 0:1] + tf.sqrt(trajectories[i][..., 1:2]) * samples_standard_normal
            samples_transposed = tf.transpose(samples, perm=[0, 2, 1])
            samples_transposed_reshaped = tf.reshape(samples_transposed, shape=[-1, self.state_dim])

            rewards = (self.discount_factor ** i) * self.reward_model.build(samples_transposed_reshaped, self.build_policy(samples_transposed_reshaped))
            rewards_reshaped = tf.reshape(rewards, shape=[-1, no_samples, 1])
            costs.append(-tf.reduce_mean(tf.squeeze(rewards_reshaped, axis=-1), axis=-1))

        loss = tf.reduce_mean(tf.reduce_sum(tf.stack(costs, axis=-1), axis=-1))
        return loss

    def unroll(self, states_actions):
        assert states_actions.shape.as_list() == [None, self.state_dim + self.action_dim]
        trajectories = []
        in_no_samples = 2
        out_no_samples = 2

        # Posterior predictive distributions
        ppd = tf.stack([m.posterior_predictive_distribution(states_actions) for m in self.model], axis=1)
        trajectories.append(ppd)

        # Unroll
        for _ in range(self.unroll_steps - 1):
            in_samples_standard_normal = tf.random_normal(shape=([self.batch_size] + ppd.shape.as_list()[1:-1] + [in_no_samples]), dtype=tf.float64)
            #in_samples_standard_normal = tf.random_normal(shape=tf.shape(tf.placeholder(shape=(ppd.shape.as_list()[:-1] + [in_no_samples]), dtype=tf.float64)), dtype=tf.float64)

            in_samples = ppd[..., 0:1] + tf.sqrt(ppd[..., 1:2]) * in_samples_standard_normal
            in_sampled_states = tf.reshape(tf.transpose(in_samples, perm=[0, 2, 1]), shape=[-1, 3])
            in_sampled_states_actions = tf.concat([in_sampled_states, self.build_policy(in_sampled_states)], axis=-1)

            ppd = tf.stack([m.posterior_predictive_distribution(in_sampled_states_actions) for m in self.model], axis=1)
            ppd_reshaped = tf.reshape(ppd, shape=[-1, out_no_samples, self.state_dim, 2])
            ppd_reshaped_transposed = tf.transpose(ppd_reshaped, perm=[0, 2, 1, 3])

            out_samples_standard_normal = tf.random_normal(shape=([self.batch_size] + ppd_reshaped_transposed.shape.as_list()[1:-1] + [out_no_samples]), dtype=tf.float64)
            #out_samples_standard_normal = tf.random_normal(shape=tf.shape(tf.placeholder(shape=(ppd_reshaped_transposed.shape.as_list()[:-1] + [out_no_samples]), dtype=tf.float64)), dtype=tf.float64)

            out_samples = ppd_reshaped_transposed[..., 0:1] + tf.sqrt(ppd_reshaped_transposed[..., 1:2]) * out_samples_standard_normal
            ppd = tf.concat(tf.nn.moments(tf.reshape(out_samples, shape=[-1, self.state_dim, in_no_samples * out_no_samples]), axes=[-1], keep_dims=True), axis=-1)
            trajectories.append(ppd)

        return trajectories

    # For testing purposes!!
    def get_next_states(self, states_actions):
        from prototype8.dmlac.real_env_pendulum import real_env_pendulum_state
        state_model = real_env_pendulum_state()

        samples = []
        for _ in range(self.no_samples):
            samples.append(state_model.build(states_actions[:, 0:3], states_actions[:, 3:4]))
        samples = tf.stack(samples, axis=0)
        return samples
    # For testing purposes!!

    def unroll2(self, states):
        assert states.shape.as_list() == [None, self.state_dim]
        self.reward_model = real_env_pendulum_reward()#Use true model.
        trajectories = [tf.tile(tf.expand_dims(states, axis=1), [1, self.no_samples, 1])]
        costs = []

        # Action
        self.actions = self.build_policy2(states)

        # Posterior predictive distributions
        rewards = self.reward_model.build(states, self.actions)
        costs.append(-rewards)
        states_actions = tf.concat([states, self.actions], axis=-1)
        ppd = tf.stack([self.model[i].posterior_predictive_distribution2(states_actions, i) for i in range(len(self.model))], axis=1)
        particles = tfd.MultivariateNormalDiag(loc=ppd[..., 0], scale_diag=tf.sqrt(ppd[..., 1])).sample(self.no_samples)
        '''
        particles = self.get_next_states(states_actions)# For testing purposes!!
        '''

        for unroll_step in range(self.unroll_steps - 1):
            print 'unrolling step:', unroll_step
            particles_transposed = tf.transpose(particles, perm=[1, 0, 2])
            trajectories.append(particles_transposed)

            particles_transposed_flattened = tf.reshape(particles_transposed, shape=[-1, self.state_dim])
            actions = self.build_policy2(particles_transposed_flattened)

            rewards = self.reward_model.build(particles_transposed_flattened, actions)
            rewards = tf.reshape(rewards, shape=[-1, self.no_samples, 1])
            rewards = tf.reduce_mean(pow(self.discount_factor, unroll_step + 1) * rewards, axis=1)
            costs.append(-rewards)

            states_actions = tf.concat([particles_transposed_flattened, actions], axis=-1)
            ppd = tf.stack([self.model[i].posterior_predictive_distribution2(states_actions, i) for i in range(len(self.model))], axis=1)
            ppd = tf.reshape(ppd, shape=[-1, self.no_samples, self.state_dim, 2])

            random_selections = np.random.multinomial(self.no_samples, [1./self.no_samples]*self.no_samples)
            particles = []
            for i in range(len(random_selections)):
                if random_selections[i] > 0:
                    particles.append(tfd.MultivariateNormalDiag(loc=ppd[:, i, :, 0], scale_diag=tf.sqrt(ppd[:, i, :, 1])).sample(random_selections[i]))
            particles = tf.concat(particles, axis=0)
            '''
            particles = self.get_next_states(tf.reshape(states_actions, shape=[-1, self.no_samples, self.state_dim + self.action_dim])[:, 0, :])# For testing purposes!!
            '''

        particles_transposed = tf.transpose(particles, perm=[1, 0, 2])
        trajectories.append(particles_transposed)

        costs = tf.reduce_sum(tf.concat(costs, axis=-1), axis=-1)
        loss = tf.reduce_mean(costs)

        return trajectories, loss

    def act(self, sess, states, epoch):
        if epoch <= 1000000:
            actions = np.random.uniform(-2., 2., 1)
        else:
            states = np.atleast_2d(states)
            actions = sess.run(self.actions, feed_dict={self.states:states})[0]
        return actions

    # For testing purposes
    def train2(self, sess, states):
        sess.run(self.opt, feed_dict={self.states:states})

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

    def train_policy(self, sess, states, epoch):
        if epoch <= -1:
            return
        feed_dict = {self.states:states, self.batch_size:states.shape[0]}
        for m in self.model:
            feed_dict[m.prior_mu] = m.mu
            feed_dict[m.prior_sigma] = m.sigma
        for _ in range(1):
            print 'starting opt.',
            sess.run(self.opt, feed_dict=feed_dict)
            print 'finished opt.'

    # For testing purposes
    def build_policy2(self, states):
        assert states.shape.as_list() == [None, self.state_dim]

        actions = tf.matmul(tf.ones_like(states[:, 0:1]), tf.expand_dims(self.test_policy[self.test_policy_idx, :], axis=-1))

        #actions = tf.expand_dims(tf.tile(self.test_policy[self.test_policy_idx, :], [self.batch_size*self.no_samples]), axis=-1)
        self.test_policy_idx += 1
        return actions

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

    def visualize_trajectories2(self, sess):
        import matplotlib.pyplot as plt
        import sys
        sys.path.append('..')
        from prototype8.dmlac.real_env_pendulum import get_next_state
        from tf_bayesian_model import random_seed_state

        dims = len(self.model)

        policy = np.random.uniform(-2., 2., self.unroll_steps)
        seed_state = random_seed_state()

        # Real trajectory.
        state = np.copy(seed_state)
        real_trajectory = [np.expand_dims(np.copy(state), axis=0)]
        for action in policy:
            state = get_next_state(state, action)
            real_trajectory.append(state)
        real_trajectory = np.concatenate(real_trajectory, axis=0)

        # Model trajectory
        feed_dict={self.states:seed_state[np.newaxis, ...], self.batch_size:1, self.test_policy:policy[..., np.newaxis]}
        for m in self.model:
            feed_dict[m.prior_mu] = m.mu
            feed_dict[m.prior_sigma] = m.sigma
        trajectories = sess.run(self.trajectories, feed_dict=feed_dict)
        assert len(trajectories) == self.unroll_steps + 1
        assert len(trajectories) == len(real_trajectory)
        trajectories = np.concatenate(trajectories, axis=0)

        for i in range(dims):
            plt.subplot(1, dims, i + 1)
            plt.grid()
            for j in range(self.no_samples):
                plt.scatter(np.arange(len(trajectories[:, j, i])), trajectories[:, j, i], color='r')
            plt.plot(np.arange(len(real_trajectory[:, i])), real_trajectory[:, i])

        plt.show()

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

    # For testing purposes
    def pretrain(self, sess, pretrain_epochs):
        env = gym.make('Pendulum-v0')

        for epoch in range(pretrain_epochs):
            batch = []
            state = env.reset()

            while True:
                action = np.random.uniform(-2., 2., 1)
                next_state, reward, done, _ = env.step(action)
                batch.append([np.atleast_2d(state), np.atleast_2d(action), np.atleast_2d(next_state)])
                state = np.copy(next_state)
                if done == True:
                    states = np.concatenate([b[0] for b in batch], axis=0)
                    actions = np.concatenate([b[1] for b in batch], axis=0)
                    next_states = np.concatenate([b[2] for b in batch], axis=0)
                    self.train_dynamics(sess, states, actions, next_states)
                    print 'Pretraining dynamics.', 'Epoch', epoch, 'of', pretrain_epochs
                    break

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-samples", type=int, default=50)
    parser.add_argument("--unroll-steps", type=int, default=20)
    parser.add_argument("--replay-mem-size", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--pretrain-epochs", type=int, default=100)
    args = parser.parse_args()
    
    print args

    env = gym.make('Pendulum-v0')

    # Initialize the agent
    psb = policy_search_bayesian(state_dim=env.observation_space.shape[0],
                                 action_dim=env.action_space.shape[0],
                                 observation_space_low=env.observation_space.low,
                                 observation_space_high=env.observation_space.high,
                                 no_basis=(6**4)+1,
                                 action_bound_low=env.action_space.low,
                                 action_bound_high=env.action_space.high,
                                 unroll_steps=args.unroll_steps,
                                 no_samples=args.no_samples,
                                 discount_factor=.9)

    # Initialize the memory
    #memory = Memory(args.replay_mem_size)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #psb.pretrain(sess, args.pretrain_epochs)
        state = env.reset()
        total_rewards = 0.0
        epoch = 1
        batch = []
        for time_steps in range(30000):
            #env.render()
            # Get action and step in environment
            action = psb.act(sess, state, epoch)
            next_state, reward, done, _ = env.step(action)
            total_rewards += float(reward)

            # Append to the batch
            #memory.add([np.atleast_2d(state), np.atleast_2d(action), reward, np.atleast_2d(next_state), done])

            batch.append([state, action, reward, next_state, done])

            '''
            # Training step
            batch = memory.sample(args.batch_size)
            states = np.concatenate([b[0] for b in batch], axis=0)
            #psb.train2(sess, states)
            psb.train_policy(sess, states, epoch)
            '''

            # s <- s'
            state = np.copy(next_state)

            if done == True:
                print 'time steps', time_steps, 'epoch', epoch, 'total rewards', total_rewards
                epoch += 1
                total_rewards = 0.

                B = batch
                states = np.stack([b[0] for b in B], axis=0)
                actions = np.stack([b[1] for b in B], axis=0)
                rewards = np.array([b[2] for b in B])
                next_states = np.stack([b[3] for b in B], axis=0)
                dones = np.array([float(b[4]) for b in B])
                psb.train_dynamics(sess, states, actions, next_states)
                psb.visualize_trajectories2(sess)
                psb.visualize_trajectories(sess)
                #psb.train_policy(sess, states, epoch)

                batch = []
                state = env.reset()

if __name__ == '__main__':
    main()
