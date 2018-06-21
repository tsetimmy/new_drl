import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.distributions as tfd

import gym

from tf_bayesian_model import bayesian_model, hyperparameter_search

import sys
sys.path.append('..')
from prototype8.dmlac.real_env_pendulum import real_env_pendulum_reward

from utils import Memory

class blr_model:
    def __init__(self, x_dim, y_dim, state_dim, action_dim, observation_space_low,
                 observation_space_high, action_bound_low, action_bound_high, unroll_steps,
                 no_samples, no_basis, discount_factor, train_policy_batch_size, train_policy_iterations,
                 hyperparameters, debugging_plot):
        
        assert x_dim == state_dim + action_dim
        assert len(hyperparameters) == y_dim
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

        self.hyperparameters = hyperparameters
        self.debugging_plot = debugging_plot

        self.policy_scope = 'policy_scope'
        self.policy_reuse_vars = None

        self.models = [bayesian_model(self.x_dim, self.observation_space_low, self.observation_space_high,
                                      self.action_bound_low, self.action_bound_high, self.no_basis,
                                      *self.hyperparameters[i]) for i in range(self.y_dim)]

        self.states = tf.placeholder(shape=[None, self.state_dim], dtype=tf.float64)
        self.batch_size = tf.shape(self.states)[0]
        #self.batch_size = 3

        self.actions = self.build_policy(self.states)

        self.cum_xx = [tf.tile(tf.expand_dims(model.cum_xx_pl, axis=0), [self.batch_size * self.no_samples, 1, 1]) for model in self.models]
        self.cum_xy = [tf.tile(tf.expand_dims(model.cum_xy_pl, axis=0), [self.batch_size * self.no_samples, 1, 1]) for model in self.models]

        states_tiled = tf.tile(tf.expand_dims(self.states, axis=1), [1, self.no_samples, 1])
        states_tiled_reshape = tf.reshape(states_tiled, shape=[-1, self.state_dim])

        self.unroll(states_tiled_reshape)

    def unroll(self, states):
        self.reward_model = real_env_pendulum_reward()#Use true model.
        costs = []
        self.next_states = []
        for unroll_step in range(self.unroll_steps):
            print 'unrolling:', unroll_step
            if self.debugging_plot == True:
                actions = self.build_policy2(states)
            else:
                actions = self.build_policy(states)

            # Reward
            rewards = (self.discount_factor ** unroll_step) * self.reward_model.build(states, actions)
            rewards = tf.reshape(rewards, shape=[-1, self.no_samples, 1])
            costs.append(-rewards)

            states_actions = tf.concat([states, actions], axis=-1)
            bases = [model.approx_rbf_kern_basis(states_actions) for model in self.models]
            mus, sigmas = zip(*[self.mu_sigma(self.cum_xx[y], self.cum_xy[y], self.models[y].s, self.models[y].noise_sd) for y in range(self.y_dim)])

            mu_pred, sigma_pred = [tf.concat(e, axis=-1) for e in zip(*[self.prediction(mu, sigma, basis, model.noise_sd)
                                                                      for mu, sigma, basis, model in zip(mus, sigmas, bases, self.models)])]

            next_states = tfd.MultivariateNormalDiag(loc=mu_pred, scale_diag=tf.sqrt(sigma_pred)).sample()

            if self.debugging_plot == True:
                self.next_states.append(tf.reshape(next_states, shape=[-1, self.no_samples, self.state_dim]))

            '''
            for y in range(self.y_dim):
                self.update_posterior(bases[y], next_states[..., y:y+1], y)
            '''

            states = next_states

        if self.debugging_plot == False:
            costs = tf.concat(costs, axis=-1)
            self.loss = tf.reduce_mean(tf.reduce_sum(tf.reduce_mean(costs, axis=1), axis=-1))
            self.opt = tf.train.AdamOptimizer().minimize(self.loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'policy_scope'))

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

    def update(self, sess, X=None, y=None, memory=None):
        if memory is not None:
            states = np.stack([e[0] for e in memory], axis=0)
            actions = np.stack([e[1] for e in memory], axis=0)
            y = np.stack([e[3] for e in memory], axis=0)
            X = np.concatenate([states, actions], axis=-1)

        for i in range(self.y_dim):
            self.models[i].update(sess, X, y[..., i])

    def act(self, sess, state):
        state = np.atleast_2d(state)
        action = sess.run(self.actions, feed_dict={self.states:state})
        return action[0]

    def train(self, sess, memory):
        feed_dict = {}
        for model in self.models:
            feed_dict[model.cum_xx_pl] = model.cum_xx
            feed_dict[model.cum_xy_pl] = model.cum_xy

        for it in range(self.train_policy_iterations):
            batch = memory.sample(self.train_policy_batch_size)
            states = np.stack([b[0] for b in batch], axis=0)
            feed_dict[self.states] = states

            try:
                loss, _ = sess.run([self.loss, self.opt], feed_dict=feed_dict)
                print 'iteration:', it, 'loss:', loss
            except:
                print 'training step failed.'

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

    def build_policy2(self, states):
        try:
            self.policy
        except:
            self.idx = 0
            self.policy = tf.placeholder(shape=[self.unroll_steps, 1], dtype=tf.float64)

        action = self.policy[self.idx:self.idx+1, ...]
        tile_size = tf.shape(states)[0]

        action_tiled = tf.tile(action, [tile_size, 1])
        self.idx += 1

        return action_tiled

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--unroll-steps", type=int, default=25)
    parser.add_argument("--no-samples", type=int, default=20)
    parser.add_argument("--no-basis", type=int, default=256)
    parser.add_argument("--discount-factor", type=float, default=.9)
    parser.add_argument("--replay-mem-size", type=int, default=1000000)
    parser.add_argument("--train-policy-batch-size", type=int, default=32)
    parser.add_argument("--train-policy-iterations", type=int, default=30)
    parser.add_argument("--replay-start-size-epochs", type=int, default=2)
    args = parser.parse_args()
    
    print args

    env = gym.make('Pendulum-v0')

    # Gather data to train hyperparameters
    data = []
    for _ in range(2):
        state = env.reset()
        while True:
            action = np.random.uniform(env.action_space.low, env.action_space.high, 1)
            next_state, reward, done, _ = env.step(action)
            data.append([state, action, next_state])
            state = np.copy(next_state)
            if done:
                break

    states = np.stack([d[0] for d in data], axis=0)
    actions = np.stack([d[1] for d in data], axis=0)
    next_states = np.stack([d[2] for d in data], axis=0)

    states_actions = np.concatenate([states, actions], axis=-1)

    # Train the hyperparameters
    hs = [hyperparameter_search(dim=env.observation_space.shape[0]+env.action_space.shape[0])
          for _ in range(env.observation_space.shape[0])]
    hyperparameters = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        batch_size = 32
        iterations = 50000
        idxs = [np.random.randint(len(states_actions), size=batch_size) for _ in range(iterations)]
        for i in range(len(hs)):
            hs[i].train_hyperparameters(sess, states_actions, next_states[:, i], idxs)
            hyperparameters.append(sess.run([hs[i].length_scale, hs[i].signal_sd, hs[i].noise_sd]))

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
                    train_policy_iterations=args.train_policy_iterations,
                    hyperparameters=hyperparameters,
                    debugging_plot=False)

    # Initialize the memory
    memory = Memory(args.replay_mem_size)
    memory2 = []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # Update the model with data used from training hyperparameters
        blr.update(sess, states_actions, next_states)
        state = env.reset()
        total_rewards = 0.0
        epoch = 1
        for time_steps in range(30000):
            action = blr.act(sess, state)
            next_state, reward, done, _ = env.step(action)
            total_rewards += float(reward)

            # Append to the batch
            memory.add([state, action, reward, next_state, done])
            memory2.append([state, action, reward, next_state, done])

            # s <- s'
            state = np.copy(next_state)

            if done == True:
                print 'time steps', time_steps, 'epoch', epoch, 'total rewards', total_rewards

                # Update the memory
                blr.update(sess, memory=memory2)

                # Train the policy
                blr.train(sess, memory)

                epoch += 1
                total_rewards = 0.
                state = env.reset()
                memory2 = []

def plotting_experiment():
    import matplotlib.pyplot as plt
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--unroll-steps", type=int, default=25)
    parser.add_argument("--no-samples", type=int, default=20)
    parser.add_argument("--no-basis", type=int, default=256)
    parser.add_argument("--discount-factor", type=float, default=.9)
    parser.add_argument("--replay-mem-size", type=int, default=1000000)
    parser.add_argument("--train-policy-batch-size", type=int, default=32)
    parser.add_argument("--train-policy-iterations", type=int, default=30)
    parser.add_argument("--replay-start-size-epochs", type=int, default=2)
    args = parser.parse_args()
    
    print args

    env = gym.make('Pendulum-v0')

    epochs = 3
    train_size = (epochs - 1) * 200
    policy = []

    # Gather data to train hyperparameters
    data = []
    for _ in range(epochs):
        state = env.reset()
        while True:
            action = np.random.uniform(env.action_space.low, env.action_space.high, 1)
            next_state, reward, done, _ = env.step(action)
            data.append([state, action, next_state])
            state = np.copy(next_state)
            if done:
                break

    states = np.stack([d[0] for d in data], axis=0)
    actions = np.stack([d[1] for d in data], axis=0)
    next_states = np.stack([d[2] for d in data], axis=0)

    states_actions = np.concatenate([states, actions], axis=-1)

    x_train = states_actions[:train_size, ...]
    y_train = next_states[:train_size, ...]
    x_test = states_actions[train_size:, ...]
    y_test = next_states[train_size:, ...]

    # Train the hyperparameters
    hs = [hyperparameter_search(dim=env.observation_space.shape[0]+env.action_space.shape[0])
          for _ in range(env.observation_space.shape[0])]
    hyperparameters = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        batch_size = 32
        iterations = 50000
        idxs = [np.random.randint(len(x_train), size=batch_size) for _ in range(iterations)]
        for i in range(len(hs)):
            hs[i].train_hyperparameters(sess, x_train, y_train[:, i], idxs)
            hyperparameters.append(sess.run([hs[i].length_scale, hs[i].signal_sd, hs[i].noise_sd]))

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
                    train_policy_iterations=args.train_policy_iterations,
                    hyperparameters=hyperparameters,
                    debugging_plot=True)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # Update the model with data used from training hyperparameters
        blr.update(sess, states_actions, next_states)

        # Plotting experiment
        policy = actions[-200:-200+blr.unroll_steps, ...]
        seed_state = x_test[0, :3]

        feed_dict = {}
        for model in blr.models:
            feed_dict[model.cum_xx_pl] = model.cum_xx
            feed_dict[model.cum_xy_pl] = model.cum_xy
        feed_dict[blr.states] = seed_state[np.newaxis, ...]
        feed_dict[blr.policy] = policy

        next_states = sess.run(blr.next_states, feed_dict=feed_dict)

        seed_state = seed_state[np.newaxis, ...][np.newaxis, ...]
        seed_state = np.tile(seed_state, [1, blr.no_samples, 1])

        next_states = np.concatenate(next_states, axis=0)
        next_states = np.concatenate([seed_state, next_states], axis=0)

        for i in range(3):
            plt.subplot(1, 3, i+1)
            for j in range(blr.no_samples):
                print next_states[:, j, i]
                plt.plot(np.arange(len(next_states[:, j, i])), next_states[:, j, i], color='r')
            plt.plot(np.arange(len(x_test[:1+blr.unroll_steps, i])), x_test[:1+blr.unroll_steps, i])
            plt.grid()

        plt.show()

if __name__ == '__main__':
    #main()
    plotting_experiment()
