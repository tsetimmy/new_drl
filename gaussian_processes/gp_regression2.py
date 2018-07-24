import numpy as np
from scipy.optimize import minimize
#import tensorflow as tf
from gp_np import gaussian_process
#from gp_regression import gp_model as gp_model_tf
from hyperparameter_optimizer import log_marginal_likelihood, batch_sek

import sys
sys.path.append('..')
#from custom_environments.environment_state_functions import mountain_car_continuous_state_function
from custom_environments.environment_reward_functions import mountain_car_continuous_reward_function
from prototype8.dmlac.real_env_pendulum import real_env_pendulum_reward

import argparse
import gym
import random
import copy

class gp_model:
    def __init__(self, environment, x_dim, y_dim, state_dim, action_dim, action_space_low, action_space_high,
                 unroll_steps, no_samples, discount_factor, train_set_size):#, hyperparameters, x_train, y_train):
        assert environment in ['Pendulum-v0', 'MountainCarContinuous-v0']
        assert x_dim == state_dim + action_dim
        assert len(action_space_low.shape) == 1
        np.testing.assert_equal(-action_space_low, action_space_high)
        self.environment = environment
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_space_low = action_space_low
        self.action_space_high = action_space_high

        self.unroll_steps = unroll_steps
        self.no_samples = no_samples
        self.discount_factor = discount_factor
        self.train_set_size = train_set_size
        #self.hyperparameters = hyperparameters#Redundant; remove later.
        #self.x_train = x_train#Redundant; remove later.
        #self.y_train = y_train#Redundant; remove later.
        #self._clip_training_set()

        #self.models = [gaussian_process(self.x_dim, *self.hyperparameters[i], x_train=self.x_train, y_train=self.y_train[..., i:i+1]) for i in range(self.y_dim)]#Redundant; remove later.

        #Use real reward function
        if self.environment == 'Pendulum-v0':
            self.reward_function = real_env_pendulum_reward()
        elif self.environment == 'MountainCarContinuous-v0':
            self.reward_function = mountain_car_continuous_reward_function()

        #Neural network initialization
        self.hidden_dim = 32

        self.w0 = np.concatenate([np.random.normal(size=[self.train_set_size*self.state_dim, self.state_dim]), np.random.uniform(-3e-3, 3e-3, size=[1, self.state_dim])], axis=0)
        self.w1 = np.concatenate([np.random.normal(size=[2*self.state_dim, self.hidden_dim]), np.random.uniform(-3e-3, 3e-3, size=[1, self.hidden_dim])], axis=0)
        self.w2 = np.concatenate([np.random.normal(size=[self.hidden_dim, self.hidden_dim]), np.random.uniform(-3e-3, 3e-3, size=[1, self.hidden_dim])], axis=0)
        self.w3 = np.concatenate([np.random.normal(size=[self.hidden_dim, self.action_dim]), np.random.uniform(-3e-3, 3e-3, size=[1, self.action_dim])], axis=0)

        self.thetas = self._pack([self.w0, self.w1, self.w2, self.w3])

        self.sizes = [[self.train_set_size*self.state_dim + 1, self.state_dim],
                      [2*self.state_dim + 1, self.hidden_dim],
                      [self.hidden_dim + 1, self.hidden_dim],
                      [self.hidden_dim + 1, self.action_dim]]
        w0, w1, w2, w3 = self._unpack(self.thetas, self.sizes)
        np.testing.assert_equal(w0, self.w0)
        np.testing.assert_equal(w1, self.w1)
        np.testing.assert_equal(w2, self.w2)
        np.testing.assert_equal(w3, self.w3)

    def _clip_training_set(self):
        assert len(self.x_train) == len(self.y_train)
        if len(self.x_train) > self.train_set_size:
            idx = np.random.randint(len(self.x_train), size=self.train_set_size)
            self.x_train = np.copy(self.x_train[idx])
            self.y_train = np.copy(self.y_train[idx])

    def _pack(self, thetas):
        return np.concatenate([theta.flatten() for theta in thetas])

    def _unpack(self, thetas, sizes):
        sidx = 0
        weights = []
        for size in sizes:
            i, j = size
            w = thetas[sidx:sidx+i*j].reshape([i, j])
            sidx += i*j
            weights.append(w)
        return weights

    def _fit(self, X, Xt, yt, hyperparameters):
        options = {'maxiter': 1, 'disp': True}
        _res = minimize(self._loss, self.thetas, method='powell', args=(X, Xt, yt, hyperparameters), options=options)
        assert self.thetas.shape == _res.x.shape
        self.thetas = np.copy(_res.x)

    def _loss(self, thetas, X, Xt, yt, hyperparameters):
        assert len(X.shape) == 2
        assert X.shape[-1] == self.state_dim
        assert len(Xt) == len(yt)
        assert len(Xt.shape) == 2
        assert len(yt.shape) == 2
        assert Xt.shape[-1] == self.state_dim + self.action_dim
        assert yt.shape[-1] == self.state_dim
        assert len(hyperparameters) == self.state_dim
        assert len(Xt) == self.train_set_size

        rng_state = np.random.get_state()
        np.random.seed(1)

        n = len(Xt)
        X = np.copy(X)
        Xt = np.copy(Xt)
        yt = np.copy(yt)
        hyperparameters = np.copy(hyperparameters)

        X = np.expand_dims(X, axis=1)
        X = np.tile(X, [1, self.no_samples, 1])
        X = np.reshape(X, [-1, self.state_dim])

        Xt = np.tile(Xt[np.newaxis, ...], [len(X), 1, 1])
        yt = np.tile(yt[np.newaxis, ...], [len(X), 1, 1])

        rewards = []
        state = np.copy(X)
        for unroll_step in range(self.unroll_steps):

            Ls, alphas = self._hyperstate(Xt, yt, hyperparameters)

            action = self._forward(thetas, state, alphas)

            reward = self.reward_function.build_np(state, action)
            rewards.append((self.discount_factor**unroll_step)*reward)

            state_action = np.concatenate([state, action], axis=-1)

            mu_vec = []
            sigma_vec = []
            for i in range(self.state_dim):
                length_scale, signal_sd, noise_sd = hyperparameters[i]
                Xtest = np.expand_dims(state_action, axis=1)
                #Xtest = np.tile(Xtest, [1, 2, 1])
                K = batch_sek(Xt, Xt, signal_sd, length_scale)
                #L = np.linalg.cholesky(K + np.tile(noise_sd**2*np.eye(n)[np.newaxis, ...], [len(K), 1, 1]))
                L = Ls[i]
                v = np.linalg.solve(L, batch_sek(Xt, Xtest, signal_sd, length_scale))
                mu = np.matmul(np.transpose(v, [0, 2, 1]), np.linalg.solve(L, yt[..., i:i+1]))
                sigma = batch_sek(Xtest, Xtest, signal_sd, length_scale) - np.matmul(np.transpose(v, [0, 2, 1]), v)

                mu = np.squeeze(mu, axis=-1)
                sigma = np.stack([np.diag(s) for s in sigma], axis=0)

                mu_vec.append(mu)
                sigma_vec.append(sigma)
            mu_vec = np.concatenate(mu_vec, axis=-1)
            sigma_vec = np.concatenate(sigma_vec, axis=-1)

            state = np.stack([np.random.multivariate_normal(mean=mean, cov=np.diag(cov)) for mean, cov in zip(mu_vec, sigma_vec)], axis=0)

            # Update information state. FIFO strategy.
            Xt = np.concatenate([Xt[:, 1:, :], np.expand_dims(state_action, axis=1)], axis=1)
            yt = np.concatenate([yt[:, 1:, :], np.expand_dims(state, axis=1)], axis=1)


        rewards = np.concatenate(rewards, axis=-1)
        rewards = np.sum(rewards, axis=-1)
        loss = -np.mean(rewards)
        np.random.set_state(rng_state)
        return loss

        '''
        rewards = []
        state = np.copy(X)
        for unroll_step in range(self.unroll_steps):
            action = self._forward(thetas, state)
            reward = self.reward_function.build_np(state, action)
            rewards.append((self.discount_factor**unroll_step)*reward)

            mu, sigma = zip(*[model.predict(np.concatenate([state, action], axis=-1)) for model in self.models])

            mu = np.concatenate(mu, axis=-1)
            sigma = np.stack([np.diag(s) for s in sigma], axis=-1)

            state = np.stack([np.random.multivariate_normal(mean=mean, cov=np.diag(cov)) for mean, cov in zip(mu, sigma)], axis=0)

        rewards = np.stack(rewards, axis=-1)
        rewards = np.sum(rewards, axis=-1)
        loss = -np.mean(rewards)
        print loss
        return loss
        '''

    #TODO: Can this be optimized further?
    def _hyperstate(self, Xt, yt, hyperparameters):
        alphas = []
        Ls = []
        for i in range(self.state_dim):
            length_scale, signal_sd, noise_sd = hyperparameters[i]
            K = batch_sek(Xt, Xt, signal_sd, length_scale)
            L = np.linalg.cholesky(K + np.tile(noise_sd**2*np.eye(self.train_set_size)[np.newaxis, ...], [len(K), 1, 1]))
            alpha = np.linalg.solve(np.transpose(L, [0, 2, 1]), np.linalg.solve(L, yt[..., i:i+1]))
            Ls.append(L)
            alphas.append(alpha)
        alphas = np.concatenate(alphas, axis=-1)
        return Ls, alphas

    def _forward(self, thetas, X, hyperstate):
        assert len(X.shape) == 2
        assert X.shape[-1] == self.state_dim
        X = np.copy(X)

        w0, w1, w2, w3 = self._unpack(thetas, self.sizes)

        hyperstate = np.reshape(hyperstate, [len(hyperstate), -1])
        hyperstate = self._add_bias(hyperstate)
        hyperstate_embeddding = self._relu(np.matmul(hyperstate, w0))

        state_hyperstate = np.concatenate([X, hyperstate_embeddding], axis=-1)
        state_hyperstate = self._add_bias(state_hyperstate)

        h1 = self._relu(np.matmul(state_hyperstate, w1))
        h1 = self._add_bias(h1)

        h2 = self._relu(np.matmul(h1, w2))
        h2 = self._add_bias(h2)

        out = np.tanh(np.matmul(h2, w3))
        out = out * self.action_space_high#action bounds.

        return out

    def _add_bias(self, X):
        assert len(X.shape) == 2
        return np.concatenate([X, np.ones([len(X), 1])], axis=-1)

    def _relu(self, X):
        return np.maximum(X, 0.)

    def set_training_data(self, x_train, y_train):
        assert x_train.shape[0] == y_train.shape[0]
        assert len(x_train.shape) == 2
        assert len(y_train.shape) == 2
        assert x_train.shape[-1] == self.x_dim
        assert y_train.shape[-1] == self.y_dim

        self.x_train = np.copy(x_train)
        self.y_train = np.copy(y_train)

        for i in range(len(self.models)):
            self.models[i].set_training_data(np.copy(x_train), np.copy(y_train[..., i:i+1]))

    def predict(self, state, action):
        mu, sigma = zip(*[model.predict(np.concatenate([state, action], axis=-1)) for model in self.models])
        mu = np.concatenate(mu, axis=-1)
        sigma = np.stack([np.diag(s) for s in sigma], axis=-1)

        return mu, sigma
#-----------------------------------------------------------------------------------------------#

def gather_data(env, epochs):
    data = []
    for epoch in range(epochs):
        state = env.reset()
        while True:
            action = np.random.uniform(env.action_space.low, env.action_space.high, 1)
            next_state, reward, done, _ = env.step(action)
            data.append([state, action, reward, next_state, done])
            state = np.copy(next_state)
            if done:
                break
    return data
    #states, actions, next_states = [np.stack(e, axis=0) for e in zip(*data)]
    #return states, actions, next_states

def unpack(data_buffer):
    states, actions, _, next_states, _ = zip(*data_buffer)
    states, actions, next_states = [np.stack(ele, axis=0) for ele in [states, actions, next_states]]
    states_actions = np.concatenate([states, actions], axis=-1)
    return states_actions, next_states

def distance(batch, datum):
    states_actions, _ = unpack(batch)
    state_action = np.concatenate([datum[0], datum[1]], axis=-1)

    norm_distances = 0.
    for i in xrange(len(states_actions)):
        norm_distances += np.linalg.norm(states_actions[i] - state_action)
    return norm_distances

def select_data(data_buffer, train_set_size):
    random_sample_size = int(train_set_size * .1)
    remainder_size = train_set_size - random_sample_size
    data_buffer_copy = copy.deepcopy(data_buffer)

    batch = [data_buffer_copy.pop(random.randrange(len(data_buffer_copy))) for _ in xrange(random_sample_size)]

    for _ in range(remainder_size):
        distances = []
        for i in range(len(data_buffer_copy)):
            distances.append(distance(batch, data_buffer_copy[i]))

        idx = np.argmax(distances)
        batch.append(data_buffer_copy.pop(idx))

    return unpack(batch)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default='Pendulum-v0')
    parser.add_argument("--unroll-steps", type=int, default=100)
    parser.add_argument("--no-samples", type=int, default=1)
    parser.add_argument("--discount-factor", type=float, default=.95)
    parser.add_argument("--gather-data-epochs", type=int, default=1, help='Epochs for initial data gather.')
    parser.add_argument("--train-hp-iterations", type=int, default=2000)
    parser.add_argument("--train-policy-batch-size", type=int, default=30)
    parser.add_argument("--train-set-size", type=int, default=50)
    args = parser.parse_args()

    print args

    env = gym.make(args.environment)

    # Gather data.
    #states, actions, next_states = gather_data(env, args.gather_data_epochs)
    #states_actions = np.concatenate([states, actions], axis=-1)
    data_buffer = gather_data(env, args.gather_data_epochs)
    states_actions, next_states = unpack(data_buffer)
    '''
    states, actions, _, next_states, _ = zip(*data_buffer)
    states, actions, next_states = [np.stack(ele, axis=0) for ele in [states, actions, next_states]]
    states_actions = np.concatenate([states, actions], axis=-1)
    '''

    '''
    # Train hyperparameters.
    gpm_tf = gp_model_tf (x_dim=env.observation_space.shape[0]+env.action_space.shape[0],
                          y_dim=env.observation_space.shape[0],
                          state_dim=env.observation_space.shape[0],
                          action_dim=env.action_space.shape[0],
                          observation_space_low=env.observation_space.low,
                          observation_space_high=env.observation_space.high,
                          action_bound_low=env.action_space.low,
                          action_bound_high=env.action_space.high,
                          unroll_steps=2,#Not used
                          no_samples=2,#Not used
                          discount_factor=.95,#Not used
                          train_policy_batch_size=2,#Not used
                          train_policy_iterations=2)#Not used
    gpm_tf.set_training_data(np.copy(states_actions), np.copy(next_states))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        gpm_tf.train_hyperparameters(sess, iterations=args.train_hp_iterations)
        hyperparameters = [sess.run([model.length_scale, model.signal_sd, model.noise_sd]) for model in gpm_tf.models]
    del gpm_tf
    '''

    hyperparameters = []
    for i in range(env.observation_space.shape[0]):
        theta0 = np.array([1., 1., 1.])
        options = {'maxiter': args.train_hp_iterations, 'disp': True}
        _res = minimize(log_marginal_likelihood, theta0, method='nelder-mead', args=(states_actions, next_states[:, i:i+1]), options=options)
        hyperparameters.append(_res.x)
    hyperparameters = np.stack(hyperparameters, axis=0)

    # Initialize the model.
    gpm = gp_model(environment=args.environment,
                   x_dim=env.observation_space.shape[0]+env.action_space.shape[0],
                   y_dim=env.observation_space.shape[0],
                   state_dim=env.observation_space.shape[0],
                   action_dim=env.action_space.shape[0],
                   action_space_low=env.action_space.low,
                   action_space_high=env.action_space.high,
                   unroll_steps=args.unroll_steps,
                   no_samples=args.no_samples,
                   discount_factor=args.discount_factor,
                   train_set_size=args.train_set_size)
                   #hyperparameters=hyperparameters,#Redundant; remove later.
                   #x_train=np.copy(states_actions),#Redundant; remove later.
                   #y_train=np.copy(next_states))#Redundant; remove later.

    #x_train = np.copy(states_actions)#Redundant; remove later.
    #y_train = np.copy(next_states)#Redundant; remove later.

#    # Quick plotting experiment (for sanity check).
#    import matplotlib.pyplot as plt
#
#    if args.environment == 'Pendulum-v0':
#        states, actions, next_states = gather_data(env, 1)
#    elif args.environment == 'MountainCarContinuous-v0':
#        import sys
#        sys.path.append('..')
#        from custom_environments.environment_state_functions import mountain_car_continuous_state_function
#        from custom_environments.environment_reward_functions import mountain_car_continuous_reward_function
#
#        state_function = mountain_car_continuous_state_function()
#        reward_function = mountain_car_continuous_reward_function()
#
#        seed_state = np.concatenate([np.random.uniform(low=-.6, high=-.4, size=1), np.zeros(1)])[np.newaxis, ...]
#        i = 0
#        while True:
#            print 'Finding... iteration:', i
#            i += 1
#            states = []
#            next_states = []
#            state = np.copy(seed_state)
#            policy = np.random.uniform(env.action_space.low, env.action_space.high, env._max_episode_steps)
#            found = False
#
#            for a in policy:
#                states.append(np.copy(state))
#                action = np.atleast_2d(a)
#                reward = reward_function.step_np(state, action)
#                next_state = state_function.step_np(state, action)
#                next_states.append(np.copy(next_state))
#                state = np.copy(next_state)
#
#                if reward[0] > 50.: found = True
#
#            if found: break
#
#        states = np.concatenate(states, axis=0)
#        actions = np.copy(policy[..., np.newaxis])
#        next_states = np.concatenate(next_states, axis=0)
#
#    states0 = np.copy(states)
#    actions0 = np.copy(actions)
#    next_states0 = np.copy(next_states)
#
#    count = 0
#    increments = 10
#    for i in range(0, len(x_train), increments):
#        states = np.copy(states0)
#        actions = np.copy(actions0)
#        next_states = np.copy(next_states0)
#        size = np.minimum(i + increments, len(x_train))
#        idx = np.random.randint(len(x_train), size=size)
#        gpm.set_training_data(np.copy(x_train[idx]), np.copy(y_train[idx]))
#
#        count += 1
#        plt.figure(count)
#        plt.clf()
#
#        try:
#            mu, sigma = gpm.predict(states, actions)
#
#            #---#
#            for i in range(env.observation_space.shape[0]):
#                plt.subplot(2, env.observation_space.shape[0], i+1)
#                plt.grid()
#                plt.plot(np.arange(len(next_states)), next_states[:, i])
#                plt.errorbar(np.arange(len(mu)), mu[:, i], yerr=np.sqrt(sigma[:, i]), color='m', ecolor='g')
#
#            #---#
#            no_lines = 50
#
#            seed_state = np.copy(states[0:1, ...])
#            seed_state = np.tile(seed_state, [no_lines, 1])
#
#            states = []
#            state = np.copy(seed_state)
#            for action, i in zip(actions, range(len(actions))):
#                print i
#                mu, sigma = gpm.predict(state, np.tile(action[np.newaxis, ...], [no_lines, 1]))
#                state = np.stack([np.random.multivariate_normal(mean=mean, cov=np.diag(cov)) for mean, cov in zip(mu, sigma)], axis=0)
#                states.append(state)
#            states = np.stack(states, axis=-1)
#
#            for i in range(env.observation_space.shape[0]):
#                plt.subplot(2, env.observation_space.shape[0], env.observation_space.shape[0]+i+1)
#                for j in range(no_lines):
#                    plt.plot(np.arange(len(states[j, i, :])), states[j, i, :], color='r')
#                plt.plot(np.arange(len(next_states[:, i])), next_states[:, i])
#                plt.grid()
#            plt.title(str(size))
#            plt.show(block=False)
#        except Exception as e:
#            print(e)
#    plt.show()
#    exit()

    # Try fitting the model.
    init_states = np.stack([env.reset() for _ in range(args.train_policy_batch_size)], axis=0)
    #gpm._fit(init_states, np.copy(states_actions), np.copy(next_states), hyperparameters)

    # Test the model on the environment.
    for epoch in range(1000):
        Xt, yt = select_data(data_buffer, args.train_set_size)
        _, hyperstate = gpm._hyperstate(Xt[np.newaxis, ...], yt[np.newaxis, ...], hyperparameters)
        gpm._fit(init_states, Xt, yt, hyperparameters)
        data_buffer = []#Clears every episode.
        total_rewards = 0.
        state = env.reset()
        while True:
            #env.render()
            action = gpm._forward(gpm.thetas, state[np.newaxis, ...], hyperstate)[0]
            next_state, reward, done, _ = env.step(action)
            data_buffer.append([state, action, reward, next_state, done])
            total_rewards += float(reward)
            state = np.copy(next_state)
            if done:
                print 'epoch:', epoch, 'total_rewards:', total_rewards
                break

if __name__ == '__main__':
    main()
