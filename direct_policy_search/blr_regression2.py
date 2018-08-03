import numpy as np
from scipy.optimize import minimize
import argparse

import sys
sys.path.append('..')

import tensorflow as tf
from custom_environments.generateANN_env import ANN
#from custom_environments.environment_state_functions import mountain_car_continuous_state_function
from custom_environments.environment_reward_functions import mountain_car_continuous_reward_function
from prototype8.dmlac.real_env_pendulum import real_env_pendulum_reward

from more_gradfree_experiments import posterior
from gaussian_processes.gp_regression2 import unpack
from utils import gather_data, gather_data2

import gym
import pickle

def _basis(X, random_matrix, bias, basis_dim, length_scale, signal_sd):
    x_omega_plus_bias = np.matmul(X, (1./length_scale)*random_matrix) + bias
    z = signal_sd * np.sqrt(2./basis_dim) * np.cos(x_omega_plus_bias)
    return z

class RegressionWrapper:
    def __init__(self, input_dim, basis_dim, length_scale=1., signal_sd=1., noise_sd=5e-4, prior_sd=1., rffm_seed=1, train_hp_iterations=2000, noise_sd_clip_threshold=5e-5):
        self.input_dim = input_dim
        self.basis_dim = basis_dim
        self.length_scale = length_scale
        self.signal_sd = signal_sd
        self.noise_sd = noise_sd
        self.prior_sd = prior_sd
        self.hyperparameters = np.array([self.length_scale, self.signal_sd, self.noise_sd, self.prior_sd])

        self.rffm_seed = rffm_seed
        self.train_hp_iterations = train_hp_iterations
        self.noise_sd_clip_threshold = noise_sd_clip_threshold

        self._init_statistics()

        rng_state = np.random.get_state()
        np.random.seed(self.rffm_seed)

        self.random_matrix = np.random.normal(size=[self.input_dim, self.basis_dim])
        self.bias = np.random.uniform(low=0., high=2.*np.pi, size=[self.basis_dim])

        np.random.set_state(rng_state)

    def _init_statistics(self):
        self.XX = np.zeros([self.basis_dim, self.basis_dim])
        self.Xy = np.zeros([self.basis_dim, 1])

    def _update(self, X, y):
        assert len(X.shape) == 2
        assert len(y.shape) == 2
        assert X.shape[0] == y.shape[0]

        basis = _basis(X, self.random_matrix, self.bias, self.basis_dim, self.length_scale, self.signal_sd)
        self.XX += np.matmul(basis.T, basis)
        self.Xy += np.matmul(basis.T, y)

    def _train_hyperparameters(self, X, y):
        thetas = np.array([self.length_scale, self.signal_sd, self.noise_sd, self.prior_sd])
        options = {'maxiter': self.train_hp_iterations, 'disp': True}
        _res = minimize(self._log_marginal_likelihood, thetas, method='nelder-mead', args=(X, y), options=options)
        self.length_scale, self.signal_sd, self.noise_sd, self.prior_sd = _res.x
        self.noise_sd = np.maximum(self.noise_sd, self.noise_sd_clip_threshold)
        self.hyperparameters = np.array([self.length_scale, self.signal_sd, self.noise_sd, self.prior_sd])
        print self.length_scale, self.signal_sd, self.noise_sd, self.prior_sd

    def _log_marginal_likelihood(self, thetas, X, y):
        try:
            length_scale, signal_sd, noise_sd, prior_sd = thetas
            noise_sd_clipped = np.maximum(noise_sd, self.noise_sd_clip_threshold)

            basis = _basis(X, self.random_matrix, self.bias, self.basis_dim, length_scale, signal_sd)
            N = len(basis.T)
            XX = np.matmul(basis.T, basis)
            Xy = np.matmul(basis.T, y)

            wn, Vn, V0, tmp = posterior(XX, Xy, noise_sd_clipped, prior_sd)

            s1, logdet1 = np.linalg.slogdet(V0)
            s2, logdet2 = np.linalg.slogdet(Vn)
            assert s1 == 1 and s2 == 1

            lml = .5*(-N*np.log(noise_sd_clipped**2) - logdet1 + logdet2 - np.matmul(y.T, y)[0, 0]/noise_sd_clipped**2 + np.matmul(np.matmul(Xy.T, tmp.T), Xy)[0, 0]/noise_sd_clipped**2)
            loss = -lml
            return loss
        except:
            return np.inf

    def _reset_statistics(self, X, y):
        self._init_statistics()
        self._update(X, y)

    def _predict(self, X):
        basis = _basis(X, self.random_matrix, self.bias, self.basis_dim, self.length_scale, self.signal_sd)
        mu, sigma, _, _ = posterior(self.XX, self.Xy, self.noise_sd, self.prior_sd)
        predict_mu = np.matmul(basis, mu)
        predict_sigma = self.noise_sd**2 + np.sum(np.multiply(np.matmul(basis, sigma), basis), axis=-1, keepdims=True)
        return predict_mu, predict_sigma

class Agent:
    def __init__(self, environment, x_dim, y_dim, state_dim, action_dim, observation_space_low, observation_space_high,
                 action_space_low, action_space_high, unroll_steps, no_samples, discount_factor, rffm_seed=1, basis_dim=256):
        assert environment in ['Pendulum-v0', 'MountainCarContinuous-v0']
        assert x_dim == state_dim + action_dim
        assert len(action_space_low.shape) == 1
        np.testing.assert_equal(-action_space_low, action_space_high)
        self.environment = environment
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.observation_space_low = observation_space_low
        self.observation_space_high = observation_space_high
        self.action_space_low = action_space_low
        self.action_space_high = action_space_high

        self.unroll_steps = unroll_steps
        self.no_samples = no_samples
        self.discount_factor = discount_factor
        self.rffm_seed = rffm_seed
        self.basis_dim = basis_dim

        self.count = 0
        self.mod_interval = 500

        #Initialize random matrix
        rng_state = np.random.get_state()
        np.random.seed(self.rffm_seed)
        self.random_matrix = np.random.normal(size=[self.x_dim, self.basis_dim])
        self.bias = np.random.uniform(low=0., high=2.*np.pi, size=[self.basis_dim])
        np.random.set_state(rng_state)

        #Use real reward function
        if self.environment == 'Pendulum-v0':
            #self.reward_function = real_env_pendulum_reward()
            self.reward_function = ANN(self.state_dim+self.action_dim, 1)
            self.placeholders_reward = [tf.placeholder(shape=v.shape, dtype=tf.float64)
                                        for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.reward_function.scope)]
            self.assign_ops0 = [v.assign(pl) for v, pl in zip(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.reward_function.scope),
                                self.placeholders_reward)]
        elif self.environment == 'MountainCarContinuous-v0':
            self.reward_function = mountain_car_continuous_reward_function()

        self.hidden_dim = 32
        self.hyperstate_dim = self.state_dim*self.basis_dim*(self.basis_dim+1)/2+self.state_dim*self.basis_dim
        self.w0 = np.concatenate([np.random.normal(size=[self.hyperstate_dim, self.state_dim]), np.random.uniform(-3e-3, 3e-3, size=[1, self.state_dim])], axis=0)
        self.w1 = np.concatenate([np.random.normal(size=[2*self.state_dim, self.hidden_dim]), np.random.uniform(-3e-3, 3e-3, size=[1, self.hidden_dim])], axis=0)
        self.w2 = np.concatenate([np.random.normal(size=[self.hidden_dim, self.hidden_dim]), np.random.uniform(-3e-3, 3e-3, size=[1, self.hidden_dim])], axis=0)
        self.w3 = np.concatenate([np.random.normal(size=[self.hidden_dim, self.action_dim]), np.random.uniform(-3e-3, 3e-3, size=[1, self.action_dim])], axis=0)

        self.thetas = self._pack([self.w0, self.w1, self.w2, self.w3])

        self.sizes = [[self.hyperstate_dim + 1, self.state_dim],
                      [2*self.state_dim + 1, self.hidden_dim],
                      [self.hidden_dim + 1, self.hidden_dim],
                      [self.hidden_dim + 1, self.action_dim]]

        w0, w1, w2, w3 = self._unpack(self.thetas, self.sizes)
        np.testing.assert_equal(w0, self.w0)
        np.testing.assert_equal(w1, self.w1)
        np.testing.assert_equal(w2, self.w2)
        np.testing.assert_equal(w3, self.w3)

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

    def _forward(self, thetas, X, hyperstate):
        w0, w1, w2, w3 = self._unpack(thetas, self.sizes)

        wn, Vn = hyperstate

        batch_size, state_dim, _, _ = Vn.shape
        indices = np.triu_indices(self.basis_dim, 1)
        for i in range(batch_size):
            for j in range(state_dim):
                Vn[i, j][indices] = np.nan
        Vn = Vn[~np.isnan(Vn)]
        Vn = np.reshape(Vn, [batch_size, state_dim, -1])

        hyperstate = np.concatenate([Vn, np.squeeze(wn, axis=-1)], axis=-1)
        hyperstate = np.reshape(hyperstate, [len(hyperstate), -1])
        hyperstate = self._add_bias(hyperstate)
        hyperstate_embeddding = np.tanh(np.matmul(hyperstate, w0))

        state_hyperstate = np.concatenate([X, hyperstate_embeddding], axis=-1)
        state_hyperstate = self._add_bias(state_hyperstate)

        h1 = np.tanh(np.matmul(state_hyperstate, w1))
        h1 = self._add_bias(h1)

        h2 = np.tanh(np.matmul(h1, w2))
        h2 = self._add_bias(h2)

        out = np.tanh(np.matmul(h2, w3))
        out = out * self.action_space_high#action bounds.

        return out

    def _add_bias(self, X):
        assert len(X.shape) == 2
        return np.concatenate([X, np.ones([len(X), 1])], axis=-1)

    def _relu(self, X):
        return np.maximum(X, 0.)

    def _fit(self, X, XXtr, Xytr, hyperparameters, sess):
        assert len(XXtr) == self.state_dim
        assert len(Xytr) == self.state_dim
        assert len(hyperparameters) == self.state_dim

        A = []
        for i in xrange(self.state_dim):
            _, _, noise_sd, prior_sd = hyperparameters[i]
            V0 = prior_sd**2*np.eye(self.basis_dim)
            noise = noise_sd**2*np.linalg.inv(V0)
            tmp = np.linalg.inv(noise + XXtr[i])
            A.append(tmp)
        A = np.stack(A, axis=0)

        X = np.expand_dims(X, axis=1)
        X = np.tile(X, [1, self.no_samples, 1])
        X = np.reshape(X, [-1, self.state_dim])

        XXtr = np.tile(XXtr[np.newaxis, ...], [len(X), 1, 1, 1])
        Xytr = np.tile(Xytr[np.newaxis, ...], [len(X), 1, 1, 1])
        A = np.tile(A[np.newaxis, ...], [len(X), 1, 1, 1])

        options = {'maxiter': 1, 'disp': True}
        _res = minimize(self._loss, self.thetas, method='powell', args=(X, XXtr, Xytr, A, hyperparameters, sess), options=options)
        assert self.thetas.shape == _res.x.shape
        self.thetas = np.copy(_res.x)

    def _loss(self, thetas, X, XXtr, Xytr, A=[], hyperparameters=None, sess=None):
        rng_state = np.random.get_state()
        np.random.seed(2)

        rewards = []
        state = X
        for unroll_step in xrange(self.unroll_steps):
            Vns = []
            wns = []
            for i in xrange(self.state_dim):
                length_scale, signal_sd, noise_sd, prior_sd = hyperparameters[i]
                Vn = noise_sd**2*A[:, i, ...]
                wn = np.matmul(A[:, i, ...], Xytr[:, i, ...])
                Vns.append(Vn)
                wns.append(wn)

            action = self._forward(thetas, state, hyperstate=[np.stack(wns, axis=1), np.stack(Vns, axis=1)])

            if self.environment == 'Pendulum-v0':
                reward = self.reward_function.build_np(sess, state, action)
            elif self.environment == 'MountainCarContinuous-v0':
                reward = self.reward_function.build_np(state, action)
            rewards.append((self.discount_factor**unroll_step)*reward)

            state_action = np.concatenate([state, action], axis=-1)

            means = []
            covs = []
            bases = []
            for i in xrange(self.state_dim):
                length_scale, signal_sd, noise_sd, prior_sd = hyperparameters[i]
                basis = _basis(state_action, self.random_matrix, self.bias, self.basis_dim, length_scale, signal_sd)
                bases.append(basis)
                basis = np.expand_dims(basis, axis=1)
                
                pred_mu = np.squeeze(np.matmul(basis, wns[i]))
                pred_sigma = noise_sd**2 + np.squeeze(np.matmul(np.matmul(basis, Vns[i]), np.transpose(basis, [0, 2, 1])))

                means.append(pred_mu)
                covs.append(pred_sigma)
            means = np.stack(means, axis=-1)
            covs = np.stack(covs, axis=-1)

            state = np.stack([np.random.multivariate_normal(mean=mean, cov=np.diag(cov)) for mean, cov in zip(means, covs)], axis=0)
            state = np.clip(state, self.observation_space_low, self.observation_space_high)

            bases = np.stack(bases, axis=1)
            bases = np.expand_dims(bases, axis=2)
            bases_transpose = np.transpose(bases, [0, 1, 3, 2])

            XXtr += np.matmul(bases_transpose, bases)
            state_expand_dims = state[..., np.newaxis][..., np.newaxis]
            Xytr += np.matmul(bases_transpose, state_expand_dims)

            tmp = np.matmul(bases, A)
            A -= np.matmul(np.matmul(A, bases_transpose), tmp) /\
                 (1. + np.matmul(tmp, bases_transpose))

        rewards = np.concatenate(rewards, axis=-1)
        rewards = np.sum(rewards, axis=-1)
        loss = -np.mean(rewards)
        np.random.set_state(rng_state)

        if (self.count)%self.mod_interval==0 or\
           (self.count-1)%self.mod_interval==0 or\
           (self.count-2)%self.mod_interval==0 or\
           (self.count-3)%self.mod_interval==0 or\
           (self.count-4)%self.mod_interval==0 or\
           (self.count-5)%self.mod_interval==0 or\
           (self.count-6)%self.mod_interval==0 or\
           (self.count-7)%self.mod_interval==0 or\
           (self.count-8)%self.mod_interval==0 or\
           (self.count-9)%self.mod_interval==0 or\
           (self.count-10)%self.mod_interval==0:
           print 'count:', self.count, 'loss:', loss
        self.count += 1
        return loss

def main_loop():
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default='Pendulum-v0')
    parser.add_argument("--unroll-steps", type=int, default=200)
    parser.add_argument("--discount-factor", type=float, default=.995)
    parser.add_argument("--gather-data-epochs", type=int, default=2, help='Epochs for initial data gather.')
    parser.add_argument("--train-hp-iterations", type=int, default=2000*10)
    parser.add_argument("--train-policy-batch-size", type=int, default=30)
    parser.add_argument("--no-samples", type=int, default=1)
    parser.add_argument("--basis-dim", type=int, default=45)
    parser.add_argument("--rffm-seed", type=int, default=1)
    args = parser.parse_args()

    print args

    env = gym.make(args.environment)

    agent = Agent(environment=env.spec.id,
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
    regression_wrappers = [RegressionWrapper(input_dim=env.observation_space.shape[0]+env.action_space.shape[0],
                                             basis_dim=args.basis_dim,
                                             length_scale=1.,
                                             signal_sd=1.,
                                             noise_sd=5e-4,
                                             prior_sd=1.,
                                             rffm_seed=args.rffm_seed,
                                             train_hp_iterations=args.train_hp_iterations,
                                             noise_sd_clip_threshold=5e-5)
                           for _ in range(env.observation_space.shape[0])]

    flag = False
    data_buffer = gather_data(env, args.gather_data_epochs)

    init_states = np.stack([env.reset() for _ in range(args.train_policy_batch_size)], axis=0)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if args.environment == 'Pendulum-v0':
            weights = pickle.load(open('../custom_environments/weights/pendulum_reward.p', 'rb'))
            sess.run(agent.assign_ops0, feed_dict=dict(zip(agent.placeholders_reward, weights)))
        for epoch in range(1000):
            #Train hyperparameters and update systems model.
            states_actions, next_states = unpack(data_buffer)
            for i in range(env.observation_space.shape[0]):
                if flag == False:
                    regression_wrappers[i]._train_hyperparameters(states_actions, next_states[:, i:i+1])
                    regression_wrappers[i]._reset_statistics(states_actions, next_states[:, i:i+1])
                else:
                    regression_wrappers[i]._update(states_actions, next_states[:, i:i+1])
            if len(data_buffer) >= 3000: flag = True
            if flag: data_buffer = []

            #Fit policy network.
            XX, Xy, hyperparameters = [np.stack(ele, axis=0) for ele in zip(*[[rw.XX, rw.Xy, rw.hyperparameters] for rw in regression_wrappers])]
            agent._fit(init_states, XX, Xy, hyperparameters, sess)

            #Get hyperstate
            wns, Vns, _, _ = zip(*[posterior(rw.XX, rw.Xy, rw.noise_sd, rw.prior_sd) for rw in regression_wrappers])
            hyperstate = [np.stack(ele, axis=0)[np.newaxis, ...] for ele in [wns, Vns]]

            total_rewards = 0.
            state = env.reset()
            while True:
                #env.render()
                action = agent._forward(agent.thetas, state[np.newaxis, ...], hyperstate)[0]
                next_state, reward, done, _ = env.step(action)
                data_buffer.append([state, action, reward, next_state, done])
                total_rewards += float(reward)
                state = np.copy(next_state)
                if done:
                    print 'epoch:', epoch, 'total_rewards:', total_rewards
                    break

def plotting_experiments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default='Pendulum-v0')
    parser.add_argument("--train-hp-iterations", type=int, default=2000)
    args = parser.parse_args()
    print args

    env = gym.make(args.environment)

    predictors = []
    for i in range(env.observation_space.shape[0]):
        predictors.append(RegressionWrapper(input_dim=env.observation_space.shape[0]+env.action_space.shape[0], basis_dim=128*2, length_scale=1.,
                                          signal_sd=1., noise_sd=5e-4, prior_sd=1., rffm_seed=1, train_hp_iterations=args.train_hp_iterations, noise_sd_clip_threshold=5e-5))

    states, actions, next_states = gather_data(env, 5, unpack=True)
    states_actions = np.concatenate([states, actions], axis=-1)

    # Quick plotting experiment (for sanity check).
    import matplotlib.pyplot as plt
    if args.environment == 'Pendulum-v0':
        states2, actions2, next_states2 = gather_data(env, 1, unpack=True)
    elif args.environment == 'MountainCarContinuous-v0':
        from utils import mcc_get_success_policy
        states2, actions2, next_states2 = mcc_get_success_policy(env)
    states_actions2 = np.concatenate([states2, actions2], axis=-1)

    plt.figure()
    for i in range(env.observation_space.shape[0]):
        plt.subplot(2, env.observation_space.shape[0], i+1)

        predictors[i]._train_hyperparameters(states_actions, next_states[:, i:i+1])
        predictors[i]._update(states_actions, next_states[:, i:i+1])
        predict_mu, predict_sigma = predictors[i]._predict(states_actions2)

        plt.plot(np.arange(len(next_states2[:, i:i+1])), next_states2[:, i:i+1])
        plt.errorbar(np.arange(len(predict_mu)), predict_mu, yerr=np.sqrt(predict_sigma), color='m', ecolor='g')
        plt.grid()

    traj = []
    no_lines = 50
    state = np.tile(np.copy(states2[0:1, ...]), [no_lines, 1])
    for a in actions2:
        action = np.tile(a[np.newaxis, ...], [no_lines, 1])
        state_action = np.concatenate([state, action], axis=-1)

        mu_vec = []
        sigma_vec = []
        for i in range(env.observation_space.shape[0]):
            predict_mu, predict_sigma = predictors[i]._predict(state_action)
            mu_vec.append(predict_mu)
            sigma_vec.append(predict_sigma)

        mu_vec = np.concatenate(mu_vec, axis=-1)
        sigma_vec = np.concatenate(sigma_vec, axis=-1)

        state = np.stack([np.random.multivariate_normal(mu, np.diag(sigma)) for mu, sigma in zip(mu_vec, sigma_vec)], axis=0)
        traj.append(np.copy(state))

    traj = np.stack(traj, axis=-1)

    for i in range(env.observation_space.shape[0]):
        plt.subplot(2, env.observation_space.shape[0], env.observation_space.shape[0]+i+1)
        for j in range(no_lines):
            y = traj[j, i, :]
            plt.plot(np.arange(len(y)), y, color='r')

        plt.plot(np.arange(len(next_states2[..., i])), next_states2[..., i])
        plt.grid()

    plt.show()

if __name__ == '__main__':
    #plotting_experiments()
    main_loop()
