import numpy as np
import scipy
from scipy.optimize import minimize
import scipy.linalg as spla
import argparse

import sys

import warnings

import cma
from choldate import cholupdate

from kusanagi.shell import experiment_utils, cartpole, double_cartpole, pendulum
from functools import partial

def _basis(X, random_matrix, bias, basis_dim, length_scale, signal_sd):
    x_omega_plus_bias = np.matmul(X, (1./length_scale)*random_matrix) + bias
    z = signal_sd * np.sqrt(2./basis_dim) * np.cos(x_omega_plus_bias)
    return z

class RegressionWrapper:
    def __init__(self, input_dim, basis_dim, length_scale=1., signal_sd=1., noise_sd=5e-4, prior_sd=1., rffm_seed=1, train_hp_iterations=2000, matern_param=np.inf):
        self.input_dim = input_dim
        self.basis_dim = basis_dim
        self.length_scale = length_scale
        self.signal_sd = signal_sd
        self.noise_sd = noise_sd
        self.prior_sd = prior_sd
        self.hyperparameters = np.array([self.length_scale, self.signal_sd, self.noise_sd, self.prior_sd])
        self.c = 1e-6
        #self.c = 0.

        self.rffm_seed = rffm_seed
        self.train_hp_iterations = train_hp_iterations

        self._init_statistics()

        rng_state = np.random.get_state()
        np.random.seed(self.rffm_seed)

        self.random_matrix = np.random.normal(size=[self.input_dim, self.basis_dim])
        if matern_param != np.inf:
            df = 2. * (matern_param + .5)
            u = np.random.chisquare(df, size=[self.basis_dim,])
            self.random_matrix = self.random_matrix * np.sqrt(df / u)
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

        self.Llower = spla.cholesky((self.noise_sd/self.prior_sd)**2*np.eye(self.basis_dim) + self.XX, lower=True)

    def _train_hyperparameters(self, X, y):
        warnings.filterwarnings('error')

        thetas = np.copy(np.array([self.length_scale, self.signal_sd, self.noise_sd, self.prior_sd]))
        options = {'maxiter': self.train_hp_iterations, 'disp': True}
        _res = minimize(self._log_marginal_likelihood, thetas, method='powell', args=(X, y), options=options)
        results = np.copy(_res.x)

        self.length_scale, self.signal_sd, self.noise_sd, self.prior_sd = results
        self.length_scale = np.abs(self.length_scale)
        self.signal_sd = np.abs(self.signal_sd)
        #self.noise_sd = np.abs(self.noise_sd)
        self.noise_sd = np.sqrt(self.noise_sd**2 + self.c*self.prior_sd**2)
        self.prior_sd = np.abs(self.prior_sd)
        self.hyperparameters = np.array([self.length_scale, self.signal_sd, self.noise_sd, self.prior_sd])
        print(self.length_scale, self.signal_sd, self.noise_sd, self.prior_sd)

    def _log_marginal_likelihood(self, thetas, X, y):
        try:
            length_scale, signal_sd, noise_sd, prior_sd = thetas

            noise_sd2 = np.sqrt(noise_sd**2 + self.c*prior_sd**2)

            basis = _basis(X, self.random_matrix, self.bias, self.basis_dim, np.abs(length_scale), np.abs(signal_sd))
            N = len(basis.T)
            XX = np.matmul(basis.T, basis)
            Xy = np.matmul(basis.T, y)

            tmp0 = (noise_sd2/prior_sd)**2*np.eye(self.basis_dim) + XX

            Llower = spla.cholesky(tmp0, lower=True)
            LinvXy = spla.solve_triangular(Llower, Xy, lower=True)
            tmp = np.matmul(LinvXy.T, LinvXy)

            s, logdet = np.linalg.slogdet(np.eye(self.basis_dim) + (prior_sd/noise_sd2)**2*XX)
            if s != 1:
                print('logdet is <= 0. Returning 10e100.')
                return 10e100

            lml = .5*(-N*np.log(noise_sd2**2) - logdet + (-np.matmul(y.T, y)[0, 0] + tmp[0, 0])/noise_sd2**2)
            #loss = -lml + (length_scale**2 + signal_sd**2 + noise_sd_abs**2 + prior_sd**2)*1.5
            loss = -lml
            return loss
        except Exception as e:
            print('------------')
            print(e, 'Returning 10e100.')
            print('************')
            return 10e100

    def _reset_statistics(self, X, y):
        self._init_statistics()
        self._update(X, y)

    def _predict(self, X):
        basis = _basis(X, self.random_matrix, self.bias, self.basis_dim, self.length_scale, self.signal_sd)

        tmp0 = spla.solve_triangular(self.Llower, basis.T, lower=True).T
        predict_sigma = np.square(tmp0).sum(axis=-1, keepdims=True) * self.noise_sd**2 + self.noise_sd**2

        tmp1 = spla.solve_triangular(self.Llower, self.Xy, lower=True)
        predict_mu = np.matmul(tmp0, tmp1)

        return predict_mu, predict_sigma

class Agent:
    def __init__(self, env, x_dim, y_dim, state_dim, action_dim, observation_space_low, observation_space_high,
                 action_space_low, action_space_high, unroll_steps, no_samples, discount_factor, random_matrices, biases, basis_dims,
                 hidden_dim=32, update_hyperstate=1, policy_use_hyperstate=1, learn_diff=0):
        assert x_dim == state_dim + action_dim
        assert len(action_space_low.shape) == 1
        np.testing.assert_equal(-action_space_low, action_space_high)
        self.env = env
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
        self.random_matrices = random_matrices
        self.biases = biases
        self.basis_dims = basis_dims
        self.hidden_dim = hidden_dim
        self.update_hyperstate = update_hyperstate
        self.policy_use_hyperstate = policy_use_hyperstate
        self.learn_diff = learn_diff

        #self.hyperstate_dim = sum([(basis_dim*(basis_dim+1))/2 + basis_dim for basis_dim in self.basis_dims])
        self.hyperstate_dim = sum([basis_dim*(basis_dim+1) for basis_dim in self.basis_dims])

        self.random_projection_matrix = np.random.normal(loc=0., scale=1./np.sqrt(self.state_dim), size=[self.hyperstate_dim, self.state_dim])

        input_dim = self.state_dim
        if self.policy_use_hyperstate == 1:
            input_dim *= 2

        self.w1 = np.concatenate([np.random.normal(size=[input_dim, self.hidden_dim]), np.random.uniform(-3e-3, 3e-3, size=[1, self.hidden_dim])], axis=0)
        self.w2 = np.concatenate([np.random.normal(size=[self.hidden_dim, self.hidden_dim]), np.random.uniform(-3e-3, 3e-3, size=[1, self.hidden_dim])], axis=0)
        self.w3 = np.concatenate([np.random.normal(size=[self.hidden_dim, self.action_dim]), np.random.uniform(-3e-3, 3e-3, size=[1, self.action_dim])], axis=0)

        self.thetas = self._pack([self.w1, self.w2, self.w3])

        self.sizes = [[input_dim + 1, self.hidden_dim],
                      [self.hidden_dim + 1, self.hidden_dim],
                      [self.hidden_dim + 1, self.action_dim]]

        w1, w2, w3 = self._unpack(self.thetas, self.sizes)
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
        w1, w2, w3 = self._unpack(thetas, self.sizes)

        #Perform a simple random projection on the hyperstate.
        if self.policy_use_hyperstate == 1:
            hyperstate = np.concatenate([np.concatenate([np.reshape(Llowers, [len(Llowers), -1]), np.reshape(Xytr, [len(Xytr), -1])], axis=-1) for Llowers, Xytr in zip(*hyperstate)], axis=-1)
            hyperstate = np.tanh(hyperstate/50000.)
            hyperstate_embedding = np.matmul(hyperstate, self.random_projection_matrix)
            hyperstate_embedding = np.tanh(hyperstate_embedding)

            state_hyperstate = np.concatenate([X, hyperstate_embedding], axis=-1)
            policy_net_input = self._add_bias(state_hyperstate)
        else:
            policy_net_input = self._add_bias(X)

        h1 = np.tanh(np.matmul(policy_net_input, w1))
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

    def _fit(self, cma_maxiter, X, XXtr, Xytr, hyperparameters):
        warnings.filterwarnings('ignore', message='.*scipy.linalg.solve\nIll-conditioned matrix detected. Result is not guaranteed to be accurate.\nReciprocal.*')
        assert len(XXtr) == self.state_dim
        assert len(Xytr) == self.state_dim
        assert len(hyperparameters) == self.state_dim

        #if self.use_mean_reward == 1: print('Warning: use_mean_reward is set to True but this flag is not used by this function.')

        X = X.copy()
        XXtr = [ele.copy() for ele in XXtr]
        Xytr = [ele.copy() for ele in Xytr]
        hyperparameters = [ele.copy() for ele in hyperparameters]

        X = X[:, None, ...]
        X = np.tile(X, [1, self.no_samples, 1])
        X = np.reshape(X, [-1, self.state_dim])

        Llowers = [spla.cholesky((hp[-2]/hp[-1])**2*np.eye(basis_dim) + XX, lower=True) for hp, basis_dim, XX in zip(hyperparameters, self.basis_dims, XXtr)]
        Llowers = [np.tile(ele[np.newaxis, ...], [len(X), 1, 1]) for ele in Llowers]
        Xytr = [np.tile(ele[np.newaxis, ...], [len(X), 1, 1]) for ele in Xytr]

        options = {'maxiter': cma_maxiter, 'verb_disp': 1, 'verb_log': 0}
        res = cma.fmin(self._loss, self.thetas, 2., args=(X.copy(), [ele.copy() for ele in Llowers], [ele.copy() for ele in Xytr], [ele.copy() for ele in hyperparameters]), options=options)
        self.thetas = res[0].copy()

    def _predict(self, Llower, Xytr, basis, noise_sd):
        tmp0 = solve_triangular(Llower, basis.transpose([0, 2, 1])).transpose([0, 2, 1])
        pred_sigma = np.square(tmp0).sum(axis=-1)*noise_sd**2+noise_sd**2

        tmp1 = solve_triangular(Llower, Xytr)
        pred_mu = np.matmul(tmp0, tmp1).squeeze(axis=-1)
        return pred_mu, pred_sigma

    def _loss(self, thetas, X, Llowers, Xytr, hyperparameters=None):
        rng_state = np.random.get_state()
        X = X.copy()
        Llowers = [ele.copy() for ele in Llowers]
        Xytr = [ele.copy() for ele in Xytr]
        hyperparameters = [ele.copy() for ele in hyperparameters]
        try:
            np.random.seed(2)

            rewards = []
            state = X
            for unroll_step in range(self.unroll_steps):
                action = self._forward(thetas, state, hyperstate=[Llowers, Xytr])
                state_action = np.concatenate([state, action], axis=-1)

                means = []
                covs = []
                bases = []
                for i in range(self.state_dim):
                    length_scale, signal_sd, noise_sd, prior_sd = hyperparameters[i]
                    basis = _basis(state_action, self.random_matrices[i], self.biases[i], self.basis_dims[i], length_scale, signal_sd)
                    basis = np.expand_dims(basis, axis=1)
                    bases.append(basis)
                    pred_mu, pred_sigma = self._predict(Llowers[i], Xytr[i], basis, noise_sd)
                    means.append(pred_mu)
                    covs.append(pred_sigma)
                means = np.concatenate(means, axis=-1)
                covs = np.concatenate(covs, axis=-1)

                state_ = means + np.sqrt(covs) * np.random.standard_normal(size=covs.shape)

                if self.learn_diff:
                    state_tmp = state.copy()
                    #state = np.clip(state + state_, self.observation_space_low, self.observation_space_high)
                    state_ = state - state_tmp
                else:
                    #state_ = np.clip(state_, self.observation_space_low, self.observation_space_high)
                    state = state_.copy()

                reward = -self.env.loss_func(state)
                rewards.append((self.discount_factor**unroll_step)*reward)

                if self.update_hyperstate == 1 or self.policy_use_hyperstate == 1:
                    y = state_.copy()[..., None, None]
                    for i in range(self.state_dim):
                        Llowers[i] = Llowers[i].transpose([0, 2, 1])
                    for i in range(self.state_dim):
                        for j in range(len(Llowers[i])):
                            cholupdate(Llowers[i][j], bases[i][j, 0].copy())
                        Xytr[i] += np.matmul(bases[i].transpose([0, 2, 1]), y[:, i, ...])
                    for i in range(self.state_dim):
                        Llowers[i] = Llowers[i].transpose([0, 2, 1])

            rewards = np.stack(rewards, axis=-1).sum(axis=-1)
            loss = -np.mean(rewards)
            np.random.set_state(rng_state)
            return loss
        except Exception as e:
            np.random.set_state(rng_state)
            print(e, 'Returning 10e100')
            return 10e100

def solve_triangular(A, b):
    assert len(A.shape) == len(b.shape)
    assert len(A.shape) >= 3
    assert A.shape[:-1] == b.shape[:-1]
    A = np.copy(A)
    b = np.copy(b)

    bs = list(A.shape[:-2])
    dimA = list(A.shape[-2:])
    dimb = list(b.shape[-2:])

    A = np.reshape(A, [-1]+dimA)
    b = np.reshape(b, [-1]+dimb)

    results = [scipy.linalg.solve_triangular(_A, _b, lower=True) for _A, _b in zip(A, b)]
    results = np.stack(results, axis=0)
    results = np.reshape(results, bs+dimb)

    return results

def update_hyperstate(agent, hyperstate, hyperparameters, datum, dim, learn_diff):
    state, action, reward, next_state = [np.atleast_2d(np.copy(dat)) for dat in datum]
    Llowers, Xy = [list(ele) for ele in hyperstate]
    assert len(Llowers) == len(hyperparameters)
    assert len(Xy) == len(hyperparameters)
    assert len(hyperparameters) == dim
    state_action = np.concatenate([state, action], axis=-1)
    y = next_state - state if learn_diff else next_state

    for i in range(len(Llowers)):
        Llowers[i] = Llowers[i].transpose([0, 2, 1])
    for i, hp in zip(list(range(dim)), hyperparameters):
        length_scale, signal_sd, noise_sd, prior_sd = hp
        basis = _basis(state_action, agent.random_matrices[i], agent.biases[i], agent.basis_dims[i], length_scale, signal_sd)
        cholupdate(Llowers[i][0], basis[0].copy())
        Xy[i] += np.matmul(basis[:, None, :].transpose([0, 2, 1]), y[:, None, :][..., i:i+1])
    for i in range(len(Llowers)):
        Llowers[i] = Llowers[i].transpose([0, 2, 1])

    return [Llowers, Xy]

def unpack(data_buffer):
    states, actions, rewards, next_states = [np.stack(ele, axis=0) for ele in list(zip(*data_buffer))]
    states_actions = np.concatenate([states, actions], axis=-1)
    return states_actions, states, rewards, next_states

def scrub_data(environment, data_buffer, warn):
    if environment == 'MountainCarContinuous-v0':
        states, actions, rewards, next_states, dones = [np.stack(ele, axis=0) for ele in zip(*data_buffer)]
        for i in range(len(next_states)):
            if next_states[i, 0] == -1.2 and next_states[i, 1] == 0.:
                states = states[:i, ...]
                actions = actions[:i, ...]
                rewards = rewards[:i, ...]
                next_states = next_states[:i, ...]
                dones = dones[:i, ...]
                if warn: print('Warning: training data is cut short because the cart hit the left wall!')
                break
        data_buffer = list(zip(states, actions, rewards, next_states, dones))
    return data_buffer

def main_loop():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, choices=['cartpole', 'double_cartpole', 'pendulum'], default='cartpole')
    parser.add_argument("--discount_factor", type=float, default=.995)
    parser.add_argument("--gather_data_epochs", type=int, default=2, help='Epochs for initial data gather.')
    parser.add_argument("--train_hp_iterations", type=int, default=2000*10)
    parser.add_argument("--train_policy_batch_size", type=int, default=30)
    parser.add_argument("--no_samples", type=int, default=1)
    parser.add_argument("--basis_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--rffm_seed", type=int, default=1)
    parser.add_argument("--Agent", type=str, choices=['', '2'], default='')
    parser.add_argument("--max_train_hp_datapoints", type=int, default=20000)
    parser.add_argument("--update_hyperstate", type=int, default=1)
    parser.add_argument("--policy_use_hyperstate", type=int, default=1)
    parser.add_argument("--cma_maxiter", type=int, default=1000)
    parser.add_argument("--learn_diff", type=int, choices=[0, 1], default=0)
    args = parser.parse_args()

    print(sys.argv)
    print(args)
    from blr_regression2_sans_hyperstate_kusanagi import Agent2

    if args.env == 'cartpole':
        params = cartpole.default_params()
        cost = partial(cartpole.cartpole_loss, **params['cost'])
        env = cartpole.Cartpole(loss_func=cost, **params['plant'])
        max_steps = 25
        maxA = 10.
    elif args.env == 'double_cartpole':
        params = double_cartpole.default_params()
        cost = partial(double_cartpole.double_cartpole_loss, **params['cost'])
        env = double_cartpole.DoubleCartpole(loss_func=cost, **params['plant'])
        max_steps = 30
        maxA = 20.
    elif args.env == 'pendulum':
        params = pendulum.default_params()
        cost = partial(pendulum.pendulum_loss, **params['cost'])
        env = pendulum.Pendulum(loss_func=cost, **params['plant'])
        max_steps = 40
        maxA = 2.5
    else:
        raise Exception('Unknown environment.')

    regression_wrappers = [RegressionWrapper(input_dim=env.observation_space.shape[0]+env.action_space.shape[0],
                                             basis_dim=args.basis_dim,
                                             length_scale=1.,
                                             signal_sd=1.,
                                             noise_sd=5e-1,
                                             prior_sd=1.,
                                             rffm_seed=args.rffm_seed,
                                             train_hp_iterations=args.train_hp_iterations)
                           for _ in range(env.observation_space.shape[0])]
    agent = eval('Agent'+args.Agent)(env=env,
                                     x_dim=env.observation_space.shape[0]+env.action_space.shape[0],
                                     y_dim=env.observation_space.shape[0],
                                     state_dim=env.observation_space.shape[0],
                                     action_dim=env.action_space.shape[0],
                                     observation_space_low=env.observation_space.low,
                                     observation_space_high=env.observation_space.high,
                                     action_space_low=np.array([-maxA]),
                                     action_space_high=np.array([maxA]),
                                     unroll_steps=max_steps,
                                     no_samples=args.no_samples,
                                     discount_factor=args.discount_factor,
                                     random_matrices=[rw.random_matrix for rw in regression_wrappers],
                                     biases=[rw.bias for rw in regression_wrappers],
                                     basis_dims=[rw.basis_dim for rw in regression_wrappers],
                                     hidden_dim=args.hidden_dim,
                                     #learn_reward=args.learn_reward,
                                     #use_mean_reward=args.use_mean_reward,
                                     update_hyperstate=args.update_hyperstate,
                                     policy_use_hyperstate=args.policy_use_hyperstate,
                                     learn_diff=args.learn_diff)

    flag = False
    from utils import get_data3
    data_buffer = get_data3(env, trials=args.gather_data_epochs, max_steps=max_steps, maxA=maxA)

    init_states = np.stack([env.reset() for _ in range(args.train_policy_batch_size)], axis=0)

    for epoch in range(1000):
        #Train hyperparameters and update systems model.
        states_actions, states, rewards, next_states = unpack(data_buffer)
        targets = np.concatenate([next_states - states if args.learn_diff else next_states, rewards], axis=-1) 
        for i in range(env.observation_space.shape[0]):
            if flag == False:
                regression_wrappers[i]._train_hyperparameters(states_actions, targets[:, i:i+1])
                regression_wrappers[i]._reset_statistics(states_actions, targets[:, i:i+1])
            else:
                regression_wrappers[i]._update(states_actions, targets[:, i:i+1])
        if len(data_buffer) >= args.max_train_hp_datapoints: flag = True
        if flag: data_buffer = []
        tmp_data_buffer = []

        #Fit policy network.
        XX, Xy, hyperparameters = list(zip(*[[rw.XX, rw.Xy, rw.hyperparameters] for rw in regression_wrappers]))
        agent._fit(args.cma_maxiter, init_states.copy(), [ele.copy() for ele in XX], [ele.copy() for ele in Xy], [ele.copy() for ele in hyperparameters])

        #Get hyperstate & hyperparameters
        hyperstate = list(zip(*[[spla.cholesky(rw.XX.copy()+(rw.noise_sd/rw.prior_sd)**2*np.eye(rw.basis_dim), lower=True)[None, ...], rw.Xy.copy()[None, ...]] for rw in regression_wrappers]))

        total_rewards = 0.
        state = env.reset()
        steps = 0
        while True:
            #env.render()
            action = agent._forward(agent.thetas, state[None, ...], hyperstate)[0]
            next_state, cost, done, _ = env.step(action)
            reward = -cost
            steps += 1

            #hyperstate = update_hyperstate_old(agent, XX, hyperstate, hyperparameters, [state, action, reward, next_state, done], agent.state_dim+agent.learn_reward, args.learn_diff)
            hyperstate = update_hyperstate(agent, hyperstate, hyperparameters, [state, action, reward, next_state], agent.state_dim, args.learn_diff)

            tmp_data_buffer.append([state, action, reward, next_state])
            total_rewards += float(reward)
            state = next_state.copy()
            if done or steps >= max_steps:
                print('epoch:', epoch, 'total_rewards:', total_rewards)
                data_buffer.extend(tmp_data_buffer)
                break

if __name__ == '__main__':
    main_loop()
