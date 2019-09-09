import numpy as np
import scipy
from scipy.optimize import minimize
import scipy.linalg as spla
import argparse

import sys
sys.path.append('..')
import warnings

from choldate import cholupdate
from kusanagi.shell import experiment_utils, cartpole, double_cartpole, pendulum
from functools import partial


#from blr_regression2 import _basis, RegressionWrapperReward, solve_triangular, unpack, scrub_data
from blr_regression2_kusanagi import _basis, solve_triangular, unpack

from morw import MultiOutputRegressionWrapper

import uuid
import os

class Agent:
    def __init__(self, env, x_dim, y_dim, state_dim, action_dim, observation_space_low, observation_space_high,
                 action_space_low, action_space_high, unroll_steps, no_samples, discount_factor, random_matrix_state, bias_state, basis_dim_state, hidden_dim=32, update_hyperstate=1, policy_use_hyperstate=1, learn_diff=0, dump_model=0):
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

        self.random_matrix_state = random_matrix_state
        self.bias_state = bias_state
        self.basis_dim_state = basis_dim_state

        self.hidden_dim = hidden_dim
        self.update_hyperstate = update_hyperstate
        self.policy_use_hyperstate = policy_use_hyperstate
        self.learn_diff = learn_diff

        self.dump_model = dump_model

        self.uid = str(uuid.uuid4())
        self.epoch = 0

        self.hyperstate_dim = self.basis_dim_state * (self.basis_dim_state + self.state_dim)

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

    def _forward(self, thetas, X, hyperstate_params):
        w1, w2, w3 = self._unpack(thetas, self.sizes)

        #Perform a simple random projection on the hyperstate.
        if self.policy_use_hyperstate == 1:
            Llower_state, Xytr_state = hyperstate_params
            hyperstate = np.concatenate([Llower_state.reshape([len(Llower_state), -1]),
                                         Xytr_state.reshape([len(Xytr_state), -1])],
                                         axis=-1)
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

    def _fit(self, cma_maxiter, X, XXtr_state, Xytr_state, hyperparameters_state):
        warnings.filterwarnings('ignore', message='.*scipy.linalg.solve\nIll-conditioned matrix detected. Result is not guaranteed to be accurate.\nReciprocal.*')
        assert XXtr_state.shape == (self.basis_dim_state, self.basis_dim_state)
        assert Xytr_state.shape == (self.basis_dim_state, self.state_dim)

        #Copy the arrays (just to be safe no overwriting occurs).
        X = X.copy()
        XXtr_state = XXtr_state.copy()
        Xytr_state = Xytr_state.copy()
        hyperparameters_state = hyperparameters_state.copy()

        X = np.expand_dims(X, axis=1)
        X = np.tile(X, [1, self.no_samples, 1])
        X = np.reshape(X, [-1, self.state_dim])

        #State
        Llower_state = spla.cholesky((hyperparameters_state[-2]/hyperparameters_state[-1])**2*np.eye(self.basis_dim_state) + XXtr_state, lower=True)
        Llower_state = np.tile(Llower_state, [len(X), 1, 1])

        XXtr_state = np.tile(XXtr_state, [len(X), 1, 1])
        Xytr_state = np.tile(Xytr_state, [len(X), 1, 1])

        import cma
        options = {'maxiter': cma_maxiter, 'verb_disp': 1, 'verb_log': 0}
        print('Before calling cma.fmin')
        res = cma.fmin(self._loss, self.thetas, 2., args=(X.copy(),
                                                          Llower_state.copy(),
                                                          XXtr_state.copy(),
                                                          Xytr_state.copy(),
                                                          hyperparameters_state), options=options)
        self.thetas = res[0].copy()
        if self.dump_model:
            print('Unique identifier:', self.uid)
            directory = './models/'
            if not os.path.exists(directory):
                os.makedirs(directory)
            with open(directory+self.uid+'_epoch:'+str(self.epoch)+'.p', 'wb') as fp:
                pickle.dump(self.thetas, fp)
            self.epoch += 1

    def _predict(self, Llower, Xytr, basis, noise_sd):
        LinvXT = solve_triangular(Llower, basis.transpose([0, 2, 1]))
        sigma = np.sum(np.square(LinvXT), axis=1)*noise_sd**2+noise_sd**2
        #tmp0 = solve_triangular(Llower, basis.transpose([0, 2, 1])).transpose([0, 2, 1])
        tmp0 = LinvXT.transpose([0, 2, 1])
        tmp1 = solve_triangular(Llower, Xytr)
        mu = np.matmul(tmp0, tmp1).squeeze(axis=1)
        return mu, sigma

    def _loss(self, thetas, X, Llower_state, XXtr_state, Xytr_state, hyperparameters_state):
        X = X.copy()
        Llower_state = Llower_state.copy()
        XXtr_state = XXtr_state.copy()
        Xytr_state = Xytr_state.copy()
        hyperparameters_state = hyperparameters_state.copy()
        rng_state = np.random.get_state()
        #try:
        np.random.seed(2)

        rewards = []
        state = X
        for unroll_step in range(self.unroll_steps):
            action = self._forward(thetas, state, hyperstate_params=[Llower_state, Xytr_state])
            state_action = np.concatenate([state, action], axis=-1)

            length_scale, signal_sd, noise_sd, prior_sd = hyperparameters_state
            basis_state = _basis(state_action, self.random_matrix_state, self.bias_state, self.basis_dim_state, length_scale, signal_sd)
            basis_state = basis_state[:, None, ...]
            mu, sigma = self._predict(Llower_state, Xytr_state, basis_state, noise_sd)
            state_ = mu + np.sqrt(sigma) * np.random.standard_normal(size=mu.shape)

            if self.learn_diff:
                state_tmp = state.copy()
                state = np.clip(state + state_, self.observation_space_low, self.observation_space_high)
                state_ = state - state_tmp
            else:
                state_ = np.clip(state_, self.observation_space_low, self.observation_space_high)
                state = state_.copy()

            reward = -self.env.loss_func(state)
            rewards.append((self.discount_factor**unroll_step)*reward)

            if self.update_hyperstate == 1 or self.policy_use_hyperstate == 1:
                #Update state hyperstate
                Llower_state = Llower_state.transpose([0, 2, 1])
                for i in range(len(Llower_state)):
                    cholupdate(Llower_state[i], basis_state[i, 0].copy())
                Llower_state = Llower_state.transpose([0, 2, 1])
                Xytr_state += np.matmul(basis_state.transpose([0, 2, 1]), state_[..., None, :])

        rewards = np.stack(rewards, axis=-1).sum(axis=-1)
        loss = -np.mean(rewards)
        np.random.set_state(rng_state)
        return loss
        #except Exception as e:
            #np.random.set_state(rng_state)
            #print e, 'Returning 10e100'
            #return 10e100

def update_hyperstate(agent, hyperstate_params, hyperparameters_state, datum, learn_diff):
    state, action, reward, next_state, _ = [np.atleast_2d(np.copy(dat)) for dat in datum]
    Llower_state, Xy_state = hyperstate_params

    state_action = np.concatenate([state, action], axis=-1)
    state_ = next_state - state if learn_diff else next_state

    basis_state = _basis(state_action, agent.random_matrix_state, agent.bias_state, agent.basis_dim_state, hyperparameters_state[0], hyperparameters_state[1])
    Llower_state = Llower_state.transpose([0, 2, 1])
    for i in range(len(Llower_state)):
        cholupdate(Llower_state[i], basis_state[i].copy())
    Llower_state = Llower_state.transpose([0, 2, 1])
    Xy_state += np.matmul(basis_state[..., None, :].transpose([0, 2, 1]), state_[..., None, :])

    return [Llower_state, Xy_state]

def main_loop():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, choices=['cartpole', 'double_cartpole', 'pendulum'], default='cartpole')
    parser.add_argument("--discount_factor", type=float, default=.995)
    parser.add_argument("--gather_data_epochs", type=int, default=3, help='Epochs for initial data gather.')
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
    parser.add_argument("--dump_model", type=int, choices=[0, 1], default=0)
    args = parser.parse_args()

    print(sys.argv)
    print(args)
    from blr_regression2_sans_hyperstate_kusanagi_multioutput import Agent2

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


    regression_wrapper_state = MultiOutputRegressionWrapper(input_dim=env.observation_space.shape[0]+env.action_space.shape[0],
                                                            output_dim=env.observation_space.shape[0],
                                                            basis_dim=args.basis_dim,
                                                            length_scale=1.,
                                                            signal_sd=1.,
                                                            noise_sd=5e-4,
                                                            prior_sd=1.,
                                                            rffm_seed=args.rffm_seed,
                                                            train_hp_iterations=args.train_hp_iterations)
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

                                     random_matrix_state=regression_wrapper_state.random_matrix,
                                     bias_state=regression_wrapper_state.bias,
                                     basis_dim_state=regression_wrapper_state.basis_dim,




                                     hidden_dim=args.hidden_dim,
                                     update_hyperstate=args.update_hyperstate,
                                     policy_use_hyperstate=args.policy_use_hyperstate,
                                     learn_diff=args.learn_diff,
                                     dump_model=args.dump_model)


    #I have to work on the classes before working on the code below.
    flag = False
    from utils import get_data3
    data_buffer = get_data3(env, trials=args.gather_data_epochs, max_steps=max_steps, maxA=maxA)

    init_states = np.stack([env.reset() for _ in range(args.train_policy_batch_size)], axis=0)


    for epoch in range(1000):
        #Train hyperparameters and update systems model.
        states_actions, states, rewards, next_states = unpack(data_buffer)

        next_states_train = next_states.copy() - states.copy() if args.learn_diff else next_states.copy()

        if flag == False:
            regression_wrapper_state._train_hyperparameters(states_actions, next_states_train)
            regression_wrapper_state._reset_statistics(states_actions, next_states_train)
        else:
            regression_wrapper_state._update(states_actions, next_states_train)

        if len(data_buffer) >= args.max_train_hp_datapoints: flag = True
        if flag: data_buffer = []
        tmp_data_buffer = []

        #Fit policy network.
        #XX, Xy, hyperparameters = zip(*[[rw.XX, rw.Xy, rw.hyperparameters] for rw in regression_wrappers])
        #eval('agent.'+args.fit_function)(args.cma_maxiter, np.copy(init_states), [np.copy(ele) for ele in XX], [np.copy(ele) for ele in Xy], [np.copy(ele) for ele in hyperparameters], sess)
        agent._fit(args.cma_maxiter,
                   init_states.copy(),
                   regression_wrapper_state.XX.copy(),
                   regression_wrapper_state.Xy.copy(),
                   regression_wrapper_state.hyperparameters.copy())

        #Get hyperstate & hyperparameters
        hyperstate_params = [regression_wrapper_state.Llower.copy()[None, ...],
                             regression_wrapper_state.Xy.copy()[None, ...]]
        total_rewards = 0.
        state = env.reset()
        steps = 0
        while True:
            #env.render()
            action = agent._forward(agent.thetas, state[np.newaxis, ...], hyperstate_params)[0]
            next_state, cost, done, _ = env.step(action)
            reward = -cost
            steps += 1

            hyperstate_params = update_hyperstate(agent,
                                                  hyperstate_params,
                                                  regression_wrapper_state.hyperparameters.copy(),
                                                  [state, action, reward, next_state, done],
                                                  args.learn_diff)

            tmp_data_buffer.append([state, action, reward, next_state, done])
            total_rewards += float(reward)
            state = next_state.copy()
            if done or steps >= max_steps:
                print('epoch:', epoch, 'total_rewards:', total_rewards)
                data_buffer.extend(tmp_data_buffer)
                break

if __name__ == '__main__':
    main_loop()
