import numpy as np
import scipy
from scipy.optimize import minimize
import scipy.linalg as spla
import argparse

import sys
sys.path.append('..')

import tensorflow as tf
from custom_environments.generateANN_env import ANN
#from custom_environments.environment_state_functions import mountain_car_continuous_state_function
from custom_environments.environment_reward_functions import mountain_car_continuous_reward_function
from prototype8.dmlac.real_env_pendulum import real_env_pendulum_reward

from utils import gather_data

import gym
import pybullet_envs
import pickle
import warnings

from choldate import cholupdate

from blr_regression2 import _basis, RegressionWrapperReward, solve_triangular, unpack, scrub_data

from morw import MultiOutputRegressionWrapper

class Agent:
    def __init__(self, environment, x_dim, y_dim, state_dim, action_dim, observation_space_low, observation_space_high,
                 action_space_low, action_space_high, unroll_steps, no_samples, discount_factor, random_matrix_state, bias_state,
                 basis_dim_state, random_matrix_reward, bias_reward, basis_dim_reward, hidden_dim=32, learn_reward=0,
                 use_mean_reward=0, update_hyperstate=1, policy_use_hyperstate=1, learn_diff=0):
        #assert environment in ['Pendulum-v0', 'MountainCarContinuous-v0']
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

        self.random_matrix_state = random_matrix_state
        self.bias_state = bias_state
        self.basis_dim_state = basis_dim_state
        self.random_matrix_reward = random_matrix_reward
        self.bias_reward = bias_reward
        self.basis_dim_reward = basis_dim_reward

        self.hidden_dim = hidden_dim
        self.learn_reward = learn_reward
        self.use_mean_reward = use_mean_reward
        self.update_hyperstate = update_hyperstate
        self.policy_use_hyperstate = policy_use_hyperstate
        self.learn_diff = learn_diff

        if self.environment == 'Pendulum-v0' and self.learn_reward == 0:
            #self.reward_function = real_env_pendulum_reward()
            self.reward_function = ANN(self.state_dim+self.action_dim, 1)
            self.placeholders_reward = [tf.placeholder(shape=v.shape, dtype=tf.float64)
                                        for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.reward_function.scope)]
            self.assign_ops0 = [v.assign(pl) for v, pl in zip(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.reward_function.scope),
                                self.placeholders_reward)]
        elif self.environment == 'MountainCarContinuous-v0' and self.learn_reward == 0:
            self.reward_function = mountain_car_continuous_reward_function()

        self.hyperstate_dim = self.basis_dim_state * (self.basis_dim_state + self.state_dim)
        if self.learn_reward == 1: self.hyperstate_dim += self.basis_dim_reward * (self.basis_dim_reward + 1)

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
            Llower_state, Xytr_state, Llower_reward, Xytr_reward = hyperstate_params
            hyperstate = np.concatenate([Llower_state.reshape([len(Llower_state), -1]),
                                         Xytr_state.reshape([len(Xytr_state), -1]),
                                         Llower_reward.reshape([len(Llower_reward), -1]),
                                         Xytr_reward.reshape([len(Xytr_reward), -1])],
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

    def _fit(self, cma_maxiter, X, XXtr_state, Xytr_state, hyperparameters_state, XXtr_reward, Xytr_reward, hyperparameters_reward, sess):
        warnings.filterwarnings('ignore', message='.*scipy.linalg.solve\nIll-conditioned matrix detected. Result is not guaranteed to be accurate.\nReciprocal.*')
        assert XXtr_state.shape == (self.basis_dim_state, self.basis_dim_state)
        assert Xytr_state.shape == (self.basis_dim_state, self.state_dim)
        assert XXtr_reward.shape == (self.basis_dim_reward, self.basis_dim_reward)
        assert Xytr_reward.shape == (self.basis_dim_reward, 1)
        assert hyperparameters_state.shape == hyperparameters_reward.shape

        if self.use_mean_reward == 1: print 'Warning: use_mean_reward is set to True but this flag is not used by this function.'

        #Copy the arrays (just to be safe no overwriting occurs).
        X = X.copy()
        XXtr_state = XXtr_state.copy()
        Xytr_state = Xytr_state.copy()
        hyperparameters_state = hyperparameters_state.copy()
        XXtr_reward = XXtr_reward.copy()
        Xytr_reward = Xytr_reward.copy()
        hyperparameters_reward = hyperparameters_reward.copy()

        X = np.expand_dims(X, axis=1)
        X = np.tile(X, [1, self.no_samples, 1])
        X = np.reshape(X, [-1, self.state_dim])

        #State
        Llower_state = spla.cholesky((hyperparameters_state[-2]/hyperparameters_state[-1])**2*np.eye(self.basis_dim_state) + XXtr_state, lower=True)
        Llower_state = np.tile(Llower_state, [len(X), 1, 1])

        XXtr_state = np.tile(XXtr_state, [len(X), 1, 1])
        Xytr_state = np.tile(Xytr_state, [len(X), 1, 1])

        #Reward
        if self.learn_reward:
            Llower_reward = spla.cholesky((hyperparameters_reward[-2]/hyperparameters_reward[-1])**2*np.eye(self.basis_dim_reward) + XXtr_reward, lower=True)
            Llower_reward = np.tile(Llower_reward, [len(X), 1, 1])

            XXtr_reward = np.tile(XXtr_reward, [len(X), 1, 1])
            Xytr_reward = np.tile(Xytr_reward, [len(X), 1, 1])

        import cma
        options = {'maxiter': cma_maxiter, 'verb_disp': 1, 'verb_log': 0}
        print 'Before calling cma.fmin'
        res = cma.fmin(self._loss, self.thetas, 2., args=(X.copy(),
                                                          Llower_state.copy(),
                                                          XXtr_state.copy(),
                                                          Xytr_state.copy(),
                                                          hyperparameters_state,
                                                          Llower_reward.copy() if self.learn_reward else None,
                                                          XXtr_reward.copy() if self.learn_reward else None,
                                                          Xytr_reward.copy() if self.learn_reward else None,
                                                          hyperparameters_reward if self.learn_reward else None,
                                                          sess), options=options)
        self.thetas = np.copy(res[0])

    def _predict(self, Llower, Xytr, basis, noise_sd):
        LinvXT = solve_triangular(Llower, basis.transpose([0, 2, 1]))
        sigma = np.sum(np.square(LinvXT), axis=1)*noise_sd**2+noise_sd**2
        tmp0 = solve_triangular(Llower, basis.transpose([0, 2, 1])).transpose([0, 2, 1])
        tmp1 = solve_triangular(Llower, Xytr)
        mu = np.matmul(tmp0, tmp1).squeeze(axis=1)
        return mu, sigma

    def _loss(self, thetas, X, Llower_state, XXtr_state, Xytr_state, hyperparameters_state, Llower_reward, XXtr_reward, Xytr_reward, hyperparameters_reward, sess=None):
        rng_state = np.random.get_state()
        X = X.copy()
        Llower_state = Llower_state.copy()
        XXtr_state = XXtr_state.copy()
        Xytr_state = Xytr_state.copy()
        hyperparameters_state = hyperparameters_state.copy()
        if self.learn_reward:
            Llower_reward = Llower_reward.copy()
            XXtr_reward = XXtr_reward.copy()
            Xytr_reward = Xytr_reward.copy()
            hyperparameters_reward = hyperparameters_reward.copy()

        #try:
        np.random.seed(2)

        rewards = []
        state = X
        for unroll_step in xrange(self.unroll_steps):
            action = self._forward(thetas, state, hyperstate_params=[Llower_state, Xytr_state, Llower_reward, Xytr_reward])
            state_action = np.concatenate([state, action], axis=-1)

            reward, basis_reward = self._reward(state, action, state_action, sess, Llower_reward, Xytr_reward, hyperparameters_reward)
            rewards.append((self.discount_factor**unroll_step)*reward)

            length_scale, signal_sd, noise_sd, prior_sd = hyperparameters_reward
            basis_state = _basis(state_action, self.random_matrix_state, self.bias_state, self.basis_dim_state, length_scale, signal_sd)
            basis_state = basis_state[:, None, ...]
            mu, sigma = self._predict(Llower_state, Xytr_state, basis_state, noise_sd)
            state_ = mu + np.sqrt(sigma) * np.random.normal(size=mu.shape)

            if self.learn_diff:
                state_tmp = state.copy()
                state = np.clip(state + state_, self.observation_space_low, self.observation_space_high)
                state_ = state - state_tmp
            else:
                state_ = np.clip(state_, self.observation_space_low, self.observation_space_high)
                state = state_.copy()

            if self.update_hyperstate == 1 and self.policy_use_hyperstate == 1:
                #Update state hyperstate
                Llower_state = Llower_state.transpose([0, 2, 1])
                for i in range(len(Llower_state)):
                    cholupdate(Llower_state[i], basis_state[i, 0].copy())
                Llower_state = Llower_state.transpose([0, 2, 1])
                Xytr_state += np.matmul(basis_state.transpose([0, 2, 1]), state_[..., None, :])

                #Update reward hyperstate
                if self.learn_reward:
                    Llower_reward = Llower_reward.transpose([0, 2, 1])
                    for i in range(len(Llower_reward)):
                        cholupdate(Llower_reward[i], basis_reward[i, 0].copy())
                    Llower_reward = Llower_reward.transpose([0, 2, 1])
                Xytr_reward += np.matmul(basis_reward.transpose([0, 2, 1]), reward[..., None, :])

        rewards = np.concatenate(rewards, axis=-1)
        rewards = np.sum(rewards, axis=-1)
        loss = -np.mean(rewards)
        np.random.set_state(rng_state)
        return loss
        #except Exception as e:
            #np.random.set_state(rng_state)
            #print e, 'Returning 10e100'
            #return 10e100

    def _update_hyperstate(self, XXold, XXnew, Xyold, Xynew, Llowerold, var_ratio):
        var_diag = var_ratio*np.eye(XXnew.shape[-1])
        XX = []
        Xy = []
        Llower = []
        for i in range(len(XXnew)):
            try:
                tmp = scipy.linalg.cholesky(XXnew[i] + var_diag, lower=True)
                XX.append(XXnew[i].copy())
                Xy.append(Xynew[i].copy())
                Llower.append(tmp.copy())
            except Exception as e:
                XX.append(XXold[i].copy())
                Xy.append(Xyold[i].copy())
                Llower.append(Llowerold[i].copy())
        XX = np.stack(XX, axis=0)
        Xy = np.stack(Xy, axis=0)
        Llower = np.stack(Llower, axis=0)
        return XX, Xy, Llower

    def _reward(self, state, action, state_action, sess, Llower, Xy, hyperparameters):
        basis = None
        if self.environment == 'Pendulum-v0' and self.learn_reward == 0:
            reward = self.reward_function.build_np(sess, state, action)
        elif self.environment == 'MountainCarContinuous-v0' and self.learn_reward == 0:
            reward = self.reward_function.build_np(state, action)
        else:
            #state_action = np.concatenate([state, action], axis=-1)
            length_scale, signal_sd, noise_sd, prior_sd = hyperparameters
            basis = _basis(state_action, self.random_matrix_reward, self.bias_reward, self.basis_dim_reward, length_scale, signal_sd)
            basis = basis[:, None, ...]
            mu, sigma = self._predict(Llower, Xy, basis, noise_sd)
            if self.use_mean_reward == 1: sigma = np.zeros_like(sigma)
            reward = mu + np.sqrt(sigma) * np.random.standard_normal(size=mu.shape)
        return reward, basis

def update_hyperstate(agent, hyperstate_params, hyperparameters_state, hyperparameters_reward, datum, learn_diff):
    state, action, reward, next_state, _ = [np.atleast_2d(np.copy(dat)) for dat in datum]
    Llower_state, Xy_state, Llower_reward, Xy_reward = hyperstate_params

    state_action = np.concatenate([state, action], axis=-1)
    state_ = next_state - state if learn_diff else next_state

    basis_state = _basis(state_action, agent.random_matrix_state, agent.bias_state, agent.basis_dim_state, hyperparameters_state[0], hyperparameters_state[1])
    Llower_state = Llower_state.transpose([0, 2, 1])
    for i in range(len(Llower_state)):
        cholupdate(Llower_state[i], basis_state[i].copy())
    Llower_state = Llower_state.transpose([0, 2, 1])
    Xy_state += np.matmul(basis_state[..., None, :].transpose([0, 2, 1]), state_[..., None, :])

    basis_reward = _basis(state_action, agent.random_matrix_reward, agent.bias_reward, agent.basis_dim_reward, hyperparameters_reward[0], hyperparameters_reward[1])
    Llower_reward = Llower_reward.transpose([0, 2, 1])
    for i in range(len(Llower_reward)):
        cholupdate(Llower_reward[i], basis_reward[i].copy())
    Llower_reward = Llower_reward.transpose([0, 2, 1])
    Xy_reward += np.matmul(basis_reward[..., None, :].transpose([0, 2, 1]), reward[..., None, :])

    return [Llower_state, Xy_state, Llower_reward, Xy_reward]

def main_loop():
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default='Pendulum-v0')
    parser.add_argument("--unroll-steps", type=int, default=200)
    parser.add_argument("--discount-factor", type=float, default=.995)
    parser.add_argument("--gather-data-epochs", type=int, default=3, help='Epochs for initial data gather.')
    parser.add_argument("--train-hp-iterations", type=int, default=2000*10)
    parser.add_argument("--train-policy-batch-size", type=int, default=30)
    parser.add_argument("--no-samples", type=int, default=1)
    parser.add_argument("--basis-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--rffm-seed", type=int, default=1)
    parser.add_argument("--Agent", type=str, choices=['', '2'], default='')
    #parser.add_argument("--fit-function", type=str, choices=['_fit', '_fit_cma'], default='_fit')
    parser.add_argument("--learn-reward", type=int, choices=[0, 1], default=1)
    parser.add_argument("--max-train-hp-datapoints", type=int, default=20000)
    parser.add_argument("--matern-param-reward", type=float, default=np.inf)
    parser.add_argument("--basis-dim-reward", type=int, default=600)
    parser.add_argument("--use-mean-reward", type=int, default=0)
    parser.add_argument("--update-hyperstate", type=int, default=1)
    parser.add_argument("--policy-use-hyperstate", type=int, default=1)
    parser.add_argument("--cma-maxiter", type=int, default=1000)
    parser.add_argument("--learn-diff", type=int, choices=[0, 1], default=0)
    args = parser.parse_args()

    print sys.argv
    print args
    from blr_regression2_sans_hyperstate_multioutput import Agent2

    env = gym.make(args.environment)

    regression_wrapper_state = MultiOutputRegressionWrapper(input_dim=env.observation_space.shape[0]+env.action_space.shape[0],
                                                            output_dim=env.observation_space.shape[0],
                                                            basis_dim=args.basis_dim,
                                                            length_scale=1.,
                                                            signal_sd=1.,
                                                            noise_sd=5e-4,
                                                            prior_sd=1.,
                                                            rffm_seed=args.rffm_seed,
                                                            train_hp_iterations=args.train_hp_iterations)
    regression_wrapper_reward = RegressionWrapperReward(environment=args.environment,
                                                        input_dim=env.observation_space.shape[0]+env.action_space.shape[0],
                                                        basis_dim=args.basis_dim_reward,
                                                        length_scale=1.,
                                                        signal_sd=1.,
                                                        noise_sd=5e-4,
                                                        prior_sd=1.,
                                                        rffm_seed=args.rffm_seed,
                                                        train_hp_iterations=args.train_hp_iterations,
                                                        matern_param=args.matern_param_reward)
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

                                     random_matrix_state=regression_wrapper_state.random_matrix,
                                     bias_state=regression_wrapper_state.bias,
                                     basis_dim_state=regression_wrapper_state.basis_dim,


                                     random_matrix_reward=regression_wrapper_reward.random_matrix,
                                     bias_reward=regression_wrapper_reward.bias,
                                     basis_dim_reward=regression_wrapper_reward.basis_dim,

                                     #random_matrices=[rw.random_matrix for rw in regression_wrappers],
                                     #biases=[rw.bias for rw in regression_wrappers],
                                     #basis_dims=[rw.basis_dim for rw in regression_wrappers],

                                     hidden_dim=args.hidden_dim,
                                     learn_reward=args.learn_reward,
                                     use_mean_reward=args.use_mean_reward,
                                     update_hyperstate=args.update_hyperstate,
                                     policy_use_hyperstate=args.policy_use_hyperstate,
                                     learn_diff=args.learn_diff)


    #I have to work on the classes before working on the code below.
    flag = False
    data_buffer = gather_data(env, args.gather_data_epochs)
    data_buffer = scrub_data(args.environment, data_buffer, True)

    init_states = np.stack([env.reset() for _ in range(args.train_policy_batch_size)], axis=0)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if args.environment == 'Pendulum-v0' and args.learn_reward == 0:
            weights = pickle.load(open('../custom_environments/weights/pendulum_reward.p', 'rb'))
            sess.run(agent.assign_ops0, feed_dict=dict(zip(agent.placeholders_reward, weights)))
        for epoch in range(1000):
            #Train hyperparameters and update systems model.
            states_actions, states, rewards, next_states = unpack(data_buffer)

            next_states_train = next_states.copy() - states.copy() if args.learn_diff else next_states.copy()
            rewards_train = rewards.copy()

            if flag == False:
                regression_wrapper_state._train_hyperparameters(states_actions, next_states_train)
                regression_wrapper_state._reset_statistics(states_actions, next_states_train)
                regression_wrapper_reward._train_hyperparameters(states_actions, rewards_train)
                regression_wrapper_reward._reset_statistics(states_actions, rewards_train)
            else:
                regression_wrapper_state._update(states_actions, next_states_train)
                regression_wrapper_reward._update(states_actions, rewards_train)

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
                       regression_wrapper_state.hyperparameters.copy(),
                       regression_wrapper_reward.XX.copy(),
                       regression_wrapper_reward.Xy.copy(),
                       regression_wrapper_reward.hyperparameters.copy(),
                       sess)

            #Get hyperstate & hyperparameters
            hyperstate_params = [regression_wrapper_state.Llower.copy()[None, ...],
                                 regression_wrapper_state.Xy.copy()[None, ...],
                                 regression_wrapper_reward.Llower.copy()[None, ...],
                                 regression_wrapper_reward.Xy.copy()[None, ...]]
            total_rewards = 0.
            state = env.reset()
            while True:
                #env.render()
                action = agent._forward(agent.thetas, state[np.newaxis, ...], hyperstate_params)[0]
                next_state, reward, done, _ = env.step(action)

                hyperstate_params = update_hyperstate(agent,
                                                      hyperstate_params,
                                                      regression_wrapper_state.hyperparameters.copy(),
                                                      regression_wrapper_reward.hyperparameters.copy(),
                                                      [state, action, reward, next_state, done],
                                                      args.learn_diff)

                tmp_data_buffer.append([state, action, reward, next_state, done])
                total_rewards += float(reward)
                state = np.copy(next_state)
                if done:
                    print 'epoch:', epoch, 'total_rewards:', total_rewards
                    data_buffer.extend(scrub_data(args.environment, tmp_data_buffer, False))
                    break

if __name__ == '__main__':
    main_loop()
