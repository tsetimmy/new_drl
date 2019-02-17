import numpy as np
from choldate import cholupdate

def get_data(env, max_steps, action_high):
    action_low = -action_high

    states = []
    actions = []
    costs = []
    next_states = []
    state = env.reset()
    steps = 0
    while True:
        #env.render()
        action = np.random.random(env.action_space.shape) * (action_high - action_low) + action_low

        next_state, cost, done, _ = env.step(action)
        steps += 1
        #print (cost, env.loss_func(env.state_noiseless[None, ...]))

        states.append(state)
        actions.append(action)
        costs.append(cost)
        next_states.append(next_state)

        state = next_state.copy()
        if done or steps >= max_steps:
            break

    states = np.stack(states, axis=0)
    actions = np.stack(actions, axis=0)
    costs = np.stack(costs, axis=0)
    next_states = np.stack(next_states, axis=0)
    return states, actions, -costs, next_states

def get_data2(env, trials=1, max_steps=25, maxA=10):
    return [np.concatenate(ele, axis=0) for ele in list(zip(*[get_data(env, max_steps, maxA) for _ in range(trials)]))]

def get_data3(env, trials=1, max_steps=25, maxA=10):
    states, actions, rewards, next_states = get_data2(env, trials, max_steps, maxA)
    data = [[state, action, reward, next_state] for state, action, reward, next_state in zip(states, actions, rewards, next_states)]
    return data

from blr_regression2_kusanagi import RegressionWrapper, RegressionWrapperReward, _basis, solve_triangular

class RegressionWrapper2(RegressionWrapper):
    def __init__(self, input_dim, basis_dim, length_scale=1., signal_sd=1., noise_sd=5e-4, prior_sd=1., rffm_seed=1, train_hp_iterations=2000, matern_param=np.inf):
        RegressionWrapper.__init__(self, input_dim, basis_dim, length_scale, signal_sd, noise_sd, prior_sd, rffm_seed, train_hp_iterations, matern_param)

    def _reset_statistics(self, X, y, update_hyperstate):
        self._init_statistics()
        self._update(X, y, update_hyperstate)

    def _update(self, X, y, update_hyperstate):
        RegressionWrapper._update(self, X, y)
        if update_hyperstate:
            self.XX_tiled = np.tile(self.XX[np.newaxis, ...], [50, 1, 1])
            self.Xy_tiled = np.tile(self.Xy[np.newaxis, ...], [50, 1, 1])
            self.Llower_tiled = np.tile(self.Llower[np.newaxis, ...], [50, 1, 1])

    def _predict(self, X, update_hyperstate):
        if update_hyperstate:
            basis = _basis(X, self.random_matrix, self.bias, self.basis_dim, self.length_scale, self.signal_sd)
            basis = np.expand_dims(basis, axis=1)

            LinvXT = solve_triangular(self.Llower_tiled, np.transpose(basis, [0, 2, 1]))
            pred_sigma = np.sum(np.square(LinvXT), axis=1)*self.noise_sd**2+self.noise_sd**2
            tmp0 = np.transpose(solve_triangular(self.Llower_tiled, np.transpose(basis, [0, 2, 1])), [0, 2, 1])
            tmp1 = solve_triangular(self.Llower_tiled, self.Xy_tiled)
            pred_mu = np.matmul(tmp0, tmp1)
            pred_mu = np.squeeze(pred_mu, axis=-1)
            return pred_mu, pred_sigma
        else:
            return RegressionWrapper._predict(self, X)

    def _update_hyperstate(self, X, y, update_hyperstate):
        if update_hyperstate:
            basis = _basis(X, self.random_matrix, self.bias, self.basis_dim, self.length_scale, self.signal_sd)

            self.Llower_tiled = self.Llower_tiled.transpose([0, 2, 1])
            assert len(self.Llower_tiled) == len(basis)
            for i in range(len(self.Llower_tiled)):
                cholupdate(self.Llower_tiled[i], basis[i].copy())
            self.Llower_tiled = self.Llower_tiled.transpose([0, 2, 1])

            self.Xy_tiled += np.matmul(basis[:, None, :].transpose([0, 2, 1,]), y[:, None, :])

class RegressionWrapperReward2(RegressionWrapperReward):
    def __init__(self, environment, input_dim, basis_dim, length_scale=1., signal_sd=1., noise_sd=5e-4, prior_sd=1., rffm_seed=1, train_hp_iterations=2000, matern_param=np.inf):
        RegressionWrapperReward.__init__(self, environment, input_dim, basis_dim, length_scale, signal_sd, noise_sd, prior_sd, rffm_seed, train_hp_iterations, matern_param)

    def _reset_statistics(self, X, y, update_hyperstate):
        self._init_statistics()
        self._update(X, y, update_hyperstate)

    def _update(self, X, y, update_hyperstate):
        RegressionWrapperReward._update(self, X, y)
        if update_hyperstate:
            self.XX_tiled = np.tile(self.XX[np.newaxis, ...], [50, 1, 1])
            self.Xy_tiled = np.tile(self.Xy[np.newaxis, ...], [50, 1, 1])
            self.Llower_tiled = np.tile(self.Llower[np.newaxis, ...], [50, 1, 1])

    def _predict(self, X, update_hyperstate):
        if update_hyperstate:
            basis = _basis(X, self.random_matrix, self.bias, self.basis_dim, self.length_scale, self.signal_sd)
            basis = np.expand_dims(basis, axis=1)

            LinvXT = solve_triangular(self.Llower_tiled, np.transpose(basis, [0, 2, 1]))
            pred_sigma = np.sum(np.square(LinvXT), axis=1)*self.noise_sd**2+self.noise_sd**2
            tmp0 = np.transpose(solve_triangular(self.Llower_tiled, np.transpose(basis, [0, 2, 1])), [0, 2, 1])
            tmp1 = solve_triangular(self.Llower_tiled, self.Xy_tiled)
            pred_mu = np.matmul(tmp0, tmp1)
            pred_mu = np.squeeze(pred_mu, axis=-1)
            return pred_mu, pred_sigma
        else:
            return RegressionWrapperReward._predict(self, X)

    def _update_hyperstate(self, X, y, update_hyperstate):
        if update_hyperstate:
            basis = _basis(X, self.random_matrix, self.bias, self.basis_dim, self.length_scale, self.signal_sd)

            self.Llower_tiled = self.Llower_tiled.transpose([0, 2, 1])
            assert len(self.Llower_tiled) == len(basis)
            for i in range(len(self.Llower_tiled)):
                cholupdate(self.Llower_tiled[i], basis[i].copy())
            self.Llower_tiled = self.Llower_tiled.transpose([0, 2, 1])

            self.Xy_tiled += np.matmul(basis[:, None, :].transpose([0, 2, 1,]), y[:, None, :])
