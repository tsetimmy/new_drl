import numpy as np
from scipy.optimize import minimize
import argparse
import gym

from utils import gather_data, gather_data2

class RandomFourierFeatureMapper:
    def __init__(self, input_dim, output_dim, stddev=1., seed=1):
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._stddev = stddev
        self._seed = seed

    def map(self, input_tensor, signal_sd=1., stddev=None, output_dim=None):
        assert len(input_tensor.shape) == 2
        assert input_tensor.shape[-1] == self._input_dim

        if stddev is not None:
            _stddev = stddev
        else:
            _stddev = self._stddev

        if output_dim is not None:
            _output_dim = output_dim
        else:
            _output_dim = self._output_dim

        rng_state = np.random.get_state()#Get rng state.
        np.random.seed(self._seed)

        omega_matrix_shape = [self._input_dim, _output_dim]
        bias_shape = [_output_dim]

        omega_matrix = np.random.normal(scale=1./_stddev, size=omega_matrix_shape)
        bias = np.random.uniform(low=0., high=2.*np.pi, size=bias_shape)

        x_omega_plus_bias = np.matmul(input_tensor, omega_matrix) + bias
        z = signal_sd * np.sqrt(2./_output_dim) * np.cos(x_omega_plus_bias)

        np.random.set_state(rng_state)#Set rng state.
        return z

def basis_func(rffm, X, output_dim, signal_sd, length_scale):
    return rffm.map(X, signal_sd=signal_sd, stddev=length_scale, output_dim=int(np.round(output_dim)))

def posterior(XX, Xy, noise_sd, prior_sd):
    V0 = prior_sd**2*np.eye(len(XX))
    tmp = np.linalg.inv(noise_sd**2*np.linalg.inv(V0) + XX)
    Vn = noise_sd**2*tmp
    wn = np.matmul(tmp, Xy)
    return wn, Vn, V0, tmp

def log_marginal_likelihood(thetas, rffm, X, y, output_dim, noise_sd_clip_threshold=None):
    assert len(thetas) == 4
    length_scale, signal_sd, noise_sd, prior_sd = thetas

    try:
        bases = basis_func(rffm, X, output_dim, signal_sd, length_scale)

        if noise_sd_clip_threshold is None:
            noise_sd_clipped = noise_sd
        else:
            noise_sd_clipped = np.maximum(noise_sd, noise_sd_clip_threshold)

        N = len(bases.T)
        XX = np.matmul(bases.T, bases)
        Xy = np.matmul(bases.T, y)

        wn, Vn, V0, tmp = posterior(XX, Xy, noise_sd_clipped, prior_sd)

        s1, logdet1 = np.linalg.slogdet(V0)
        s2, logdet2 = np.linalg.slogdet(Vn)
        assert s1 == 1 and s2 == 1

        lml = .5*(-N*np.log(noise_sd_clipped**2) - logdet1 + logdet2 - np.matmul(y.T, y)[0, 0]/noise_sd_clipped**2 + np.matmul(np.matmul(Xy.T, tmp.T), Xy)[0, 0]/noise_sd_clipped**2)
        loss = -lml
        return loss
    except:
        return np.inf

class predictor:
    def __init__(self, dim, XX=None, Xy=None, length_scale=1., signal_sd=1., noise_sd=1., prior_sd=1.):
        self.dim = dim

        if XX is not None:
            self.XX = XX
        else:
            self.XX = np.zeros([dim, dim])

        if Xy is not None:
            self.Xy = Xy
        else:
            self.Xy = np.zeros([dim, 1])

        self.length_scale = length_scale
        self.signal_sd = signal_sd
        self.noise_sd = noise_sd
        self.prior_sd = prior_sd

    def update(self, rffm, X, y):
        bases = basis_func(rffm, X, self.dim, self.signal_sd, self.length_scale)

        self.XX += np.matmul(bases.T, bases)
        self.Xy += np.matmul(bases.T, y)

    def predict(self, rffm, X):
        bases = basis_func(rffm, X, self.dim, self.signal_sd, self.length_scale)

        mu, sigma, _, _ = posterior(self.XX, self.Xy, self.noise_sd, self.prior_sd)

        predict_mu = np.matmul(bases, mu)
        predict_sigma = self.noise_sd**2 + np.sum(np.multiply(np.matmul(bases, sigma), bases), axis=-1, keepdims=True)

        return predict_mu, predict_sigma

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default='Pendulum-v0')
    parser.add_argument("--train-hp-iterations", type=int, default=2000)
    args = parser.parse_args()

    print args

    env = gym.make(args.environment)

    states, actions, next_states = gather_data(env, 3, unpack=True)
    states_actions = np.concatenate([states, actions], axis=-1)

    output_dim = 128*2
    noise_sd_clip_threshold = 5e-5
    rffm = RandomFourierFeatureMapper(states_actions.shape[-1], int(output_dim))

    hyperparameters = []
    for i in range(env.observation_space.shape[0]):
        thetas0 = np.array([1., 1., 5e-4, 1.])
        options = {'maxiter': args.train_hp_iterations, 'disp': True}
        _res = minimize(log_marginal_likelihood, thetas0, method='nelder-mead', args=(rffm, states_actions, next_states[:, i:i+1], output_dim, noise_sd_clip_threshold), options=options)
        length_scale, signal_sd, noise_sd, prior_sd = _res.x
        hyperparameters.append([length_scale, signal_sd, np.maximum(noise_sd, noise_sd_clip_threshold), prior_sd])
    print hyperparameters

    # Quick plotting experiment (for sanity check).
    import matplotlib.pyplot as plt
    if args.environment == 'Pendulum-v0':
        states2, actions2, next_states2 = gather_data(env, 1, unpack=True)
    elif args.environment == 'MountainCarContinuous-v0':
        from utils import mcc_get_success_policy
        states2, actions2, next_states2 = mcc_get_success_policy(env)
    states_actions2 = np.concatenate([states2, actions2], axis=-1)

    predictors = []

    for i in range(env.observation_space.shape[0]):
        plt.subplot(2, env.observation_space.shape[0], i+1)
        length_scale, signal_sd, noise_sd, prior_sd = hyperparameters[i]

        predict = predictor(output_dim, length_scale=length_scale, signal_sd=signal_sd, noise_sd=noise_sd, prior_sd=prior_sd)
        predict.update(rffm, states_actions, next_states[:, i:i+1])
        predict_mu, predict_sigma = predict.predict(rffm, states_actions2)
        predictors.append(predict)

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
            predict_mu, predict_sigma = predictors[i].predict(rffm, state_action)
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
    main()
