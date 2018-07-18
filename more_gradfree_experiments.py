import numpy as np
from scipy.optimize import minimize
import argparse
import gym
from gaussian_processes.gp_regression2 import gather_data

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

def log_marginal_likelihood(thetas, rffm, X, y):
    assert len(thetas) == 5
    length_scale, signal_sd, noise_sd, prior_sd, output_dim = thetas

    try:
        bases = rffm.map(X, signal_sd=signal_sd, stddev=length_scale, output_dim=int(np.round(output_dim)))

        N = len(bases.T)
        XX = np.matmul(bases.T, bases)
        Xy = np.matmul(bases.T, y)

        V0 = prior_sd**2*np.eye(N)
        tmp = np.linalg.inv(noise_sd**2*np.linalg.inv(V0) + XX)
        Vn = noise_sd**2*tmp
        wn = np.matmul(tmp, Xy)

        s1, logdet1 = np.linalg.slogdet(V0)
        s2, logdet2 = np.linalg.slogdet(Vn)
        assert s1 == 1 and s2 == 1

        lml = .5*(-N*np.log(noise_sd**2) - logdet1 + logdet2 - np.matmul(y.T, y)[0, 0]/noise_sd**2 + np.matmul(np.matmul(Xy.T, tmp.T), Xy)[0, 0]/noise_sd**2)
        loss = -lml
        print loss
        return loss
    except:
        return np.inf

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default='Pendulum-v0')
    args = parser.parse_args()

    print args

    env = gym.make(args.environment)

    states, actions, next_states = gather_data(env, 2)
    states_actions = np.concatenate([states, actions], axis=-1)

    output_dim = 64
    rffm = RandomFourierFeatureMapper(states_actions.shape[-1], int(output_dim))

    for i in range(env.observation_space.shape[0]):
        thetas0 = np.array([.316, 1., 1., 1., output_dim])
        options = {'maxiter': 2000, 'disp': True}
        _res = minimize(log_marginal_likelihood, thetas0, method='nelder-mead', args=(rffm, states_actions, next_states[:, i:i+1]), options=options)
        print _res.x



























def log_marginal_likelihood3(thetas, bf, X, y):
    length_scale, noise_sd, prior_sd = thetas
    try:
        bases = bf(X, 20, np.array([-20.]), np.array([20.]), sigma=length_scale)

        N = len(bases.T)
        XX = np.matmul(bases.T, bases)
        Xy = np.matmul(bases.T, y)

        V0 = prior_sd**2*np.eye(N)
        tmp = np.linalg.inv(noise_sd**2*np.linalg.inv(V0) + XX)
        Vn = noise_sd**2*tmp
        wn = np.matmul(tmp, Xy)

        s1, logdet1 = np.linalg.slogdet(V0)
        s2, logdet2 = np.linalg.slogdet(Vn)
        assert s1 == 1 and s2 == 1

        lml = .5*(-N*np.log(noise_sd**2) - logdet1 + logdet2 - np.matmul(y.T, y)[0, 0]/noise_sd**2 + np.matmul(np.matmul(Xy.T, tmp.T), Xy)[0, 0]/noise_sd**2)
        loss = -lml
        print loss
        return loss
    except:
        return np.inf

def mu_sigma(noise_sd, prior_sd, basisFunctions, X, y, length_scale):
    #assert len(thetas) == 5
    #length_scale, signal_sd, noise_sd, prior_sd, output_dim = thetas

    try:
        #bases = rffm.map(X, signal_sd=signal_sd, stddev=length_scale, output_dim=int(np.round(output_dim)))

        bases = basisFunctions(X, 20, np.array([-20.]), np.array([20.]), sigma=length_scale)

        N = len(bases.T)
        XX = np.matmul(bases.T, bases)
        Xy = np.matmul(bases.T, y)

        V0 = prior_sd**2*np.eye(N)
        tmp = np.linalg.inv(noise_sd**2*np.linalg.inv(V0) + XX)
        Vn = noise_sd**2*tmp
        wn = np.matmul(tmp, Xy)
        return wn, Vn
    except:
        return np.inf, np.inf

def main2():
    from direct_policy_search.univariate_bayes_basis_function import basisFunctions
    import matplotlib.pyplot as plt
    X = np.random.uniform(-20, 0, 100)
    y = 5. * np.sin(X) / X #+ np.random.normal(scale=.1, size=100)

    X = X[..., np.newaxis]
    y = y[..., np.newaxis]
    plt.scatter(X, y)

    length_scale = 1.
    noise_sd = 1.
    prior_sd = 1.
    thetas0 = np.array([length_scale, noise_sd, prior_sd])
    options = {'maxiter': 2000, 'disp': True}
    _res = minimize(log_marginal_likelihood3, thetas0, method='nelder-mead', args=(basisFunctions, X, y), options=options)
    length_scale, noise_sd, prior_sd = _res.x
    print length_scale, noise_sd, prior_sd

    mu, sigma = mu_sigma(noise_sd, prior_sd, basisFunctions, X, y, length_scale)

    X_test = np.linspace(-20, 20, 100)[..., np.newaxis]
    bases_test = basisFunctions(X_test, 20, np.array([-20]), np.array([20.]), sigma=length_scale)

    predict = np.matmul(bases_test, mu)

    predict_sig = noise_sd**2 + np.sum(np.multiply(np.matmul(bases_test, sigma), bases_test), axis=-1, keepdims=True)

    plt.errorbar(X_test, predict, yerr=np.sqrt(predict_sig), color='m', ecolor='g')
    #plt.plot(X_test, predict)
    
    plt.grid()
    plt.show()

if __name__ == '__main__':
    main()
    #main2()


