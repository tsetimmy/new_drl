import numpy as np
import scipy
from scipy.optimize import minimize
from hyperparameter_optimizer import log_marginal_likelihood, squared_exponential_kernel
import warnings

class RegressionWrappers:
    def __init__(self, input_dim, length_scale=.0001, signal_sd=1., noise_sd=1.):
        self.input_dim = input_dim
        self.length_scale = length_scale
        self.signal_sd = signal_sd
        self.noise_sd = noise_sd

        self.hyperparameters = np.array([self.length_scale, self.signal_sd, self.noise_sd])

    def _train_hyperparameters(self, X, y):
        '''
        warnings.filterwarnings('error')
        thetas = np.copy(self.hyperparameters)
        options = {'maxiter': 1, 'disp': True}
        _res = minimize(log_marginal_likelihood, thetas, method='powell', args=(X, y), options=options)
        self.hyperparameters = np.copy(_res.x)
        self.length_scale, self.signal_sd, self.noise_sd = self.hyperparameters
        print self.hyperparameters
        '''
        self.length_scale = -2.04393657e-02
        #self.signal_sd = 5.23822283e+00
        self.signal_sd = 20.
        self.noise_sd = 1.10004099e-03

    def _predict(self, Xt, X, y):
        K = squared_exponential_kernel(X, X, self.signal_sd, self.length_scale) + self.noise_sd**2*np.eye(len(X))
        k = squared_exponential_kernel(X, Xt, self.signal_sd, self.length_scale)

        with warnings.catch_warnings():
            warnings.simplefilter('default')
            tmp = scipy.linalg.solve(K.T, k).T

        mu = np.matmul(tmp, y)
        sigma = squared_exponential_kernel(Xt, Xt, self.signal_sd, self.length_scale) - np.matmul(tmp, k)

        return mu, sigma

def main():
    import matplotlib.pyplot as plt
    X = np.random.uniform(-4., 4., size=[10000, 1])
    #X = np.concatenate([X, np.zeros([1, 1])])
    y = f(X)# + np.random.normal(loc=0., scale=.5, size=[len(X), 1])
    X_test = np.linspace(-5., 5., 1000)[..., np.newaxis]

    regression_wrapper = RegressionWrappers(input_dim=1)
    regression_wrapper._train_hyperparameters(X, y)
    mu, sigma = regression_wrapper._predict(X_test, X, y)

    mu = np.squeeze(mu, axis=-1)
    sd = np.sqrt(np.diag(sigma))

    plt.gca().fill_between(X_test.flat, mu-3*sd, mu+3*sd, color="#dddddd")
    plt.plot(X_test, mu, 'r--')

    plt.scatter(X, y)
    plt.plot(X_test, f(X_test))

    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
