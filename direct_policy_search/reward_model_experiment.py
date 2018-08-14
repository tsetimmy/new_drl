import numpy as np
import scipy
from scipy.optimize import minimize
import matplotlib.pyplot as plt

import warnings

#import gym

def squared_exponential_kernel(a, b, signal_sd, length_scale):
    sqdist = squared_distance(a, b)
    kernel = np.square(signal_sd) * np.exp(-.5*sqdist/np.square(length_scale))
    return kernel

def squared_distance(a, b):
    sqdist = np.sum(np.square(a), axis=-1, keepdims=True) +\
             -2. * np.matmul(a, b.T) +\
             np.sum(np.square(b), axis=-1, keepdims=True).T
    return sqdist

def log_marginal_likelihood(thetas, X, y):
    assert len(thetas) == 3
    warnings.filterwarnings('error')
    try:
        length_scale, signal_sd, noise_sd = thetas
        noise_sd_clipped = np.maximum(1e-6, noise_sd)
        K = squared_exponential_kernel(X, X, signal_sd, length_scale) + np.square(noise_sd_clipped)*np.eye(len(X))
        sign, logdet = np.linalg.slogdet(K)
        if sign != 1:
            return 10e100
        loss = .5*(logdet+np.matmul(y.T, scipy.linalg.solve(K, y))[0, 0])
        print loss
        return loss
    except:
        return 10e100

def predict(x_train, y_train, length_scale, signal_sd, noise_sd, x_test):
    K = squared_exponential_kernel(x_train, x_train, signal_sd, length_scale) + np.square(noise_sd)*np.eye(len(x_train))
    k = squared_exponential_kernel(x_test, x_train, signal_sd, length_scale)
    tmp = scipy.linalg.solve(K.T, k.T).T
    mu = np.matmul(tmp, y_train)
    sigma = squared_exponential_kernel(x_test, x_test, signal_sd, length_scale) - np.matmul(tmp, k.T)

    return mu, sigma

def f(x):
    return np.sign(x)

def main():
    #Training points
    #X = np.linspace(-5, 5, 500)[..., np.newaxis]
    X = np.random.uniform(-5, 5, size=[50, 1])
    y = f(X)

    #Testing points
    Xt = np.linspace(-10, 10, 100)[..., np.newaxis]

    #Train hyperparameters
    _res = minimize(log_marginal_likelihood, np.ones(3), method='powell', args=(X, y), options={'disp': True})
    print _res.x
    length_scale, signal_sd, noise_sd = _res.x
    noise_sd = np.maximum(1e-6, noise_sd)

    mu, sigma = predict(X, y, length_scale, signal_sd, noise_sd, Xt)
    mu = np.squeeze(mu, axis=-1)
    sd = np.sqrt(np.diag(sigma))

    plt.gca().fill_between(Xt.flat, mu-3*sd, mu+3*sd, color="#dddddd")
    plt.plot(Xt, mu, 'r--')

    plt.scatter(X, y)
    plt.plot(Xt, f(Xt))

    plt.grid()
    plt.show()

if __name__ == '__main__':
    main()
