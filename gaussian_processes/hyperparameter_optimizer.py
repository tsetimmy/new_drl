import numpy as np
from scipy.optimize import minimize

def squared_distance(a, b):
    sqdist = np.sum(np.square(a), axis=-1, keepdims=True) +\
             -2. * np.matmul(a, b.T) +\
             np.sum(np.square(b), axis=-1, keepdims=True).T
    return sqdist

def squared_exponential_kernel_prime(a, b, signal_sd, length_scale):
    sqdist = squared_distance(a, b)
    exp_sqdist = np.exp(-.5*sqdist/np.square(length_scale))
    return np.multiply(np.square(signal_sd)*exp_sqdist, sqdist) / length_scale**3,\
           2.*signal_sd*exp_sqdist

def squared_exponential_kernel(a, b, signal_sd, length_scale):
    sqdist = squared_distance(a, b)
    kernel = np.square(signal_sd) * np.exp(-.5*sqdist/np.square(length_scale))
    return kernel

def log_marginal_likelihood_prime(thetas, X, y):
    assert len(thetas) == 3
    length_scale, signal_sd, noise_sd = thetas

    kernel = squared_exponential_kernel(X, X, signal_sd, length_scale) + np.square(noise_sd)*np.eye(len(X))
    kernel_inv = np.linalg.inv(kernel)
    alpha = np.matmul(kernel_inv, y)
    ak = np.matmul(alpha, alpha.T) - kernel_inv

    elements = list(squared_exponential_kernel_prime(X, X, signal_sd, length_scale)) + [2.*noise_sd*np.eye(len(X))]
    f = lambda x, y: -.5*np.trace(np.matmul(x, y))

    grads = np.array([f(ak, ele) for ele in elements])
    return grads

def log_marginal_likelihood(thetas, X, y):
    assert len(thetas) == 3
    length_scale, signal_sd, noise_sd = thetas

    try:
        L = np.linalg.cholesky(squared_exponential_kernel(X, X, signal_sd, length_scale) + np.square(noise_sd)*np.eye(len(X)))
    except:
        #print 'singular matrix.'
        return np.inf
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    lml = -.5*(np.matmul(y.T, alpha)[0, 0] + np.sum(np.log(np.diag(L))))
    loss = -lml
    #print loss
    return loss
