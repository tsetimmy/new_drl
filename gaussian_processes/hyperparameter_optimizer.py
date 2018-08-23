import numpy as np
import scipy.linalg as la
import warnings

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

    K = squared_exponential_kernel(X, X, signal_sd, length_scale) + noise_sd**2*np.eye(len(X))

    try:
        tmp0 = la.solve(K, y)
    except Exception as e:
        if 'Ill-conditioned matrix detected.' in str(e):
            with warnings.catch_warnings():
                warnings.simplefilter('default')
                tmp0 = la.solve(K, y)
            if np.allclose(np.matmul(K, tmp0), y) == False:
                'np.allclose == False (Ill-conditioned matrix detected). Returning 10e100.'
                return 10e100
        else:
            print('Unhandled exception. Exception: ' + str(e) + ' Returning 10e100.')
            return 10e100

    sign, logdet = np.linalg.slogdet(K)
    if sign != 1:
        print 'Sign of logdet is not 1. Returning 10e100.'
        return 10e100

    lml = np.matmul(y.T, tmp0)[0, 0] + logdet + np.log(2.*np.pi)*len(X)
    lml *= -.5
    return -lml

def batch_sek(a, b, signal_sd, length_scale):
    sqdist = np.sum(np.square(a), axis=-1, keepdims=True) +\
             -2. * np.matmul(a, np.transpose(b, [0, 2, 1])) +\
             np.transpose(np.sum(np.square(b), axis=-1, keepdims=True), [0, 2, 1])
    kernel = np.square(signal_sd) * np.exp(-.5*sqdist/np.square(length_scale))
    return kernel
