import numpy as np
import scipy
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import warnings

import sys
sys.path.append('..')
sys.path.append('../..')

from blr_regression2 import RegressionWrapper as rw2

def f(X):
    y = np.heaviside(X, 1.) * 100.
    return y

def rational_quadratic_kernel(a, b, hyperparameters):
    signal_sd, length_scale, alpha = hyperparameters

    sqdist = squared_distance(a, b)

    kernel = signal_sd**2*np.power((1. + sqdist/(2.*np.abs(alpha)*length_scale**2)), -np.abs(alpha))

    return kernel


def matern_kernel(a, b, hyperparameters):
    signal_sd, length_scale = hyperparameters

    sqdist = squared_distance(a, b)

    #v=1/2
    kernel = signal_sd**2*np.exp(-np.sqrt(sqdist)/np.abs(length_scale))

    #v=3/2
    #kernel = signal_sd**2*(1. + np.sqrt(3.*sqdist)/np.abs(length_scale))*np.exp(-np.sqrt(3.*sqdist)/np.abs(length_scale))

    #v=5/2
    #kernel = signal_sd**2*(1. + np.sqrt(5.*sqdist)/np.abs(length_scale) + (5.*sqdist)/(3.*length_scale**2))*np.exp(-np.sqrt(5.*sqdist)/np.abs(length_scale))
    return kernel

def periodic_kernel(a, b, hyperparameters):
    s, p, l = hyperparameters

    a_ = np.tile(np.expand_dims(a, axis=1), [1, len(b), 1])
    b_ = np.tile(b[np.newaxis, ...], [len(a), 1, 1])

    c = np.sum(np.square(np.sin((a_ - b_)/p)), axis=-1)
    tmp1 = np.exp(-2.*c/l**2)

    d = squared_distance(a, b)
    tmp2 = np.exp(-.5*d/l**2)
    
    ret = s**2*np.multiply(tmp1, tmp2)
    return ret

def squared_distance(a, b):
    sqdist = np.sum(np.square(a), axis=-1, keepdims=True) +\
             -2. * np.matmul(a, b.T) +\
             np.sum(np.square(b), axis=-1, keepdims=True).T
    return np.maximum(sqdist, 0.)

def squared_exponential_kernel(a, b, hyperparameters, *unused):
    signal_sd, length_scale = hyperparameters
    sqdist = squared_distance(a, b)
    kernel = np.square(signal_sd) * np.exp(-.5*sqdist/np.square(length_scale))
    return kernel

def log_marginal_likelihood(thetas, X, y, kern):
    K = kernel(X, X, thetas[:-1], kern) + thetas[-1]**2*np.eye(len(X))

    try:
        Llower = scipy.linalg.cholesky(K, lower=True)
    except Exception as e:
        print e
        return 10e100

    Linv_y = scipy.linalg.solve_triangular(Llower, y, lower=True)

    sign, logdet = np.linalg.slogdet(K)
    if sign != 1:
        print 'Sign of logdet is not 1. Returning 10e100.'
        return 10e100

    lml = np.matmul(Linv_y.T, Linv_y)[0, 0] + logdet + np.log(2.*np.pi)*len(X)
    lml *= -.5
    return -lml

def kernel(x, y, hyperparameters, kern='matern'):
    if kern == 'rbf':
        return squared_exponential_kernel(x, y, hyperparameters)
    elif kern == 'periodic':
        return periodic_kernel(x, y, hyperparameters)
    elif kern == 'matern':
        return matern_kernel(x, y, hyperparameters)
    elif kern == 'rq':
        return rational_quadratic_kernel(x, y, hyperparameters)

class RegressionWrappers:
    def __init__(self, input_dim, kern='rbf'):
        assert kern in ['rbf', 'periodic', 'matern', 'rq']
        print kern
        self.input_dim = input_dim
        self.kern = kern

        if self.kern == 'rbf':
            self.hyperparameters = np.ones(3)
        elif self.kern == 'periodic':
            self.hyperparameters = np.ones(4)
        elif self.kern == 'matern':
            self.hyperparameters = np.ones(3)
            self.hyperparameters[0] = 2.
            self.hyperparameters[1] = 1.
            self.hyperparameters[2] = .0001
        elif self.kern == 'rq':
            self.hyperparameters = np.ones(4)

    def _train_hyperparameters(self, X, y):
        warnings.filterwarnings('error')
        '''
        import cma
        thetas = np.copy(self.hyperparameters)
        options = {'maxiter': 1000, 'verb_disp': 1, 'verb_log': 0}
        res = cma.fmin(log_marginal_likelihood, thetas, 2., args=(X, y, self.kern), options=options)
        self.hyperparameters = np.copy(res[0])
        '''
        thetas = np.copy(self.hyperparameters)
        options = {'maxiter': 1000, 'disp': True}
        _res = minimize(log_marginal_likelihood, thetas, method='powell', args=(X, y, self.kern), options=options)
        self.hyperparameters = np.copy(_res.x)
        print self.hyperparameters

    def _predict(self, Xt, X, y):
        print self.kern
        K = kernel(X, X, self.hyperparameters[:-1], self.kern) + self.hyperparameters[-1]**2*np.eye(len(X))
        k = kernel(X, Xt, self.hyperparameters[:-1], self.kern)

        Llower = scipy.linalg.cholesky(K, lower=True)
        Linv_k = scipy.linalg.solve_triangular(Llower, k, lower=True)
        Linv_y = scipy.linalg.solve_triangular(Llower, y, lower=True)

        mu = np.matmul(Linv_k.T, Linv_y)
        kss = kernel(Xt, Xt, self.hyperparameters[:-1], self.kern)
        sigma = kss  - np.matmul(Linv_k.T, Linv_k)

        return mu, sigma

def main():
    kern = 'matern'
    X = np.random.uniform(-1.8, 0., size=[998*3, 1])
    X = np.concatenate([X, np.zeros([5, 1])])
    y = f(X) - np.abs(np.random.normal(loc=0., scale=.1, size=X.shape))
    X_test = np.linspace(-2., 2., 50)[..., np.newaxis]

    #GPs
#    regression_wrapper = RegressionWrappers(input_dim=1, kern=kern)
#    #regression_wrapper._train_hyperparameters(X, y)
#    mu, sigma = regression_wrapper._predict(X_test, X, y)
#
#    '''
#    plt.figure()
#
#    L = scipy.linalg.cholesky(kernel(X_test, X_test, regression_wrapper.hyperparameters[:-1], kern) + regression_wrapper.hyperparameters[-1]*np.eye(len(X_test)), lower=True)
#    f_prior = np.matmul(L, np.random.normal(size=[len(X_test), 20]))
#    plt.plot(X_test, f_prior)
#    plt.grid()
#    plt.show()
#    '''
#    
#
#    plt.figure()
#    mu = np.squeeze(mu, axis=-1)
#    sd = np.sqrt(np.diag(sigma))
#
#    plt.gca().fill_between(X_test.flat, mu-3*sd, mu+3*sd, color="#dddddd")
#    plt.plot(X_test, mu, 'r--')
#
#    plt.scatter(X, y)
#    plt.plot(X_test, f(X_test))
#
#    plt.grid()
#    import uuid
#    #plt.savefig(str(uuid.uuid4()) + '.pdf')
#    plt.show()
#    exit()

    input_dim = 1
    feature_dim = 100
    random_projection_matrix = np.random.normal(loc=0., scale=1./np.sqrt(feature_dim), size=[input_dim, feature_dim])

    #RFFMs
    rw = rw2(input_dim=feature_dim, basis_dim=512, matern_param=0.)
    rw._train_hyperparameters(np.matmul(X, random_projection_matrix), y)
    rw.length_scale = 5.
    rw.signal_sd = 1.
    rw.noise_sd = 1.
    rw.prior_sd = 15.
    rw.hyperparameters = np.array([rw.length_scale, rw.signal_sd, rw.noise_sd, rw.prior_sd])

    rw._reset_statistics(np.matmul(X, random_projection_matrix), y)
    mu, sigma = rw._predict(np.matmul(X_test, random_projection_matrix))

    plt.figure()
    mu = np.squeeze(mu, axis=-1)
    sd = np.squeeze(np.sqrt(sigma), axis=-1)

    plt.gca().fill_between(X_test.flat, mu-3*sd, mu+3*sd, color="#dddddd")
    plt.plot(X_test, mu, 'r--')

    plt.scatter(X, y)
    plt.plot(X_test, f(X_test))

    plt.grid()
    import uuid
    #plt.savefig(str(uuid.uuid4()) + '.pdf')
    plt.show()




if __name__ == '__main__':
    main()
