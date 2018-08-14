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
    assert (len(thetas) - 1) % 2 == 0
    warnings.filterwarnings('error')
    try:
        noise_sd = thetas[0]
        noise_sd_clipped = np.maximum(1e-4, noise_sd)
        K = kernel(X, X, np.reshape(thetas[1:], [-1, 2])) + np.square(noise_sd_clipped)*np.eye(len(X))
        sign, logdet = np.linalg.slogdet(K)
        if sign != 1:
            return 10e100
        loss = .5*(logdet+np.matmul(y.T, scipy.linalg.solve(K, y))[0, 0])
        print loss
        return loss
    except:
        return 10e100

def kernel(a, b, hyperparameters):
    assert len(hyperparameters.shape) == 2
    assert hyperparameters.shape[-1] == 2
    return sum([squared_exponential_kernel(a, b, hp[0], hp[1]) for hp in hyperparameters])

def predict(x_train, y_train, hyperparameters, noise_sd, x_test):
    K = kernel(x_train, x_train, hyperparameters) + np.square(noise_sd)*np.eye(len(x_train))
    k = kernel(x_test, x_train, hyperparameters)
    tmp = scipy.linalg.solve(K.T, k.T).T
    mu = np.matmul(tmp, y_train)
    sigma = kernel(x_test, x_test, hyperparameters) - np.matmul(tmp, k.T)

    return mu, sigma

def _basis(X, random_matrix, bias, hyperparameters, basis_dim):
    Z = []
    for rm, b, hp in zip(random_matrix, bias, hyperparameters):
        length_scale, signal_sd = hp
        x_omega_plus_bias = np.matmul(X, (1./length_scale)*rm) + b
        z = signal_sd * np.sqrt(2./basis_dim) * np.cos(x_omega_plus_bias)
        Z.append(z)
    Z = np.concatenate(Z, axis=-1)
    return Z

def log_marginal_likelihood2(thetas, X, y, random_matrix, bias, basis_dim):
    assert (len(thetas) - 2) % 2 == 0
    warnings.filterwarnings('error')
    try:
        noise_sd = thetas[0]
        prior_sd = thetas[1]
        
        basis = _basis(X, random_matrix, bias, np.reshape(thetas[2:], [-1, 2]), basis_dim)

        N = len(basis.T)
        XX = np.matmul(basis.T, basis)
        Xy = np.matmul(basis.T, y)

        tmp0 = (noise_sd/prior_sd)**2*np.eye(basis_dim*len(random_matrix)) + XX
        tmp = np.matmul(Xy.T, scipy.linalg.solve(tmp0.T, Xy))

        s, logdet = np.linalg.slogdet(np.eye(basis_dim*len(random_matrix)) + (prior_sd/noise_sd)**2*XX)
        if s != 1:
            print 'logdet is <= 0. Returning 10e100.'
            return 10e100

        lml = .5*(-N*np.log(noise_sd**2) - logdet + (-np.matmul(y.T, y)[0, 0] + tmp[0, 0])/noise_sd**2)
        loss = -lml
        #print loss
        return loss
    except:
        return 10e100

def f(x):
    return np.sign(np.sign(x) + 1.)*100.

def main():
    #Training points
    #X = np.linspace(-5, 0, 20)[..., np.newaxis]
    X = np.random.uniform(-5, 0., size=[25, 1])
    X2 = np.zeros(25)[..., np.newaxis]
    X = np.concatenate([X, X2], axis=0)

    no_kernel_terms = 20
    y = f(X) + np.random.normal(size=X.shape)

    #Testing points
    Xt = np.linspace(-10, 10, 100)[..., np.newaxis]

    #Train hyperparameters
    _res = minimize(log_marginal_likelihood, np.ones(1+no_kernel_terms*2), method='powell', args=(X, y), options={'disp': True})
    print _res.x
    noise_sd = _res.x[0]
    hyperparameters = np.copy(_res.x[1:]).reshape([-1, 2])
    noise_sd = np.maximum(1e-4, noise_sd)

    mu, sigma = predict(X, y, hyperparameters, noise_sd, Xt)
    mu = np.squeeze(mu, axis=-1)
    sd = np.sqrt(np.diag(sigma))

    plt.gca().fill_between(Xt.flat, mu-3*sd, mu+3*sd, color="#dddddd")
    plt.plot(Xt, mu, 'r--')

    plt.scatter(X, y)
    plt.plot(Xt, f(Xt))

    plt.grid()
    plt.show()

def predict2(X, y, hyperparameters, noise_sd, prior_sd, Xt, random_matrix, bias, basis_dim):
    basis = _basis(X, random_matrix, bias, hyperparameters, basis_dim)
    XX = np.matmul(basis.T, basis)
    Xy = np.matmul(basis.T, y)

    basis = _basis(Xt, random_matrix, bias, hyperparameters, basis_dim)

    tmp = (noise_sd/prior_sd)**2*np.eye(len(XX)) + XX
    predict_sigma = noise_sd**2 + np.sum(np.multiply(basis, noise_sd**2*scipy.linalg.solve(tmp, basis.T).T), axis=-1, keepdims=True)
    predict_mu = np.matmul(basis, scipy.linalg.solve(tmp, Xy))

    return predict_mu, predict_sigma

def main2():
    #Training points
    #X = np.linspace(-5, 0, 20)[..., np.newaxis]
    X = np.random.uniform(-5, 0, size=[500, 1])
    X2 = np.zeros(5)[..., np.newaxis]
    X = np.concatenate([X, X2], axis=0)

    x_dim = 1
    basis_dim = 256
    no_kernel_terms = 1
    y = f(X)

    #Initialize the random matrix
    rng_state = np.random.get_state()
    random_matrix = np.random.normal(size=[no_kernel_terms, x_dim, basis_dim])
    bias = np.random.uniform(low=0., high=2.*np.pi, size=[no_kernel_terms, basis_dim])
    np.random.set_state(rng_state)

    #Testing points
    Xt = np.linspace(-10, 10, 100)[..., np.newaxis]

    #Train hyperparameters
    _res = minimize(log_marginal_likelihood2, np.ones(2+no_kernel_terms*2), method='powell', args=(X, y, random_matrix, bias, basis_dim), options={'disp': True})
    print _res.x
    noise_sd = _res.x[0]
    prior_sd = _res.x[1]
    hyperparameters = np.copy(_res.x[2:]).reshape([-1, 2])


    mu, sigma = predict2(X, y, hyperparameters, noise_sd, prior_sd, Xt, random_matrix, bias, basis_dim)

    mu = np.squeeze(mu, axis=-1)
    #sd = np.sqrt(np.diag(sigma))
    sd = np.sqrt(np.squeeze(sigma, axis=-1))

    plt.gca().fill_between(Xt.flat, mu-3*sd, mu+3*sd, color="#dddddd")
    plt.plot(Xt, mu, 'r--')

    plt.scatter(X, y)
    plt.plot(Xt, f(Xt))

    plt.grid()
    plt.show()
    #import uuid
    #plt.savefig(str(uuid.uuid4())+'.pdf')

def main3():
    pass

if __name__ == '__main__':
    #main()
    #main2()
    main3()
