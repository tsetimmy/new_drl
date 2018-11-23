import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize
import warnings

import sys
sys.path.append('..')
from direct_policy_search.blr_regression2 import RegressionWrapper, _basis

import matplotlib.pyplot as plt

class MultiOutputRegressionWrapper(RegressionWrapper):
    def __init__(self, input_dim, output_dim, basis_dim, length_scale=1., signal_sd=1., noise_sd=5e-4, prior_sd=1., rffm_seed=1, train_hp_iterations=2000, matern_param=np.inf):
        self.output_dim = output_dim
        RegressionWrapper.__init__(self, input_dim, basis_dim, length_scale, signal_sd, noise_sd, prior_sd, rffm_seed, train_hp_iterations, matern_param)

    def _init_statistics(self):
        self.XX = np.zeros([self.basis_dim, self.basis_dim])
        self.Xy = np.zeros([self.basis_dim, self.output_dim])

    def _update(self, X, y):
        assert len(X.shape) == 2
        assert len(y.shape) == 2
        assert X.shape[0] == y.shape[0]

        basis = _basis(X, self.random_matrix, self.bias, self.basis_dim, self.length_scale, self.signal_sd)
        self.XX += np.matmul(basis.T, basis)
        self.Xy += np.matmul(basis.T, y)

        #TODO: perform a rank-1 cholesky update?
        self.Llower = la.cholesky((self.noise_sd/self.prior_sd)**2*np.eye(self.basis_dim) + self.XX, lower=True)

    def _train_hyperparameters(self, X, y):
        warnings.filterwarnings('error')

        thetas = np.copy(np.array([self.length_scale, self.signal_sd, self.noise_sd, self.prior_sd]))
        options = {'maxiter': self.train_hp_iterations, 'disp': True}
        _res = minimize(self._log_marginal_likelihood, thetas, method='powell', args=(X, y), options=options)
        results = np.copy(_res.x)

        self.length_scale, self.signal_sd, self.noise_sd, self.prior_sd = results
        self.length_scale = np.abs(self.length_scale)
        self.signal_sd = np.abs(self.signal_sd)
        #self.noise_sd = np.abs(self.noise_sd)
        self.noise_sd = np.sqrt(self.noise_sd**2 + self.c*self.prior_sd**2)
        self.prior_sd = np.abs(self.prior_sd)
        self.hyperparameters = np.array([self.length_scale, self.signal_sd, self.noise_sd, self.prior_sd])
        print self.length_scale, self.signal_sd, self.noise_sd, self.prior_sd

    def _log_marginal_likelihood(self, thetas, X, y):
        try:
            length_scale, signal_sd, noise_sd, prior_sd = thetas

            noise_sd2 = np.sqrt(noise_sd**2 + self.c*prior_sd**2)

            basis = _basis(X, self.random_matrix, self.bias, self.basis_dim, np.abs(length_scale), np.abs(signal_sd))
            N = len(basis.T)
            XX = np.matmul(basis.T, basis)
            Xy = np.matmul(basis.T, y)

            tmp0 = (noise_sd2/prior_sd)**2*np.eye(self.basis_dim) + XX

            Llower = la.cholesky(tmp0, lower=True)
            LinvXy = la.solve_triangular(Llower, Xy, lower=True)
            tmp = np.sum(np.square(LinvXy))

            s, logdet = np.linalg.slogdet(np.eye(self.basis_dim) + (prior_sd/noise_sd2)**2*XX)
            if s != 1:
                print 'logdet is <= 0. Returning 10e100.'
                return 10e100

            lml = .5*(-N*np.log(noise_sd2**2)*self.output_dim - logdet*self.output_dim + (-np.sum(np.square(y)) + tmp)/noise_sd2**2)
            loss = -lml
            return loss
        except Exception as e:
            print '------------'
            print e, 'Returning 10e100.'
            print '************'
            return 10e100

    def _predict(self, X):
        basis = _basis(X, self.random_matrix, self.bias, self.basis_dim, self.length_scale, self.signal_sd)

        predict_sigma = np.sum(np.square(la.solve_triangular(self.Llower, basis.T, lower=True)), axis=0) * self.noise_sd**2 + self.noise_sd**2
        predict_sigma = predict_sigma[..., np.newaxis]
        tmp0 = la.solve_triangular(self.Llower, basis.T, lower=True).T
        tmp1 = la.solve_triangular(self.Llower, self.Xy, lower=True)
        predict_mu = np.matmul(tmp0, tmp1)

        return predict_mu, predict_sigma

def main():

    X = np.random.uniform(-20., 20., size=[100, 1])
    y0 = np.cos(X)
    '''
    y1 = np.sin(X)
    y2 = np.sin(X) / X
    y3 = np.abs(np.cos(X))
    y4 = np.abs(np.sin(X))
    y5 = np.abs(np.sin(X) / X)
    '''

    #Y = np.concatenate([y0, y1, y2, y3, y4, y5], axis=-1)
    Y = np.concatenate([y0], axis=-1)
    Y += np.random.normal(loc=0., scale=0.5, size=Y.shape)

    rw = RegressionWrapper(1, 256)
    rw._train_hyperparameters(X, Y)
    rw._reset_statistics(X, Y)

    morw = MultiOutputRegressionWrapper(X.shape[-1], Y.shape[-1], 256)
    morw._train_hyperparameters(X, Y)
    morw._reset_statistics(X, Y)

    X_test = np.linspace(-20, 20, 100)[..., None]

    predict_mu0, predict_sigma0 = rw._predict(X_test)
    predict_mu1, predict_sigma1= morw._predict(X_test)

    print predict_mu0.shape
    print predict_mu1.shape
    print predict_sigma0.shape
    print predict_sigma1.shape


    for i in range(predict_mu1.shape[-1]):
        plt.figure()
        plt.errorbar(X_test, predict_mu0[:, i:i+1], yerr=np.sqrt(predict_sigma0[:, i:i+1]), color='m', ecolor='g')
        plt.grid()

        plt.figure()
        plt.errorbar(X_test, predict_mu1[:, i:i+1], yerr=np.sqrt(predict_sigma1[:, i:i+1]), color='r', ecolor='b')
        plt.grid()

        plt.show()





if __name__ == '__main__':
    main()
