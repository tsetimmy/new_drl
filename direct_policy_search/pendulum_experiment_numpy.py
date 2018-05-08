import numpy as np
import matplotlib.pyplot as plt

from tf_bayesian_model import get_training_data
from univariate_bayes_basis_function import basisFunctions, update

import sys
sys.path.append('..')
from prototype8.dmlac.real_env_pendulum import get_next

def plot_model_data(mu, sigma, number_of_basis):
    number_of_lines = 6
    lines = np.random.multivariate_normal(mu, sigma, number_of_lines)

    u = np.linspace(-2., 2., 100)
    th = 0.
    thdot = 8.

    # Real model
    costh = []
    for i in range(len(u)):
        a, b, c = get_next(th, thdot, u[i])
        costh.append(a)
    plt.scatter(u, costh)

    # Bayesian model
    states = []
    for i in range(len(u)):
        states.append([np.cos(th), np.sin(th), thdot, u[i]])
    states = np.stack(states, axis=0)
    states_basis = basisFunctions(states, number_of_basis, low=np.array([-1., -1., -8., -2.]), high=np.array([1., 1., 8., 2.]))

    for line in lines:
        y = np.matmul(states_basis, line)
        plt.scatter(u, y)

    plt.grid()
    plt.show()

def pendulum_experiment_numpy():
    sigma = .01
    number_of_basis = 20
    training_points = 400*10

    noise_sd = .2
    prior_precision = 2.
    likelihood_sd = noise_sd

    xtrain, ytrain = get_training_data(training_points)
    xtrain_basis = basisFunctions(xtrain, number_of_basis, low=np.array([-1., -1., -8., -2.]), high=np.array([1., 1., 8., 2.]), sigma=sigma)

    prior_mean = np.zeros(number_of_basis)
    prior_sigma = np.eye(number_of_basis) / prior_precision

    mu, sigma = update(xtrain_basis, ytrain, 1. / noise_sd ** 2, prior_mean, prior_sigma)

    plot_model_data(mu, sigma, number_of_basis)

def main():
    pendulum_experiment_numpy()

if __name__ == '__main__':
    main()
