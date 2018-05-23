import numpy as np
import matplotlib.pyplot as plt

from tf_bayesian_model import get_training_data
from univariate_bayes_basis_function import basisFunctions, update

import sys
sys.path.append('..')
from prototype8.dmlac.real_env_pendulum import get_next

def plot_model_data2(mu, sigma, number_of_basis, sigma_basis):
    number_of_lines = 6
    lines = np.random.multivariate_normal(mu, sigma, number_of_lines)

    th = np.linspace(-np.pi, np.pi, 10)
    thdot = np.linspace(-8., 8., 20)
    u = np.linspace(-2., 2., 10)

    # Real model
    costh = []
    sinth = []
    newthdot = []
    for i in range(len(u)):
        for j in range(len(thdot)):
            for k in range(len(th)):
                a, b, c = get_next(th[k], thdot[j], u[i])
                costh.append(a)
                sinth.append(b)
                newthdot.append(c)

    plt.scatter(np.arange(len(costh)) / 10., costh)
    #plt.scatter(np.arange(len(sinth)) / 10., sinth)
    #plt.scatter(np.arange(len(newthdot)) / 10., newthdot)
    #plt.grid()

    # Bayesian model
    states = []
    for i in range(len(u)):
        for j in range(len(thdot)):
            for k in range(len(th)):
                states.append([np.cos(th[k]), np.sin(th[k]), thdot[j], u[i]])
    states = np.stack(states, axis=0)
    states_basis = basisFunctions(states, number_of_basis, low=np.array([-1., -1., -8., -2.]), high=np.array([1., 1., 8., 2.]), sigma=sigma_basis)

    for line in lines:
        y = np.matmul(states_basis, line)
        plt.scatter(np.arange(len(y)) / 10., y)

    plt.grid()
    plt.show()

def plot_model_data(mu, sigma, number_of_basis, sigma_basis):
    number_of_lines = 6
    lines = np.random.multivariate_normal(mu, sigma, number_of_lines)

    th = np.linspace(-np.pi, np.pi, 30)
    thdot = np.linspace(-8., 8., 30)
    u = np.linspace(-2., 2., 30)

    # Real model
    costh = []

    for i in range(len(th)):
        for j in range(len(thdot)):
            for k in range(len(u)):
                a, b, c = get_next(th[i], thdot[j], u[k])
                costh.append(a)

    plt.scatter(np.arange(len(costh)) / 10., costh)

    # Bayesian model
    states = []
    for i in range(len(th)):
        for j in range(len(thdot)):
            for k in range(len(u)):
                states.append([np.cos(th[i]), np.sin(th[i]), thdot[j], u[k]])
    states = np.stack(states, axis=0)
    states_basis = basisFunctions(states, number_of_basis, low=np.array([-1., -1., -8., -2.]), high=np.array([1., 1., 8., 2.]), sigma=sigma_basis)

    for line in lines:
        y = np.matmul(states_basis, line)
        plt.scatter(np.arange(len(y)) / 10., y)

    plt.grid()
    plt.show()

def pendulum_experiment_numpy():
    sigma_basis = .09
    number_of_basis = (5**4)+1
    training_points = 400*5

    noise_sd = .2
    prior_precision = 2.
    likelihood_sd = noise_sd

    xtrain, ytrain = get_training_data(training_points)
    xtrain_basis = basisFunctions(xtrain, number_of_basis, low=np.array([-1., -1., -8., -2.]), high=np.array([1., 1., 8., 2.]), sigma=sigma_basis)

    prior_mean = np.zeros(number_of_basis)
    prior_sigma = np.eye(number_of_basis) / prior_precision

    mu, sigma = update(xtrain_basis, ytrain, 1. / noise_sd ** 2, prior_mean, prior_sigma)

    plot_model_data(mu, sigma, number_of_basis, sigma_basis)
    plot_model_data2(mu, sigma, number_of_basis, sigma_basis)

def main():
    pendulum_experiment_numpy()

if __name__ == '__main__':
    main()
