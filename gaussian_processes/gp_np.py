import numpy as np
import scipy
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import warnings

import sys
sys.path.append('..')
from utils import gather_data, gather_data3

from direct_policy_search.blr_regression2 import RegressionWrapper as RWL

import uuid
import pickle
import gym
import pybullet_envs
import argparse

def gather_data4(env, epochs, data_points, train=True, unpack=False):
    if env.spec.id in ['Pendulum-v0', 'MountainCarContinuous-v0']:
        return gather_data(env, epochs=epochs, unpack=unpack)
    elif train == True:
        return gather_data3(env, data_points=data_points, unpack=unpack)
    else:
        data = []
        count = 0
        while True:
            state = env.reset()
            while True:
                action = np.random.uniform(low=env.action_space.low, high=env.action_space.high)
                next_state, reward, done, _ = env.step(action)
                data.append([state, action, reward, next_state, done])
                state = np.copy(next_state)
                if done:
                    count += 1
                    break
            if count == epochs:
                break
        if unpack == False:
            return data
        else:
            states, actions, rewards, next_states = [np.stack(ele, axis=0) for ele in zip(*data)[:-1]]
            return states, actions, rewards[..., np.newaxis], next_states

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
        print e, 'Returning 10e100.'
        return 10e100

    tmp = np.sum(np.square(scipy.linalg.solve_triangular(Llower, y, lower=True)))
    sign, logdet = np.linalg.slogdet(K)

    if sign != 1:
        print 'Sign of logdet is not 1. Returning 10e100.'
        return 10e100

    lml =  tmp + logdet + np.log(2.*np.pi)*len(X)
    lml *= -.5
    return -lml

def kernel(x, y, hyperparameters, kern='matern'):
    dictionary = {'rbf': 'squared_exponential', 'periodic': 'periodic', 'matern': 'matern', 'rq': 'rational_quadratic'}
    kern = dictionary[kern] + '_kernel'
    return eval(kern)(x, y, hyperparameters)

class RegressionWrappers:
    def __init__(self, input_dim, kern='rbf'):
        assert kern in ['rbf', 'periodic', 'matern', 'rq']
        self.HP = {'rbf': np.ones(3), 'periodic': np.ones(4), 'matern': np.ones(3), 'rq': np.ones(4)}
        self.input_dim = input_dim
        self.kern = kern
        self.hyperparameters = self.HP[self.kern]

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
        K = kernel(X, X, self.hyperparameters[:-1], self.kern) + self.hyperparameters[-1]**2*np.eye(len(X))
        k = kernel(X, Xt, self.hyperparameters[:-1], self.kern)

        Llower = scipy.linalg.cholesky(K, lower=True)
        Llower_k = scipy.linalg.solve_triangular(Llower, k, lower=True)

        mu = np.matmul(Llower_k.T, scipy.linalg.solve_triangular(Llower, y, lower=True))
        sigma = kernel(Xt, Xt, self.hyperparameters[:-1], self.kern) - np.matmul(Llower_k.T, Llower_k)

#        with warnings.catch_warnings():
#            warnings.simplefilter('default')
#            tmp = scipy.linalg.solve(K.T, k).T
#
#        mu = np.matmul(tmp, y)
#        sigma = kernel(Xt, Xt, self.hyperparameters[:-1], self.kern) - np.matmul(tmp, k)

        return mu, sigma

def main3():
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default='Pendulum-v0')
    parser.add_argument("--matern-param", type=float, default=np.inf)
    args = parser.parse_args()

    print args

    env = gym.make(args.environment)

    states, actions, _, next_states = gather_data4(env, epochs=3, data_points=400, train=True, unpack=True)
    states_actions = np.concatenate([states, actions], axis=-1)

    regression_wrappers = [RegressionWrappers(env.observation_space.shape[0], kern='matern')  for _ in range(env.observation_space.shape[0])]

    for i in range(len(regression_wrappers)):
        print 'Training hyperparameters of '+str(i)+'th dimension.'
        regression_wrappers[i]._train_hyperparameters(states_actions, next_states[:, i:i+1])

    states2, actions2, _, next_states2 = gather_data4(env, epochs=1, data_points=None, train=False, unpack=True)
    states_actions2 = np.concatenate([states2, actions2], axis=-1)

    for i in range(len(regression_wrappers)):
        print i
        mu, sigma = regression_wrappers[i]._predict(states_actions2, states_actions, next_states[:, i:i+1])
        mu = np.squeeze(mu, axis=-1)
        sd = np.sqrt(np.abs(np.diag(sigma)))
        plt.figure()
        plt.errorbar(np.arange(len(mu)), mu, yerr=sd, color='m', ecolor='g')
        plt.plot(np.arange(len(next_states2)), next_states2[:, i])
        plt.grid()
        plt.savefig(str(i)+'.pdf')
        #plt.show()


def main2():

    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default='Pendulum-v0')
    #parser.add_argument("--path", type=str, default='')
    args = parser.parse_args()

    print args

    uid = str(uuid.uuid4())
    env = gym.make(args.environment)

    #states_actions, rewards, states_actions2, rewards2 = pickle.load(open(args.path, 'rb'))

    states, actions, rewards, _ = gather_data(env, 3, unpack=True)
    states_actions = np.concatenate([states, actions], axis=-1)

#    rbf = RegressionWrappers(input_dim=states_actions.shape[-1], kern='rbf')
#    rbf._train_hyperparameters(states_actions, rewards)
#
    matern = RegressionWrappers(input_dim=states_actions.shape[-1], kern='matern')
    matern._train_hyperparameters(states_actions, rewards)
#
#    rq = RegressionWrappers(input_dim=states_actions.shape[-1], kern='rq')
#    rq._train_hyperparameters(states_actions, rewards)

    states2, actions2, rewards2, _ = gather_data(env, 1, unpack=True)
    states_actions2 = np.concatenate([states2, actions2], axis=-1)

    pickle.dump([states_actions, rewards, states_actions2, rewards2], open(uid+'.p', 'wb'))

#    mu, sigma = rbf._predict(states_actions2, states_actions, rewards)
#
#    mu = np.squeeze(mu, axis=-1)
#    sd = np.sqrt(np.diag(sigma))
#
#    plt.errorbar(np.arange(len(mu)), mu, yerr=sd, color='m', ecolor='g')
#
    mu, sigma = matern._predict(states_actions2, states_actions, rewards)

    mu = np.squeeze(mu, axis=-1)
    sd = np.sqrt(np.diag(sigma))

    plt.errorbar(np.arange(len(mu)), mu, yerr=sd, color='y', ecolor='c')
#
#    mu, sigma = rq._predict(states_actions2, states_actions, rewards)
#
#    mu = np.squeeze(mu, axis=-1)
#    sd = np.sqrt(np.diag(sigma))
#
#    plt.errorbar(np.arange(len(mu)), mu, yerr=sd, color='b', ecolor='g')

    rwl = RWL(input_dim=states_actions.shape[-1], basis_dim=1024)
    rwl._train_hyperparameters(states_actions, rewards)
    rwl._reset_statistics(states_actions, rewards)

    mu, sigma = rwl._predict(states_actions2)
    plt.errorbar(np.arange(len(mu)), mu, yerr=np.sqrt(sigma), color='r', ecolor='k')


    rwl2 = RWL(input_dim=states_actions.shape[-1], basis_dim=1024, matern_param=0.)
    rwl2._train_hyperparameters(states_actions, rewards)
    rwl2._reset_statistics(states_actions, rewards)

    mu, sigma = rwl2._predict(states_actions2)
    plt.errorbar(np.arange(len(mu)), mu, yerr=np.sqrt(sigma), color='g', ecolor='b')

    plt.scatter(np.arange(len(rewards2)), rewards2)

    plt.grid()
    plt.title(uid)
    #plt.show()
    plt.savefig(uid+'.pdf')

def main():
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
    #main()
    #main2()
    main3()
