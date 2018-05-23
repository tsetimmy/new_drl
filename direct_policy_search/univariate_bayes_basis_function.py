import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import multivariate_normal
from scipy.stats import norm

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def basisFunctions(xtrain, numberOfBasis=20, low=np.array([-1.]), high=np.array([1.]), sigma=20.):
    xtrain = np.atleast_1d(xtrain)
    if xtrain.ndim == 1:
        xtrain = xtrain[..., np.newaxis]

    assert numberOfBasis > 1
    assert xtrain.shape[-1] == len(low)
    np.testing.assert_array_equal(-low, high)

    numberOfBasis -= 1
    numberOfBasisOriginal = numberOfBasis
    grid_intervals = int(np.ceil(numberOfBasis ** (1. / len(low))))
    numberOfBasis = grid_intervals ** len(low)

    if numberOfBasis != numberOfBasisOriginal:
        print 'Warning, number of basis is', numberOfBasis

    grid = [np.linspace(low[i], high[i], grid_intervals) for i in range(len(low))]
    means = np.meshgrid(*grid)
    means = np.stack([m.flatten() for m in means], axis=-1)
    assert len(means) == numberOfBasis

    means = means.T
    norm_of_difference = np.square(np.linalg.norm(xtrain, axis=-1, keepdims=True)) + (-2. * np.matmul(xtrain, means)) +\
                         np.square(np.linalg.norm(means, axis=0, keepdims=True))
    bases = np.exp(-norm_of_difference / 2. * pow(sigma, 2))
    bases = np.concatenate([np.ones((len(xtrain), 1)), bases], axis=-1)
    return bases

def plotSampleLines(mu, sigma, numberOfLines, dataPoints, numberOfBasis):
    x = np.linspace(-1., 1., 50)
    xbasis = basisFunctions(x, numberOfBasis)
    lines = np.random.multivariate_normal(mu, sigma, numberOfLines)

    for line in lines:
        y = np.matmul(xbasis, line)
        plt.plot(x, y)
    
    plt.scatter(dataPoints[0], dataPoints[1])
    plt.grid()
    plt.show()

def update(xtrain, ytrain, likelihoodPrecision, priorMu, priorSigma):
    xtrain = np.atleast_2d(xtrain)
    #xtrain = np.stack([np.ones_like(xtrain), xtrain], axis=-1)
    ytrain = np.atleast_1d(ytrain)

    sigma = np.linalg.inv(np.linalg.inv(priorSigma) + likelihoodPrecision * np.matmul(xtrain.T, xtrain))
    mu = np.matmul(np.matmul(sigma, np.linalg.inv(priorSigma)), priorMu) + \
         likelihoodPrecision * np.matmul(np.matmul(sigma, xtrain.T), ytrain)
    
    return mu, sigma

def univariate_bayes():
    a0 = -.3
    a1 = .5
    numberOfBasis = 20

    trainingPoints = 100
    noiseSD = .2
    priorPrecision = 2.
    likelihoodSD = noiseSD

    #Generate the training points
    xtrain = np.random.uniform(-1., 1., size=trainingPoints)
    xtrain2 = basisFunctions(xtrain, numberOfBasis=numberOfBasis)
    #ytrain = (a0 + a1 * xtrain) + np.random.normal(loc=0., scale=noiseSD, size=trainingPoints)
    ytrain = np.sin(10.*xtrain) + np.random.normal(loc=0., scale=noiseSD, size=trainingPoints)

    #Plot prior
    priorMean = np.zeros(numberOfBasis)
    priorSigma = np.eye(numberOfBasis) / priorPrecision
    #contourPlot(priorMean, priorSigma, [a0, a1])

    #Plot sample lines

    iterations = 1
    mu = priorMean
    sigma = priorSigma
    for i in range(iterations):
        #Plot the likelihood
        #contourPlotLikelihood(ytrain[i], xtrain[i], likelihoodSD, [a0, a1])

        #Update prior to posterior
        mu, sigma = update(xtrain2[i], ytrain[i], 1. / noiseSD**2, mu, sigma)
        #contourPlot(mu, sigma, [a0, a1])

        #Plot sample lines
        plotSampleLines(mu, sigma, 6, [xtrain[0 : i + 1], ytrain[0 : i + 1]], numberOfBasis)

    mu, sigma = update(xtrain2, ytrain, 1. / noiseSD**2, priorMean, priorSigma)
    #contourPlot(mu, sigma, [a0, a1])
    plotSampleLines(mu, sigma, 6, [xtrain, ytrain], numberOfBasis)

def multivariate_domain_bayes():
    #True parameters
    a0 = .1
    a1 = -.2
    a2 = .5

    noise_sd = .2
    prior_precision = 2.
    likelihood_sd = noise_sd
    number_of_lines = 7

    #Generate the training points
    training_points = 100*800
    xtrain = np.random.uniform(-1., 1., size=[training_points, 2])
    ytrain = a0 + a1 * xtrain[:, 0] + a2 * xtrain[:, 1] + np.random.normal(loc=0., scale=noise_sd, size=training_points)

    prior_mean = np.zeros(3)
    prior_sigma = np.eye(3) / prior_precision

    xtrain = np.concatenate([np.ones([len(xtrain), 1]), xtrain], axis=-1)
    mu, sigma = update(xtrain, ytrain, 1. / noise_sd ** 2, prior_mean, prior_sigma)

    #Mesh grid
    X = np.linspace(-1., 1., 100)
    Y = np.linspace(-1., 1., 100)
    X, Y = np.meshgrid(X, Y)

    Z = a0 * a1 * X + a2 * Y

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    #Plot the experiments
    lines = np.random.multivariate_normal(mu, sigma, number_of_lines)

    for line in lines:
        Z = line[0] + line[1] * X + line[2] * Y
        surf = ax.plot_surface(X, Y, Z, cmap=cm.jet, linewidth=0, antialiased=False)

    ax.scatter(xtrain[:,1], xtrain[:,2], ytrain)

    # Customize the z axis.
    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

def multivariate_domain_nonlinear_bayes():
    noise_sd = .2
    prior_precision = 2.
    likelihood_sd = noise_sd
    number_of_lines = 1
    no_basis = 50
    sigma_basis = 4.5

    #Generate the training points
    training_points = 100*4
    xtrain = np.random.uniform(-1., 1., size=[training_points, 2])
    ytrain = np.sin(4.*np.sqrt(xtrain[:, 0]**2 + xtrain[:, 1]**2)) + np.random.normal(loc=0., scale=noise_sd, size=training_points)
    xtrain2 = basisFunctions(xtrain, numberOfBasis=no_basis, low=np.array([-1., -1.]), high=np.array([1., 1.]), sigma=sigma_basis)

    prior_mean = np.zeros(no_basis)
    prior_sigma = np.eye(no_basis) / prior_precision

    mu, sigma = update(xtrain2, ytrain, 1. / noise_sd ** 2, prior_mean, prior_sigma)

    #Mesh grid
    X = np.linspace(-1., 1., 100)
    Y = np.linspace(-1., 1., 100)
    X, Y = np.meshgrid(X, Y)

    Z = np.sin(4.*np.sqrt(X**2 + Y**2))

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    #surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    #Plot the experiments
    lines = np.random.multivariate_normal(mu, sigma, number_of_lines)

    for line in lines:
        basis = basisFunctions(np.stack([X.flatten(), Y.flatten()], axis=-1), numberOfBasis=no_basis, low=np.array([-1., -1.]), high=np.array([1., 1.]), sigma=sigma_basis)
        Z = np.matmul(basis, line).reshape(X.shape)
        surf = ax.plot_surface(X, Y, Z, cmap=cm.jet, linewidth=0, antialiased=False)

    #ax.scatter(xtrain[:,0], xtrain[:,1], ytrain)

    # Customize the z axis.
    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

def main():
    #univariate_bayes()
    #multivariate_domain_bayes()
    multivariate_domain_nonlinear_bayes()

if __name__ == '__main__':
    #basisFunctions(np.zeros([100, 4]), numberOfBasis=20, low=np.array([-1., -2., -3., -4.]), high=np.array([1., 2., 3., 4.]))
    main()
