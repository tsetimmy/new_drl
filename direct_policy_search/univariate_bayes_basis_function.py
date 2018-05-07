import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import multivariate_normal
from scipy.stats import norm

def basisFunctions(xtrain, numberOfBasis=7):
    assert numberOfBasis > 1
    numberOfBasis -= 1
    sigma = 20.
    means = np.linspace(-1., 1., numberOfBasis)

    basis = np.zeros((len(xtrain), numberOfBasis))

    for i in range(len(xtrain)):
        for j in range(numberOfBasis):
            basis[i, j] = np.exp(-pow(xtrain[i] - means[j], 2) / 2. * pow(sigma, 2))

    basis = np.concatenate([np.ones((len(xtrain), 1)), basis], axis=-1)
    return basis

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

def main():
    univariate_bayes()

if __name__ == '__main__':
    main()
