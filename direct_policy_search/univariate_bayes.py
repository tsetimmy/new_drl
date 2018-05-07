import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import multivariate_normal
from scipy.stats import norm

def plotSampleLines(mu, sigma, numberOfLines, dataPoints):
    x = np.linspace(-1., 1., 2)
    lines = np.random.multivariate_normal(mu, sigma, numberOfLines)

    for line in lines:
        y = line[0] + line[1] * x
        plt.plot(x, y)
    
    plt.scatter(dataPoints[0], dataPoints[1])
    plt.grid()
    plt.show()

def update(xtrain, ytrain, likelihoodPrecision, priorMu, priorSigma):
    xtrain = np.atleast_1d(xtrain)
    xtrain = np.stack([np.ones_like(xtrain), xtrain], axis=-1)
    ytrain = np.atleast_1d(ytrain)

    sigma = np.linalg.inv(np.linalg.inv(priorSigma) + likelihoodPrecision * np.matmul(xtrain.T, xtrain))

    mu = np.matmul(np.matmul(sigma, np.linalg.inv(priorSigma)), priorMu) + \
         likelihoodPrecision * np.matmul(np.matmul(sigma, xtrain.T), ytrain)
    
    return mu, sigma

def contourPlotLikelihood(y, x, std, truth=None):
    xv, yv = np.meshgrid(np.linspace(-1., 1., 10*5), np.linspace(-1., 1., 10*5))
    data = np.stack([xv.flatten(), yv.flatten()], axis=-1)

    p = norm.pdf(y - (data[:, 0] + data[:, 1] * x), loc=0, scale=std).reshape(xv.shape)
    cp = plt.contourf(xv, yv, p, cmap=cm.jet)
    plt.colorbar(cp)
    if truth is not None:
        assert len(truth) == 2
        plt.scatter(truth[0], truth[1], marker='x')
    plt.show()

def contourPlot(mean, std, truth=None):
    xv, yv = np.meshgrid(np.linspace(-1., 1., 10*5), np.linspace(-1., 1., 10*5))
    data = np.stack([xv.flatten(), yv.flatten()], axis=-1)

    p = multivariate_normal.pdf(data, mean, std).reshape(xv.shape)

    cp = plt.contourf(xv, yv, p, cmap=cm.jet)
    plt.colorbar(cp)
    if truth is not None:
        assert len(truth) == 2
        plt.scatter(truth[0], truth[1], marker='x')
    plt.show()

def univariate_bayes():
    a0 = -.3
    a1 = .5

    trainingPoints = 100
    noiseSD = .2
    priorPrecision = 2.
    likelihoodSD = noiseSD

    #Generate the training points
    xtrain = np.random.uniform(-1., 1., size=trainingPoints)
    ytrain = (a0 + a1 * xtrain) + np.random.normal(loc=0., scale=noiseSD, size=trainingPoints)

    #Plot prior
    priorMean = np.zeros(2)
    priorSigma = np.eye(2) / priorPrecision
    contourPlot(priorMean, priorSigma, [a0, a1])

    #Plot sample lines

    iterations = 2
    mu = priorMean
    sigma = priorSigma
    for i in range(iterations):
        #Plot the likelihood
        contourPlotLikelihood(ytrain[i], xtrain[i], likelihoodSD, [a0, a1])

        #Update prior to posterior
        mu, sigma = update(xtrain[i], ytrain[i], 1. / noiseSD**2, mu, sigma)
        contourPlot(mu, sigma, [a0, a1])

        #Plot sample lines
        plotSampleLines(mu, sigma, 6, [xtrain[0 : i + 1], ytrain[0 : i + 1]])

    mu, sigma = update(xtrain, ytrain, 1. / noiseSD**2, priorMean, priorSigma)
    contourPlot(mu, sigma, [a0, a1])
    plotSampleLines(mu, sigma, 6, [xtrain, ytrain])

def main():
    univariate_bayes()

if __name__ == '__main__':
    main()
