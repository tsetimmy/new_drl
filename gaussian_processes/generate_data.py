from __future__ import division
import numpy as np

# This is the true unknown function we are trying to approximate
f = lambda x: np.sin(0.9*x).flatten()

N = 10         # number of training points.
n = 50         # number of test points.
s = 0.00005    # noise variance.

# Sample some input points and noisy versions of the function evaluated at
# these points. 
X = np.random.uniform(-5, 5, size=(N,1))
y = f(X) + s*np.random.randn(N)

# points we're going to make predictions at.
Xtest = np.linspace(-5, 5, n).reshape(-1,1)


# Dump the data into a pickle file
import pickle
data = [X, y, Xtest]
pickle.dump( data, open( "data.p", "wb" ) )
