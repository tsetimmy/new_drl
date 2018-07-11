import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(x, 0.)

'''
class nelder_mead_experiment:
    def __init__(self):
        self.batch_size = 12
        self.state_dim = 2
        self.action_dim = 1
        self.X = np.concatenate([np.random.uniform(size=[self.batch_size, self.state_dim]), np.ones([self.batch_size, 1])], axis=-1)

        self.h1 = np.random.normal(size=[self.state_dim+1, 256])
        self.h2 = np.random.normal(size=[256, 256])
        self.o = np.random.normal(size=[256, self.action_dim])

    def relu(self, x):
        return np.maximum(x, 0.)

    def forward(self, X, l1, l2, o):
        h1 = self.relu(np.matmul(X, l1))
        h2 = self.relu(np.matmul(h1, l2))
        out = np.matmul(h2, o)
        return out

    def loss(self, thetas, X, y):
        out = self.forward(X, *thetas.tolist())
        return np.mean(np.sum(out, axis=-1))
'''

class nelder_mead_experiment2:
    def __init__(self):
        self.batch_size = 100

        self.X = np.linspace(-2., 2., self.batch_size)
        self.y = np.sin(self.X) + 5e-5 * np.random.randn(self.batch_size)

        self.Xin = np.concatenate([self.X[..., np.newaxis], np.ones([self.batch_size, 1])], axis=-1)
        self.h1 = np.random.normal(size=[1+1, 32])
        self.h2 = np.random.normal(size=[32, 32])
        self.o = np.random.normal(size=[32, 1])

        self.thetas = np.concatenate([self.h1.flatten(), self.h2.flatten(), self.o.flatten()])

        self.it = 0

    def unpack(self, thetas):
        h1 = thetas[:2*32].reshape([2, 32])
        h2 = thetas[2*32:2*32+32*32].reshape([32, 32])
        h3 = thetas[2*32+32*32:2*32+32*32+32].reshape([32, 1])

        return [h1, h2, h3]

    def relu(self, x):
        return np.maximum(x, 0.)

    def forward(self, X, l1, l2, o):
        h1 = self.relu(np.matmul(X, l1))
        h2 = self.relu(np.matmul(h1, l2))
        out = np.matmul(h2, o)
        return out

    def loss(self, thetas, X, y):
        out = np.squeeze(self.forward(X, *self.unpack(thetas)), axis=-1)
        print np.mean(np.square(out - y))
        return np.mean(np.square(out - y))

    def fit(self):
        options = {'maxiter': 100000, 'disp': True}

        _res = minimize(self.loss, self.thetas, method='powell', args=(self.Xin, self.y), options=options)
        res = self.unpack(_res.x)
        self.h1 = np.copy(res[0])
        self.h2 = np.copy(res[1])
        self.o = np.copy(res[2])

    def plot(self):
        out = self.forward(self.Xin, self.h1, self.h2, self.o)
        
        plt.plot(self.X, self.y)
        plt.plot(self.X, out)
        plt.grid()
        plt.show()



def main():
    nme = nelder_mead_experiment2()
    nme.plot()
    nme.fit()
    nme.plot()

        




if __name__ == '__main__':
    main()
