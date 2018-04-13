import numpy as np
import scipy.stats as stats

class alphas:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols

        self.alphas = np.ones([rows, cols])

    def update(self, row, col):
        self.alphas[row, col] += 1

class transitions:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols




def main():
    x = np.random.uniform(size=10)
    x = x / sum(x)
    alpha = np.random.randint(low=1, high=3, size=10)

    #print stats.dirichlet.pdf(x, alpha)
    alpha[-1] = 10
    alpha[0] = 1
    print alpha
    dude = stats.dirichlet.rvs(alpha, size=10)
    print dude.shape
    print np.sum(dude, axis=-1)

    b = [[a() for i in range(100)] for i in range(100)]
    c = np.array(b)
    c[0, 0].pp()


    



if __name__ == '__main__':
    main()
