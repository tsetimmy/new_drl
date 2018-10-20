import numpy as np
import scipy.linalg as la
import time


def main():
    batch_size = 1000000
    dim = 256
    X = np.random.normal(size=[batch_size, dim])
    A = np.matmul(X.T, X)

    Xtest = np.random.normal(size=[batch_size, dim])
    Ainv = la.solve(A, np.eye(dim), sym_pos=True)

    cho_factor = la.cho_factor(A)
    cholA = la.cholesky(A)

    #Method 0
    start = time.time()
    predict0 = np.sum(np.multiply(np.matmul(Xtest, Ainv), Xtest), axis=-1)
    time0 = time.time() - start
    '''
    predict0 = np.sum(np.multiply(Xtest, la.solve(A, Xtest.T, sym_pos=True).T), axis=-1)
    predict0_2 = np.sum(np.multiply(Xtest, la.cho_solve((cholA, False), Xtest.T).T), axis=-1)
    '''

    #Method 1
    start = time.time()
    predict0_1 = np.sum(np.multiply(Xtest, la.cho_solve(cho_factor, Xtest.T).T), axis=-1)
    time1 = time.time() - start

    #Method 2
    start = time.time()
    predict1 = np.sum(np.square(la.solve_triangular(cholA, Xtest.T)), axis=0)
    time2 = time.time() - start

    '''
    for p0, p0_1, p0_2, p1 in zip(predict0, predict0_1, predict0_2, predict1):
        print p0, p0_1, p0_2, p1
    print np.allclose(predict0, predict0_1)
    print np.allclose(predict0, predict0_2)
    print np.allclose(predict0, predict1)
    '''

    '''
    for p0, p1 in zip(predict0, predict1):
        print p0, p1
    '''
    print time0
    print time1
    print time2













if __name__ == '__main__':
    main()
