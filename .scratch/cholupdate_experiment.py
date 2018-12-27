import numpy as np
import scipy.linalg as spla
from choldate import cholupdate
import time

batch_size = 1000
samples = 100
dim = 150

X = np.random.normal(size=[batch_size, samples, dim])
M = np.matmul(X.transpose([0, 2, 1]), X) + np.tile(np.eye(X.shape[-1])[None, ...], [len(X), 1, 1])

u = np.random.normal(size=[batch_size, 1, dim])
U = np.matmul(u.transpose([0, 2, 1]), u)

MU = M + U

A = np.linalg.cholesky(M).transpose([0, 2, 1])

start = time.time()
B = np.linalg.cholesky(MU)
print time.time() - start
B = B.transpose([0, 2, 1])

start = time.time()
for i in range(batch_size):
    cholupdate(A[i], u[i, 0].copy())
print time.time() - start

start = time.time()
C = [spla.cholesky(MU[i]) for i in range(batch_size)]
print time.time() - start
C = np.stack(C, axis=0)

print np.allclose(A, B)
print np.allclose(A, C)
print np.allclose(B, C)
