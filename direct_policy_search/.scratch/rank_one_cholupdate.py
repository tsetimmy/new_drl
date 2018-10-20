import numpy as np
import scipy.linalg as la
import time

def cholupdate(R, x):
      p = np.size(x)
      x = x.T
      for k in range(p):
        r = np.sqrt(R[k, k]**2 + x[k]**2)
        c = r / R[k, k]
        s = x[k] / R[k, k]
        R[k, k] = r
        R[k, k+1:p] = (R[k, k+1:p] + s * x[k+1:p]) / c
        x[k+1:p] = c * x[k+1:p] - s * R[k, k+1:p]
      return R

def cholupdate2(R, x):
      p = np.size(x[0])
      #x = x.T
      for k in range(p):
        r = np.sqrt(R[..., k, k]**2 + x[..., k]**2)
        c = r / R[..., k, k]
        s = x[..., k] / R[..., k, k]
        R[..., k, k] = r
        R[..., k, k+1:p] = (R[..., k, k+1:p] + s[..., np.newaxis] * x[..., k+1:p]) / c[..., np.newaxis]
        x[..., k+1:p] = c[..., np.newaxis] * x[..., k+1:p] - s[..., np.newaxis] * R[..., k, k+1:p]
      return R

batch_size = 10000
dim = 100*3

A = []
B = []
U = []
for _ in range(10):
    v = np.random.normal(size=[batch_size, dim])
    a = np.matmul(v.T, v)
    A.append(a)

    u = np.random.normal(size=[batch_size/2, dim])
    U.append(u)
    b = a + np.matmul(u.T, u)
    B.append(b)

A = np.stack(A, axis=0)
U = np.stack(U, axis=0)
B = np.stack(B, axis=0)

cholA = []
for a in A:
    cholA.append(la.cholesky(a))
cholA = np.stack(cholA)


A_tmp = np.copy(A)
start = time.time()
for i in range(U.shape[1]):
    u = U[:, i:i+1, :]
    A_tmp += np.matmul(np.transpose(u, [0, 2, 1]), u)
print time.time() - start

start = time.time()
for i in range(U.shape[1]):
    cholA = cholupdate2(cholA, U[:, i, :])
print time.time() - start
exit()

print A.shape
print B.shape

print np.allclose(np.matmul(np.transpose(cholA, [0, 2, 1]), cholA), B)

cholB = []
for b in B:
    cholB.append(la.cholesky(b))
cholB = np.stack(cholB)

print np.allclose(cholA, cholB)
