import numpy as np
import scipy.linalg as la

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
dim = 100

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
    cholA.append(la.cholesky(a, lower=True))
cholA = np.stack(cholA)


for i in range(U.shape[1]):
    cholA = np.transpose(cholupdate2(np.transpose(cholA, [0, 2, 1]), U[:, i, :]), [0, 2, 1])

print A.shape
print B.shape

print np.allclose(np.matmul(cholA, np.transpose(cholA, [0, 2, 1])), B)

cholB = []
for b in B:
    cholB.append(la.cholesky(b, lower=True))
cholB = np.stack(cholB)

print np.allclose(cholA, cholB)
