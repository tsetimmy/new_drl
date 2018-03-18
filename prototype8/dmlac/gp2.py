#import sys
#sys.path.append('../..')
#from tgp.np.gaussian_process import GaussianProcess as GP
import numpy as np
import tensorflow as tf
#Previous bugs were to:
#1) Passing in ls as a single float rather than a list of floats
#2) Singular matrix due to small variance on the diagonal
#3) Not using float64 in the GP procedure (not 100% sure this was actually a bug)

class DoubleAdamOptimizer(tf.train.AdamOptimizer):
  def _valid_dtypes(self):
    return set([tf.float32, tf.float64])

class multivariate_gaussian_process:
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.gps = [GaussianProcess(dim=self.input_shape[-1], ls=[1.]*self.input_shape[-1], noise=.1) for i in range(self.output_shape[-1])]

    def build(self, x, y, xtest):
        output, _ = self.build_and_get_opt(x, y, xtest)
        return output

    def build_and_get_opt(self, x, y, xtest):
        assert x.shape.as_list() == self.input_shape
        assert y.shape.as_list() == self.output_shape
        assert xtest.shape.as_list() == self.input_shape

        y_splits = tf.split(y, self.output_shape[-1], axis=-1)
        assert len(self.gps) == len(y_splits)

        output = []
        opts = []
        for i in range(len(self.gps)):
            L, y_, var_, opt_op, grad_ls, grad_amp = self.gps[i].build(x, y_splits[i], xtest)
            output.append(y_)
            opts.append(opt_op)
        output = tf.concat(output, axis=-1)
        return output, opts

class GaussianProcess(object):
    #def __init__(self, ls=.1, amp=.1, noise=1e-2, sess=tf.Session(), dim=1):
    def __init__(self, ls=.1, amp=.1, noise=1e-2, dim=1):
      #self.sess = sess
      self.tmp = []
      self.ls = tf.Variable(np.array(ls), dtype=tf.float64)
      self.amp = tf.Variable(np.array(amp), dtype=tf.float64)
      self.noise = tf.Variable(np.array(noise), dtype=tf.float64)
      self.opt = DoubleAdamOptimizer(
        learning_rate=tf.constant(1e-3, tf.float64),
        beta1=tf.constant(0.9, tf.float64),
        beta2=tf.constant(0.999, tf.float64),
        epsilon=tf.constant(1e-8, tf.float64)
      )

      # construct loss computation graph
      '''
      self.xp = tf.placeholder(tf.float64, [None, dim])
      self.yp = tf.placeholder(tf.float64, [None, 1])
      self.x_p = tf.placeholder(tf.float64, [None, dim])
      self.L, self.y_, self.var_, self.opt_op, self.grad_ls, self.grad_amp = self.build(self.xp, self.yp, self.x_p)
      '''

    def build(self, xp, yp, x_p):
      xp = tf.cast(xp, tf.float64)
      yp = tf.cast(yp, tf.float64)
      x_p = tf.cast(x_p, tf.float64)

      L = self.construct_loss_graph(xp, yp)
      # construct prediction computation graph
      y_, var_ = self.construct_prediction_graph(xp, yp, x_p)

      opt_op = self.opt.minimize(L, var_list=[self.ls, self.amp])
      grad_ls = tf.gradients(L, self.ls)
      grad_amp = tf.gradients(L, self.amp)
      #self.init = tf.initialize_all_variables()
      #return L, y_, var_, opt_op, grad_ls, grad_amp
      return tf.cast(L, tf.float32), tf.cast(y_, tf.float32), tf.cast(var_, tf.float32), opt_op, [tf.cast(grad, tf.float32) for grad in grad_ls], [tf.cast(grad, tf.float32) for grad in grad_amp]
      
    def construct_covariance_graph(self, xs, ys=None):
      add_noise = True if ys is None else False
      ys = xs if ys is None else ys
      # Compute covariance matrix K
      xsq = tf.reduce_sum(tf.square(xs), 1)
      ysq = tf.reduce_sum(tf.square(ys), 1)
      xsq = tf.reshape(xsq, tf.stack([tf.shape(xsq)[0], 1]))
      ysq = tf.reshape(ysq, tf.stack([1, tf.shape(ysq)[0]]))
      sqdist = xsq + ysq - 2*tf.matmul(xs, tf.transpose(ys))
      K = tf.square(self.amp) * tf.exp(-0.5*sqdist)
      if add_noise:
        ones = tf.ones(tf.stack([tf.shape(xs)[0]]), dtype=tf.float64)
        K = K + tf.diag(ones)*(1. + tf.square(self.noise))
        # compute loss
        Ki = tf.matrix_inverse(K)
        self.tmp.append([K,Ki])
        return K, Ki
      else:
        return K
      
    def construct_loss_graph(self, x, y):
      xs = x/self.ls
      K, Ki = self.construct_covariance_graph(xs)
      yT = tf.transpose(y)
      Kiy = tf.matmul(Ki, y)
      lK = tf.log(tf.matrix_determinant(K))
      L = tf.matmul(yT, Kiy) + lK
      ones = tf.ones(tf.stack([tf.shape(xs)[0]]), dtype=tf.float64)
      L = L/tf.reduce_sum(ones) * 0.5
      return L

    def construct_prediction_graph(self, x, y, x_):
      xs = x/self.ls
      xs_ = x_/self.ls
      _, Ki = self.construct_covariance_graph(xs)
      K_ = self.construct_covariance_graph(xs, xs_)
      # compute variance
      K_T = tf.transpose(K_)
      K_Ki = tf.matmul(K_T, Ki)
      var_ = tf.square(self.amp) - tf.reduce_sum(K_T * K_Ki, 1)
      # compute prediction
      y_ = tf.matmul(K_T, tf.matmul(Ki, y))
      return y_, var_
    
    def solve(self, X, y, epochs=20, batch_size=50, train=True):
      self.X = X
      self.y = y
      #self.sess.run(self.init)
      if train:
        iterations = epochs * len(X) / batch_size
        epochiter = iterations/epochs
        for i in xrange(iterations):
          idx = np.random.choice(np.arange(len(X)), batch_size, replace=False)
          X_mini = X[idx]
          y_mini = y[idx]
          fd = {self.xp: X_mini, self.yp: y_mini}
          '''
          if i % epochiter == 0:
            print self.sess.run(self.L, feed_dict=fd)
          '''
          self.sess.run(self.opt_op, fd)

    def predict(self, X_):
      fd = {self.xp: self.X, self.yp: self.y, self.x_p: X_}
      y_ = self.sess.run(self.y_, feed_dict=fd)
      var_ = self.sess.run(self.var_, feed_dict=fd)
      return y_, var_
        
def main():
  import matplotlib.pyplot as plt
  dim = 1
  X = np.random.uniform(0, 1, (10, dim))
  y = (np.sin((X[:, 0])*10) - X[:,0]*3).reshape(-1, 1)

  ls = [.1]*dim
  sess = tf.Session()
  gp = GaussianProcess(dim=dim, sess=sess, ls=ls, noise=0.1)
  sess.run(tf.initialize_all_variables())
  gp.solve(X, y, epochs=100, batch_size=len(X), train=True)
  print sess.run(gp.ls)
  
  X_ = np.linspace(0, 1, 1000).reshape(-1, 1)
  y_, var_ = gp.predict(X_)
  std_ = np.sqrt(var_).reshape(-1, 1)

  plt.scatter(X, y)
  plt.plot(X_, y_, c='r')
  plt.plot(X_, y_+2*std_, c='g')
  plt.plot(X_, y_-2*std_, c='g')
  plt.grid()
  plt.show()
  print tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
  
def main2():
    gp2 = multivariate_gaussian_process([None, 4], [None, 3])

    x = tf.placeholder(tf.float64, [None, 4])
    y = tf.placeholder(tf.float64, [None, 3])
    xtest = tf.placeholder(tf.float64, [None, 4])

    gp2.build(x, y, xtest)

    
if __name__ == "__main__":
  #main()
  main2()
