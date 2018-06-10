import numpy as np
import tensorflow as tf
import uuid

class gaussian_process:
    def __init__(self, x_dim, x_train_data=None, y_train_data=None):
        self.x_dim = x_dim
        self.x_train_data = x_train_data
        self.y_train_data = y_train_data

        self.uuid = str(uuid.uuid4())

        # Hyperparameters.
        self.length_scale = tf.get_variable(name='length_scale'+self.uuid, shape=[], dtype=tf.float64,
                                            initializer=tf.constant_initializer(.316))
        self.signal_sd = tf.get_variable(name='signal_sd'+self.uuid, shape=[], dtype=tf.float64,
                                               initializer=tf.constant_initializer(1.))
        self.noise_sd = tf.get_variable(name='noise_sd'+self.uuid, shape=[], dtype=tf.float64,
                                              initializer=tf.constant_initializer(1.))

        # Placholders.
        self.x_train = tf.placeholder(shape=[None, self.x_dim], dtype=tf.float64)
        self.y_train = tf.placeholder(shape=[None, 1], dtype=tf.float64)
        self.x_test = tf.placeholder(shape=[None, self.x_dim], dtype=tf.float64)

        self.n = tf.shape(self.x_train)[0]

        # Get predictive distribution and log marginal likelihood (Algorithm 2.1 in the GP book).
        L = tf.cholesky(self.squared_exponential_kernel(self.x_train, self.x_train) +\
                        tf.multiply(tf.square(self.noise_sd), tf.eye(self.n, dtype=tf.float64)))
        v = tf.linalg.solve(L, self.squared_exponential_kernel(self.x_train, self.x_test))

        self.mu = tf.matmul(tf.transpose(v), tf.linalg.solve(L, self.y_train))
        self.var = self.squared_exponential_kernel(self.x_test, self.x_test) - tf.matmul(tf.transpose(v), v)

        alpha = tf.linalg.solve(tf.transpose(L), tf.linalg.solve(L, self.y_train))
        self.log_marginal_likelihood = -.5 * tf.matmul(tf.transpose(self.y_train), alpha)[0, 0] +\
                                       -.5 * tf.reduce_sum(tf.log(tf.diag_part(L))) +\
                                       -.5 * tf.cast(self.n, dtype=tf.float64) * np.log(2. * np.pi)


        self.opt = tf.train.AdamOptimizer().minimize(-self.log_marginal_likelihood, var_list=[self.length_scale, self.signal_sd, self.noise_sd])
        #self.opt = tf.train.GradientDescentOptimizer(.1).minimize(-self.log_marginal_likelihood, var_list=[self.length_scale, self.signal_sd, self.noise_sd])

        '''
        n2 = tf.shape(self.x_test)[0]
        K_ = self.squared_exponential_kernel(self.x_test, self.x_test)
        self.L1_ = tf.linalg.cholesky(K_ + tf.eye(n2, dtype=tf.float64)*1e-6)
        self.L2_ = tf.linalg.cholesky(K_ + 1e-6*tf.eye(n2, dtype=tf.float64) - tf.matmul(tf.transpose(v), v))
        '''

        #self.get_prediction(self.x_test)

    def build_mu_var(self, x_test):
        assert x_test.shape.as_list() == self.x_test.shape.as_list()

        L = tf.cholesky(self.squared_exponential_kernel(self.x_train, self.x_train) +\
                        tf.multiply(tf.square(self.noise_sd), tf.eye(self.n, dtype=tf.float64)))
        v = tf.linalg.solve(L, self.squared_exponential_kernel(self.x_train, x_test))

        mu = tf.matmul(tf.transpose(v), tf.linalg.solve(L, self.y_train))
        var = self.squared_exponential_kernel(x_test, x_test) - tf.matmul(tf.transpose(v), v)

        return mu, var

    def get_prediction(self, x_test):
        assert x_test.shape.as_list() == self.x_test.shape.as_list()
        mu, var = self.build_mu_var(x_test)
        sd = tf.expand_dims(tf.sqrt(tf.diag_part(var)), axis=-1)
        return mu, sd

    def squared_exponential_kernel(self, a, b):
        sqdist = tf.reduce_sum(tf.square(a), axis=-1, keep_dims=True) +\
                 -2. * tf.matmul(a, tf.transpose(b)) +\
                 tf.transpose(tf.reduce_sum(tf.square(b), axis=-1, keep_dims=True))
        kernel = tf.square(self.signal_sd) * tf.exp(-.5 * (1. / tf.square(self.length_scale)) * sqdist)
        return kernel
    
    def set_training_data(self, x_train_data, y_training_data):
        self.x_train_data = x_train_data
        self.y_training_data = y_train_data

    def predict(self, sess, x_test):
        assert len(self.x_train_data.shape) == 2
        assert len(self.y_train_data.shape) == 2
        assert self.x_train_data.shape[0] == self.y_train_data.shape[0]
        assert self.x_train_data.shape[1] == self.x_dim
        assert self.y_train_data.shape[1] == 1
        feed_dict={self.x_train:self.x_train_data,
                   self.y_train:self.y_train_data,
                   self.x_test:x_test}
        return sess.run([self.mu, self.var, self.L1_, self.L2_], feed_dict=feed_dict)

def main():
    N = 10         # number of training points.
    n = 50         # number of test points.
    s = 0.00005    # noise variance.

    f = lambda x: np.sin(0.9*x).flatten()

    X = np.random.uniform(-5, 5, size=(N,1))
    y = f(X) + s * np.random.randn(N)

    Xtest = np.linspace(-5, 5, n).reshape(-1,1)
    '''
    import pickle
    data = pickle.load(open('save.p', 'rb'))
    X, y, Xtest = data
    '''


    gp = gaussian_process(1, x_train_data=X, y_train_data=y[..., np.newaxis])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(5000):
            feed_dict={gp.x_train:gp.x_train_data,
                       gp.y_train:gp.y_train_data}
            _, loss = sess.run([gp.opt, gp.log_marginal_likelihood], feed_dict=feed_dict)
            print loss

        mu, var, L1_, L2_ = gp.predict(sess, Xtest)

        mu = np.squeeze(mu, axis=-1)
        s = np.sqrt(np.diag(var))

        print s


    import matplotlib.pyplot as pl

    # PLOTS:
    pl.figure(1)
    pl.clf()
    pl.plot(X, y, 'r+', ms=20)
    pl.plot(Xtest, f(Xtest), 'b-')
    pl.gca().fill_between(Xtest.flat, mu-3*s, mu+3*s, color="#dddddd")
    pl.plot(Xtest, mu, 'r--', lw=2)
    #pl.savefig('predictive.png', bbox_inches='tight')
    pl.grid()
    pl.title('Mean predictions plus 3 st.deviations')
    pl.axis([-5, 5, -3, 3])


    # draw samples from the prior at our test points.
    #L = np.linalg.cholesky(K_ + 1e-6*np.eye(n))
    f_prior = np.dot(L1_, np.random.normal(size=(n,10)))
    pl.figure(2)
    pl.clf()
    pl.plot(Xtest, f_prior)
    pl.title('Ten samples from the GP prior')
    pl.axis([-5, 5, -3, 3])
    #pl.savefig('prior.png', bbox_inches='tight')
    pl.grid()

    # draw samples from the posterior at our test points.
    #L = np.linalg.cholesky(K_ + 1e-6*np.eye(n) - np.dot(Lk.T, Lk))
    f_post = mu.reshape(-1,1) + np.dot(L2_, np.random.normal(size=(n,10)))
    pl.figure(3)
    pl.clf()
    pl.plot(Xtest, f_post)
    pl.title('Ten samples from the GP posterior')
    pl.axis([-5, 5, -3, 3])
    #pl.savefig('post.png', bbox_inches='tight')
    pl.grid()

    pl.show()


if __name__ == '__main__':
    main()
