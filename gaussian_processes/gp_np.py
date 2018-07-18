import numpy as np

class gaussian_process:
    def __init__(self, x_dim, length_scale, signal_sd, noise_sd, x_train=None, y_train=None):
        self.x_dim = x_dim
        self.length_scale = length_scale
        self.signal_sd = signal_sd
        self.noise_sd = noise_sd
        self.x_train = x_train
        self.y_train = y_train

    def squared_exponential_kernel(self, a, b):
        sqdist = np.sum(np.square(a), axis=-1, keepdims=True) +\
                 -2. * np.matmul(a, b.T) +\
                 np.sum(np.square(b), axis=-1, keepdims=True).T
        kernel = np.square(self.signal_sd) * np.exp(-.5*sqdist/np.square(self.length_scale))
        return kernel

    def set_training_data(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test):
        K = self.squared_exponential_kernel(self.x_train, self.x_train)
        L = np.linalg.cholesky(K + np.square(self.noise_sd)*np.eye(len(K)))
        v = np.linalg.solve(L, self.squared_exponential_kernel(self.x_train, x_test))

        mu = np.matmul(v.T, np.linalg.solve(L, self.y_train))
        sigma = self.squared_exponential_kernel(x_test, x_test) - np.matmul(v.T, v)

        return mu, sigma

def f(X):
    return np.abs(-np.square(X) + 4.)

def main():
    import tensorflow as tf
    import matplotlib.pyplot as plt
    from gp_tf import gaussian_process as gaussian_process_tf

    X = np.random.uniform(-4., 4., size=[100, 1])
    y = f(X) + np.random.normal(loc=0., scale=.5, size=[len(X), 1])
    X_test = np.linspace(-5., 5., 1000)[..., np.newaxis]

    gp_tf = gaussian_process_tf(1, X, y)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(5000):
            feed_dict={gp_tf.x_train:gp_tf.x_train_data,
                       gp_tf.y_train:gp_tf.y_train_data}
            _, loss = sess.run([gp_tf.opt, gp_tf.log_marginal_likelihood], feed_dict=feed_dict)
            #print loss
        hyperparameters = sess.run([gp_tf.length_scale, gp_tf.signal_sd, gp_tf.noise_sd])
    del gp_tf

    gp_np = gaussian_process(1, *hyperparameters, x_train=X, y_train=y)
    mu, sigma = gp_np.predict(X_test)
    mu = np.squeeze(mu, axis=-1)
    sd = np.sqrt(np.diag(sigma))

    plt.gca().fill_between(X_test.flat, mu-3*sd, mu+3*sd, color="#dddddd")
    plt.plot(X_test, mu, 'r--')

    plt.scatter(X, y)
    plt.plot(X_test, f(X_test))

    plt.grid()
    plt.show()

if __name__ == '__main__':
    main()
