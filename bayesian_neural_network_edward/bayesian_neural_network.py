import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import edward as ed

from edward.models import Normal

class bayesian_dynamics_model:
    def __init__(self, input_size, output_size, hidden_size=20):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        # Declare placholder.
        self.x = tf.placeholder(shape=[None, self.input_size], dtype=tf.float32)
        self.y_ph = tf.placeholder(shape=[None, self.output_size], dtype=tf.float32)

        # Declare weights.
        self.W_0 = Normal(loc=tf.zeros([self.input_size, self.hidden_size]), scale=tf.ones([self.input_size, self.hidden_size]))
        self.W_1 = Normal(loc=tf.zeros([self.hidden_size, self.hidden_size]), scale=tf.ones([self.hidden_size, self.hidden_size]))
        self.W_2 = Normal(loc=tf.zeros([self.hidden_size, self.output_size]), scale=tf.ones([self.hidden_size, self.output_size]))

        self.b_0 = Normal(loc=tf.zeros(self.hidden_size), scale=tf.ones(self.hidden_size))
        self.b_1 = Normal(loc=tf.zeros(self.hidden_size), scale=tf.ones(self.hidden_size))
        self.b_2 = Normal(loc=tf.zeros(self.output_size), scale=tf.ones(self.output_size))

        # Output of computational graph.
        nn_out = self.build(self.x, self.W_0, self.W_1, self.W_2, self.b_0, self.b_1, self.b_2)
        self.y = Normal(loc=nn_out, scale=tf.ones_like(nn_out) * .1)

        # Variables.
        self.qW_0 = Normal(loc=tf.get_variable('qW_0/loc', [self.input_size, self.hidden_size]),
                           scale=tf.nn.softplus(tf.get_variable('qW_0/scale', [self.input_size, self.hidden_size])))
        self.qW_1 = Normal(loc=tf.get_variable('qW_1/loc', [self.hidden_size, self.hidden_size]),
                           scale=tf.nn.softplus(tf.get_variable('qW_1/scale', [self.hidden_size, self.hidden_size])))
        self.qW_2 = Normal(loc=tf.get_variable('qW_2/loc', [self.hidden_size, self.output_size]),
                           scale=tf.nn.softplus(tf.get_variable('qW_2/scale', [self.hidden_size, self.output_size])))

        self.qb_0 = Normal(loc=tf.get_variable('qb_0/loc', [self.hidden_size]),
                           scale=tf.nn.softplus(tf.get_variable('qb_0/scale', [self.hidden_size])))
        self.qb_1 = Normal(loc=tf.get_variable('qb_1/loc', [self.hidden_size]),
                           scale=tf.nn.softplus(tf.get_variable('qb_1/scale', [self.hidden_size])))
        self.qb_2 = Normal(loc=tf.get_variable('qb_2/loc', [self.output_size]),
                           scale=tf.nn.softplus(tf.get_variable('qb_2/scale', [self.output_size])))

        # Keep track of the variables in a list.
        self.random_vars = [self.qW_0, self.qW_1, self.qW_2, self.qb_0, self.qb_1, self.qb_2]

        # Sample of the posterior model.
        self.sample_model = [var.sample() for var in self.random_vars]

        # Sample functions from variational model to visualize fits.
        self.mus = self.build(self.x, self.qW_0.sample(), self.qW_1.sample(), self.qW_2.sample(), self.qb_0.sample(), self.qb_1.sample(), self.qb_2.sample())

    def initialize_inference(self, n_iter=1000*5, n_samples=5):
        self.inference = ed.KLqp({self.W_0: self.qW_0, self.b_0: self.qb_0,
                                  self.W_1: self.qW_1, self.b_1: self.qb_1,
                                  self.W_2: self.qW_2, self.b_2: self.qb_2}, data={self.y: self.y_ph})
        self.inference.initialize(n_iter=n_iter, n_samples=n_samples)

    def rbf(self, x):
        return tf.exp(-tf.square(x))

    def function(self, x):
        return np.sin(x)

    def build(self, x, W_0, W_1, W_2, b_0, b_1, b_2):
        '''Builds the computational graph.'''

        h_0 = self.rbf(tf.matmul(x, W_0) + b_0)
        h_1 = self.rbf(tf.matmul(h_0, W_1) + b_1)
        out = tf.matmul(h_1, W_2) + b_2

        return out

    def generate_toy_data(self, noise_sd=.1, size=50):
        x = np.random.uniform(-3., 3., size)
        y1 = np.cos(x) + np.random.normal(0, noise_sd, size=size)
        y2 = np.sin(x) + np.random.normal(0, noise_sd, size=size)

        y = np.stack([y1, y2], axis=-1)

        return x[..., np.newaxis], y

    def get_batch(self, noise_sd=.1, size=50):
        x = np.random.uniform(-3., 3., size)
        y = self.function(x) + np.random.normal(0, noise_sd, size=size)

        return x[..., np.newaxis], y[..., np.newaxis]

    def visualize(self, sess, xeval, animate=False):
        plt.cla()
        plt.scatter(xeval, self.function(xeval))
        for _ in range(10):
            yeval = sess.run(self.mus, feed_dict={self.x:xeval})
            plt.plot(xeval, yeval)
        plt.grid()
        if animate == False:
            plt.show()
        else:
            plt.pause(1. / 60.)

def multi_batch_demo():
    model = bayesian_dynamics_model(1, 1)
    #x, y = model.generate_toy_data()

    xeval = np.linspace(-3., 3., 100)[..., np.newaxis]

    #sess = ed.get_session()
    #tf.global_variables_initializer().run()
    inference = ed.KLqp({model.W_0: model.qW_0, model.b_0: model.qb_0,
                         model.W_1: model.qW_1, model.b_1: model.qb_1,
                         model.W_2: model.qW_2, model.b_2: model.qb_2}, data={model.y: model.y_ph})
    inference.initialize(n_iter=1000*5, n_samples=5)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Plot the prior
        model.visualize(sess, xeval)

        # Train the model
        for _ in range(1000*5):
            x_batch, y_batch = model.get_batch(size=np.random.randint(low=100))
            info_dict = inference.update({model.x: x_batch, model.y_ph: y_batch})
            inference.print_progress(info_dict)

            # Visualize the evolution of the posterior plots
            #model.visualize(sess, xeval, animate=True)

        # Plot the posterior
        model.visualize(sess, xeval)

def single_batch_demo():
    model = bayesian_dynamics_model(1, 2)
    x, y = model.generate_toy_data()

    xeval = np.linspace(-3., 3., 100)[..., np.newaxis]

    #sess = ed.get_session()
    #tf.global_variables_initializer().run()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Plot the prior
        plt.scatter(x, y[:, 0])
        for _ in range(10):
            yeval = sess.run(model.mus, feed_dict={model.x:xeval})
            plt.plot(xeval, yeval[:, 0])
        plt.grid()
        plt.show()

        plt.scatter(x, y[:, 1])
        for _ in range(10):
            yeval = sess.run(model.mus, feed_dict={model.x:xeval})
            plt.plot(xeval, yeval[:, 1])
        plt.grid()
        plt.show()

        # Train the model
        inference = ed.KLqp({model.W_0: model.qW_0, model.b_0: model.qb_0,
                             model.W_1: model.qW_1, model.b_1: model.qb_1,
                             model.W_2: model.qW_2, model.b_2: model.qb_2}, data={model.x: x, model.y: y})
        inference.run(n_iter=1000, n_samples=5)

        # Plot the posterior
        plt.scatter(x, y[:, 0])
        for _ in range(10):
            yeval = sess.run(model.mus, feed_dict={model.x:xeval})
            plt.plot(xeval, yeval[:, 0])
        plt.grid()
        plt.show()
    
        plt.scatter(x, y[:, 1])
        for _ in range(10):
            yeval = sess.run(model.mus, feed_dict={model.x:xeval})
            plt.plot(xeval, yeval[:, 1])
        plt.grid()
        plt.show()

if __name__ == '__main__':
    #multi_batch_demo()
    single_batch_demo()
