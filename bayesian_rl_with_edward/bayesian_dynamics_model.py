import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import edward as ed

from edward.models import Normal

class bayesian_dynamics_model:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = 200

        # Declare placholder.
        self.x = tf.placeholder(shape=[None, self.input_size], dtype=tf.float32)

        # Declare weights.
        self.W_0 = Normal(loc=tf.zeros([self.input_size, self.hidden_size]), scale=tf.ones([self.input_size, self.hidden_size]))
        self.W_1 = Normal(loc=tf.zeros([self.hidden_size, self.output_size]), scale=tf.ones([self.hidden_size, self.output_size]))

        self.b_0 = Normal(loc=tf.zeros(self.hidden_size), scale=tf.ones(self.hidden_size))
        self.b_1 = Normal(loc=tf.zeros(self.output_size), scale=tf.ones(self.output_size))

        # Output of computational graph.
        self.y = Normal(loc=self.build(self.x, self.W_0, self.W_1, self.b_0, self.b_1), scale=tf.ones_like(self.x) * .1)

        # Variables.
        self.qW_0 = Normal(loc=tf.get_variable('qW_0/loc', [self.input_size, self.hidden_size]),
                           scale=tf.nn.softplus(tf.get_variable('qW_0/scale', [self.input_size, self.hidden_size])))
        self.qW_1 = Normal(loc=tf.get_variable('qW_1/loc', [self.hidden_size, self.output_size]),
                           scale=tf.nn.softplus(tf.get_variable('qW_1/scale', [self.hidden_size, self.output_size])))

        self.qb_0 = Normal(loc=tf.get_variable('qb_0/loc', [self.hidden_size]),
                           scale=tf.nn.softplus(tf.get_variable('qb_0/scale', [self.hidden_size])))
        self.qb_1 = Normal(loc=tf.get_variable('qb_1/loc', [self.output_size]),
                           scale=tf.nn.softplus(tf.get_variable('qb_1/scale', [self.output_size])))

        # Sample functions from variational model to visualize fits.
        self.mus = self.build(self.x, self.qW_0.sample(), self.qW_1.sample(), self.qb_0.sample(), self.qb_1.sample())




    def build(self, x, W_0, W_1, b_0, b_1):
        '''Builds the computational graph.'''
        h_0 = tf.nn.tanh(tf.matmul(x, W_0) + b_0)
        out = tf.matmul(h_0, W_1) + b_1
        return out

    def generate_toy_data(self, noise_sd=.1, size=50):
        x = np.random.uniform(-3., 3., size)
        y = np.cos(x) + np.random.normal(0, noise_sd, size=size)

        return x[..., np.newaxis], y[..., np.newaxis]




        




        

def main():
    model = bayesian_dynamics_model(1, 1)
    x, y = model.generate_toy_data()

    xeval = np.linspace(-3., 3., 100)[..., np.newaxis]

    #sess = ed.get_session()
    #tf.global_variables_initializer().run()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Plot the prior
        plt.scatter(x, y)
        for _ in range(10):
            yeval = sess.run(model.mus, feed_dict={model.x:xeval})
            plt.plot(xeval, yeval)
        plt.grid()
        plt.show()

        # Train the model
        inference = ed.KLqp({model.W_0: model.qW_0, model.b_0: model.qb_0,
                             model.W_1: model.qW_1, model.b_1: model.qb_1}, data={model.x:x, model.y: y})
        inference.run(n_iter=1000, n_samples=5)

        # Plot the posterior
        plt.scatter(x, y)
        for _ in range(10):
            yeval = sess.run(model.mus, feed_dict={model.x:xeval})
            plt.plot(xeval, yeval)
        plt.grid()
        plt.show()
    





if __name__ == '__main__':
    main()
