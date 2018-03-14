import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

class multivariate_gaussian_process:
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

        #Placeholders for inputs
        self.x = tf.placeholder(shape=self.input_shape, dtype=tf.float64)
        self.xtest = tf.placeholder(shape=self.input_shape, dtype=tf.float64)
        self.y = tf.placeholder(shape=self.output_shape, dtype=tf.float64)
        y_splits = tf.split(self.y, self.output_shape[-1], axis=-1)

        #Allocate the gps
        self.output = tf.concat([gaussian_process(self.input_shape).build(self.x, self.xtest, y) for y in y_splits], axis=-1)

class gaussian_process:
    def __init__(self, xshape):
        self.xshape = xshape

    def build(self, x, xtest, y):
        assert x.shape.as_list() == self.xshape
        assert xtest.shape.as_list() == self.xshape
        assert y.shape.as_list() == [None, 1]

        #Placeholders
        #self.x = tf.placeholder(shape=self.xshape, dtype=tf.float64)

        #Kernel
        kernel = self.squared_exponential_kernel(x, x)

        #Cholesky decomposition
        self.noise_variance = 5e-5#to be optimized
        L = tf.cholesky(kernel + tf.diag(self.noise_variance * tf.ones_like(tf.reduce_sum(x, axis=-1))))

        #Placeholders for test points
        #self.xtest = tf.placeholder(shape=self.xshape, dtype=tf.float64)
        #self.y = tf.placeholder(shape=[None, 1], dtype=tf.float64)

        #Compute the mean at the test points
        Lk = tf.linalg.solve(L, self.squared_exponential_kernel(x, xtest))
        mu = tf.matmul(tf.transpose(Lk), tf.linalg.solve(L, y))

        #Compute the variance at the test points
        K_ = self.squared_exponential_kernel(xtest, xtest)
        var = K_ - tf.matmul(tf.transpose(Lk), Lk)
        std = tf.sqrt(tf.diag_part(var))
        return mu

    def squared_exponential_kernel(self, a, b):
        self.kernel_param = .1#to be optimized
        sqdist = tf.reduce_sum(tf.square(a), axis=-1, keep_dims=True) +\
                 -2. * tf.matmul(a, tf.transpose(b)) +\
                 tf.transpose(tf.reduce_sum(tf.square(b), axis=-1, keep_dims=True))
        self.sqdist = sqdist
        return tf.exp(-.5 * (1./self.kernel_param) * sqdist)

def main1():
    f = lambda x: np.sin(0.9*x).flatten()
    import pickle
    data = pickle.load( open( "data.p", "rb" ) ) 
    X = data[0]
    y = data[1][..., np.newaxis]
    Xtest = data[2]

    gp = gaussian_process(xshape=[None, X.shape[-1]])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        K_, mu, s, Lk, var = sess.run([gp.K_, gp.mu, gp.std, gp.Lk, gp.var], feed_dict={gp.x:X, gp.y:y, gp.xtest:Xtest})
        mu = np.squeeze(mu, axis=-1)

    plt.figure(1)
    plt.clf()
    plt.plot(X, y, 'r+', ms=20)
    plt.plot(Xtest, f(Xtest), 'b-')
    plt.gca().fill_between(Xtest.flat, mu-3*s, mu+3*s, color="#dddddd")
    plt.plot(Xtest, mu, 'r--', lw=2)
    plt.title('Mean predictions plus 3 st.deviations')
    plt.axis([-5, 5, -3, 3])
    plt.show()


def main2():
    import gym
    gp = multivariate_gaussian_process(input_shape=[None, 4], output_shape=[None, 3])

    env = gym.make('Pendulum-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound_high = env.action_space.high
    action_bound_low = env.action_space.low

    inputs = []
    outputs = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        state = env.reset()

        while True:
            env.render()
            action = np.random.uniform(low=action_bound_low[0], high=action_bound_high[0], size=action_dim)
            inputs.append(np.concatenate([state, action], axis=0))
            state1, reward, done, _ = env.step(action)
            outputs.append(state1)

            if len(inputs) == 10000:
                break

            state = np.copy(state1)

            if done == True:
                state = env.reset()
        
        x = np.stack(inputs)
        xtest = np.copy(x)
        y = np.stack(outputs)
        print sess.run(gp.output, feed_dict={gp.x:x, gp.xtest:xtest, gp.y:y})
    
if __name__ == '__main__':
    main2()
