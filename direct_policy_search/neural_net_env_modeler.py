import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim

import sys
sys.path.append('..')
from prototype8.dmlac.mlp_env_modeler import mlp_env_modeler
from prototype8.dmlac.real_env_pendulum import get_next

from tf_bayesian_model import get_training_data

class env_model:
    def __init__(self, output_size):
        self.mlp_env = mlp_env_modeler(output_size, True)

        self.states = tf.placeholder(shape=[None, 3], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None, 1], dtype=tf.float32)

        self.prediction = self.mlp_env.build(self.states, self.actions)

        self.target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

        self.loss = tf.reduce_mean(tf.squeeze(tf.square(self.prediction - self.target), axis=-1))


        self.solver = tf.train.AdamOptimizer(1e-3).minimize(self.loss)


def pendulum_experiment():
    training_points = 400*5
    xtrain, ytrain = get_training_data(training_points)
    ytrain = np.stack(ytrain, axis=0)[..., np.newaxis]
    model = env_model(ytrain.shape[-1])

    th = np.linspace(-1., 1., 30)
    thdot = np.linspace(-8., 8., 30)
    u = np.linspace(-2., 2., 30)

    states = []
    for i in range(len(th)):
        for j in range(len(thdot)):
            for k in range(len(u)):
                states.append([np.cos(th[i]), np.sin(th[i]), thdot[j], u[k]])
    states = np.stack(states, axis=0)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for _ in range(100*10):
            loss, _ = sess.run([model.loss, model.solver], feed_dict={model.states:xtrain[:, 0:3], model.actions:xtrain[:, 3:4], model.target:ytrain})
            #print loss

        predictions = sess.run(model.prediction, feed_dict={model.states:states[:, 0:3], model.actions:states[:, 3:4]})

    # Real model
    costh = []
    for i in range(len(th)):
        for j in range(len(thdot)):
            for k in range(len(u)):
                a, b, c = get_next(th[i], thdot[j], u[k])
                costh.append(a)

    plt.scatter(np.arange(len(costh)) / 10., costh)

    # Neural networks model
    plt.scatter(np.arange(len(predictions)) / 10., predictions)
    plt.grid()
    plt.show()

def main():
    pendulum_experiment()

if __name__ == '__main__':
    main()
