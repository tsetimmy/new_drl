import numpy as np
import tensorflow as tf

import gym

from gp_tf import gaussian_process

import matplotlib.pyplot as plt

class gp_model:
    def __init__(self, x_dim, y_dim, x_train_data=None, y_train_data=None):
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.x_train_data = x_train_data
        self.y_train_data = y_train_data

        self.models = [gaussian_process(self.x_dim, self.x_train_data, self.y_train_data)\
                       for i in range(self.y_dim)]

        self.outputs = [[model.mu, model.var] for model in self.models]

    def set_training_data(self, x_train_data, y_train_data):
        assert x_train_data.shape[0] == y_train_data.shape[0]
        assert len(x_train_data.shape) == 2
        assert len(y_train_data.shape) == 2
        assert x_train_data.shape[-1] == self.x_dim
        assert y_train_data.shape[-1] == self.y_dim

        self.x_train_data = np.copy(x_train_data)
        self.y_train_data = np.copy(y_train_data)

        for i in range(len(self.models)):
            self.models[i].x_train_data = self.x_train_data
            self.models[i].y_train_data = self.y_train_data[:, i:i+1]

    def predict(self, sess, x_test):
        assert len(x_test.shape) == 2
        assert x_test.shape[-1] == self.x_train_data.shape[-1]

        self.train_hyperparamters(sess)

        feed_dict = {}
        for i in range(len(self.models)):
            feed_dict[self.models[i].x_train] = self.models[i].x_train_data
            feed_dict[self.models[i].y_train] = self.models[i].y_train_data
            feed_dict[self.models[i].x_test] = x_test

        outputs = sess.run(self.outputs, feed_dict=feed_dict)

        means = np.concatenate([output[0] for output in outputs], axis=-1)
        sds = np.stack([np.sqrt(np.diag(output[1])) for output in outputs], axis=-1)

        return means, sds

    def train_hyperparamters(self, sess, iterations=20000):
        total_size = len(self.x_train_data)
        batch_size = 64
        for it in range(iterations):
            idx = np.random.randint(total_size, size=batch_size)
            for i in range(len(self.models)):
                feed_dict = {self.models[i].x_train:self.models[i].x_train_data[idx, ...],
                             self.models[i].y_train:self.models[i].y_train_data[idx, ...]}
                _, loss = sess.run([self.models[i].opt, self.models[i].log_marginal_likelihood], feed_dict=feed_dict)
                print 'i:', i, 'iterations:', it, 'loss:', -loss, '|',
            print ''

        '''
        for i in range(len(self.models)):
            feed_dict = {self.models[i].x_train:self.models[i].x_train_data,
                         self.models[i].y_train:self.models[i].y_train_data}
            for it in range(iterations):
                _, loss = sess.run([self.models[i].opt, self.models[i].log_marginal_likelihood], feed_dict=feed_dict)
                print 'i:', i, 'iterations:', it, 'loss:', -loss
        '''

def plotting_experiment1():
    gpm = gp_model(x_dim=4, y_dim=3)

    env = gym.make('Pendulum-v0')

    epochs = 4
    train_size = (epochs - 1) * 200

    data = []
    for epoch in range(epochs):
        state = env.reset()
        while True:
            action = np.random.uniform(env.action_space.low, env.action_space.high, 1)
            next_state, reward, done, _ = env.step(action)
            data.append([state, action, next_state])
            state = np.copy(next_state)
            if done:
                break

    states = np.stack([d[0] for d in data], axis=0)
    actions = np.stack([d[1] for d in data], axis=0)
    next_states = np.stack([d[2] for d in data], axis=0)

    states_actions = np.concatenate([states, actions], axis=-1)

    x_train = states_actions[:train_size, ...]
    y_train = next_states[:train_size, ...]
    x_test = states_actions[train_size:, ...]
    y_test = next_states[train_size:, ...]

    gpm.set_training_data(x_train, y_train)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        means, sds = gpm.predict(sess, x_test)

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.grid()
        plt.plot(np.arange(len(y_test)), y_test[:, i])
        plt.errorbar(np.arange(len(means)), means[:, i], yerr=sds[:, i], color='m', ecolor='g')
    plt.show()

def main():
    plotting_experiment1()

if __name__ == '__main__':
    main()
