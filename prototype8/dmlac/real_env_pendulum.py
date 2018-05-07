import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def get_next(th, thdot, u):
    newthdot = thdot + (-30. * np.sin(th + np.pi) + 3. * u) / 20.
    newth = th + newthdot / 20.
    return np.cos(newth), np.sin(newth), newthdot

class real_env_pendulum_state:
    def __init__(self, input_shape=[None, 3], action_shape=[None, 1]):
        self.input_shape = input_shape
        self.action_shape = action_shape

    def build(self, states, actions):
        assert states.shape.as_list() == self.input_shape
        assert actions.shape.as_list() == self.action_shape

        newthdot = states[:, -1] + (15. * states[:, 1] + 3. * actions[:, 0]) / 20.
        newcosth = tf.clip_by_value(states[:, 0] * tf.cos(newthdot / 20.) - states[:, 1] * tf.sin(newthdot / 20.), -1.+1e-6, 1.-1e-6)
        newsinth = tf.clip_by_value(states[:, 1] * tf.cos(newthdot / 20.) + states[:, 0] * tf.sin(newthdot / 20.), -1.+1e-6, 1.-1e-6)
        newthdot = tf.clip_by_value(newthdot, -8., 8.)

        return tf.stack([newcosth, newsinth, newthdot], axis=-1)

class real_env_pendulum_reward:
    def __init__(self, input_shape=[None, 3], action_shape=[None, 1]):
        self.input_shape = input_shape
        self.action_shape = action_shape

    def build(self, states, actions):
        assert states.shape.as_list() == self.input_shape
        assert actions.shape.as_list() == self.action_shape

        cossgn = (tf.sign(states[:, 0])+1.)/2.
        sinsgn = (tf.sign(states[:, 1])+1.)/2.

        th = tf.asin(tf.minimum(tf.abs(states[:, 1]), 1.-1e-6))

        th += sinsgn * tf.abs(cossgn - 1.) * (np.pi - 2. * th) +\
              tf.abs(sinsgn - 1.) * cossgn * (-2. * th) +\
              tf.abs(sinsgn - 1.) * tf.abs(cossgn - 1.) * -np.pi

        rewards = -(tf.square(th) + .1*tf.square(states[:, -1]) + .001*tf.square(actions[:, 0]))
        return tf.expand_dims(rewards, axis=-1)#, tf.minimum(tf.abs(states[:, 1]), 1.-1e-6)

def main():
    u = np.linspace(-2., 2., 10)
    thdot = np.linspace(-8., 8., 20)
    th = np.linspace(0., 2.*np.pi, 10)

    costh = []
    sinth = []
    newthdot = []
    X = []
    x = 0.
    counter = 0
    for i in range(len(u)):
        for j in range(len(thdot)):
            for k in range(len(th)):
                print counter
                counter += 1
                a, b, c = get_next(th[k], thdot[j], u[i])
                costh.append(a)
                sinth.append(b)
                newthdot.append(c)
                X.append(x)
                x += .1

    plt.scatter(X, costh)
    #plt.scatter(X, sinth)
    #plt.scatter(X, newthdot)
    plt.grid()
    plt.show()

if __name__ == '__main__':
    #testing1 = real_env_pendulum_state()
    #testing2 = real_env_pendulum_reward()
    main()
