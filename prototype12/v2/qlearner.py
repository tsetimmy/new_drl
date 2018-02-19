import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

from gated_network import gated_convolution2

import sys
sys.path.append('../..')

class qlearner(gated_convolution2):
    def __init__(self,
                 shape,
                 nummap,
                 numfactors,
                 learning_rate_recon,
                 w,
                 s,
                 a_size):
        gated_convolution2.__init__(self,
                                    shape,
                                    nummap,
                                    numfactors,
                                    learning_rate_recon,
                                    w,
                                    s,
                                    a_size)

        factors_x = self.get_factors_via_convolution(self.x,
                                                     self.numfactors,
                                                     self.w,
                                                     self.s,
                                                     self.scope1,
                                                     reuse=True)
        factors_y = self.get_factors_via_convolution(self.y,
                                                     self.numfactors,
                                                     self.w,
                                                     self.s,
                                                     self.scope2,
                                                     reuse=True)
        hidden, _ = self.get_hidden_factors(factors_x, factors_y, self.factors_a)

        self.conv2 = slim.conv2d(activation_fn=tf.nn.relu,
                                 inputs=hidden,
                                 num_outputs=64,
                                 kernel_size=[4, 4],
                                 stride=[2, 2],
                                 padding='VALID')

        self.conv3 = slim.conv2d(activation_fn=tf.nn.relu,
                                 inputs=self.conv2,
                                 num_outputs=64,
                                 kernel_size=[3, 3],
                                 stride=[1, 1],
                                 padding='VALID')

        self.fc1 = slim.fully_connected(slim.flatten(self.conv3), 512, activation_fn=tf.nn.relu)
        self.Q = slim.fully_connected(self.fc1, a_size, activation_fn=None)

        self.actions2 = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions2_onehot = tf.one_hot(self.actions2, a_size, dtype=tf.float32)

        self.targetQ = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        self.responsible_output = tf.reduce_sum(tf.multiply(self.Q, self.actions2_onehot), axis=1, keep_dims=True)
        difference = tf.abs(self.responsible_output - self.targetQ)
        quadratic_part = tf.clip_by_value(difference, 0.0, 1.0)
        linear_part = difference - quadratic_part
        errors = (0.5 * tf.square(quadratic_part)) + linear_part
        self.q_loss = tf.reduce_sum(errors) 

        self.update_model = tf.train.RMSPropOptimizer(learning_rate=.00025, decay=.95, epsilon=.01).minimize(self.q_loss)

    def get_action(self, sess, state_old, state, action):
        state_old = state_old.astype(np.float64) / 255.
        state = state.astype(np.float64) / 255.
        Q = sess.run(self.Q, feed_dict={self.x:state_old, self.y:state, self.actions:action})
        return np.argmax(Q)

    def get_Q1(self, sess, states, states1, actions, tnet):
        states = states.astype(np.float64) / 255.
        states1 = states1.astype(np.float64) / 255.
        Q1 = sess.run(tnet.Q, feed_dict={tnet.x:states, tnet.y:states1, tnet.actions:actions})
        return Q1

    def train(self, sess, states_old, states, actions, actions2, targetQ):
        states_old = states_old.astype(np.float64) / 255.
        states = states.astype(np.float64) / 255.
        _, q_loss = sess.run([self.update_model, self.q_loss], feed_dict={self.x:states_old,
                                                                          self.y:states,
                                                                          self.actions:actions,
                                                                          self.actions2:actions2,
                                                                          self.targetQ:targetQ})

        return _, q_loss

    def train_feature_extractor(self, sess, replay_buffer, batch_size=100, iterations=1):
        state_len_max = self.shape[-1]
        cum_loss = 0.
        cum_a_loss = 0.
        for it in range(iterations):
            batch = np.array(replay_buffer.sample(batch_size))

            states = np.concatenate(batch[:, 0], axis=0).astype(np.float64) / 255.
            actions = np.concatenate(batch[:, 1], axis=0)

            _, recon_loss, recon_x, recon_y, recon_action_loss = self.run2(sess, states[:,:,:,1:1+state_len_max], actions[:,1:], states[:,:,:,2:])

            cum_loss += recon_loss
            cum_a_loss += recon_action_loss
        return cum_loss, cum_a_loss, iterations

def main0():
    dim = 84
    size = 4
    qnet = qlearner(shape=[None, dim, dim, size], nummap=128, numfactors=128, learning_rate_recon=.001, w=8, s=1, a_size=3)

if __name__ == '__main__':
    main0()

