import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

class qnetwork:
    def __init__(self, input_shape=[None, 4], action_size=2, state_model=None, scope=None):
        with tf.variable_scope(scope):
            self.states = tf.placeholder(shape=input_shape, dtype=tf.float32)
            self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
            self.actions_onehot = tf.one_hot(self.actions, action_size, dtype=tf.float32)

            # Fully connected layers
            fc1 = slim.fully_connected(self.states, 256, activation_fn=tf.nn.relu, scope='fc1')
            fc2 = slim.fully_connected(fc1, 256, activation_fn=tf.nn.relu, scope='fc2')

            streamA, streamV = tf.split(fc2, 2, 1)
            self.A = slim.fully_connected(streamA, action_size, activation_fn=None, scope='advantage')
            self.V = slim.fully_connected(streamV, 1, activation_fn=None, scope='value')

            self.Q = self.V + tf.subtract(self.A, tf.reduce_mean(self.A, axis=1, keep_dims=True))

            '''
            fc1 = slim.fully_connected(self.states, 256, activation_fn=tf.nn.relu)
            fc2 = slim.fully_connected(fc1, 256, activation_fn=tf.nn.relu)
            self.Q = slim.fully_connected(fc2, action_size, activation_fn=None)
            '''

            # Loss function
            self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
            self.responsible_output = tf.reduce_sum(tf.multiply(self.Q, self.actions_onehot), axis=1, keep_dims=False)
            self.loss = tf.reduce_mean(tf.square(self.responsible_output - self.targetQ))
            # Optimizer
            self.update_model = tf.train.AdamOptimizer().minimize(self.loss)

            if state_model is not None:
                fc1_sm = slim.fully_connected(state_model, 256, activation_fn=tf.nn.relu, scope='fc1', reuse=True)
                fc2_sm = slim.fully_connected(fc1_sm, 256, activation_fn=tf.nn.relu, scope='fc2', reuse=True)

                streamA_sm, streamV_sm = tf.split(fc2_sm, 2, 1)
                A_sm = slim.fully_connected(streamA_sm, action_size, activation_fn=None, scope='advantage', reuse=True)
                V_sm = slim.fully_connected(streamV_sm, 1, activation_fn=None, scope='value', reuse=True)

                self.Q_state_model = V_sm + tf.subtract(A_sm, tf.reduce_mean(A_sm, axis=1, keep_dims=True))

    def get_action(self, sess, observation):
        Q = sess.run(self.Q, feed_dict={self.states:observation[np.newaxis, ...]})
        return np.argmax(Q)

    def train(self, sess, batch, learning_rate, target_qnet, states=None, actions=None, rewards=None, states1=None, dones=None):
        if batch is not None:
            states = np.vstack(batch[:, 0])
            actions = np.array(batch[:, 1])
            rewards = batch[:, 2]
            states1 = np.vstack(batch[:, 3])
            dones = batch[:, 4]

        Q1 = sess.run(target_qnet.Q, feed_dict={target_qnet.states:states1})
        Q1 = np.amax(Q1, axis=1, keepdims=False)

        assert len(states) == len(actions)
        assert len(states) == len(rewards)
        assert len(states) == len(states1)
        assert len(states) == len(dones)
        for k in range(len(states)):
            if not dones[k]:
                Q1[k] = rewards[k] + learning_rate * Q1[k]
            else:
                Q1[k] = rewards[k]

        sess.run(self.update_model, feed_dict={self.states:states, self.actions:actions, self.targetQ:Q1})
