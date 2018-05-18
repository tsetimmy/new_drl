import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

class qnetwork:
    def __init__(self, input_shape=[None, 4], action_size=2, scope=None):
        with tf.variable_scope(scope):
            self.scope = scope
            self.input_shape = input_shape
            self.action_size = action_size

            # Placeholders
            self.states = tf.placeholder(shape=input_shape, dtype=tf.float32)
            self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
            self.actions_onehot = tf.one_hot(self.actions, action_size, dtype=tf.float32)

            # Fully connected layers
            fc1 = slim.fully_connected(self.states, 256, activation_fn=tf.nn.relu)
            fc2 = slim.fully_connected(fc1, 256, activation_fn=tf.nn.relu)
            self.q = slim.fully_connected(fc2, action_size, activation_fn=None)

            # Loss function
            self.target_q = tf.placeholder(shape=[None], dtype=tf.float32)
            self.responsible_output = tf.reduce_sum(tf.multiply(self.q, self.actions_onehot), axis=1, keep_dims=False)
            self.loss = tf.reduce_mean(tf.square(self.responsible_output - self.target_q))

            # Optimizer
            self.update_model = tf.train.AdamOptimizer().minimize(self.loss)

    def act(self, sess, state):
        q = sess.run(self.q, feed_dict={self.states:state[np.newaxis, ...]})
        return np.argmax(q)

    def train(self, sess, batch, discount_factor, tnet):
        assert len(batch) > 0
        states = np.vstack(batch[:, 0])
        actions = np.array(batch[:, 1])
        rewards = batch[:, 2]
        next_states = np.vstack(batch[:, 3])
        dones = batch[:, 4]

        next_q = sess.run(tnet.q, feed_dict={tnet.states:next_states})
        next_q = rewards + (1. - dones.astype(np.float32)) * discount_factor * np.amax(next_q, axis=1, keepdims=False)

        sess.run(self.update_model, feed_dict={self.states:states, self.actions:actions, self.target_q:next_q})

def main():
    qnet = qnetwork()
    
if __name__ == '__main__':
    main()
