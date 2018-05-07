import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

class qnetwork:
    def __init__(self, input_shape=[None, 4], action_size=2, scope=None):
        with tf.variable_scope(scope):
            self.scope = scope
            self.input_shape = input_shape
            self.action_size = action_size
            self.states = tf.placeholder(shape=input_shape, dtype=tf.float32)
            self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
            self.actions_onehot = tf.one_hot(self.actions, action_size, dtype=tf.float32)
            self.actions_onehot2 = tf.expand_dims(self.actions_onehot, axis=2)
            self.states2 = tf.expand_dims(self.states, axis=1)


            self.vec = slim.flatten(tf.matmul(self.actions_onehot2, self.states2))

            self.xavier_init = tf.contrib.layers.xavier_initializer()
            self.w = tf.Variable(self.xavier_init([self.vec.shape.as_list()[-1], 1]))

            self.Q = tf.matmul(self.vec, self.w)

            self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
            Q = tf.reduce_sum(self.Q, axis=-1)

            self.loss = tf.reduce_mean(tf.square(Q - self.targetQ))
            self.update_model = tf.train.AdamOptimizer().minimize(self.loss)

    def get_action(self, sess, observation):
        values = []
        for a in range(self.action_size):
            Q = sess.run(self.Q, feed_dict={self.states:observation[np.newaxis, ...], self.actions:np.array([a])})
            values.append(Q)
        return np.argmax(values)

    def train(self, sess, batch, learning_rate, target_qnet, states=None, actions=None, rewards=None, states1=None, dones=None):
        if batch is not None:
            states = np.vstack(batch[:, 0])
            actions = np.array(batch[:, 1])
            rewards = batch[:, 2]
            states1 = np.vstack(batch[:, 3])
            dones = batch[:, 4]

        values = []
        for a in range(self.action_size):
            Q1 = sess.run(target_qnet.Q, feed_dict={target_qnet.states:states1, target_qnet.actions:np.ones(len(batch))*a})
            values.append(Q1)

        Q1 = np.concatenate(values, axis=-1)
        Q1 = np.amax(Q1, axis=1, keepdims=False)
        print Q1.shape

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

def main():
    qnet = qnetwork(scope='qnet')

if __name__ == '__main__':
    main()
