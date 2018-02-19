import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

class qnetwork2:
    def __init__(self, input_shape=[None, 4], action_size=2):
        assert len(input_shape) == 2
        #Parameters
        self.input_shape = input_shape
        self.action_size = action_size
        self.states = tf.placeholder(shape=input_shape, dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, action_size, dtype=tf.float32)

        #Declare the variables that will be used to make the computational graph
        self.declare_variables()

        #Build the computational graph
        self.Q = self.build_computational_graph(self.states)

    def declare_variables(self):
        #Initialize variable initializer
        self.xavier_init = tf.contrib.layers.xavier_initializer()

        #Fully connected 1
        self.w_fc1 = tf.Variable(self.xavier_init([self.input_shape[-1], 256]))
        self.b_fc1 = tf.Variable(tf.zeros([256]))

        #Fully connected 2
        self.w_fc2 = tf.Variable(self.xavier_init([256, 256]))
        self.b_fc2 = tf.Variable(tf.zeros([256]))

        #Value stream
        self.w_V = tf.Variable(self.xavier_init([128, self.action_size]))
        self.b_V = tf.Variable(tf.zeros([self.action_size]))

        #Action stream
        self.w_A = tf.Variable(self.xavier_init([128, 1]))
        self.b_A = tf.Variable(tf.zeros([1]))

    def build_computational_graph(self, states):
        fc1 = tf.nn.relu(tf.matmul(states, self.w_fc1) + self.b_fc1)
        fc2 = tf.nn.relu(tf.matmul(fc1, self.w_fc2) + self.b_fc2)

        streamA, streamV = tf.split(fc2, 2, 1)

        A = tf.matmul(streamA, self.w_A) + self.b_A
        V = tf.matmul(streamV, self.w_V) + self.b_V

        Q = V + tf.subtract(A, tf.reduce_mean(A, axis=1, keep_dims=True))

        return Q

class qnetwork:
    def __init__(self, input_shape=[None, 4], action_size=2, scope=None):
        with tf.variable_scope(scope):
            self.scope = scope
            self.input_shape = input_shape
            self.action_size = action_size
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

            #Without dueling
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

    def init_states2(self):
        with tf.variable_scope(self.scope):
            self.states2 = tf.placeholder(shape=self.input_shape, dtype=tf.float32)
            self.actions2 = tf.placeholder(shape=[None], dtype=tf.int32)
            self.actions_onehot2 = tf.one_hot(self.actions2, self.action_size, dtype=tf.float32)

            fc1_s2 = slim.fully_connected(self.states2, 256, activation_fn=tf.nn.relu, scope='fc1', reuse=True)
            fc2_s2 = slim.fully_connected(fc1_s2, 256, activation_fn=tf.nn.relu, scope='fc2', reuse=True)

            streamA_s2, streamV_s2= tf.split(fc2_s2, 2, 1)
            A_s2 = slim.fully_connected(streamA_s2, self.action_size, activation_fn=None, scope='advantage', reuse=True)
            V_s2 = slim.fully_connected(streamV_s2, 1, activation_fn=None, scope='value', reuse=True)

            self.Q_s2 = V_s2 + tf.subtract(A_s2, tf.reduce_mean(A_s2, axis=1, keep_dims=True))

            self.responsible_output2 = tf.reduce_sum(tf.multiply(self.Q_s2, self.actions_onehot2), axis=1, keep_dims=False)


    def init_state_model(self, state_model):
        with tf.variable_scope(self.scope):
            fc1_sm = slim.fully_connected(state_model, 256, activation_fn=tf.nn.relu, scope='fc1', reuse=True)
            fc2_sm = slim.fully_connected(fc1_sm, 256, activation_fn=tf.nn.relu, scope='fc2', reuse=True)

            streamA_sm, streamV_sm = tf.split(fc2_sm, 2, 1)
            A_sm = slim.fully_connected(streamA_sm, self.action_size, activation_fn=None, scope='advantage', reuse=True)
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

def main():
    qnet2 = qnetwork2()
    
if __name__ == '__main__':
    main()
