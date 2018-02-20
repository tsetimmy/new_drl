import tensorflow as tf
import numpy as np

class gated_dqn:
    def __init__(self, s_shape, a_size, numfactors, learning_rate=.95):
        self.s_shape = s_shape
        self.a_size = a_size
        self.numfactors = numfactors
        self.learning_rate = learning_rate

        self.states = tf.placeholder(shape=s_shape, dtype=tf.float32)
        self.states_ = tf.placeholder(shape=s_shape, dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, a_size, dtype=tf.float32)
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)

        self.declare_variables()
        self.recon_s, self.recon_s_, self.Q = self.build_computational_graph(self.states, self.states_)

        self.get_loss(self.states, self.states_, self.actions_onehot, self.targetQ, self.recon_s, self.recon_s_, self.Q)

    def get_loss(self, states, states_, actions_onehot, targetQ, recon_s, recon_s_, Q):
        #Reconstruction losses
        sloss = tf.reduce_mean(tf.reduce_sum(recon_s - states, axis=-1))
        sloss_ = tf.reduce_mean(tf.reduce_sum(recon_s_ - states_, axis=-1))

        #Q losses
        responsible_output = tf.reduce_sum(tf.multiply(actions_onehot, Q), axis=-1)
        qloss = tf.reduce_mean(tf.square(targetQ - responsible_output))

        return sloss, sloss_, qloss

    def declare_variables(self):
        #Declare initializer
        self.xavier_init = tf.contrib.layers.xavier_initializer()

        #Variables for states
        self.wfs = tf.Variable(self.xavier_init([self.s_shape[-1], self.numfactors]))
        self.bfs = tf.Variable(tf.zeros([self.s_shape[-1]]))

        #Variables for states_
        self.wfs_ = tf.Variable(self.xavier_init([self.s_shape[-1], self.numfactors]))
        self.bfs_ = tf.Variable(tf.zeros([self.s_shape[-1]]))

        #Variables for Q((s, s_), a)
        self.wfq1 = tf.Variable(self.xavier_init([512 , self.numfactors]))
        self.bfq1f = tf.Variable(tf.zeros([512]))
        self.bfq1b = tf.Variable(tf.zeros([self.numfactors]))

        self.wfq2 = tf.Variable(self.xavier_init([512 , 512]))
        self.bfq2f = tf.Variable(tf.zeros([512]))
        self.bfq2b = tf.Variable(tf.zeros([512]))

        self.wfq3 = tf.Variable(self.xavier_init([self.a_size, 512]))
        self.bfq3f = tf.Variable(tf.zeros([self.a_size]))
        self.bfq3b = tf.Variable(tf.zeros([512]))

    def build_computational_graph(self, states, states_):
        #Factors
        fs = tf.matmul(states, self.wfs)
        fs_ = tf.matmul(states_, self.wfs_)

        #Compute Q-values
        fc1 = tf.nn.relu(tf.matmul(tf.multiply(fs, fs_), tf.transpose(self.wfq1)) + self.bfq1f)
        fc2 = tf.nn.relu(tf.matmul(fc1, tf.transpose(self.wfq2)) + self.bfq2f)
        q = tf.matmul(fc2, tf.transpose(self.wfq3)) + self.bfq3f

        #Backward pass
        fc2_back = tf.nn.relu(tf.matmul(q, self.wfq3) + self.bfq3b)
        fc1_back = tf.nn.relu(tf.matmul(fc2_back, self.wfq2) + self.bfq2b)
        fq = tf.nn.relu(tf.matmul(fc1_back, self.wfq1) + self.bfq1b)

        #Recon states
        recon_s = tf.matmul(tf.multiply(fq, fs_), tf.transpose(self.wfs)) + self.bfs

        #Recon states_
        recon_s_ = tf.matmul(tf.multiply(fq, fs), tf.transpose(self.wfs_)) + self.bfs_

        return recon_s, recon_s_, q


def main():
    a = gated_dqn([None, 10], 4, 256)
if __name__ == '__main__':
    main()
