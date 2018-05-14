import numpy as np
import tensorflow as tf
import edward as ed

from edward.models import Normal

class bayesian_dynamics_mdoel:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = 200

        # Declare placholder.
        self.x = tf.placeholder(shape=[None, self.input_size], dtype=tf.float32)

        # Declare weights.
        self.W_0 = Normal(loc=tf.zeros([self.input_size, self.hidden_size]), scale=tf.ones([self.input_size, self.hidden_size]))
        self.W_1 = Normal(loc=tf.zeros([self.hidden_size, self.hidden_size]), scale=tf.ones([self.hidden_size, self.hidden_size]))
        self.W_2 = Normal(loc=tf.zeros([self.hidden_size, self.output_size]), scale=tf.ones([self.hidden_size, self.output_size]))
        self.b_0 = Normal(loc=tf.zeros(self.hidden_size), scale=tf.ones(self.hidden_size))
        self.b_1 = Normal(loc=tf.zeros(self.hidden_size), scale=tf.ones(self.hidden_size))
        self.b_2 = Normal(loc=tf.zeros(self.output_size), scale=tf.ones(self.output_size))

        # Output of computational graph.
        self.y = self.build(self.x)

    def build(self, x):
        '''Builds the computational graph.'''
        h_0 = tf.nn.sigmoid(tf.matmul(x, self.W_0) + self.b_0)
        h_1 = tf.nn.sigmoid(tf.matmul(h_0, self.W_1) + self.b_1)
        out = tf.matmul(h_1, self.W_2) + self.b_2
        return out




        

def main():
    model = bayesian_dynamics_mdoel(4, 3)

if __name__ == '__main__':
    main()
