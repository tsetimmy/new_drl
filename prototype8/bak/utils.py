import numpy as np
import tensorflow as tf
import random

def update_target_graph(from_scope, to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

def log(x):
    return tf.log(tf.maximum(x, 1e-6))

def lrelu(x, alpha=.2):
  return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

def sample_z(batch_size, latent_size):
    #return np.random.uniform(-1., 1., [batch_size, latent_size])
    return np.random.normal(0., 1., [batch_size, latent_size])

class Memory:
    def __init__(self, size):
       self.max_size = size
       self.mem = []

    def add(self, element):
        self.mem.append(element)

        if len(self.mem) > self.max_size:
            self.mem.pop(0)

    def sample(self, size):
        size = min(size, len(self.mem))
        return random.sample(self.mem, size)
