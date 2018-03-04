import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

import sys
sys.path.append('..')
from gan import CGAN


class model_gan_wrapper:
    def __init__(self, input_shape=[None, 3], action_size=1, latent_size=4, a_type='continuous'):
        self.input_shape = input_shape
        self.action_size = action_size
        self.latent_size = latent_size
        self.a_type = a_type
        assert self.a_type in ['continuous', 'discrete']

        #State and reward models
        self.smodel = CGAN(input_shape=input_shape, action_size=action_size, latent_size=latent_size,
                           gen_input_shape=input_shape, continuous_action=(self.a_type=='continuous'))
        self.rmodel = CGAN(input_shape=input_shape, action_size=action_size, latent_size=latent_size,
                           gen_input_shape=[None, 1], continuous_action=(self.a_type=='continuous'))

    def get_prediction_state(self, states, actions):
        if self.a_type == 'discrete':
            actions = tf.one_hot(actions, self.action_size, dtype=tf.float32)
        return self.smodel.generate(states, actions, Z)

    def get_prediction_reward(self, states, actions):
        if self.a_type == 'discrete':
            actions = tf.one_hot(actions, self.action_size, dtype=tf.float32)
        return self.rmodel.generate(states, actions, Z)

    def smodel_loss(self.

def main():
    model = model_gan_wrapper()

if __name__ == '__main__':
    main()

