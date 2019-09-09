import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import gym
import argparse

import pickle

import sys
sys.path.append('..')
from prototype8.dmlac.real_env_pendulum import real_env_pendulum_reward, real_env_pendulum_state
#from environment_reward_functions import mountain_car_continuous_reward_function
#from environment_state_functions import mountain_car_continuous_state_function

import uuid

class ANN:
    def __init__(self, input_dim, output_dim, train_weights=False):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.train_weights = train_weights

        self.scope = str(uuid.uuid4())
        self.reuse = None

        if self.train_weights:
            self.inputs = tf.placeholder(shape=[None, self.input_dim], dtype=tf.float64)
            self.outputs = self.build_graph(self.inputs)

            self.targets = tf.placeholder(shape=[None, self.output_dim], dtype=tf.float64)

            self.loss = tf.reduce_mean(tf.reduce_mean(tf.square(self.outputs - self.targets), axis=-1))
            self.opt = tf.train.AdamOptimizer().minimize(self.loss)
            
        else:
            self.inputs = tf.placeholder(shape=[None, self.input_dim], dtype=tf.float64)
            self.outputs = self.build_graph(self.inputs)

    def build_graph(self, inputs):
        fc1 = slim.fully_connected(inputs, 256, activation_fn=tf.nn.relu, scope=self.scope+'/fc1', reuse=self.reuse)
        fc2 = slim.fully_connected(fc1, 256, activation_fn=tf.nn.relu, scope=self.scope+'/fc2', reuse=self.reuse)
        outputs = slim.fully_connected(fc2, self.output_dim, activation_fn=None, scope=self.scope+'/output', reuse=self.reuse)
        self.reuse = True

        return outputs

    def build(self, states, actions):
        states_actions = tf.concat([states, actions], axis=-1)
        assert states_actions.shape.as_list() == [None, self.input_dim]
        
        return self.build_graph(states_actions)

    def build_np(self, sess, states, actions):
        assert len(states.shape) == 2
        assert len(actions.shape) == 2
        assert states.shape[0] == actions.shape[0]

        states_actions = np.concatenate([states, actions], axis=-1)
        rewards = sess.run(self.outputs, feed_dict={self.inputs:states_actions})
        assert len(rewards.shape) == 2
        assert states.shape[0] == rewards.shape[0]
        assert rewards.shape[-1] == 1

        return rewards

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default='MountainCarContinuous-v0')
    parser.add_argument("--data-size", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=5000)
    #parser.add_argument("--goal-position", type=float, default=-.4)
    args = parser.parse_args()

    print(args)

    env = gym.make(args.env)
    ann = ANN(env.observation_space.shape[0]+env.action_space.shape[0], 1, train_weights=True)

    reward_function = real_env_pendulum_reward()
    #state_function = real_env_pendulum_state()

    high = np.array([np.pi, 1.])
    states = np.random.uniform(low=-high, high=high, size=[args.data_size, len(high)])
    states = np.stack([np.cos(states[:, 0]), np.sin(states[:, 0]), states[:, 1]], axis=-1)
    actions = np.random.uniform(env.action_space.low, env.action_space.high, size=[args.data_size, env.action_space.shape[0]])

    rewards = reward_function.step_np(states, actions)


    #saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for it in range(args.iterations):
            for i in range(0, args.data_size, args.batch_size):
                inputs = np.concatenate([states[i:i+args.batch_size, ...], actions[i:i+args.batch_size, ...]], axis=-1)
                targets = rewards[i:i+args.batch_size, ...]
                loss, _ = sess.run([ann.loss, ann.opt], feed_dict={ann.inputs:inputs, ann.targets:targets})
                if it % 1000 == 0:
                    print('iterations:', it, 'i:', i, 'loss:', loss)

        #print sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
        #saver.save(sess, './weights/pendulum_reward.ckpt')
        pickle.dump(sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)), open('./weights/pendulum_reward.p', 'wb'))
    #print str(reward_function.goal_position_np)

if __name__ == '__main__':
    main()
