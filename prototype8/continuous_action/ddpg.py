import numpy as np

import gym
import tensorflow as tf
import tensorflow.contrib.slim as slim
import argparse

import random

import sys
sys.path.append('../../')
from utils import Memory
from utils import update_target_graph2
from utils import OrnsteinUhlenbeckActionNoise

class actor:
    def __init__(self, state_shape=[None, 100], action_shape=[None, 1], output_bound_low=[-1.], output_bound_high=[1.], scope=None):
        with tf.variable_scope(scope):
            self.scope = scope
            self.action_bound = tf.constant(output_bound_high, dtype=tf.float32)
            self.states = tf.placeholder(shape=state_shape, dtype=tf.float32)
            batch_size = tf.cast(tf.shape(self.states)[0], tf.float32)
            fc1 = slim.fully_connected(self.states, 400, activation_fn=tf.nn.relu, scope='fc1')
            fc2 = slim.fully_connected(fc1, 300, activation_fn=tf.nn.relu, scope='fc2')
            self.W = tf.Variable(tf.random_uniform([300, action_shape[-1]], -3e-3, 3e-3))
            self.b = tf.Variable(tf.random_uniform([action_shape[-1]], -3e-3, 3e-3))
            self.action = tf.multiply(tf.nn.tanh(tf.matmul(fc2, self.W) + self.b), self.action_bound)

            # Optimizer
            self.dQ_by_da = tf.placeholder(shape=action_shape, dtype=tf.float32)
            self.parameters = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope)
            self.grads = tf.gradients(self.action, self.parameters, -self.dQ_by_da)
            self.grads_normalized = list(map(lambda x: tf.div(x, batch_size), self.grads))
            assert len(self.grads_normalized) == len(self.parameters)
            self.opt = tf.train.AdamOptimizer(1e-4).apply_gradients(zip(self.grads_normalized, self.parameters))

    def get_action(self, states):
        fc1_ = slim.fully_connected(states, 400, activation_fn=tf.nn.relu, scope=self.scope + '/fc1', reuse=True)
        fc2_ = slim.fully_connected(fc1_, 300, activation_fn=tf.nn.relu, scope=self.scope + '/fc2', reuse=True)
        return tf.multiply(tf.nn.tanh(tf.matmul(fc2_, self.W) + self.b), self.action_bound)

class critic:
    def __init__(self, state_shape=[None, 100], action_shape=[None, 1], scope=None):
        with tf.variable_scope(scope):
            self.scope = scope
            self.states = tf.placeholder(shape=state_shape, dtype=tf.float32)
            self.actions = tf.placeholder(shape=action_shape, dtype=tf.float32)
            fc1 = slim.fully_connected(self.states, 400, activation_fn=tf.nn.relu, scope='fc1')

            xavier_init = tf.contrib.layers.xavier_initializer()

            # State variables
            self.W_s = tf.Variable(xavier_init([400, 300]))
            # Action variables
            self.W_a = tf.Variable(xavier_init([action_shape[-1], 300]))
            # Bias
            self.b_hidden = tf.Variable(xavier_init([300]))

            hidden = tf.nn.relu(tf.matmul(fc1, self.W_s) + tf.matmul(self.actions, self.W_a) + self.b_hidden)

            # Output layer
            self.W = tf.Variable(tf.random_uniform([300, 1], -3e-3, 3e-3))
            self.b = tf.Variable(tf.random_uniform([1], -3e-3, 3e-3))
            self.Q = tf.matmul(hidden, self.W) + self.b

            # Loss function
            self.targetQ = tf.placeholder(shape=[None, 1], dtype=tf.float32)
            self.loss = tf.reduce_mean(tf.square(self.targetQ - self.Q))
            # Optimizer
            self.critic_solver = tf.train.AdamOptimizer(1e-3).minimize(self.loss)

            # Get gradients (for actor network)
            self.grads = tf.gradients(self.Q, self.actions)

    def get_Q(self, states, actions):
            fc1 = slim.fully_connected(states, 400, activation_fn=tf.nn.relu, scope=self.scope + '/fc1', reuse=True)
            hidden = tf.nn.relu(tf.matmul(fc1, self.W_s) + tf.matmul(actions, self.W_a) + self.b_hidden)
            return tf.matmul(hidden, self.W) + self.b

def clip(action, high, low):
    return np.minimum(np.maximum(action, low), high)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default='Pendulum-v0')
    parser.add_argument("--action-dim", type=int, default=1)
    parser.add_argument("--state-dim", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=30000)
    parser.add_argument('--tau', help='soft target update parameter', default=0.001)
    parser.add_argument("--action-bound", type=float, default=1.)
    parser.add_argument("--replay-mem-size", type=int, default=1000000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=.99)
    args = parser.parse_args()

    # Initialize environment
    env = gym.make(args.environment)
    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    #assert args.action_dim == 1
    args.action_bound_high = env.action_space.high
    args.action_bound_low = env.action_space.low

    assert len(args.action_bound_high) == len(args.action_bound_low)
    for i in range(len(args.action_bound_high)):
        assert args.action_bound_high[i] == -args.action_bound_low[i]
    print(args)

    # Networks
    actor_source = actor(state_shape=[None, args.state_dim], action_shape=[None, args.action_dim], output_bound_low=args.action_bound_low, output_bound_high=args.action_bound_high, scope='actor_source')
    critic_source = critic(state_shape=[None, args.state_dim], action_shape=[None, args.action_dim], scope='critic_source')
    actor_target = actor(state_shape=[None, args.state_dim], action_shape=[None, args.action_dim], output_bound_low=args.action_bound_low, output_bound_high=args.action_bound_high, scope='actor_target')
    critic_target = critic(state_shape=[None, args.state_dim], action_shape=[None, args.action_dim], scope='critic_target')

    # Update and copy operators
    update_target_actor = update_target_graph2('actor_source', 'actor_target', args.tau)
    update_target_critic = update_target_graph2('critic_source', 'critic_target', args.tau)

    copy_target_actor = update_target_graph2('actor_source', 'actor_target', 1.)
    copy_target_critic = update_target_graph2('critic_source', 'critic_target', 1.)

    # Replay memory
    memory = Memory(args.replay_mem_size)

    # Actor noise
    actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(args.action_dim))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(copy_target_critic)
        sess.run(copy_target_actor)

        for epoch in range(args.epochs):
            state = env.reset()
            total_rewards = 0.0
            while True:
                #env.render()
                # Choose an action
                action = sess.run(actor_source.action, feed_dict={actor_source.states:state[np.newaxis, ...]})[0] + actor_noise()
                if args.environment == 'LunarLanderContinuous-v2':
                    action = clip(action, args.action_bound_high, args.action_bound_low)
                # Execute action
                state1, reward, done, _ = env.step(action)
                total_rewards += float(reward)
                # Store tuple in replay memory
                memory.add([state[np.newaxis, ...], action[np.newaxis, ...], reward, state1[np.newaxis, ...], done])

                # Training step
                batch = np.array(memory.sample(args.batch_size))
                assert len(batch) > 0
                states = np.concatenate(batch[:, 0], axis=0)
                actions = np.concatenate(batch[:, 1], axis=0)
                rewards = batch[:, 2]
                states1 = np.concatenate(batch[:, 3], axis=0)
                dones = batch[:, 4]

                # Update the critic
                actions1 = sess.run(actor_target.action, feed_dict={actor_target.states:states1})
                targetQ = np.squeeze(sess.run(critic_target.Q, feed_dict={critic_target.states:states1, critic_target.actions:actions1}), axis=-1)
                targetQ = rewards + (1. - dones.astype(np.float32)) * args.gamma * targetQ
                targetQ = targetQ[..., np.newaxis]
                _, critic_loss = sess.run([critic_source.critic_solver, critic_source.loss], feed_dict={critic_source.states:states, critic_source.actions:actions, critic_source.targetQ:targetQ})

                # Update the actor
                critic_grads = sess.run(critic_source.grads, feed_dict={critic_source.states:states, critic_source.actions:actions})[0]# Grab gradients from critic
                _ = sess.run(actor_source.opt, feed_dict={actor_source.states:states, actor_source.dQ_by_da:critic_grads})

                # Update target networks
                sess.run(update_target_critic)
                sess.run(update_target_actor)

                state = np.copy(state1)
                if done == True:
                    print 'epoch', epoch, 'total rewards', total_rewards
                    break


    '''
    for v in tf.all_variables():
        print v
    '''


if __name__ == '__main__':
    main()
