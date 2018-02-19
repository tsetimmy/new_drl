#Another (experimental) version of DDGP
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

class actor_critic:
    def __init__(self, state_shape=[None, 100], action_shape=[None, 1], output_bound=1., scope=None):
        #Parameters
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.output_bound = output_bound
        self.scope = scope

        #Initialize placeholder
        self.states = tf.placeholder(shape=state_shape, dtype=tf.float32)
        self.states_ = tf.placeholder(shape=state_shape, dtype=tf.float32)
        self.actions_ = tf.placeholder(shape=action_shape, dtype=tf.float32)

        #Initialize xavier init
        self.xavier_init = tf.contrib.layers.xavier_initializer()

        #Batch size
        self.batch_size = tf.cast(tf.shape(self.states)[0], tf.float32)
        
        #Initialize actor & critic
        self.init_actor()
        self.init_critic()
        self.init_critic_opt()
        self.init_actor_opt()
        

    def init_actor(self):
        with tf.variable_scope(self.scope + '/actor'):
            fc1 = slim.fully_connected(self.states, 400, activation_fn=tf.nn.relu)
            fc2 = slim.fully_connected(fc1, 300, activation_fn=tf.nn.relu)
            W = tf.Variable(tf.random_uniform([300, self.action_shape[-1]], -3e-3, 3e-3))
            b = tf.Variable(tf.random_uniform([self.action_shape[-1]], -3e-3, 3e-3))
            self.actions = tf.nn.tanh(tf.matmul(fc2, W) + b)

    def init_actor_opt(self):
        actor_parameters = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope + '/actor')
        grads = tf.gradients(self.Q, actor_parameters)
        grads_normalized = list(map(lambda x: tf.div(x, self.batch_size), grads))
        assert len(grads_normalized) == len(actor_parameters)
        self.actor_solver = tf.train.AdamOptimizer(1e-4).apply_gradients(zip(grads_normalized, actor_parameters))

        #---#
        self.dQ_by_da = tf.placeholder(shape=self.action_shape, dtype=tf.float32)
        self.grads2 = tf.gradients(self.actions, actor_parameters, -self.dQ_by_da)
        self.grads_normalized2 = list(map(lambda x: tf.div(x, self.batch_size), self.grads2))
        self.opt2 = tf.train.AdamOptimizer(1e-4).apply_gradients(zip(self.grads_normalized2, actor_parameters))
        #---#

    def init_critic(self):
        with tf.variable_scope(self.scope + '/critic'):
            fc1 = slim.fully_connected(self.states, 400, activation_fn=tf.nn.relu, scope='fc1')

            # State variables
            W_s = tf.Variable(self.xavier_init([400, 300]))
            # Action variables
            W_a = tf.Variable(self.xavier_init([self.action_shape[-1], 300]))
            # Bias
            b_hidden = tf.Variable(self.xavier_init([300]))

            hidden = tf.nn.relu(tf.matmul(fc1, W_s) + tf.matmul(self.actions, W_a) + b_hidden)

            # Output layer
            W = tf.Variable(tf.random_uniform([300, 1], -3e-3, 3e-3))
            b = tf.Variable(tf.random_uniform([1], -3e-3, 3e-3))
            self.Q = tf.matmul(hidden, W) + b

        fc1_ = slim.fully_connected(self.states_, 400, activation_fn=tf.nn.relu, scope=self.scope + '/critic/fc1', reuse=True)
        hidden_ = tf.nn.relu(tf.matmul(fc1_, W_s) + tf.matmul(self.actions_, W_a) + b_hidden)
        self.Q_ = tf.matmul(hidden_, W) + b


        #---#
        # Get gradients (for actor network)
        self.grads_critic = tf.gradients(self.Q_, self.actions_)
        #---#


    def init_critic_opt(self):
        self.targetQ = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        self.critic_loss = tf.reduce_mean(tf.square(self.targetQ - self.Q_))
        #Get critic parameters
        critic_parameters = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope + '/critic')
        #Optimizer
        self.critic_solver = tf.train.AdamOptimizer(1e-3).minimize(self.critic_loss, var_list=critic_parameters)


def main():
    print 'in def main'
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
    args.action_bound = env.action_space.high
    print(args)

    # Networks
    actor_critic_source = actor_critic(state_shape=[None, args.state_dim], action_shape=[None, args.action_dim], output_bound=args.action_bound[0], scope='source')
    actor_critic_target = actor_critic(state_shape=[None, args.state_dim], action_shape=[None, args.action_dim], output_bound=args.action_bound[0], scope='target')
    '''
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    for v in variables:
        print v
    exit()
    '''

    # Update and copy operators
    update_target_actor = update_target_graph2('source/actor', 'target/actor', args.tau)
    update_target_critic = update_target_graph2('source/critic', 'target/critic', args.tau)

    copy_target_actor = update_target_graph2('source/actor', 'target/actor', 1.)
    copy_target_critic = update_target_graph2('source/critic', 'target/critic', 1.)

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
                action = sess.run(actor_critic_source.actions, feed_dict={actor_critic_source.states:state[np.newaxis, ...]})[0] + actor_noise()
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
                targetQ = sess.run(actor_critic_target.Q, feed_dict={actor_critic_target.states:states1})
                targetQ = np.squeeze(targetQ, axis=-1)
                targetQ = rewards + (1. - dones.astype(np.float32)) * args.gamma * targetQ
                targetQ = targetQ[..., np.newaxis]
                _, critic_loss = sess.run([actor_critic_source.critic_solver, actor_critic_source.critic_loss], feed_dict={actor_critic_source.states_:states, actor_critic_source.actions_:actions, actor_critic_source.targetQ:targetQ})

                '''
                # Update the actor
                sess.run(actor_critic_source.actor_solver, feed_dict={actor_critic_source.states:states})
                '''

                #---#
                critic_grads = sess.run(actor_critic_source.grads_critic, feed_dict={actor_critic_source.states_:states, actor_critic_source.actions_:actions})[0]# Grab gradients from critic
                _ = sess.run(actor_critic_source.opt2, feed_dict={actor_critic_source.states:states, actor_critic_source.dQ_by_da:critic_grads})
                #---#

                # Update target networks
                sess.run(update_target_critic)
                sess.run(update_target_actor)

                state = np.copy(state1)
                if done == True:
                    print 'epoch', epoch, 'total rewards', total_rewards
                    break


if __name__ == '__main__':
    main()
