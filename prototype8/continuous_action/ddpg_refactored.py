import numpy as np

import gym
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tflearn

import argparse

import random

import sys
sys.path.append('../../')
from utils import Memory
from utils import update_target_graph2
from utils import OrnsteinUhlenbeckActionNoise

class actorcritic:
    def __init__(self, state_shape=[None, 100], action_shape=[None, 1], output_bound_low=[-1.], output_bound_high=[1.], learning_rate=.99, tau=.001):
        self.actor_src = actor(state_shape, action_shape, output_bound_low, output_bound_high, 'actor_src')
        self.critic_src = critic(state_shape, action_shape, 'critic_src')
        self.actor_tar = actor(state_shape, action_shape, output_bound_low, output_bound_high, 'actor_tar')
        self.critic_tar = critic(state_shape, action_shape, 'critic_tar')

        self.states = tf.placeholder(shape=state_shape, dtype=tf.float32)
        self.actions = tf.placeholder(shape=action_shape, dtype=tf.float32)
        self.next_states = tf.placeholder(shape=state_shape, dtype=tf.float32)
        self.rewards = tf.placeholder(shape=[None], dtype=tf.float32)
        self.dones = tf.placeholder(shape=[None], dtype=tf.float32)

        self.action_out = self.actor_src.build(self.states, None)
        self.q = self.critic_src.build(self.states, self.actions, None)
        self.q_src = self.critic_src.build(self.states, self.action_out, True)
        self.q_tar = self.critic_tar.build(self.next_states, self.actor_tar.build(self.next_states, None), None)

        #Critic loss and optimizer
        self.closs = tf.reduce_mean(tf.square(self.rewards + (1. - self.dones) * learning_rate * tf.reduce_sum(self.q_tar, axis=-1) - tf.reduce_sum(self.q, axis=-1)))
        self.critic_solver = tf.train.AdamOptimizer(1e-3).minimize(self.closs, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'critic_src'))

        #Actor loss and optimizer
        self.aloss = -tf.reduce_mean(tf.reduce_sum(self.q_src, axis=-1))
        self.actor_solver = tf.train.AdamOptimizer(1e-4).minimize(self.aloss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'actor_src'))

        #Paramter assertions
        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'actor_src') +\
                 tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'critic_src') +\
                 tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'actor_tar') +\
                 tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'critic_tar')
        assert len(params) == len(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
        assert params == tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        # Update and copy operators
        self.update_target_actor = update_target_graph2('actor_src', 'actor_tar', tau)
        self.update_target_critic = update_target_graph2('critic_src', 'critic_tar', tau)

        self.copy_target_actor = update_target_graph2('actor_src', 'actor_tar', 1.)
        self.copy_target_critic = update_target_graph2('critic_src', 'critic_tar', 1.)

    def copy_target(self, sess):
        sess.run(self.copy_target_critic)
        sess.run(self.copy_target_actor)

    def update_target(self, sess):
        sess.run(self.update_target_critic)
        sess.run(self.update_target_actor)

    def get_action(self, sess, states):
        feed_dict = {self.states:states}
        action = sess.run(self.action_out, feed_dict=feed_dict)[0]
        return action

    def train(self, sess, states, actions, rewards, next_states, dones):
        dones = dones.astype(np.float64)

        feed_dict = {self.states:states,
                     self.actions:actions,
                     self.rewards:rewards,
                     self.next_states:next_states,
                     self.dones:dones}
        sess.run(self.critic_solver, feed_dict=feed_dict)

        feed_dict = {self.states:states}
        sess.run(self.actor_solver, feed_dict=feed_dict)

class actor:
    def __init__(self, state_shape=[None, 100], action_shape=[None, 1], output_bound_low=[-1.], output_bound_high=[1.], scope=None):
        self.scope = scope
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.output_bound_high = output_bound_high

    def build(self, states, reuse):
        fc1 = slim.fully_connected(states, 400, activation_fn=None, scope=self.scope+'/fc1', reuse=reuse)
        fc1 = tflearn.layers.normalization.batch_normalization(fc1, scope=self.scope+'/fc1_bn', reuse=reuse)
        fc1 = tf.nn.relu(fc1)

        fc2 = slim.fully_connected(fc1, 300, activation_fn=None, scope=self.scope+'/fc2', reuse=reuse)
        fc2 = tflearn.layers.normalization.batch_normalization(fc2, scope=self.scope+'/fc2_bn', reuse=reuse)
        fc2 = tf.nn.relu(fc2)

        action_bound = tf.constant(self.output_bound_high, dtype=tf.float32)
        action = slim.fully_connected(fc2,
                                      self.action_shape[-1],
                                      activation_fn=tf.nn.tanh,
                                      weights_initializer=tf.initializers.random_uniform(-3e-3, 3e-3),
                                      biases_initializer=tf.initializers.random_uniform(-3e-3, 3e-3),
                                      scope=self.scope+'/out',
                                      reuse=reuse)
        action = tf.multiply(action, action_bound)
        return action

class critic:
    def __init__(self, state_shape=[None, 100], action_shape=[None, 1], scope=None):
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.scope = scope

    def build(self, states, actions, reuse):

        fc1 = slim.fully_connected(states, 400, activation_fn=None, scope=self.scope+'/fc1', reuse=reuse)
        fc1 = tflearn.layers.normalization.batch_normalization(fc1, scope=self.scope+'/fc1_bn', reuse=reuse)
        fc1 = tf.nn.relu(fc1)

        fca = slim.fully_connected(actions, 400, activation_fn=None, scope=self.scope+'/fca', reuse=reuse)
        fca = tflearn.layers.normalization.batch_normalization(fca, scope=self.scope+'/fca_bn', reuse=reuse)
        fca = tf.nn.relu(fca)

        #hidden layer
        concat = tf.concat([fc1, fca], axis=-1)
        hidden = slim.fully_connected(concat, 300, activation_fn=None, scope=self.scope+'/fch', reuse=reuse)
        hidden = tflearn.layers.normalization.batch_normalization(hidden, scope=self.scope+'/fch_bn', reuse=reuse)
        hidden =  tf.nn.relu(hidden)

        Q = slim.fully_connected(hidden,
                                 1,
                                 activation_fn=None,
                                 weights_initializer=tf.initializers.random_uniform(-3e-3, 3e-3),
                                 biases_initializer=tf.initializers.random_uniform(-3e-3, 3e-3),
                                 scope=self.scope+'/out',
                                 reuse=reuse)
        return Q



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
    ac = actorcritic(state_shape=[None, args.state_dim],
                     action_shape=[None, args.action_dim],
                     output_bound_low=args.action_bound_low,
                     output_bound_high=args.action_bound_high,
                     learning_rate=args.gamma,
                     tau=args.tau)


    # Replay memory
    memory = Memory(args.replay_mem_size)

    # Actor noise
    actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(args.action_dim))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ac.copy_target(sess)

        for epoch in range(args.epochs):
            state = env.reset()
            total_rewards = 0.0
            while True:
                #env.render()
                # Choose an action
                #action = sess.run(actor_source.action, feed_dict={actor_source.states:state[np.newaxis, ...]})[0] + actor_noise()
                action = ac.get_action(sess, state[np.newaxis, ...]) + actor_noise()
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

                ac.train(sess, states, actions, rewards, states1, dones)

                '''
                # Update the critic
                actions1 = sess.run(actor_target.action, feed_dict={actor_target.states:states1})
                targetQ = np.squeeze(sess.run(critic_target.Q, feed_dict={critic_target.states:states1, critic_target.actions:actions1}), axis=-1)
                targetQ = rewards + (1. - dones.astype(np.float32)) * args.gamma * targetQ
                targetQ = targetQ[..., np.newaxis]
                _, critic_loss = sess.run([critic_source.critic_solver, critic_source.loss], feed_dict={critic_source.states:states, critic_source.actions:actions, critic_source.targetQ:targetQ})

                # Update the actor
                critic_grads = sess.run(critic_source.grads, feed_dict={critic_source.states:states, critic_source.actions:actions})[0]# Grab gradients from critic
                _ = sess.run(actor_source.opt, feed_dict={actor_source.states:states, actor_source.dQ_by_da:critic_grads})
                '''

                # Update target networks
                ac.update_target(sess)
                '''
                sess.run(update_target_critic)
                sess.run(update_target_actor)
                '''

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
