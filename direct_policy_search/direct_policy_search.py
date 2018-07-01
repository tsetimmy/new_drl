import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import gym

import argparse
import pickle

import sys
sys.path.append('..')
from prototype8.dmlac.real_env_pendulum import real_env_pendulum_state
#from prototype8.dmlac.real_env_pendulum import real_env_pendulum_reward
from custom_environments.environment_state_functions import mountain_car_continuous_state_function
from custom_environments.environment_reward_functions import mountain_car_continuous_reward_function

from custom_environments.generateANN_env import ANN

from utils import Memory

class direct_policy_search:
    def __init__(self, state_dim, action_dim, action_bound_high, \
                 action_bound_low, unroll_length, discount_factor, \
                 gradient_descent_steps, scope):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound_high = action_bound_high
        self.action_bound_low = action_bound_low
        self.unroll_length = unroll_length
        self.discount_factor = discount_factor
        self.gradient_descent_steps = gradient_descent_steps
        self.scope = scope

        #Make sure bounds are same (assumption can be relaxed later)
        np.testing.assert_array_equal(-self.action_bound_low, self.action_bound_high)

        #Flags
        self.policy_reuse_vars = None

        '''
        self.reward_model = ANN(self.state_dim+self.action_dim, 1)
        self.placeholders_reward = [tf.placeholder(shape=v.shape, dtype=tf.float64)
                                    for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.reward_model.scope)]
        self.assign_ops0 = [v.assign(pl) for v, pl in zip(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.reward_model.scope),
                            self.placeholders_reward)]
        '''
        #self.reward_model = real_env_pendulum_reward()
        self.reward_model = mountain_car_continuous_reward_function()

        #self.state_model = real_env_pendulum_state()
        #self.state_model = mountain_car_continuous_state_function()
        self.state_model = ANN(self.state_dim+self.action_dim, self.state_dim)
        self.placeholders_state = [tf.placeholder(shape=v.shape, dtype=tf.float64) for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.state_model.scope)]
        self.assign_ops1 = [v.assign(pl) for v, pl in zip(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.state_model.scope), self.placeholders_state)]

        #Build computational graph (i.e., unroll policy)
        #self.states = tf.placeholder(shape=[None, self.state_dim], dtype=tf.float32)
        self.states = tf.placeholder(shape=[None, self.state_dim], dtype=tf.float64)

        self.action = self.build_policy(self.states)
        state = self.states
        action = self.build_policy(state)
        rewards = []
        for i in range(self.unroll_length):
            print i
            #reward = pow(self.discount_factor, i) * self.reward_model.build(state, action)
            #reward = pow(self.discount_factor, i) * self.reward_model.step_tf(state, action)
            reward = pow(self.discount_factor, i) * self.reward_model.sigmoid_approx(state, action)
            rewards.append(reward)
            state = self.state_model.build(state, action)
            #state = self.state_model.step_tf(state, action)
            action = self.build_policy(state)

        rewards = tf.reduce_sum(tf.stack(rewards, axis=-1), axis=-1)
        print 'here0'
        self.loss = -tf.reduce_mean(tf.reduce_sum(rewards, axis=-1))
        print 'here1'
        self.opt = tf.train.AdamOptimizer().minimize(self.loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope))
        print 'here2'

    def act(self, sess, states):
        states = np.atleast_2d(states)
        #print sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
        action = sess.run(self.action, feed_dict={self.states:states})
        return action[0]

    def train(self, sess, states):
        for _ in range(self.gradient_descent_steps):
            loss, _ = sess.run([self.loss, self.opt], feed_dict={self.states:states})
            #asin1, asin2, loss, _ = sess.run([self.asin1, self.asin2, self.loss, self.opt], feed_dict={self.states:states})

    def build_policy(self, states):
        assert states.shape.as_list() == [None, self.state_dim]

        #Fully connected layer 1
        fc1 = slim.fully_connected(states, 256, activation_fn=tf.nn.relu, scope=self.scope+'/fc1', reuse=self.policy_reuse_vars)

        fc2 = slim.fully_connected(fc1, 256, activation_fn=tf.nn.relu, scope=self.scope+'/fc2', reuse=self.policy_reuse_vars)

        #Output layer
        output = slim.fully_connected(fc2, self.action_dim, activation_fn=tf.nn.tanh, scope=self.scope+'/output', reuse=self.policy_reuse_vars)

        #Apply action bounds
        #action_bound = tf.constant(self.action_bound_high, dtype=tf.float32)
        action_bound = tf.constant(self.action_bound_high, dtype=tf.float64)
        policy = tf.multiply(output, action_bound)

        #Change flag
        self.policy_reuse_vars = True

        return policy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default='MountainCarContinuous-v0')
    parser.add_argument("--unroll-steps", type=int, default=20)
    parser.add_argument("--time-steps", type=int, default=30000)
    parser.add_argument("--replay-mem-size", type=int, default=1000000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--discount-factor", type=float, default=1.)
    parser.add_argument("--goal-position", type=float, default=.45)
    args = parser.parse_args()

    env = gym.make(args.environment)
    env.seed(seed=args.goal_position)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound_high = env.action_space.high
    action_bound_low = env.action_space.low

    agent = direct_policy_search(state_dim, action_dim, action_bound_high,
                                 action_bound_low, args.unroll_steps, args.discount_factor, 1, 'direct_policy_search')

    # Replay memory
    memory = Memory(args.replay_mem_size)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #weights = pickle.load(open('../custom_environments/weights/pendulum_reward.p', 'rb'))
        #weights = pickle.load(open('../custom_environments/weights/mountain_car_continuous_reward'+str(args.goal_position)+'.p', 'rb'))
        #sess.run(agent.assign_ops0, feed_dict=dict(zip(agent.placeholders_reward, weights)))
        weights = pickle.load(open('../custom_environments/weights/mountain_car_continuous_next_state.p', 'rb'))
        sess.run(agent.assign_ops1, feed_dict=dict(zip(agent.placeholders_state, weights)))
        state = env.reset()
        total_rewards = 0.0
        epoch = 1
        for time_steps in range(args.time_steps):
            env.render()
            action = agent.act(sess, state)
            next_state, reward, done, _ = env.step(action)
            total_rewards += float(reward)

            # Store tuple in replay memory
            memory.add([np.atleast_2d(state), np.atleast_2d(action), reward, np.atleast_2d(next_state), done])

            # Training step
            batch = np.array(memory.sample(args.batch_size))
            assert len(batch) > 0
            states = np.concatenate(batch[:, 0], axis=0)

            # Train the agent
            agent.train(sess, states)

            # s <- s'
            state = np.copy(next_state)

            if done == True:
                print 'time steps', time_steps, 'epoch', epoch, 'total rewards', total_rewards, 'unroll', args.unroll_steps
                epoch += 1
                total_rewards = 0.
                state = env.reset()

if __name__ == '__main__':
    main()
