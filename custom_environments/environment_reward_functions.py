import numpy as np
import gym
import tensorflow as tf

import uuid

class mountain_car_continuous_reward_function:
    def __init__(self):
        self.max_speed_tf = tf.constant(value=.07, shape=[], dtype=tf.float64)
        self.max_position_tf = tf.constant(value=.6, shape=[], dtype=tf.float64)
        self.min_position_tf = tf.constant(value=-1.2, shape=[], dtype=tf.float64)
        self.power_tf = tf.constant(value=.0015, shape=[], dtype=tf.float64)
        self.goal_position_tf = tf.constant(value=.45, shape=[], dtype=tf.float64)

    def step_np(self, state, action, done):
        return float(done) * 100. - action[0]**2 / 10.

    def step_tf(self, state, action):
        assert state.shape.as_list() == [None, 2]
        assert action.shape.as_list() == [None, 1]

        position = state[:, 0:1]
        velocity = state[:, 1:2]

        force = tf.minimum(tf.maximum(action[:, 0], -1.), 1.)
        force = tf.expand_dims(force, axis=-1)

        velocity += force * self.power_tf - .0025 * tf.cos(3. * position)
        velocity = tf.maximum(tf.minimum(velocity, self.max_speed_tf), -self.max_speed_tf)

        position += velocity
        position = tf.maximum(tf.minimum(position, self.max_position_tf), self.min_position_tf)

        velocity *= tf.sign(tf.abs((position - self.min_position_tf)) + (1. + tf.sign(velocity)))

        done = (tf.sign(tf.sign(position - self.goal_position_tf) + .5) + 1.) * .5

        return done * 100. - tf.square(action[:, 0:1]) / 10.

def main():
    env = gym.make('MountainCarContinuous-v0')

    mccrf = mountain_car_continuous_reward_function()
    states_pl = tf.placeholder(shape=[None, 2], dtype=tf.float64)
    actions_pl = tf.placeholder(shape=[None, 1], dtype=tf.float64)
    rewards_tf = mccrf.step_tf(states_pl, actions_pl)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(10000):
            state = env.reset()
            total_rewards = 0.
            while True:
                env.render()
                action = np.random.uniform(env.action_space.low, env.action_space.high)
                next_state, reward, done, _ = env.step(action)
                total_rewards += float(reward)

                #reward2 = mccrf.step_np(state, action, done)
                reward3 = sess.run(rewards_tf, feed_dict={states_pl:state[np.newaxis, ...], actions_pl:action[np.newaxis, ...]})
                reward3 = np.squeeze(reward3, axis=0)

                #np.testing.assert_almost_equal(np.array(reward), np.array(reward2))
                np.testing.assert_almost_equal(np.array(reward), np.array(reward3))

                state = np.copy(next_state)
                if done:
                    print 'epoch:', epoch, 'total_rewards:', total_rewards
                    break

if __name__ == '__main__':
    main()
