import numpy as np
import gym
import tensorflow as tf

import uuid

class mountain_cart_continuous_state_function:
    def __init__(self):
        self.max_speed_tf = tf.get_variable(name=str(uuid.uuid4()), shape=[], dtype=tf.float64, initializer=tf.constant_initializer(.07))
        self.max_position_tf = tf.get_variable(name=str(uuid.uuid4()), shape=[], dtype=tf.float64, initializer=tf.constant_initializer(.6))
        self.min_position_tf = tf.get_variable(name=str(uuid.uuid4()), shape=[], dtype=tf.float64, initializer=tf.constant_initializer(-1.2))
        self.power_tf = tf.get_variable(name=str(uuid.uuid4()), shape=[], dtype=tf.float64, initializer=tf.constant_initializer(.0015))

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

        return tf.concat([position, velocity], axis=-1)

def main():
    states_pl = tf.placeholder(shape=[None, 2], dtype=tf.float64)
    actions_pl = tf.placeholder(shape=[None, 1], dtype=tf.float64)

    mccsf = mountain_cart_continuous_state_function()
    next_states_out = mccsf.step_tf(states_pl, actions_pl)

    env = gym.make('MountainCarContinuous-v0')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(10000):
            state = env.reset()
            while True:
                #env.render()
                action = np.random.uniform(env.action_space.low, env.action_space.high)
                next_state, reward, done, _ = env.step(action)

                next_state2 = sess.run(next_states_out, feed_dict={states_pl:state[np.newaxis, ...], actions_pl:action[np.newaxis, ...]})
                next_state2 = np.squeeze(next_state2, axis=0)

                np.testing.assert_array_equal(next_state, next_state2)

                state = np.copy(next_state)
                if done:
                    break
    
if __name__ == '__main__':
    main()
