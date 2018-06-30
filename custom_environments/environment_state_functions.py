import numpy as np
import gym
import tensorflow as tf

import uuid

class mountain_car_continuous_state_function:
    def __init__(self):
        self.max_speed_np = .07
        self.max_position_np = .6
        self.min_position_np = -1.2
        self.power_np = .0015

        self.max_speed_tf = tf.constant(value=self.max_speed_np, shape=[], dtype=tf.float64)
        self.max_position_tf = tf.constant(value=self.max_position_np, shape=[], dtype=tf.float64)
        self.min_position_tf = tf.constant(value=self.min_position_np, shape=[], dtype=tf.float64)
        self.power_tf = tf.constant(value=self.power_np, shape=[], dtype=tf.float64)

    def step_np(self, state, action):
        assert len(state) == len(action)
        assert len(state.shape) == 2
        assert len(action.shape) == 2
        assert state.shape[-1] == 2
        assert action.shape[-1] == 1

        position = state[:, 0]
        velocity = state[:, 1]
        force = np.minimum(np.maximum(action[:, 0], -1.0), 1.0)

        velocity += force*self.power_np -0.0025 * np.cos(3.*position)
        velocity = np.maximum(np.minimum(velocity, self.max_speed_np), -self.max_speed_np)
        position += velocity
        position = np.maximum(np.minimum(position, self.max_position_np), self.min_position_np)

        for i in range(len(position)):
            if position[i] == self.min_position_np and velocity[i] < 0.:
                velocity[i] = 0.

        return np.stack([position, velocity], axis=-1)

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

    mccsf = mountain_car_continuous_state_function()
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
    
def main2():
    env = gym.make('MountainCarContinuous-v0')

    states = []
    actions = []
    next_states = []
    for epoch in range(2):
        state = env.reset()
        while True:
            action = np.random.uniform(env.action_space.low, env.action_space.high, 1)

            next_state, reward, done, _ = env.step(action)
            states.append(state)
            actions.append(action)
            next_states.append(next_state)

            state = np.copy(next_state)

            if done == True:
                break

    states = np.stack(states, axis=0)
    actions = np.stack(actions, axis=0)
    next_states = np.stack(next_states, axis=0)

    state_function = mountain_car_continuous_state_function()
    next_states2 = state_function.step_np(states, actions)

    np.testing.assert_almost_equal(next_states, next_states2)

if __name__ == '__main__':
    #main()
    main2()
