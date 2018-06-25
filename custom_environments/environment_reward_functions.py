import numpy as np
import gym

import tensorflow as tf

class mountain_cart_continuous_reward_function:
    def __init__(self):
        pass

    def reward_np(self, state, action, done):
        return float(done) * 100. - action[0]**2 / 10.

    def reward_tf(self, state, action, done):
        assert action.shape.as_list() == [None, 1]
        assert done.shape.as_list() == [None]

        return tf.cast(done, dtype=tf.float64) * 100. - \
               tf.square(action[:, 0]) / 10.

def main():
    env = gym.make('MountainCarContinuous-v0')

    mccrf = mountain_cart_continuous_reward_function()
    for epoch in range(10000):
        state = env.reset()
        while True:
            #env.render()
            action = np.random.uniform(env.action_space.low, env.action_space.high)
            next_state, reward, done, _ = env.step(action)

            reward2 = mccrf.reward_np(state, action, done)

            np.testing.assert_almost_equal(np.array(reward), np.array(reward2))

            state = np.copy(next_state)
            if done:
                break

if __name__ == '__main__':
    main()
