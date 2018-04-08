import tensorflow as tf
import numpy as np

class model_predictive_control:
    def __init__(self, input_shape=[None, 3], action_shape=[None, 1], action_bounds=[-1., 1.], T=4, K=50):
        self.input_shape = input_shape
        self.action_shape = action_shape
        self.action_bounds = action_bounds
        self.T = T#Number of steps to look ahead
        self.K = K#Number of sampling trials

        assert len(self.action_bounds) == 2

        self.states = tf.placeholder(shape=self.input_shape, dtype=tf.float32)
        self.actions = tf.placeholder(shape=action_shape, dtype=tf.float32)

        self.p_next_states, self.p_rewards = self.init_model(self.states, self.actions)
        print self.p_next_states.shape
        print self.p_rewards.shape

    def init_model(self, states, actions):
        from real_env_pendulum import real_env_pendulum_state, real_env_pendulum_reward

        pendulum_state = real_env_pendulum_state(self.input_shape, self.action_shape)
        pendulum_reward = real_env_pendulum_reward(self.input_shape, self.action_shape)

        p_next_states = pendulum_state.build(states, actions)
        p_rewards = pendulum_reward.build(states, actions)

        return p_next_states, p_rewards

    def action(self, sess, state):
        for k in range(self.K):
            actions = np.random.uniform(low=self.action_bounds[0],
                                        high=self.action_bounds[1],
                                        size=self.T)

            current_state = np.copy(state)
            for action in actions:
                pass






def main():
    mpc = model_predictive_control()
    
if __name__ == '__main__':
    main()
