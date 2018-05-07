import tensorflow as tf
import numpy as np

class model_predictive_control:
    def __init__(self, input_shape=[None, 3], action_shape=[None, 1], action_bounds=[-1., 1.], T=20, K=10, learning_rate=.9):
        self.input_shape = input_shape
        self.action_shape = action_shape
        self.action_bounds = action_bounds
        self.T = T#Number of steps to look ahead
        self.K = K#Number of sampling trials
        self.learning_rate = learning_rate

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
        sampled_actions = []
        rewards = [0.] * self.K
        for k in range(self.K):
            actions = np.random.uniform(low=self.action_bounds[0],
                                        high=self.action_bounds[1],
                                        size=[self.T, self.action_shape[-1]])
            sampled_actions.append(actions[0])

            gamma = 1.
            current_state = np.copy(state)
            for action in actions:
                current_state, reward = sess.run([self.p_next_states, self.p_rewards],
                                                 feed_dict={self.states:current_state, self.actions:action[..., np.newaxis]})

                rewards[k] += gamma * reward[0, 0]
                gamma *= self.learning_rate

        return sampled_actions[np.argmax(rewards)]

    def train(self, sess, states, actions, rewards, next_states, dones, *_):
        pass

def main():
    mpc = model_predictive_control()
    
if __name__ == '__main__':
    main()
