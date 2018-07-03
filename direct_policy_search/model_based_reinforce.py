import numpy as np
import gym

import sys
sys.path.append('..')
from prototype8.dmlac.real_env_pendulum import real_env_pendulum_state
from prototype8.dmlac.real_env_pendulum import real_env_pendulum_reward

def main():
    env = gym.make('Pendulum-v0')
    no_samples = 20
    unroll_steps = 200
    epoch_experience = []

    for epoch in range(1000):
        state = env.reset()
        total_rewards = 0.
        while True:
            action = agent.act(sess, state)
            next_state, reward, done, _ = env.step(action)
            total_rewards += float(reward)

            state = np.copy(next_state)

            if done:
                print 'epoch:', epoch, 'total_rewards:', total_rewards
                epoch_experience = []
                break






if __name__ == '__main__':
    main()
