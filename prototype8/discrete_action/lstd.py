import numpy as np
import gym

import argparse

import sys
sys.path.append('..')
sys.path.append('../..')
from utils import Memory

class lstd:
    def __init__(self, state_size, action_size, learning_rate):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        self.A = np.zeros((state_size * action_size, state_size * action_size))
        self.b = np.zeros((state_size * action_size, 1))
        self.w = np.zeros((state_size * action_size))
        self.count = 0.

    def updatew(self):
        self.w = np.matmul(np.linalg.pinv(self.A), self.b)

    def feature(self, state, action):
        fvec = np.zeros(self.state_size * self.action_size)
        idx = self.state_size * action
        fvec[idx : idx + self.state_size] = state
        return fvec[..., np.newaxis]

    def train(self, tup):
        state = tup[0]
        action = tup[1]
        reward = tup[2]
        next_state = tup[3]
        done = tup[4]

        vec = self.feature(state, action)
        next_vec = np.zeros(self.state_size * self.action_size)[..., np.newaxis]

        if done == False:
            next_action = self.act(next_state)
            next_vec = self.feature(next_state, next_action)

        self.A = (self.count / (self.count + 1.)) * self.A + (1. / (self.count + 1.)) * \
                 np.matmul(vec, (vec - self.learning_rate * next_vec).T)

        self.b = (self.count / (self.count + 1.)) * self.b + (1. / (self.count + 1.)) * vec * reward
        self.count += 1.

    def act(self, state):
        values = []
        for a in range(self.action_size):
            fvec = self.feature(state, a)
            values.append(np.matmul(self.w.T, fvec)[0, 0])

        return np.argmax(values)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default='CartPole-v0')
    parser.add_argument("--action-size", type=int, default=2)
    parser.add_argument("--input-shape", type=list, default=[None, 4])
    parser.add_argument("--target-update-freq", type=int, default=200)
    parser.add_argument("--epsilon-max", type=float, default=1.)
    parser.add_argument("--epsilon-min", type=float, default=.01)
    parser.add_argument("--epsilon-decay", type=float, default=.001)

    parser.add_argument("--learning-rate", type=float, default=.99)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30000)

    parser.add_argument("--replay-mem-size", type=int, default=1000000)


    args = parser.parse_args()

    env = gym.make(args.environment)
    args.action_size = env.action_space.n
    args.input_shape = [None] + list(env.observation_space.shape)

    print args

    # Initialize the controller
    controller = lstd(state_size=args.input_shape[-1],
                      action_size=args.action_size,
                      learning_rate=args.learning_rate)

    # Other parameters
    epsilon = args.epsilon_max

    # Replay memory
    memory = Memory(args.replay_mem_size)

    # Time step
    time_step = 0.

    for epoch in range(args.epochs):
        total_reward = 0
        observation = env.reset()
        for t in range(1000000):
            env.render()
            controller.updatew()
            action = controller.act(observation)
            if np.random.rand() < epsilon:
                action = np.random.randint(args.action_size)
            observation1, reward, done, info = env.step(action)
            total_reward += reward

            # Add to memory
            #memory.add([observation, action, reward, observation1, done])

            # Reduce epsilon
            time_step += 1.
            epsilon = args.epsilon_min + (args.epsilon_max - args.epsilon_min) * np.exp(-args.epsilon_decay * time_step)

            # Training step
            #batch = np.array(memory.sample(args.batch_size))
            controller.train([observation, action, reward, observation1, done])

            # Set observation
            observation = observation1

            if done:
                print"Episode finished after {} timesteps".format(t+1), 'epoch', epoch, 'total_rewards', total_reward
                break


if __name__ == '__main__':
    main()
