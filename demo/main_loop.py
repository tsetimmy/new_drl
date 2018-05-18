import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import gym

import argparse

from qnetwork import qnetwork
from utils import Memory, update_target_graph

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default='CartPole-v0')
    parser.add_argument("--action-size", type=int, default=2)
    parser.add_argument("--input-shape", type=list, default=[None, 4])
    parser.add_argument("--target-update-freq", type=int, default=200)
    parser.add_argument("--epsilon-max", type=float, default=1.)
    parser.add_argument("--epsilon-min", type=float, default=.01)
    parser.add_argument("--epsilon-decay", type=float, default=.001)

    parser.add_argument("--discount-factor", type=float, default=.99)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=300)

    parser.add_argument("--replay-mem-size", type=int, default=1000000)
    args = parser.parse_args()

    env = gym.make(args.environment)
    args.action_size = env.action_space.n
    args.input_shape = [None] + list(env.observation_space.shape)

    print args

    # Epsilon parameter
    epsilon = args.epsilon_max

    # Replay memory
    memory = Memory(args.replay_mem_size)

    # Time step
    time_step = 0.

    # Initialize the agent
    qnet = qnetwork(input_shape=args.input_shape, action_size=args.action_size, scope='qnet')
    tnet = qnetwork(input_shape=args.input_shape, action_size=args.action_size, scope='tnet')
    update_ops = update_target_graph('qnet', 'tnet')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(args.epochs):
            total_reward = 0
            state = env.reset()
            while True:
                #env.render()
                if np.random.rand() < epsilon:
                    action = np.random.randint(args.action_size)
                else:
                    action = qnet.act(sess, state)
                next_state, reward, done, _ = env.step(action)
                total_reward += reward

                # Add to memory
                memory.add([state, action, reward, next_state, done])

                # Reduce epsilon
                time_step += 1.
                epsilon = args.epsilon_min + (args.epsilon_max - args.epsilon_min) * np.exp(-args.epsilon_decay * time_step)

                # Training step
                batch = np.array(memory.sample(args.batch_size))
                qnet.train(sess, batch, args.discount_factor, tnet)

                # s <- s'
                state = np.copy(next_state)

                # Update target network
                if int(time_step) % args.target_update_freq == 0:
                    sess.run(update_ops)

                if done:
                    print 'epoch:', epoch, 'total_rewards:', total_reward
                    break

if __name__ == '__main__':
    main()
