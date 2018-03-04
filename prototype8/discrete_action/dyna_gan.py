import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
#import gym

import argparse


import sys
sys.path.append('..')
from gan import CGAN
from qnetwork import qnetwork
sys.path.append('../..')
from utils import env_interface
from utils import update_target_graph
from utils import Memory
from utils import sample_z

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-interface", type=str, default='gym!atari')
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

    parser.add_argument("--K", type=int, default=1, help='The number of steps to train the environment')
    parser.add_argument("--L", type=int, default=1, help='The number of Q-learning steps for hypothetical rollouts')
    parser.add_argument("--latent-size", type=int, default=4, help='Size of vector for Z')

    args = parser.parse_args()

    env = env_interface(args.env_interface, args.environment, pixel_feature=False, render=True)

    #args.action_size = env.action_space.n
    args.action_size = env.action_size
    args.input_shape = [None] + list(env.obs_space_shape)

    print args

    # Other parameters
    epsilon = args.epsilon_max

    # Replay memory
    memory = Memory(args.replay_mem_size)

    # Time step
    time_step = 0.

    # Initialize the GANs
    cgan_state = CGAN(input_shape=args.input_shape, action_size=args.action_size, latent_size=args.latent_size, gen_input_shape=args.input_shape)
    cgan_reward = CGAN(input_shape=args.input_shape, action_size=args.action_size, latent_size=args.latent_size, gen_input_shape=[None, 1])

    qnet = qnetwork(input_shape=args.input_shape, action_size=args.action_size, scope='qnet')
    target_qnet = qnetwork(input_shape=args.input_shape, action_size=args.action_size, scope='target_qnet')
    update_ops = update_target_graph('qnet', 'target_qnet')

    rand_no = np.random.rand()
    #env = gym.wrappers.Monitor(env, '/tmp/cartpole-experiment-' + str(rand_no), force=True, video_callable=False)
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(args.epochs):
            total_reward = 0
            observation = env.reset()
            for t in range(1000000):
                #env.render()
                action = qnet.get_action(sess, observation)
                if np.random.rand() < epsilon:
                    #action = env.action_space.sample()
                    action = np.random.randint(args.action_size)
                observation1, reward, done, info = env.step(action)
                total_reward += reward

                # Add to memory
                memory.add([observation, action, reward, observation1, done])

                # Reduce epsilon
                time_step += 1.
                epsilon = args.epsilon_min + (args.epsilon_max - args.epsilon_min) * np.exp(-args.epsilon_decay * time_step)

                # Training step
                batch = np.array(memory.sample(args.batch_size))
                qnet.train(sess, batch, args.learning_rate, target_qnet)

                # Training step: environment model
                for k in range(args.K):
                    batch = np.array(memory.sample(args.batch_size))

                    states = np.vstack(batch[:, 0])
                    actions = np.array(batch[:, 1])
                    rewards = batch[:, 2]
                    states1 = np.vstack(batch[:, 3])

                    _, D_loss_state = sess.run([cgan_state.D_solver, cgan_state.D_loss], feed_dict={cgan_state.states:states, cgan_state.actions:actions, cgan_state.Z:sample_z(len(batch), args.latent_size), cgan_state.X:states1})
                    _, G_loss_state = sess.run([cgan_state.G_solver, cgan_state.G_loss], feed_dict={cgan_state.states:states, cgan_state.actions:actions, cgan_state.Z:sample_z(len(batch), args.latent_size)})

                    _, D_loss_reward = sess.run([cgan_reward.D_solver, cgan_reward.D_loss], feed_dict={cgan_reward.states:states, cgan_reward.actions:actions, cgan_reward.Z:sample_z(len(batch), args.latent_size), cgan_reward.X:rewards[..., np.newaxis]})
                    _, G_loss_reward = sess.run([cgan_reward.G_solver, cgan_reward.G_loss], feed_dict={cgan_reward.states:states, cgan_reward.actions:actions, cgan_reward.Z:sample_z(len(batch), args.latent_size)})
                    #print D_loss_state, G_loss_state, D_loss_reward, G_loss_state

                # Training step: imagination rollouts
                if time_step == 0.:
                    print "time_step 0 here"
                if time_step >= 0.:
                    for l in range(args.L):
                        batch = np.array(memory.sample(args.batch_size))
                        assert len(batch) > 0

                        states1 = np.vstack(batch[:, 3])
                        actions = np.random.randint(args.action_size, size=len(batch))
                        dones = np.array([False] * len(batch))

                        G_sample_state = sess.run(cgan_state.G_sample, feed_dict={cgan_state.states:states1, cgan_state.actions:actions, cgan_state.Z:sample_z(len(batch), args.latent_size)})
                        G_sample_reward = sess.run(cgan_reward.G_sample, feed_dict={cgan_reward.states:states1, cgan_reward.actions:actions, cgan_reward.Z:sample_z(len(batch), args.latent_size)})
                        qnet.train(sess, None, args.learning_rate, target_qnet, states1, actions, G_sample_reward, G_sample_state, dones)

                # Set observation
                observation = observation1

                # Update?
                if int(time_step) % args.target_update_freq == 0:
                    #print "Updating target..."
                    sess.run(update_ops)

                if done:
                    print"Episode finished after {} timesteps".format(t+1), 'epoch', epoch, 'total_rewards', total_reward
                    break

    #env.close()
    #gym.upload('/tmp/cartpole-experiment-' + str(rand_no), api_key='sk_AlBXbTIgR4yaxPlvDpm61g')

if __name__ == '__main__':
    main()
