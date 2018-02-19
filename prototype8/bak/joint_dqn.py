import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import gym

import argparse

from dyna_gan import qnetwork
from gan import CGAN

from utils import sample_z
from utils import update_target_graph
from utils import Memory

class joint_qnetwork:
    def __init__(self, input_shape, action_size, latent_size, learning_rate):
        # Parameters
        self.lamb = .5
        self.learning_rate = learning_rate
        # Initialize the networks
        self.cgan_state = CGAN(input_shape=input_shape, action_size=action_size, latent_size=latent_size, gen_input_shape=input_shape)
        self.cgan_reward = CGAN(input_shape=input_shape, action_size=action_size, latent_size=latent_size, gen_input_shape=[None, 1])
        self.qnet = qnetwork(input_shape=input_shape, action_size=action_size, state_model=self.cgan_state.G_sample, scope='qnet')
        self.target_qnet = qnetwork(input_shape=input_shape, action_size=action_size, state_model=self.cgan_state.G_sample, scope='target_qnet')

        # -- Loss functions --
        # Additional placeholders for defining the loss functions
        self.rewards = tf.placeholder(shape=[None], dtype=tf.float32)
        self.dones = tf.placeholder(shape=[None], dtype=tf.float32)
        # Joint loss
        self.loss_joint = tf.reduce_mean(tf.square(tf.squeeze(self.cgan_reward.G_sample, axis=-1) + self.learning_rate * tf.reduce_max(self.target_qnet.Q_state_model, axis=-1, keep_dims=False) - self.qnet.responsible_output))
        # Q
        self.loss_Q = tf.reduce_mean(tf.square(self.rewards + (1. - self.dones) * self.learning_rate * tf.reduce_max(self.target_qnet.Q, axis=-1, keep_dims=False) - self.qnet.responsible_output))
        # State model
        self.loss_state_model = self.cgan_state.G_loss
        # Reward model
        self.loss_reward_model = self.cgan_reward.G_loss

        # -- Optimizers --
        # Q
        self.opt_Q = tf.train.AdamOptimizer().minimize(self.loss_Q + self.lamb * self.loss_joint, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'qnet'))
        # State model
        self.opt_state_model_D = self.cgan_state.D_solver
        self.opt_state_model_G = tf.train.AdamOptimizer().minimize(self.loss_state_model + self.lamb * self.loss_joint, var_list=self.cgan_state.theta_G)
        # Reward model
        self.opt_reward_model_D = self.cgan_reward.D_solver
        self.opt_reward_model_G = tf.train.AdamOptimizer().minimize(self.loss_reward_model + self.lamb * self.loss_joint, var_list=self.cgan_reward.theta_G)

        # TODO: Include separate training for the state and reward models?

    def get_action(self, sess, observation):
        Q = sess.run(self.qnet.Q, feed_dict={self.qnet.states:observation[np.newaxis, ...]})
        return np.argmax(Q)

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
    parser.add_argument("--latent-size", type=int, default=4, help='Size of vector for Z')

    args = parser.parse_args()

    env = gym.make(args.environment)
    args.action_size = env.action_space.n
    args.input_shape = [None, env.observation_space.shape[0]]
    print args

    # Other parameters
    epsilon = args.epsilon_max

    # Replay memory
    memory = Memory(args.replay_mem_size)

    # Time step
    time_step = 0.

    # Initialize the model
    jqnet = joint_qnetwork(args.input_shape, args.action_size, args.latent_size, args.learning_rate)
    update_ops = update_target_graph('qnet', 'target_qnet')
    '''
    for v in tf.all_variables():
        print v
    '''
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(args.epochs):
            total_reward = 0
            observation = env.reset()
            for t in range(1000000):
                #env.render()
                action = jqnet.get_action(sess, observation)
                if np.random.rand() < epsilon:
                    action = env.action_space.sample()
                observation1, reward, done, info = env.step(action)
                total_reward += reward

                # Add to memory
                memory.add([observation, action, reward, observation1, done])

                # Reduce epsilon
                time_step += 1.
                epsilon = args.epsilon_min + (args.epsilon_max - args.epsilon_min) * np.exp(-args.epsilon_decay * time_step)

                # Training step
                batch = np.array(memory.sample(args.batch_size))
                assert len(batch) > 0
                states = np.vstack(batch[:, 0])
                actions = np.array(batch[:, 1])
                rewards = batch[:, 2]
                states1 = np.vstack(batch[:, 3])
                dones = batch[:, 4].astype(np.float32)

                # Update Q
                _ = sess.run(jqnet.opt_Q, feed_dict={jqnet.rewards:rewards, jqnet.dones:dones, jqnet.target_qnet.states:states1, jqnet.qnet.states:states, jqnet.qnet.actions:actions, jqnet.cgan_reward.states:states, jqnet.cgan_reward.actions:actions, jqnet.cgan_reward.Z:sample_z(len(batch), args.latent_size), jqnet.cgan_state.states:states, jqnet.cgan_state.actions:actions, jqnet.cgan_state.Z:sample_z(len(batch), args.latent_size)})
                # Update state model
                _ = sess.run(jqnet.opt_state_model_D, feed_dict={jqnet.cgan_state.states:states, jqnet.cgan_state.actions:actions, jqnet.cgan_state.Z:sample_z(len(batch), args.latent_size), jqnet.cgan_state.X:states1})
                _ = sess.run(jqnet.opt_state_model_G, feed_dict={jqnet.qnet.states:states, jqnet.qnet.actions:actions, jqnet.cgan_reward.states:states, jqnet.cgan_reward.actions:actions, jqnet.cgan_reward.Z:sample_z(len(batch), args.latent_size), jqnet.cgan_state.states:states, jqnet.cgan_state.actions:actions, jqnet.cgan_state.Z:sample_z(len(batch), args.latent_size)})
                # Update reward model
                _ = sess.run(jqnet.opt_reward_model_D, feed_dict={jqnet.cgan_reward.states:states, jqnet.cgan_reward.actions:actions, jqnet.cgan_reward.Z:sample_z(len(batch), args.latent_size), jqnet.cgan_reward.X:rewards[..., np.newaxis]})
                _ = sess.run(jqnet.opt_reward_model_G, feed_dict={jqnet.qnet.states:states, jqnet.qnet.actions:actions, jqnet.cgan_reward.states:states, jqnet.cgan_reward.actions:actions, jqnet.cgan_reward.Z:sample_z(len(batch), args.latent_size), jqnet.cgan_state.states:states, jqnet.cgan_state.actions:actions, jqnet.cgan_state.Z:sample_z(len(batch), args.latent_size)})

                # Set observation
                observation = observation1

                # Update?
                if int(time_step) % args.target_update_freq == 0:
                    print "Updating target..."
                    sess.run(update_ops)

                if done:
                    print("Episode finished after {} timesteps".format(t+1))
                    print epoch, total_reward
                    break

    env.close()
    gym.upload('/tmp/cartpole-experiment-' + str(rand_no), api_key='sk_AlBXbTIgR4yaxPlvDpm61g')

if __name__ == '__main__':
    main()
