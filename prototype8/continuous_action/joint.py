import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import gym

import sys
sys.path.append('..')
sys.path.append('../..')
from gan import CGAN
from utils import Memory
from utils import update_target_graph2
from utils import OrnsteinUhlenbeckActionNoise
from utils import sample_z

from ddpg import actor, critic
import argparse

def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z

class joint_ddpg():
    def __init__(self, input_shape, action_size, latent_size, learning_rate, action_bound):
        #Parameters
        self.lamb = .5
        self.learning_rate = learning_rate

        #Initialize the networks
        self.cgan_state = CGAN(input_shape=input_shape, action_size=action_size, latent_size=latent_size, gen_input_shape=input_shape, continuous_action=True)
        self.cgan_state.init_second_stream()

        self.cgan_reward = CGAN(input_shape=input_shape, action_size=action_size, latent_size=latent_size, gen_input_shape=[None, 1], continuous_action=True)
        self.cgan_reward.init_second_stream()

        self.actor_source = actor(state_shape=input_shape, action_shape=[None, action_size], output_bound=action_bound[0], scope='actor_source')
        self.critic_source = critic(state_shape=input_shape, action_shape=[None, action_size], scope='critic_source')

        self.actor_target = actor(state_shape=input_shape, action_shape=[None, action_size], output_bound=action_bound[0], scope='actor_target')
        self.critic_target = critic(state_shape=input_shape, action_shape=[None, action_size], scope='critic_target')

        #Placeholders for defining the loss functions
        self.states = tf.placeholder(shape=input_shape, dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None, action_size], dtype=tf.float32)
        self.rewards = tf.placeholder(shape=[None], dtype=tf.float32)
        self.states1 = tf.placeholder(shape=input_shape, dtype=tf.float32)
        self.dones = tf.placeholder(shape=[None], dtype=tf.float32)
        self.batch_size = tf.cast(tf.shape(self.states)[0], tf.float32)

        # -- Loss functions --
        #Define the joint loss
        self.mu_target = self.actor_target.get_action(self.cgan_state.G_sample2)
        self.Q_target_joint = self.critic_target.get_Q(self.cgan_state.G_sample2, self.mu_target)
        self.Q_source_joint = self.critic_source.get_Q(self.cgan_state.states2, self.cgan_state.actions2)
        self.loss_joint = tf.reduce_mean(tf.squeeze(self.cgan_reward.G_sample2 + tf.square(self.Q_target_joint - self.Q_source_joint), axis=-1))

        #Define the Q loss
        Q_target = self.critic_target.get_Q(self.states1, self.actor_target.get_action(self.states1))
        Q_source = self.critic_source.get_Q(self.states, self.actions)
        self.Q_source = tf.squeeze(Q_source, axis=-1)
        self.Q_target = self.rewards + self.learning_rate * tf.squeeze(Q_target, axis=-1)
        self.loss_Q = tf.reduce_mean(tf.square(self.Q_target - self.Q_source)) + self.lamb * self.loss_joint

        #Define the state model loss
        self.loss_state_model = self.cgan_state.G_loss + self.lamb * self.loss_joint

        #Define the reward model loss
        self.loss_reward_model = self.cgan_reward.G_loss + self.lamb * self.loss_joint

        # -- Optimizers --
        #Q opt
        self.opt_Q = tf.train.AdamOptimizer().minimize(self.loss_Q, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'critic_source'))

        #State model
        self.opt_state_model_D = self.cgan_state.D_solver
        self.opt_state_model_G = tf.train.AdamOptimizer().minimize(self.loss_state_model, var_list=self.cgan_state.theta_G)

        #Reward model
        self.opt_reward_model_D = self.cgan_reward.D_solver
        self.opt_reward_model_G = tf.train.AdamOptimizer().minimize(self.loss_reward_model, var_list=self.cgan_reward.theta_G)

        # -- Optimizer for the actor --
        #Get the gradients from the critic
        self.critic_grads_M = tf.gradients(self.Q_source_joint, self.cgan_state.actions2)
        self.actions_M = self.actor_source.get_action(self.cgan_state.states2)

        self.critic_grads_B = tf.gradients(Q_source, self.actions)
        self.actions_B = self.actor_source.get_action(self.states)

        #Parameters
        self.parameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'actor_source')

        #Actor gradients
        self.dQ_by_da_M = tf.placeholder(shape=[None, action_size], dtype=tf.float32)
        self.actor_grads_M = tf.gradients(self.actions_M, self.parameters, -self.dQ_by_da_M)
        self.actor_grads_normalized_M = list(map(lambda x: tf.div(x, self.batch_size), self.actor_grads_M))
        assert len(self.actor_grads_normalized_M) == len(self.parameters)

        self.dQ_by_da_B = tf.placeholder(shape=[None, action_size], dtype=tf.float32)
        self.actor_grads_B = tf.gradients(self.actions_B, self.parameters, -self.dQ_by_da_B)
        self.actor_grads_normalized_B = list(map(lambda x: tf.div(x, self.batch_size), self.actor_grads_B))
        assert len(self.actor_grads_normalized_B) == len(self.parameters)

        self.grads = list(map(lambda M, B: M + B, self.actor_grads_normalized_M, self.actor_grads_normalized_B))
        assert len(self.grads) == len(self.parameters)

        self.opt_actor = tf.train.AdamOptimizer(1e-4).apply_gradients(zip(self.grads, self.parameters))

        '''
        for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            print v
        exit()
        '''
    def get_action(self, sess, state):
        return sess.run(self.actions_B, feed_dict={self.states:state[np.newaxis, ...]})[0]

    def train(self, sess, states_B, actions_B, rewards_B, states1_B, dones_B, states_M, actions_M, batch_size, latent_size):
        #Joint dictionary
        feed_dict_joint = {self.cgan_state.states2:states_M,
            self.cgan_state.actions2:actions_M,
            self.cgan_state.Z2:sample_z(batch_size, latent_size),
            self.cgan_reward.states2:states_M,
            self.cgan_reward.actions2:actions_M,
            self.cgan_reward.Z2:sample_z(batch_size, latent_size)}

        #Update critic
        #Dict for critic
        feed_dict_critic = {self.states:states_B,
            self.actions:actions_B,
            self.rewards:rewards_B,
            self.states1:states1_B,
            self.dones:dones_B}

        _, l_Q = sess.run([self.opt_Q,
            self.loss_Q],
            feed_dict=merge_two_dicts(feed_dict_joint,
                    feed_dict_critic))

        #Get another Z sample
        feed_dict_joint[self.cgan_state.Z2] = sample_z(batch_size, latent_size)
        feed_dict_joint[self.cgan_reward.Z2] = sample_z(batch_size, latent_size)
        
        #Update actor
        #Get gradients from the critic
        critic_grads_B = sess.run(self.critic_grads_B,
            feed_dict={self.states:states_B,
                self.actions:actions_B})[0]
        critic_grads_M = sess.run(self.critic_grads_M,
            feed_dict={self.cgan_state.states2:states_M,
                self.cgan_state.actions2:actions_M})[0]

        #Perform update
        _ = sess.run(self.opt_actor,
            feed_dict={self.dQ_by_da_M:critic_grads_M,
                self.dQ_by_da_B:critic_grads_B,
                self.cgan_state.states2:states_M,
                self.states:states_B})

        #Update state model (discriminator)
        feed_dict_state_model={self.cgan_state.states:states_B,
            self.cgan_state.actions:actions_B,
            self.cgan_state.Z:sample_z(batch_size, latent_size),
            self.cgan_state.X:states1_B}
        _ = sess.run(self.opt_state_model_D, feed_dict=feed_dict_state_model)

        #Update state model (generator)
        feed_dict_state_model.pop(self.cgan_state.X)
        feed_dict_state_model[self.cgan_state.Z] = sample_z(batch_size, latent_size)

        _ = sess.run(self.opt_state_model_G,
            feed_dict=merge_two_dicts(feed_dict_joint,
                feed_dict_state_model))

        #Get another Z sample
        feed_dict_joint[self.cgan_state.Z2] = sample_z(batch_size, latent_size)
        feed_dict_joint[self.cgan_reward.Z2] = sample_z(batch_size, latent_size)

        #Update reward model (discriminator)
        feed_dict_reward_model={self.cgan_reward.states:states_B,
            self.cgan_reward.actions:actions_B,
            self.cgan_reward.Z:sample_z(batch_size, latent_size),
            self.cgan_reward.X:rewards_B[..., np.newaxis]}
        _ = sess.run(self.opt_reward_model_D, feed_dict=feed_dict_reward_model)

        #Update reward model (generator)
        feed_dict_reward_model.pop(self.cgan_reward.X)
        feed_dict_reward_model[self.cgan_reward.Z] = sample_z(batch_size, latent_size)

        _ = sess.run(self.opt_reward_model_G,
            feed_dict=merge_two_dicts(feed_dict_joint,
                feed_dict_reward_model))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default='Pendulum-v0')
    parser.add_argument("--action-dim", type=int, default=1)
    parser.add_argument("--state-dim", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=30000)
    parser.add_argument('--tau', help='soft target update parameter', default=0.001)
    parser.add_argument("--action-bound", type=float, default=1.)
    parser.add_argument("--replay-mem-size", type=int, default=1000000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=.99)

    parser.add_argument("--latent-size", type=int, default=4, help='Size of vector for Z')

    parser.add_argument("--model", type=str, default='gan')
    args = parser.parse_args()

    assert args.model in ['gan', 'gated']
    # Initialize environment
    env = gym.make(args.environment)
    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    #assert args.action_dim == 1
    args.action_bound = env.action_space.high
    print(args)

    jointddpg, update_target_actor, update_target_critic, copy_target_actor, copy_target_critic = init_model([None, args.state_dim], args.action_dim, args.latent_size, args.learning_rate, args.action_bound, args.tau, args.model)

    '''
    #Network
    jointddpg = joint_ddpg([None, args.state_dim], args.action_dim, args.latent_size, args.learning_rate, args.action_bound)

    # Update and copy operators
    update_target_actor = update_target_graph2('actor_source', 'actor_target', args.tau)
    update_target_critic = update_target_graph2('critic_source', 'critic_target', args.tau)

    copy_target_actor = update_target_graph2('actor_source', 'actor_target', 1.)
    copy_target_critic = update_target_graph2('critic_source', 'critic_target', 1.)
    '''

    # Replay memory
    memory = Memory(args.replay_mem_size)

    # Actor noise
    actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(args.action_dim))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(copy_target_critic)
        sess.run(copy_target_actor)

        for epoch in range(args.epochs):
            state = env.reset()
            total_rewards = 0.0
            while True:
                #env.render()
                # Choose an action
                action = jointddpg.get_action(sess, state) + actor_noise()
                #action = sess.run(jointddpg.actions_B, feed_dict={jointddpg.states:state[np.newaxis, ...]})[0] + actor_noise()
                # Execute action
                state1, reward, done, _ = env.step(action)
                total_rewards += float(reward)
                # Store tuple in replay memory
                memory.add([state[np.newaxis, ...], action[np.newaxis, ...], reward, state1[np.newaxis, ...], done])

                # Training step
                batch_B = np.array(memory.sample(args.batch_size))
                assert len(batch_B) > 0
                states_B = np.concatenate(batch_B[:, 0], axis=0)
                actions_B = np.concatenate(batch_B[:, 1], axis=0)
                rewards_B = batch_B[:, 2]
                states1_B = np.concatenate(batch_B[:, 3], axis=0)
                dones_B = batch_B[:, 4]

                #Get another batch
                batch_M = np.array(memory.sample(args.batch_size))
                assert len(batch_M) > 0
                states_M = np.vstack(batch_M[:, 0])
                actions_M = np.concatenate(batch_M[:, 1], axis=0)

                jointddpg.train(sess, states_B, actions_B, rewards_B, states1_B, dones_B, states_M, actions_M, len(batch_M), args.latent_size)
                '''
                #Joint dictionary
                feed_dict_joint = {jointddpg.cgan_state.states2:states_M,\
                    jointddpg.cgan_state.actions2:actions_M,\
                    jointddpg.cgan_state.Z2:sample_z(len(batch_M), args.latent_size),\
                    jointddpg.cgan_reward.states2:states_M,\
                    jointddpg.cgan_reward.actions2:actions_M,\
                    jointddpg.cgan_reward.Z2:sample_z(len(batch_M), args.latent_size)}

                #Update critic
                #Dict for critic
                feed_dict_critic = {jointddpg.states:states_B,\
                    jointddpg.actions:actions_B,\
                    jointddpg.rewards:rewards_B,\
                    jointddpg.states1:states1_B,\
                    jointddpg.dones:dones_B}

                _, l_Q = sess.run([jointddpg.opt_Q,\
                    jointddpg.loss_Q],\
                    feed_dict=merge_two_dicts(feed_dict_joint,\
                            feed_dict_critic))

                #Get another Z sample
                feed_dict_joint[jointddpg.cgan_state.Z2] = sample_z(len(batch_M), args.latent_size)
                feed_dict_joint[jointddpg.cgan_reward.Z2] = sample_z(len(batch_M), args.latent_size)
                
                #Update actor
                #Get gradients from the critic
                critic_grads_B = sess.run(jointddpg.critic_grads_B,\
                    feed_dict={jointddpg.states:states_B,\
                        jointddpg.actions:actions_B})[0]
                critic_grads_M = sess.run(jointddpg.critic_grads_M,\
                    feed_dict={jointddpg.cgan_state.states2:states_M,\
                        jointddpg.cgan_state.actions2:actions_M})[0]

                #Perform update
                _ = sess.run(jointddpg.opt_actor,\
                    feed_dict={jointddpg.dQ_by_da_M:critic_grads_M,\
                        jointddpg.dQ_by_da_B:critic_grads_B,\
                        jointddpg.cgan_state.states2:states_M,\
                        jointddpg.states:states_B})

                #Update state model (discriminator)
                feed_dict_state_model={jointddpg.cgan_state.states:states_B,\
                    jointddpg.cgan_state.actions:actions_B,\
                    jointddpg.cgan_state.Z:sample_z(len(batch_M), args.latent_size),\
                    jointddpg.cgan_state.X:states1_B}
                _ = sess.run(jointddpg.opt_state_model_D, feed_dict=feed_dict_state_model)

                #Update state model (generator)
                feed_dict_state_model.pop(jointddpg.cgan_state.X)
                feed_dict_state_model[jointddpg.cgan_state.Z] = sample_z(len(batch_M), args.latent_size)

                _ = sess.run(jointddpg.opt_state_model_G,\
                    feed_dict=merge_two_dicts(feed_dict_joint,\
                        feed_dict_state_model))

                #Get another Z sample
                feed_dict_joint[jointddpg.cgan_state.Z2] = sample_z(len(batch_M), args.latent_size)
                feed_dict_joint[jointddpg.cgan_reward.Z2] = sample_z(len(batch_M), args.latent_size)

                #Update reward model (discriminator)
                feed_dict_reward_model={jointddpg.cgan_reward.states:states_B,\
                    jointddpg.cgan_reward.actions:actions_B,\
                    jointddpg.cgan_reward.Z:sample_z(len(batch_M), args.latent_size),\
                    jointddpg.cgan_reward.X:rewards_B[..., np.newaxis]}
                _ = sess.run(jointddpg.opt_reward_model_D, feed_dict=feed_dict_reward_model)

                #Update reward model (generator)
                feed_dict_reward_model.pop(jointddpg.cgan_reward.X)
                feed_dict_reward_model[jointddpg.cgan_reward.Z] = sample_z(len(batch_M), args.latent_size)

                _ = sess.run(jointddpg.opt_reward_model_G,\
                    feed_dict=merge_two_dicts(feed_dict_joint,\
                        feed_dict_reward_model))
                '''

                # Update target networks
                sess.run(update_target_critic)
                sess.run(update_target_actor)

                state = np.copy(state1)
                if done == True:
                    print 'epoch', epoch, 'total rewards', total_rewards
                    break

def init_model(input_shape, action_size, latent_size, learning_rate, action_bound, tau, model):
    if model == 'gan':
        jointddpg = joint_ddpg(input_shape, action_size, latent_size, learning_rate, action_bound)

        # Update and copy operators
        update_target_actor = update_target_graph2('actor_source', 'actor_target', tau)
        update_target_critic = update_target_graph2('critic_source', 'critic_target', tau)

        copy_target_actor = update_target_graph2('actor_source', 'actor_target', 1.)
        copy_target_critic = update_target_graph2('critic_source', 'critic_target', 1.)
    elif model == 'gated':
        from gated.joint_ddpg_gated import joint_ddpg_gated
        from utils import update_target_graph3
        jointddpg = joint_ddpg_gated(input_shape, action_size, learning_rate, action_bound)

        # Update and copy operators
        update_target_actor = update_target_graph3(jointddpg.actor_source_vars, jointddpg.actor_target_vars, tau)
        update_target_critic = update_target_graph3(jointddpg.critic_source_vars, jointddpg.critic_target_vars, tau)

        copy_target_actor = update_target_graph3(jointddpg.actor_source_vars, jointddpg.actor_target_vars, 1.)
        copy_target_critic = update_target_graph3(jointddpg.critic_source_vars, jointddpg.critic_target_vars, 1.)
    return jointddpg, update_target_actor, update_target_critic, copy_target_actor, copy_target_critic

if __name__ == '__main__':
    main()
        
