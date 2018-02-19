import numpy as np

import gym
import tensorflow as tf
import tensorflow.contrib.slim as slim
import argparse

import sys
sys.path.append('..')
sys.path.append('../..')
from gan import CGAN

from ddpg import actor
from ddpg import critic

from utils import update_target_graph2
from utils import Memory
from utils import OrnsteinUhlenbeckActionNoise
from utils import sample_z

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default='Pendulum-v0')
    parser.add_argument("--action-dim", type=int, default=1)
    parser.add_argument("--state-dim", type=int, default=1)
    parser.add_argument("--input-shape", type=list, default=[None, 1])
    parser.add_argument("--epochs", type=int, default=30000)
    parser.add_argument('--tau', help='soft target update parameter', default=0.001)
    parser.add_argument("--action-bound", type=float, default=1.)
    parser.add_argument("--replay-mem-size", type=int, default=1000000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=.99)

    parser.add_argument("--K", type=int, default=1, help='The number of steps to train the environment')
    parser.add_argument("--L", type=int, default=1, help='The number of Q-learning steps for hypothetical rollouts')
    parser.add_argument("--latent-size", type=int, default=4, help='Size of vector for Z')

    args = parser.parse_args()

    # Initialize environment
    env = gym.make(args.environment)
    args.state_dim = env.observation_space.shape[0]
    args.input_shape = [None, args.state_dim]
    args.action_dim = env.action_space.shape[0]
    #assert args.action_dim == 1
    args.action_bound = env.action_space.high
    print(args)

    # Networks
    actor_source = actor(state_shape=[None, args.state_dim],\
        action_shape=[None, args.action_dim],\
        output_bound=args.action_bound[0],\
        scope='actor_source')
    critic_source = critic(state_shape=[None, args.state_dim],\
        action_shape=[None, args.action_dim],\
        scope='critic_source')
    actor_target = actor(state_shape=[None, args.state_dim],\
        action_shape=[None, args.action_dim],\
        output_bound=args.action_bound[0],\
        scope='actor_target')
    critic_target = critic(state_shape=[None, args.state_dim],\
        action_shape=[None, args.action_dim],\
        scope='critic_target')

    # Initialize the GANs
    cgan_state = CGAN(input_shape=args.input_shape,\
        action_size=args.action_dim,\
        latent_size=args.latent_size,\
        gen_input_shape=args.input_shape,\
        continuous_action=True)
    cgan_reward = CGAN(input_shape=args.input_shape,\
        action_size=args.action_dim,\
        latent_size=args.latent_size,\
        gen_input_shape=[None, 1],\
        continuous_action=True)

    # Update and copy operators
    update_target_actor = update_target_graph2('actor_source', 'actor_target', args.tau)
    update_target_critic = update_target_graph2('critic_source', 'critic_target', args.tau)

    copy_target_actor = update_target_graph2('actor_source', 'actor_target', 1.)
    copy_target_critic = update_target_graph2('critic_source', 'critic_target', 1.)

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
                action = sess.run(actor_source.action,
                    feed_dict={actor_source.states:state[np.newaxis, ...]})[0] + actor_noise()
                # Execute action
                state1, reward, done, _ = env.step(action)
                total_rewards += float(reward)
                # Store tuple in replay memory
                memory.add([state[np.newaxis, ...],\
                    action[np.newaxis, ...],\
                    reward,\
                    state1[np.newaxis, ...],\
                    done])

                # Training step: update actor critic using real experience
                batch = np.array(memory.sample(args.batch_size))
                assert len(batch) > 0
                states = np.concatenate(batch[:, 0], axis=0)
                actions = np.concatenate(batch[:, 1], axis=0)
                rewards = batch[:, 2]
                states1 = np.concatenate(batch[:, 3], axis=0)
                dones = batch[:, 4]

                # Update the critic
                actions1 = sess.run(actor_target.action,\
                    feed_dict={actor_target.states:states1})
                targetQ = np.squeeze(sess.run(critic_target.Q,\
                    feed_dict={critic_target.states:states1,\
                        critic_target.actions:actions1}), axis=-1)
                targetQ = rewards + (1. - dones.astype(np.float32)) * args.gamma * targetQ
                targetQ = targetQ[..., np.newaxis]
                _, critic_loss = sess.run([critic_source.critic_solver,\
                    critic_source.loss],\
                    feed_dict={critic_source.states:states,\
                        critic_source.actions:actions,\
                        critic_source.targetQ:targetQ})

                # Update the actor
                critic_grads = sess.run(critic_source.grads,\
                    feed_dict={critic_source.states:states,\
                        critic_source.actions:actions})[0]# Grab gradients from critic
                _ = sess.run(actor_source.opt,\
                    feed_dict={actor_source.states:states,\
                        actor_source.dQ_by_da:critic_grads})

                # Update target networks
                sess.run(update_target_critic)
                sess.run(update_target_actor)

                # Training step: update the environment model using real experience (i.e., update the conditional GANs)
                for k in range(args.K):
                    batch = np.array(memory.sample(args.batch_size))

                    states = np.concatenate(batch[:, 0], axis=0)
                    actions = np.concatenate(batch[:, 1], axis=0)
                    rewards = batch[:, 2]
                    states1 = np.concatenate(batch[:, 3], axis=0)

                    _, D_loss_state = sess.run([cgan_state.D_solver, cgan_state.D_loss],\
                        feed_dict={cgan_state.states:states,\
                            cgan_state.actions:actions,\
                            cgan_state.Z:sample_z(len(batch),\
                            args.latent_size),\
                            cgan_state.X:states1})

                    _, G_loss_state = sess.run([cgan_state.G_solver,\
                        cgan_state.G_loss],\
                        feed_dict={cgan_state.states:states,\
                            cgan_state.actions:actions,\
                            cgan_state.Z:sample_z(len(batch),\
                            args.latent_size)})

                    _, D_loss_reward = sess.run([cgan_reward.D_solver,\
                        cgan_reward.D_loss],\
                        feed_dict={cgan_reward.states:states,\
                            cgan_reward.actions:actions,\
                            cgan_reward.Z:sample_z(len(batch),\
                            args.latent_size),\
                            cgan_reward.X:rewards[..., np.newaxis]})

                    _, G_loss_reward = sess.run([cgan_reward.G_solver,\
                        cgan_reward.G_loss],\
                        feed_dict={cgan_reward.states:states,\
                            cgan_reward.actions:actions,\
                            cgan_reward.Z:sample_z(len(batch),\
                            args.latent_size)})
                    #print D_loss_state, G_loss_state, D_loss_reward, G_loss_state

                # Training step: update actor critic using imagination rollouts
                for l in range(args.L):
                    batch = np.array(memory.sample(args.batch_size))
                    states_ = np.concatenate(batch[:, 3], axis=0)
                    actions = np.random.uniform(env.action_space.low[0],\
                        env.action_space.high[0],\
                        size=(len(batch),\
                        env.action_space.shape[0]))
                    dones = np.array([False] * len(batch))

                    G_sample_state = sess.run(cgan_state.G_sample,\
                        feed_dict={cgan_state.states:states_,\
                            cgan_state.actions:actions,\
                            cgan_state.Z:sample_z(len(batch),\
                            args.latent_size)})
                    G_sample_reward = sess.run(cgan_reward.G_sample,\
                        feed_dict={cgan_reward.states:states_,\
                            cgan_reward.actions:actions,\
                            cgan_reward.Z:sample_z(len(batch),\
                            args.latent_size)})
                    G_sample_reward = np.squeeze(G_sample_reward, axis=-1)

                    # Update the critic
                    actions1 = sess.run(actor_target.action,\
                        feed_dict={actor_target.states:G_sample_state})
                    targetQ = np.squeeze(sess.run(critic_target.Q,\
                        feed_dict={critic_target.states:G_sample_state,\
                            critic_target.actions:actions1}), axis=-1)
                    targetQ = G_sample_reward + (1. - dones.astype(np.float32)) * args.gamma * targetQ
                    targetQ = targetQ[..., np.newaxis]
                    _, critic_loss = sess.run([critic_source.critic_solver,\
                        critic_source.loss],\
                        feed_dict={critic_source.states:states_,\
                            critic_source.actions:actions,\
                            critic_source.targetQ:targetQ})

                    # Update the actor
                    critic_grads = sess.run(critic_source.grads,\
                        feed_dict={critic_source.states:states_,\
                            critic_source.actions:actions})[0]# Grab gradients from critic
                    _ = sess.run(actor_source.opt,\
                        feed_dict={actor_source.states:states_,\
                            actor_source.dQ_by_da:critic_grads})

                    # Update target networks
                    sess.run(update_target_critic)
                    sess.run(update_target_actor)

                state = np.copy(state1)
                if done == True:
                    print 'epoch', epoch, 'total rewards', total_rewards
                    break

if __name__ == '__main__':
    main()
