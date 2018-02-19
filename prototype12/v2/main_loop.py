import gym
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import argparse

import sys
sys.path.append('../..')
from utils import env_interface
from utils import Memory_with_compression2 as Memory
from utils import update_target_graph_vars
from utils import str2list

def main():
    #Arguments for the q-learner
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-interface", type=str, default='gym')
    parser.add_argument("--environment", type=str, default='BreakoutDeterministic-v4')
    parser.add_argument("--action-size", type=int, default=4)
    parser.add_argument("--input-shape", type=str, default='None,84,84,4')
    parser.add_argument("--state-len-max", type=int, default=4)
    parser.add_argument("--target-update-freq", type=int, default=10000)
    parser.add_argument("--epsilon-max", type=float, default=1.)
    parser.add_argument("--epsilon-min", type=float, default=.01)
    parser.add_argument("--epsilon-decay", type=int, default=1000000)
    parser.add_argument("--learning-rate", type=float, default=.95)
    parser.add_argument("--replay-start-size", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--replay-mem-size", type=int, default=1000000)
    parser.add_argument("--epochs", type=int, default=30000)

    #Arguments for the feature extractor
    #parser.add_argument("--train-fe-shape", type=str, default='None,12,12,4')
    #parser.add_argument("--stop-gradient", type=int, default=0)
    parser.add_argument("--train-fe-iterations", type=int, default=2500)
    parser.add_argument("--train-fe-batch-size", type=int, default=4)
    #parser.add_argument("--train-fe-lamb", type=float, default=0.)
    parser.add_argument("--numfactors", type=int, default=128)
    parser.add_argument("--nummap", type=int, default=128)
    parser.add_argument("--train-fe-learning-rate", type=float, default=.001)
    parser.add_argument("--w", type=int, default=8)
    parser.add_argument("--s", type=int, default=4)

    #parser.add_argument("--mode", type=str, default='cross_correlation')

    #parser.add_argument("--use-conv-after-fe", type=int, default=0)
    #parser.add_argument("--no-inputs", type=int, default=2)
    #parser.add_argument("--action-placement", type=str, default='in')
    #parser.add_argument("--tiled", type=int, default=0)
    #parser.add_argument("--use-close-to-ones-init", type=int, default=0)

    parser.add_argument("--ep-greedy-speed", type=str, default='slow')
    #Arguments for the environment interface
    parser.add_argument("--pixel-features", type=int, default=1)
    parser.add_argument("--padding", type=int, default=0)
    args = parser.parse_args()

    #Parse arguments wrt other arguments
    args.input_shape = str2list(args.input_shape)
    #args.train_fe_shape = str2list(args.train_fe_shape)
    assert args.env_interface in ['gym', 'ale', 'custom_cart', 'custom_cartpole', 'ple']
    assert args.ep_greedy_speed in ['fast', 'slow']
    #assert args.mode in ['cross_correlation', 'transformation']
    #assert args.action_placement in ['in', 'out']
    env = env_interface(args.env_interface,
                        args.environment,
                        pixel_feature=bool(args.pixel_features),
                        padding=bool(args.padding),
                        render=True)
    args.action_size = env.action_size
    if args.env_interface in ['custom_cart', 'custom_cartpole', 'ple']:
        args.input_shape = [None] + list(env.obs_space_shape) + [args.state_len_max]
    #args.train_fe_shape[-1] = args.state_len_max
    #if args.mode == 'transformation':
        #assert args.state_len_max == args.no_inputs
        #args.input_shape[-1] = 1
        #args.train_fe_shape[-1] = 1

    print args

    #Other other parameters
    steps = 0

    #Other parameters
    epsilon_lambda = .001
    epsilon = args.epsilon_max
    epsilon_rate = 0.
    if args.epsilon_decay != 0:
        epsilon_rate = ((args.epsilon_max - args.epsilon_min) / float(args.epsilon_decay))

    #Initialize replay memory
    print args.input_shape
    memory = Memory(args.replay_mem_size, args.input_shape[1:-1] + [args.state_len_max+2])

    #Initialize neural net
    from qlearner import qlearner
    qnet = qlearner(shape=args.input_shape,
                    nummap=args.nummap,
                    numfactors=args.numfactors,
                    learning_rate_recon=args.train_fe_learning_rate,
                    #no_inputs=args.no_inputs,
                    #frame_shape=args.input_shape,
                    a_size=args.action_size,
                    #stop_gradient=bool(args.stop_gradient),
                    #lamb=args.train_fe_lamb,
                    w=args.w,
                    s=args.s)
                    #use_conv_after_fe=bool(args.use_conv_after_fe),
                    #mode=args.mode,
                    #action_placement=args.action_placement,
                    #tiled=bool(args.tiled),
                    #use_close_to_ones_init=bool(args.use_close_to_ones_init))
    qnet_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    tnet = qlearner(shape=args.input_shape,
                    nummap=args.nummap,
                    numfactors=args.numfactors,
                    learning_rate_recon=args.train_fe_learning_rate,
                    #no_inputs=args.no_inputs,
                    #frame_shape=args.input_shape,
                    a_size=args.action_size,
                    #stop_gradient=bool(args.stop_gradient),
                    #lamb=args.train_fe_lamb,
                    w=args.w,
                    s=args.s)
                    #use_conv_after_fe=bool(args.use_conv_after_fe),
                    #mode=args.mode,
                    #action_placement=args.action_placement,
                    #tiled=bool(args.tiled),
                    #use_close_to_ones_init=bool(args.use_close_to_ones_init))
    tnet_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[len(qnet_vars):]

    update_ops = update_target_graph_vars(qnet_vars, tnet_vars)

    train_fe_iterations = args.train_fe_iterations
    #import time
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(update_ops)
        for epoch in range(args.epochs):
            frame = env.reset()
            total_rewards = 0.
            total_losses = 0.
            #state_old = [frame] * args.state_len_max 
            #state = [frame] * args.state_len_max 
            state_hist = [frame] * (args.state_len_max + 2)
            action_hist = [-1] * (args.state_len_max + 1)
            done = False

            #start = time.time()
            while done == False:
                if np.random.rand() < epsilon:
                    action = np.random.randint(args.action_size)
                else:
                    image_in_old = np.stack(state_hist[1:1+args.state_len_max], axis=-1)[np.newaxis, ...]
                    image_in = np.stack(state_hist[2:], axis=-1)[np.newaxis, ...]
                    action = qnet.get_action(sess, image_in_old, image_in, np.array(action_hist[1:])[np.newaxis, ...])

                frame, reward, done, _ = env.step(action)
                total_rewards += reward

                state_hist.append(frame)
                action_hist.append(action)
                state_hist.pop(0)
                action_hist.pop(0)
                assert len(state_hist) == args.state_len_max + 2
                assert len(action_hist) == args.state_len_max + 1

                #Add to memory
                memory.add([np.stack(state_hist, axis=-1)[np.newaxis, ...],
                           np.array(action_hist)[np.newaxis, ...],
                           min(1., max(-1., reward)),
                           done])

                #Reduce epsilon
                if args.ep_greedy_speed == 'slow':
                    epsilon = max(args.epsilon_min, epsilon - epsilon_rate)
                elif args.ep_greedy_speed == 'fast':
                    epsilon = args.epsilon_min + (args.epsilon_max - args.epsilon_min) * np.exp(-epsilon_lambda * float(steps))

                #Train the reconstruction loss
                if args.train_fe_iterations > 0:
                    cum_recon_loss, cum_recon_a_loss, iterations = qnet.train_feature_extractor(sess, memory, args.train_fe_batch_size, 1)
                    args.train_fe_iterations -= iterations
                    print 'cum_recon_loss', cum_recon_loss, 'cum_recon_a_loss', cum_recon_a_loss, 'iterations left', train_fe_iterations - args.train_fe_iterations, 'of', train_fe_iterations

                if steps > args.replay_start_size and args.train_fe_iterations <= 0:
                    #Training step
                    batch = np.array(memory.sample(args.batch_size))

                    states = np.concatenate(batch[:, 0], axis=0)
                    actions = np.concatenate(batch[:, 1], axis=0)
                    rewards = batch[:, 2]
                    dones = batch[:, 3]

                    Q1 = qnet.get_Q1(sess, states[:,:,:,1:1+args.state_len_max], states[:,:,:,2:], actions[:,1:], tnet)

                    targetQ = rewards + (1. - dones) * args.learning_rate * np.amax(Q1, keepdims=False, axis=1)

                    _, l = qnet.train(sess,
                                      states[:,:,:,:args.state_len_max],
                                      states[:,:,:,1:1+args.state_len_max],
                                      actions[:,:args.state_len_max],
                                      actions[:,-1],
                                      targetQ[..., np.newaxis])

                    total_losses += l

                #Increase the frame steps counter
                steps += 1
                #Check if target network is to be updated
                if steps % args.target_update_freq == 0:
                    print "Updating target..."
                    sess.run(update_ops)

                if done == True:
                    print "epoch", epoch, "total rewards", total_rewards, "total losses", total_losses
                    #print 'time:', time.time() - start
                    break

    env.close()

if __name__ == '__main__':
    main()

