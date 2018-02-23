#python nature_replicate.py --env-interface=custom_cartpole --action-size=3 --input-shape=None,36,36,2 --state-len-max=2 --ep-greedy-speed=fast --replay-start-size=-1
#import gym
#from gym import wrappers
import numpy as np
import random
import math
import tensorflow as tf
import tensorflow.contrib.slim as slim
#import scipy.signal
#from skimage.color import rgb2gray
#from matplotlib import pyplot as plt
#import pickle
#import scipy.ndimage as ndimage

import argparse

from networks import network
from networks import network2
from networks import network3
from networks import network4
import sys
sys.path.append('..')
from utils import env_interface
from utils import Memory_with_compression as Memory
from utils import update_target_graph
from utils import str2list

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-interface", type=str, default='gym')
    parser.add_argument("--environment", type=str, default='BreakoutDeterministic-v4')
    parser.add_argument("--action-size", type=int, default=4)
    parser.add_argument("--input-shape", type=str, default='None,84,84,4')
    parser.add_argument("--state-len-max", type=int, default=4)
    parser.add_argument("--target-update-freq", type=int, default=10000)

    parser.add_argument("--ep-greedy-speed", type=str, default='slow')
    parser.add_argument("--epsilon-max", type=float, default=1.)
    parser.add_argument("--epsilon-min", type=float, default=.01)
    parser.add_argument("--epsilon-decay-slow", type=int, default=1000000)

    parser.add_argument("--epsilon-decay-fast", type=float, default=.001)

    parser.add_argument("--learning-rate", type=float, default=.95)
    parser.add_argument("--replay-start-size", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--replay-mem-size", type=int, default=1000000)
    parser.add_argument("--epochs", type=int, default=30000)

    parser.add_argument("--pixel-feature", type=int, default=1)
    parser.add_argument("--padding", type=int, default=0)


    parser.add_argument("--model", type=str, default='nature')

    args = parser.parse_args()

    args.input_shape = str2list(args.input_shape)
    assert args.model in ['nature', 'gated']
    assert args.ep_greedy_speed in ['fast', 'slow']
    assert args.env_interface in ['gym', 'ale', 'custom_cart', 'custom_cartpole', 'ple']
    if args.env_interface in ['gym', 'ale']:
        env = env_interface(args.env_interface, args.environment)
    elif args.env_interface in ['custom_cart', 'custom_cartpole', 'ple']:
        env = env_interface(args.env_interface,
                            args.environment,
                            bool(args.pixel_feature),
                            bool(args.padding))
        args.input_shape = [None] + list(env.obs_space_shape) + [1]
    args.input_shape[-1] = args.state_len_max
    args.action_size = env.action_size
    assert args.state_len_max == args.input_shape[-1]
    print args

    #Other other paramters
    state_old = []
    state = []
    steps = 0

    #Other parameters
    if args.ep_greedy_speed == 'slow':
        epsilon = args.epsilon_max
        epsilon_rate = 0.
        if args.epsilon_decay_slow != 0:
            epsilon_rate = ((args.epsilon_max - args.epsilon_min) / float(args.epsilon_decay_slow))
    elif args.ep_greedy_speed == 'fast':
        epsilon = args.epsilon_max

    #Initialize replay memory
    memory = Memory(args.replay_mem_size, args.input_shape[1:])

    #Initialize neural net
    qnet, tnet, update_ops = init_network(args.input_shape, args.action_size, args.model)

    #import time
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(update_ops)
        for epoch in range(args.epochs):
            frame = env.reset()
            total_rewards = 0.
            total_losses = 0.
            state_old = []
            state = [frame] * args.state_len_max 
            done = False

            #start = time.time()
            while done == False:
                if np.random.rand() < epsilon:
                    action = np.random.randint(args.action_size)
                else:
                    image_in = np.stack(state, axis=-1)[np.newaxis, ...]
                    action = qnet.get_action(sess, image_in)

                frame, reward, done, _ = env.step(action)
                total_rewards += reward
                state_old = state[:]
                state.append(frame)
                if len(state) > args.state_len_max:
                    state = state[1:]

                #Add to memory
                memory.add([np.stack(state_old, axis=-1)[np.newaxis, ...], action, min(1., max(-1., reward)), np.stack(state, axis=-1)[np.newaxis, ...], done])

                #Reduce epsilon
                if args.ep_greedy_speed == 'slow':
                    epsilon = max(args.epsilon_min, epsilon - epsilon_rate)
                elif args.ep_greedy_speed == 'fast':
                    epsilon = args.epsilon_min + (args.epsilon_max - args.epsilon_min) * np.exp(-args.epsilon_decay_fast * float(steps))

                if steps > args.replay_start_size:
                    #Training step
                    batch = np.array(memory.sample(args.batch_size))

                    states = np.concatenate(batch[:, 0], axis=0)
                    actions = batch[:, 1]
                    rewards = batch[:, 2]
                    states1 = np.concatenate(batch[:, 3], axis=0)
                    dones = batch[:, 4]

                    l = qnet.train(sess, states, actions, rewards, states1, dones, args.learning_rate, tnet)
                    total_losses += l

                #Increase the frame steps counter
                steps += 1
                #Check if target network is to be updated
                if steps % args.target_update_freq == 0:
                    print "Updating target..."
                    sess.run(update_ops)

                if done == True:
                    print "epoch:", epoch, "total rewards", total_rewards, "total losses", total_losses, qnet.string
                    #print 'time:', time.time() - start
                    break
    env.close()

def init_network(input_shape, action_size, model):
    if model == 'nature':
        qnet = network(input_shape, action_size, 'qnet')
        tnet = network(input_shape, action_size, 'tnet')
        update_ops = update_target_graph('qnet', 'tnet')
    elif model == 'gated':
        sys.path.append('../prototype8/gated')
        sys.path.append('../prototype8/')
        from gated_regularized_qnetwork import gated_regularized_qnetwork_visual_input
        from utils import update_target_graph_vars
        qnet = gated_regularized_qnetwork_visual_input(input_shape, action_size)
        tnet = None
        update_ops = update_target_graph_vars(qnet.qnet_vars, qnet.tnet_vars)
    return qnet, tnet, update_ops

if __name__ == '__main__':
    main()
