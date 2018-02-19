#cd new_drl/prototype12/ && source ~/tensorflow/bin/activate && python qlearning_mainloop_multi_inputs.py --env-interface=custom_cart --state-len-max=2 --input-shape=None,36,36,2  --replay-start-size=-1 --train-fe-shape=None,8,8,2 --train-fe-iterations=0 --train-fe-numfactors=32 --train-fe-nummap=32 --use-conv-after-fe=1 --ep-greedy-speed=fast --train-fe-w=8 --train-fe-s=1 --mode=cross_correlation --no-inputs=2 --action-placement=in |tee ~/go0.txt
#python qlearning_mainloop_multi_inputs.py --env-interface=custom_cart --state-len-max=2 --input-shape=None,36,36,2 --replay-start-size=-1 --train-fe-shape=None,8,8,1 --train-fe-iterations=0 --train-fe-numfactors=32 --train-fe-nummap=32 --train-fe-w=8 --train-fe-s=1 --mode=transformation --use-conv-after-fe=1 --no-inputs=2 --action-placement=in --ep-greedy-speed=fast
#Result's section
#1) Effect of pretraining on the performance of the algorithm
#2) Comparison of the algorithm across a few other baselines? (DQN, A3C)
#3) Show the attention mechanism of the network at work (How?). Show the filters of the network after training?
#4) Show the transformation filters?
#->Refactor cartpole swingup to allow for pixel access
import gym
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import argparse

import sys
sys.path.append('../..')
from utils import env_interface
from utils import Memory_with_compression as Memory
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
    parser.add_argument("--train-fe-shape", type=str, default='None,12,12,4')
    parser.add_argument("--stop-gradient", type=int, default=0)
    parser.add_argument("--train-fe-iterations", type=int, default=1000)
    parser.add_argument("--train-fe-batch-size", type=int, default=100)
    parser.add_argument("--train-fe-lamb", type=float, default=0.)
    parser.add_argument("--train-fe-numfactors", type=int, default=200)
    parser.add_argument("--train-fe-nummap", type=int, default=100)
    parser.add_argument("--train-fe-learning-rate", type=float, default=.001)
    parser.add_argument("--train-fe-w", type=int, default=12)
    parser.add_argument("--train-fe-s", type=int, default=1)

    parser.add_argument("--mode", type=str, default='cross_correlation')

    parser.add_argument("--use-conv-after-fe", type=int, default=0)
    parser.add_argument("--no-inputs", type=int, default=2)
    parser.add_argument("--action-placement", type=str, default='in')
    parser.add_argument("--tiled", type=int, default=0)
    parser.add_argument("--use-close-to-ones-init", type=int, default=0)

    parser.add_argument("--ep-greedy-speed", type=str, default='slow')
    #Arguments for the environment interface
    parser.add_argument("--pixel-features", type=int, default=1)
    parser.add_argument("--padding", type=int, default=0)
    args = parser.parse_args()

    #Parse arguments wrt other arguments
    args.input_shape = str2list(args.input_shape)
    args.train_fe_shape = str2list(args.train_fe_shape)
    assert args.env_interface in ['gym', 'ale', 'custom_cart', 'custom_cartpole']
    assert args.ep_greedy_speed in ['fast', 'slow']
    assert args.mode in ['cross_correlation', 'transformation']
    assert args.action_placement in ['in', 'out']
    env = env_interface(args.env_interface,
                        args.environment,
                        pixel_feature=bool(args.pixel_features),
                        padding=bool(args.padding),
                        render=True)
    args.action_size = env.action_size
    if args.env_interface in ['custom_cart', 'custom_cartpole']:
        args.input_shape = [None] + list(env.obs_space_shape) + [args.state_len_max]
    args.train_fe_shape[-1] = args.state_len_max
    if args.mode == 'transformation':
        assert args.state_len_max == args.no_inputs
        args.input_shape[-1] = 1
        args.train_fe_shape[-1] = 1

    print args

    #Other other parameters
    state_old = []
    state = []
    steps = 0

    #Other parameters
    epsilon_lambda = .001
    epsilon = args.epsilon_max
    epsilon_rate = 0.
    if args.epsilon_decay != 0:
        epsilon_rate = ((args.epsilon_max - args.epsilon_min) / float(args.epsilon_decay))

    #Initialize replay memory
    print args.input_shape
    memory = Memory(args.replay_mem_size, args.input_shape[1:-1] + [args.state_len_max])

    #Initialize neural net
    from gated_qlearning_multi_inputs import gated_qlearning_multi_inputs
    qnet = gated_qlearning_multi_inputs(shape=args.train_fe_shape,
                                        nummap=args.train_fe_nummap,
                                        numfactors=args.train_fe_numfactors,
                                        learning_rate=args.train_fe_learning_rate,
                                        no_inputs=args.no_inputs,
                                        frame_shape=args.input_shape,
                                        a_size=args.action_size,
                                        stop_gradient=bool(args.stop_gradient),
                                        lamb=args.train_fe_lamb,
                                        w=args.train_fe_w,
                                        s=args.train_fe_s,
                                        use_conv_after_fe=bool(args.use_conv_after_fe),
                                        mode=args.mode,
                                        action_placement=args.action_placement,
                                        tiled=bool(args.tiled),
                                        use_close_to_ones_init=bool(args.use_close_to_ones_init))
    qnet_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    tnet = gated_qlearning_multi_inputs(shape=args.train_fe_shape,
                                        nummap=args.train_fe_nummap,
                                        numfactors=args.train_fe_numfactors,
                                        learning_rate=args.train_fe_learning_rate,
                                        no_inputs=args.no_inputs,
                                        frame_shape=args.input_shape,
                                        a_size=args.action_size,
                                        stop_gradient=bool(args.stop_gradient),
                                        lamb=args.train_fe_lamb,
                                        w=args.train_fe_w,
                                        s=args.train_fe_s,
                                        use_conv_after_fe=bool(args.use_conv_after_fe),
                                        mode=args.mode,
                                        action_placement=args.action_placement,
                                        tiled=bool(args.tiled),
                                        use_close_to_ones_init=bool(args.use_close_to_ones_init))
    tnet_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[len(qnet_vars):]

    update_ops = update_target_graph_vars(qnet_vars, tnet_vars)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        sess.run(update_ops)
        for epoch in range(args.epochs):
            frame = env.reset()
            total_rewards = 0.
            total_losses = 0.
            state_old = []
            state = [frame] * args.state_len_max 
            done = False

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
                memory.add([np.stack(state_old, axis=-1)[np.newaxis, ...],\
                           action,\
                           min(1., max(-1., reward)),\
                           np.stack(state, axis=-1)[np.newaxis, ...],\
                           done])

                #Reduce epsilon
                if args.ep_greedy_speed == 'slow':
                    epsilon = max(args.epsilon_min, epsilon - epsilon_rate)
                elif args.ep_greedy_speed == 'fast':
                    epsilon = args.epsilon_min + (args.epsilon_max - args.epsilon_min) * np.exp(-epsilon_lambda * float(steps))

                #Train the reconstruction loss
                if args.train_fe_iterations > 0:
                    args.train_fe_iterations -= qnet.train_feature_extractor(sess,
                                                                             memory,
                                                                             args.train_fe_batch_size,
                                                                             10,
                                                                             args.train_fe_iterations)
                    print args.train_fe_iterations

                if steps > args.replay_start_size and args.train_fe_iterations <= 0:
                    #Training step
                    batch = np.array(memory.sample(args.batch_size))

                    states = np.concatenate(batch[:, 0], axis=0)
                    actions = batch[:, 1]
                    rewards = batch[:, 2]
                    states1 = np.concatenate(batch[:, 3], axis=0)
                    dones = batch[:, 4]

                    Q1 = qnet.get_Q1(sess, states1, tnet)

                    targetQ = rewards + (1. - dones) * args.learning_rate * np.amax(Q1, keepdims=False, axis=1)

                    l, _, _ = qnet.train(sess, states, actions, targetQ[..., np.newaxis])
                    total_losses += l

                #Increase the frame steps counter
                steps += 1
                #Check if target network is to be updated
                if steps % args.target_update_freq == 0:
                    print "Updating target..."
                    sess.run(update_ops)

                if done == True:
                    print "epoch", epoch, "total rewards", total_rewards, "total losses", total_losses
                    break

    env.close()

if __name__ == '__main__':
    main()
