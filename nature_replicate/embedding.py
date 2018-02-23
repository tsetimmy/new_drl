# Add an 'embedding' option
# Add an option for reconstruction for the difference
import gym
from gym import wrappers
import numpy as np
import random
import math
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.signal
from skimage.color import rgb2gray
#from matplotlib import pyplot as plt
import pickle
import scipy.ndimage as ndimage
import sys
import blosc
import argparse

def generate_arguments(write=False):
    string ='#!/bin/bash\n#SBATCH --gres=gpu:1              # request GPU "generic resource"\n#SBATCH --cpus-per-task=6    #Maximum of CPU cores per GPU request: 6 on Cedar, 16 on Graham.\n#SBATCH --mem=60G               # memory per node\n#SBATCH --time=03-00:00            # time (DD-HH:MM)\n#SBATCH --output=%N-%j.out        # %N for node name, %j for jobID\n\nmodule load cuda cudnn python/2.7.13\nsource ~/tensorflow/bin/activate\n'

    optimizers = ['adam', 'sgd']
    embedding = [1]
    reconstruction = [1, 0]
    recon_difference = [1, 0]

    counter = 0

    for op in optimizers:
        for em in embedding:
            for recon in reconstruction:
                for recon_diff in recon_difference:
                    string2 = 'python embedding.py --optimizer=' + op + ' --embedding=' + str(em) + ' --reconstruction=' + str(recon) + ' --recon-difference=' + str(recon_diff)
                    print string + string2
                    if write == True:
                        with open("run" + str(counter) + ".sh", "w") as text_file:
                            text_file.write(string + string2)
                    counter += 1

    embedding = [0]
    for op in optimizers:
        for em in embedding:
            string2 = 'python embedding.py --optimizer=' + op + ' --embedding=' + str(em)
            if write == True:
                with open("run" + str(counter) + ".sh", "w") as text_file:
                    text_file.write(string + string2)
            counter += 1

class Memory_with_compression:
    def __init__(self, size, state_len_max):
       self.max_size = size
       self.mem = []

       self.state_len_max = state_len_max

    def add(self, element):
        ele = []
        ele.append(blosc.compress(np.reshape(element[0], 1 * 84 * 84 * self.state_len_max).tobytes(), typesize=1)) #Current state
        ele.append(element[1]) #Action
        ele.append(element[2]) #Reward
        ele.append(blosc.compress(np.reshape(element[3], 1 * 84 * 84 * self.state_len_max).tobytes(), typesize=1)) #Next state
        ele.append(element[4]) #Done
        self.mem.append(ele)

        if len(self.mem) > self.max_size:
            self.mem.pop(0)

    def sample(self, size):
        size = min(size, len(self.mem))
        elements = random.sample(self.mem, size)

        elements_decompressed = []
        for i in range(size):
            element_decompressed = []
            element_decompressed.append(np.reshape(np.fromstring(blosc.decompress(elements[i][0]), dtype=np.uint8), (1, 84, 84, self.state_len_max)))
            element_decompressed.append(elements[i][1])
            element_decompressed.append(elements[i][2])
            element_decompressed.append(np.reshape(np.fromstring(blosc.decompress(elements[i][3]), dtype=np.uint8), (1, 84, 84, self.state_len_max)))
            element_decompressed.append(elements[i][4])
            elements_decompressed.append(element_decompressed)
        return elements_decompressed

    def __del__(self):
        del self.mem

def process_frame2(frame):
    s = np.dot(frame, np.array([.299, .587, .114])).astype(np.uint8)
    s = ndimage.zoom(s, (0.4, 0.525))
    #s.resize((84, 84, 1))
    return s

class network():
    def __init__(self, input_shape, args, scope=None):
        with tf.variable_scope(scope):
            #batch_size = 32
            self.image_in = tf.placeholder(shape=input_shape, dtype=tf.uint8)
            batch_size = tf.shape(self.image_in)[0]
            self.image_in_normalized = tf.to_float(self.image_in) / 255.0
            self.conv1 = slim.conv2d(activation_fn=tf.nn.relu, inputs=self.image_in_normalized, num_outputs=32, kernel_size=[8, 8], stride=[4, 4], padding='VALID')
            self.conv2 = slim.conv2d(activation_fn=tf.nn.relu, inputs=self.conv1, num_outputs=64, kernel_size=[4, 4], stride=[2, 2], padding='VALID')
            self.conv3 = slim.conv2d(activation_fn=tf.nn.relu, inputs=self.conv2, num_outputs=64, kernel_size=[3, 3], stride=[1, 1], padding='VALID')
            self.fc1 = slim.fully_connected(slim.flatten(self.conv3), 512, activation_fn=tf.nn.relu)

            if args.embedding == 1:
                embedding_size = 16
                self.embedding = slim.fully_connected(self.fc1, embedding_size * args.action_size, activation_fn=tf.nn.relu)
                self.embedding = tf.reshape(self.embedding, shape=(-1, args.action_size, embedding_size))
                self.embedding = tf.expand_dims(self.embedding, axis=2)

                initializer = tf.contrib.layers.xavier_initializer()
                self.W = tf.Variable(initializer([1, args.action_size, embedding_size, 1]))
                self.W = tf.tile(self.W, [batch_size, 1, 1, 1])

                self.biases = tf.Variable(initializer([args.action_size]))
                self.Q = tf.matmul(self.embedding, self.W)
                self.Q = tf.reshape(self.Q, shape=(-1, args.action_size)) + self.biases
            else:
                self.Q = slim.fully_connected(self.fc1, args.action_size, activation_fn=None)

            #Define loss function for Q
            self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
            self.actions_onehot = tf.one_hot(self.actions, args.action_size, dtype=tf.float32)
            self.responsible_output = tf.reduce_sum(tf.multiply(self.Q, self.actions_onehot), axis=1, keep_dims=True)
            self.targetQ = tf.placeholder(shape=[None, 1], dtype=tf.float32)
            #self.loss = tf.reduce_sum(tf.square(self.targetQ - self.responsible_output))

            difference = tf.abs(self.responsible_output - self.targetQ)
            quadratic_part = tf.clip_by_value(difference, 0.0, 1.0)
            linear_part = difference - quadratic_part
            errors = (0.5 * tf.square(quadratic_part)) + linear_part

            if args.loss_optimizer == 'reduce_mean':
                self.Q_loss = tf.reduce_mean(errors)
            elif args.loss_optimizer == 'reduce_sum':
                self.Q_loss = tf.reduce_sum(errors)

            #Define loss function for reconstruction
            if args.reconstruction == 1:
                self.actions_onehot = tf.expand_dims(self.actions_onehot, axis=-1)
                self.actions_onehot_tiled = tf.tile(self.actions_onehot, [1, 1, embedding_size])
                self.embedding = tf.reshape(self.embedding, shape=(-1, args.action_size, embedding_size))
                self.responsible_embedding = tf.reduce_sum(tf.multiply(self.actions_onehot_tiled, self.embedding), axis=1)

                self.fc1_recon = slim.fully_connected(self.responsible_embedding, 512, activation_fn=tf.nn.relu)
                self.fc2_recon = slim.fully_connected(self.fc1_recon, 1024, activation_fn=tf.nn.relu)
                self.recon = slim.fully_connected(self.fc2_recon, 7056, activation_fn=tf.nn.sigmoid)
                self.recon = tf.reshape(self.recon, shape=[-1, 84, 84, 1])
                self.image_target = tf.placeholder(shape=[None, 84, 84, 1], dtype=tf.float32) #Not normalized here b/c it's normalized in the data
                #self.image_target = tf.placeholder(shape=[None, 84, 84, 1], dtype=tf.uint8) #Not normalized here b/c it's normalized in the data
                #self.image_target_normalized = tf.to_float(self.image_target) / 255.0
                self.recon_loss = tf.reduce_mean(tf.square(self.recon - self.image_target))

            #Total loss
            if args.reconstruction == 1:
                self.loss = self.Q_loss + .0005 * self.recon_loss
            else:
                self.loss = self.Q_loss

            #Optimizer
            if args.optimizer == 'adam':
                self.update_model = tf.train.AdamOptimizer().minimize(self.loss)
            elif args.optimizer == 'sgd':
                self.update_model = tf.train.RMSPropOptimizer(learning_rate=.00025, decay=.95, epsilon=.01).minimize(self.loss)

def update_target_graph(from_scope, to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default='BreakoutDeterministic-v4')
    parser.add_argument("--action-size", type=int, default=4)
    parser.add_argument("--input-shape", type=list, default=[None, 84, 84, 4])
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

    parser.add_argument("--optimizer", type=str, default='adam')
    parser.add_argument("--embedding", type=int, default=1)
    parser.add_argument("--reconstruction", type=int, default=1)
    parser.add_argument("--recon-difference", type=int, default=0)

    parser.add_argument("--loss-optimizer", type=str, default='reduce_mean')

    args = parser.parse_args()
    if args.optimizer not in ['adam', 'sgd']:
        raise Exception('Optimizer can only be \'adam\' or \'sgd\'')
    if args.loss_optimizer not in ['reduce_sum', 'reduce_mean']:
        raise Exception('Optimizer can only be \'reduce_sum\' or \'reduce_mean\'')
    if args.embedding == 0:
        args.reconstruction = None
        args.recon_difference = None

    #Initialize the environment
    env = gym.make(args.environment)
    args.action_size = env.action_space.n
    print args

    #Other other parameters
    state_old = []
    state = []
    steps = 0

    #Other parameters
    epsilon = args.epsilon_max
    epsilon_rate = 0.
    if args.epsilon_decay != 0:
        epsilon_rate = ((args.epsilon_max - args.epsilon_min) / float(args.epsilon_decay))

    #Initialize neural net
    qnet = network(args.input_shape, args, 'qnet')
    tnet = network(args.input_shape, args, 'tnet')
    update_ops = update_target_graph('qnet', 'tnet')
    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)
        sess.run(update_ops)
        memory = Memory_with_compression(args.replay_mem_size, args.state_len_max) #Initializing it here to be memory friendly
        for j in range(args.epochs):
            frame = process_frame2(env.reset())
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
                    Q = sess.run(qnet.Q, feed_dict={qnet.image_in:image_in})
                    action = np.argmax(Q)

                frame, reward, done, _ = env.step(action)
                frame = process_frame2(frame)
                total_rewards += reward
                state_old = state[:]
                state.append(frame)
                if len(state) > args.state_len_max:
                    state = state[1:]

                #Add to memory
                memory.add([np.stack(state_old, axis=-1)[np.newaxis, ...], action, min(1., max(-1., reward)), np.stack(state, axis=-1)[np.newaxis, ...], done])

                #Reduce epsilon
                epsilon = max(args.epsilon_min, epsilon - epsilon_rate)

                if steps > args.replay_start_size:
                    batch = np.array(memory.sample(args.batch_size))

                    states = np.concatenate(batch[:, 0], axis=0)
                    actions = batch[:, 1]
                    rewards = batch[:, 2]
                    states1 = np.concatenate(batch[:, 3], axis=0)
                    dones = batch[:, 4]

                    #Training step
                    Q1 = sess.run(tnet.Q, feed_dict={tnet.image_in:states1})
                    targetQ = rewards + args.learning_rate * np.amax(Q1, keepdims=False, axis=1)

                    assert len(dones) == targetQ.shape[0]
                    for i in range(len(dones)):
                        if dones[i] == True:
                            targetQ[i] = rewards[i]

                    if args.embedding == 1 and args.reconstruction == 1:
                        recon_input = np.copy(states1[:, :, :, -1][..., np.newaxis].astype(np.float32)) / 255.0


                        if args.recon_difference == 1:
                            recon_input = np.copy(recon_input - states1[:, :, :, -2][..., np.newaxis].astype(np.float32) / 255.0)
                        assert recon_input.shape == (len(batch), 84, 84, 1)
                        _, l = sess.run([qnet.update_model, qnet.loss], feed_dict={qnet.image_in:states, qnet.actions:actions, qnet.targetQ:targetQ[..., np.newaxis], qnet.image_target:recon_input})
                    else:
                        _, l = sess.run([qnet.update_model, qnet.loss], feed_dict={qnet.image_in:states, qnet.actions:actions, qnet.targetQ:targetQ[..., np.newaxis]})

                    total_losses += l

                #Increase the frame steps counter
                steps += 1
                #Check if target network is to be updated
                if steps % args.target_update_freq == 0:
                    print "Updating target..."
                    sess.run(update_ops)

                if done == True:
                    print "j:", j, "total rewards", total_rewards, "total losses", total_losses
                    break
    env.close()

if __name__ == '__main__':
    main()
