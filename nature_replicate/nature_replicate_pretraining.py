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

class Memory:
    def __init__(self, size):
       self.max_size = size
       self.mem = []

    def add(self, element):
        self.mem.append(element)

        if len(self.mem) > self.max_size:
            self.mem.pop(0)

    def sample(self, size):
        size = min(size, len(self.mem))
        return random.sample(self.mem, size)
    
    def __del__(self):
        del self.mem

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

def process_frame(frame):
    s = rgb2gray(frame)#Grayscale
    s = scipy.misc.imresize(s, [110, 84])#Downsample
    s = s[13:-13, :]#Crop
    #s = s.astype(np.float32) / 255.0
    return s

def process_frame2(frame):
    s = np.dot(frame, np.array([.299, .587, .114])).astype(np.uint8)
    s = ndimage.zoom(s, (0.4, 0.525))
    #s.resize((84, 84, 1))
    return s

class network():
    def __init__(self, width=84, height=84, channels=4, a_size=3, scope=None):
        with tf.variable_scope(scope):
            self.image_in = tf.placeholder(shape=[None, width, height, channels], dtype=tf.uint8)
            self.image_in_normalized = tf.to_float(self.image_in) / 255.0
            self.conv1 = slim.conv2d(activation_fn=tf.nn.relu, inputs=self.image_in_normalized, num_outputs=32, kernel_size=[8, 8], stride=[4, 4], padding='VALID')
            self.conv2 = slim.conv2d(activation_fn=tf.nn.relu, inputs=self.conv1, num_outputs=64, kernel_size=[4, 4], stride=[2, 2], padding='VALID')
            self.conv3 = slim.conv2d(activation_fn=tf.nn.relu, inputs=self.conv2, num_outputs=64, kernel_size=[3, 3], stride=[1, 1], padding='VALID')
            self.fc1 = slim.fully_connected(slim.flatten(self.conv3), 512, activation_fn=tf.nn.relu)
            self.Q = slim.fully_connected(self.fc1, a_size, activation_fn=None)

            #Define loss function
            self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
            self.actions_onehot = tf.one_hot(self.actions, a_size, dtype=tf.float32)
            self.responsible_output = tf.reduce_sum(tf.multiply(self.Q, self.actions_onehot), axis=1, keep_dims=True)
            self.targetQ = tf.placeholder(shape=[None, 1], dtype=tf.float32)
            #self.loss = tf.reduce_sum(tf.square(self.targetQ - self.responsible_output))

            difference = tf.abs(self.responsible_output - self.targetQ)
            quadratic_part = tf.clip_by_value(difference, 0.0, 1.0)
            linear_part = difference - quadratic_part
            errors = (0.5 * tf.square(quadratic_part)) + linear_part
            self.loss = tf.reduce_sum(errors)

            #self.update_model = tf.train.AdamOptimizer(learning_rate=2.5e-4).minimize(self.loss)
            self.update_model = tf.train.RMSPropOptimizer(learning_rate=.00025, decay=.95, epsilon=.01).minimize(self.loss)

            #---#
            self.S = slim.fully_connected(self.fc1, 84 * 84 * a_size, activation_fn=tf.nn.relu)
            self.S_reshape = tf.reshape(self.S, [-1, 84, 84, a_size])
            self.targetS = tf.placeholder(shape=[None, 84, 84, a_size], dtype=tf.float32)
            #self.targetS_normalized = tf.to_float(self.targetS) / 255.0
            self.loss2 = tf.reduce_sum(tf.square(self.targetS - self.S_reshape))
            self.update_model2 = tf.train.AdamOptimizer(learning_rate=2.5e-4).minimize(self.loss2)

def update_target_graph(from_scope, to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

def generate_dynamics(environment, no_samples, state_len_max=4, a_size=4):
    env = gym.make(environment)
    #X = []
    #A = []
    #Y = []
    dd = Memory(no_samples)
    samples = 0
    while no_samples > samples:
        frame = process_frame2(env.reset())
        state_old = []
        state = [frame] * state_len_max
        done = False

        while done == False:
            print "samples", samples
            action = np.random.randint(a_size)
            frame, _, done, _ = env.step(action)
            #A.append(action)
            frame = process_frame2(frame)
            state_old = state[:]
            state.append(frame)
            if len(state) > state_len_max:
                state = state[1:]

            dd.add([np.stack(state_old, axis=-1), action, np.copy(state[-1][:]), np.copy(state[-2][:])])
            #X.append(np.stack(state_old, axis=-1))
            #Y.append(np.stack(np.copy(state[-1][:] - state[-2][:]), axis=-1))
            samples += 1
            if samples >= no_samples:
              break

    env.close()
    #return np.stack(X, axis=0), np.array(A), np.stack(Y, axis=0)
    return dd

def evaluate(sess, environment, qnet, a_size, state_len_max, episodes_to_eval=10):
    env = gym.make(environment)
    total_rewards = 0.
    for _ in range(episodes_to_eval):
        frame = process_frame2(env.reset())
        state_old = []
        state = [frame] * state_len_max
        done = False

        while done == False:
            if np.random.rand() < .05:
                action = np.random.randint(a_size)
            else:
                image_in = np.stack(state, axis=-1)[np.newaxis, ...]
                Q = sess.run(qnet.Q, feed_dict={qnet.image_in:image_in})
                action = np.argmax(Q)

            frame, reward, done, _ = env.step(action)
            frame = process_frame2(frame)
            total_rewards += float(reward)
            state_old = state[:]
            state.append(frame)
            if len(state) > state_len_max:
                state = state[1:]
    env.close()
    return total_rewards / float(episodes_to_eval)

def main():
    #Initialize the environment
    environment = 'Breakout-v0'
    env = gym.make(environment)

    #Other other parameters
    width = 84
    height = 84
    channels = 4
    a_size = 4
    state_len_max = 4
    state_old = []
    state = []
    target_update_freq = 10000
    eval_freq = 250000#250000
    eval_steps = 125000
    steps = 0

    #Other parameters
    #lamb = .001
    epsilon_max = 1.
    epsilon_min = .01
    epsilon = epsilon_max
    epsilon_decay = 1000000
    epsilon_rate = 0.
    if epsilon_decay != 0:
        epsilon_rate = ((epsilon_max - epsilon_min) / float(epsilon_decay))
    learning_rate = .95
    #time_step = 0.

    #Initialize replay memory
    replay_start_size = 50000#50000
    batch_size = 32
    mem_size = 1000000
    #memory = Memory(mem_size)
    #memory.initial_populate(environment, replay_start_size, a_size, hist_len_max)
    nd = int(sys.argv[1])
    nj = 30000

    #Generate dynamics dataset
    mode = 'generate'
    dd_len = int(sys.argv[2])
    dd_batch_size = int(sys.argv[2])
    if nd > 0 and 'generate' in mode:
        dynamics_data = generate_dynamics(environment, dd_len, state_len_max, a_size)
        if mode == 'generate_and_save':
            with open('dynamics_data' + str(dd_len) + '.p', 'wb') as handle:
                pickle.dump(dynamics_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    elif mode == 'load':
        with open('dynamics_data' + str(dd_len) + '.p', 'rb') as handle:
            dynamics_data = pickle.load(handle)

    #Initialize neural net
    qnet = network(width, height, channels, a_size, 'qnet')
    tnet = network(width, height, channels, a_size, 'tnet')
    update_ops = update_target_graph('qnet', 'tnet')
    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)
        sess.run(update_ops)
        losses = []
        for d in range(nd):
            dd_batch = np.array(dynamics_data.sample(dd_batch_size))
            print len(dd_batch)
            X = np.stack(dd_batch[:, 0], axis=0)
            A = dd_batch[:, 1]

            Y1 = np.stack(dd_batch[:, 2], axis=0)
            Y2 = np.stack(dd_batch[:, 3], axis=0)

            S = sess.run(qnet.S_reshape, feed_dict={qnet.image_in:X})
            for i in range(len(S)):
                S[i, :, :, A[i]] = (Y1[i, :, :].astype(np.float32) - Y2[i, :, :].astype(np.float32)) / 255.
            _, l2 = sess.run([qnet.update_model2, qnet.loss2], feed_dict={qnet.image_in:X, qnet.targetS:S})
            losses.append(l2)
            print d, "of", nd, "dynamics loss:", l2
        dynamics_data = None
        memory = Memory_with_compression(mem_size, state_len_max) #Initializing it here to be memory friendly
        for j in range(nj):
            frame = process_frame2(env.reset())
            total_rewards = 0.
            total_losses = 0.
            state_old = []
            state = [frame] * state_len_max
            done = False

            while done == False:
                if np.random.rand() < epsilon:
                    action = np.random.randint(a_size)
                else:
                    image_in = np.stack(state, axis=-1)[np.newaxis, ...]
                    Q = sess.run(qnet.Q, feed_dict={qnet.image_in:image_in})
                    action = np.argmax(Q)

                frame, reward, done, _ = env.step(action)
                frame = process_frame2(frame)
                total_rewards += reward
                state_old = state[:]
                state.append(frame)
                if len(state) > state_len_max:
                    state = state[1:]

                #Add to memory
                memory.add([np.stack(state_old, axis=-1)[np.newaxis, ...], action, min(1., max(-1., reward)), np.stack(state, axis=-1)[np.newaxis, ...], done])

                #Reduce epsilon
                epsilon = max(epsilon_min, epsilon - epsilon_rate)

                if steps > replay_start_size:
                    #Training step
                    batch = np.array(memory.sample(batch_size))

                    states = np.concatenate(batch[:, 0], axis=0)
                    actions = batch[:, 1]
                    rewards = batch[:, 2]
                    states1 = np.concatenate(batch[:, 3], axis=0)
                    dones = batch[:, 4]

                    Q1 = sess.run(tnet.Q, feed_dict={tnet.image_in:states1})
                    targetQ = rewards + learning_rate * np.amax(Q1, keepdims=False, axis=1)

                    assert len(dones) == targetQ.shape[0]
                    for i in range(len(dones)):
                        if dones[i] == True:
                            targetQ[i] = rewards[i]

                    _, l = sess.run([qnet.update_model, qnet.loss], feed_dict={qnet.image_in:states, qnet.actions:actions, qnet.targetQ:targetQ[..., np.newaxis]})
                    total_losses += l

                #Increase the frame steps counter
                steps += 1
                #Check if target network is to be updated
                if steps % target_update_freq == 0:
                    print "Updating target..."
                    sess.run(update_ops)

                if steps % eval_freq == 0:
                  print "evaluating..."
                  eval_score = evaluate(sess, environment, qnet, a_size, state_len_max)
                  print "eval_score:", eval_score

                if done == True:
                    print "j:", j, "total rewards", total_rewards, "total losses", total_losses
                    break
    env.close()

if __name__ == '__main__':
    main()
