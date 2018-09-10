import numpy as np
import scipy.ndimage as ndimage
import tensorflow as tf
import blosc
import random
#from prototype11.atari_environment_wrapper import atari_environment
from custom_environments.cart import Cart
from custom_environments.cartpole import CartPole
from custom_environments.pygames import ple_wrapper
import gym
#from matplotlib import pyplot as plt
#import pylab
import uuid

def process_frame2(frame):
    s = np.dot(frame, np.array([.299, .587, .114])).astype(np.uint8)
    s = ndimage.zoom(s, (0.4, 0.525))
    #s.resize((84, 84, 1))
    return s

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

class Memory_with_compression:
    def __init__(self, size, shape=[84, 84, 4]):
       self.shape = [1] + shape
       self.max_size = size
       self.mem = []

    def add(self, element):
        ele = []
        ele.append(blosc.compress(np.reshape(element[0], np.prod(np.array(self.shape))).tobytes(), typesize=1)) #Current state
        ele.append(element[1]) #Action
        ele.append(element[2]) #Reward
        ele.append(blosc.compress(np.reshape(element[3], np.prod(np.array(self.shape))).tobytes(), typesize=1)) #Next state
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
            element_decompressed.append(np.reshape(np.fromstring(blosc.decompress(elements[i][0]), dtype=np.uint8), tuple(self.shape)))
            element_decompressed.append(elements[i][1])
            element_decompressed.append(elements[i][2])
            element_decompressed.append(np.reshape(np.fromstring(blosc.decompress(elements[i][3]), dtype=np.uint8), tuple(self.shape)))
            element_decompressed.append(elements[i][4])
            elements_decompressed.append(element_decompressed)
        return elements_decompressed

    def __del__(self):
        del self.mem

class Memory_with_compression2:
    def __init__(self, size, shape=[84, 84, 4]):
       self.shape = [1] + shape
       self.max_size = size
       self.mem = []

    def add(self, element):
        ele = []
        ele.append(blosc.compress(np.reshape(element[0], np.prod(np.array(self.shape))).tobytes(), typesize=1)) #States
        ele.append(element[1]) #Action
        ele.append(element[2]) #Reward
        ele.append(element[3]) #Done
        self.mem.append(ele)

        if len(self.mem) > self.max_size:
            self.mem.pop(0)

    def sample(self, size):
        size = min(size, len(self.mem))
        elements = random.sample(self.mem, size)

        elements_decompressed = []
        for i in range(size):
            element_decompressed = []
            element_decompressed.append(np.reshape(np.fromstring(blosc.decompress(elements[i][0]), dtype=np.uint8), tuple(self.shape)))
            element_decompressed.append(elements[i][1])
            element_decompressed.append(elements[i][2])
            element_decompressed.append(elements[i][3])
            elements_decompressed.append(element_decompressed)
        return elements_decompressed

    def __del__(self):
        del self.mem

def update_target_graph(from_scope, to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

def update_target_graph_vars(from_vars, to_vars):
    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

#A soft version of update target graph
def update_target_graph2(from_scope, to_scope, tau=.001):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(tf.multiply(from_var, tau) + tf.multiply(to_var, 1. - tau)))
    return op_holder

def update_target_graph3(from_vars, to_vars, tau=.001):
    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(tf.multiply(from_var, tau) + tf.multiply(to_var, 1. - tau)))
    return op_holder

def split(array, w, s):
    assert len(array.shape) == 4
    channels = array.shape[-1]
    sliced = []
    rbegin = 0; cbegin = 0;
    while rbegin + w <= array.shape[1]:
        cbegin = 0
        while cbegin + w <= array.shape[2]:
            sliced.append(array[:, rbegin:rbegin+w, cbegin:cbegin+w, :])
            cbegin += s
        rbegin += s

    sliced = np.concatenate(sliced, axis=0)
    sliced = np.reshape(sliced, (-1, w * w * channels))
    sliced = sliced.astype(np.float64) / 255.
    return sliced

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def shuffle(a):
    p = np.random.permutation(len(a))
    return a[p]

class env_interface:
    def __init__(self, interface, rom=None, pixel_feature=None, padding=False, render=True):
        assert interface in ['gym', 'gym!atari', 'ale', 'custom_cart', 'custom_cartpole', 'ple']
        if interface in ['gym', 'ale']:
            assert rom is not None
        self.interface = interface
        self.rom = rom

        if interface in ['custom_cart', 'custom_cartpole']:
            assert pixel_feature in [True, False]
            self.pixel_feature = pixel_feature
            self.padding = padding
            self.render = render

        if self.interface == 'gym':
            self.env = gym.make(self.rom)
            self.action_size = self.env.action_space.n
            self.obs_space_shape = (210, 160, 3)
        if self.interface == 'gym!atari':
            self.env = gym.make(self.rom)
            self.action_size = self.env.action_space.n
            self.obs_space_shape = self.env.observation_space.shape
        elif self.interface == 'ale':
            self.env = atari_environment(self.rom, display_screen=False)
            self.action_size = self.env.num_actions
            self.obs_space_shape = (210, 160, 3)
        elif self.interface == 'custom_cart':
            self.env = Cart(pixelFeature=self.pixel_feature, render=self.render)
            if self.env.pixelFeature:
                self.obs_space_shape = self.env.screenSize
            elif self.env.pixelFeature == False and self.padding == True:
                self.obs_space_shape = (2, 2)
            else:
                self.obs_space_shape = (2,)
            self.action_size = self.env.numActions
        elif self.interface == 'custom_cartpole':
            self.env = CartPole(pixelFeature=self.pixel_feature, render=self.render)
            if self.env.pixelFeature:
                self.obs_space_shape = self.env.screenSize
            elif self.env.pixelFeature == False and self.padding == True:
                self.obs_space_shape = (2, 2)
            else:
                self.obs_space_shape = (4,)
            self.action_size = self.env.numActions
        elif self.interface == 'ple':
            self.env = ple_wrapper(rom)
            self.obs_space_shape = tuple(self.env.screen_dims)
            self.action_size = self.env.action_size

    def reset(self):
        if self.interface == 'gym':
            frame = process_frame2(self.env.reset())
            return frame
        elif self.interface == 'gym!atari':
            frame = self.env.reset()
            return frame
        elif self.interface == 'ale':
            frame = self.env.reset()
            return frame
        elif self.interface == 'custom_cart':
            self.env = Cart(pixelFeature=self.pixel_feature, render=self.render)
            frame = self.env.getCurrentState()
            return self.pad(frame)
        elif self.interface == 'custom_cartpole':
            self.env = CartPole(pixelFeature=self.pixel_feature, render=self.render)
            frame = self.env.getCurrentState()
            return self.pad(frame)
        elif self.interface == 'ple':
            return self.env.reset()

    def step(self, action):
        if self.interface == 'gym':
            frame, reward, done, info = self.env.step(action)
            frame = process_frame2(frame)
            return frame, reward, done, info 
        if self.interface == 'gym!atari':
            frame, reward, done, info = self.env.step(action)
            return frame, reward, done, info
        elif self.interface == 'ale':
            frame, reward, done = self.env.step(action)
            return frame, float(reward), done, None
        elif self.interface == 'custom_cart':
            frame, reward, done = self.env.act(action - 1)
            return self.pad(frame) , reward, done, None
        elif self.interface == 'custom_cartpole':
            frame, reward, done = self.env.act(action - 1)
            return self.pad(frame) , reward, done, None
        elif self.interface == 'ple':
            frame, reward, done = self.env.step(action)
            return frame, reward, done, None
    
    def pad(self, frame):
        if self.padding == False:
            return frame
        assert self.pixel_feature == False
        if self.interface == 'custom_cart':
            ret = np.concatenate([frame[..., np.newaxis], np.zeros((2, 1))], axis=-1)
            return ret
        elif self.interface == 'custom_cartpole':
            return frame.reshape((2, 2,))

    def __del__(self):
        if self.interface == 'gym':
            self.env.close()

def parse_states(states, mode):
    assert mode in ['gbm', 'cc', 'gae']

    if mode == 'gbm':
        assert states.shape[-1] == 2
        return states[:, :, :, 0][..., np.newaxis], states[:, :, :, 1][..., np.newaxis]
    elif mode == 'cc' or mode == 'gae':
        assert len(states.shape) == 4
        return states, states

def parse_split_shuffle_states(states, mode, w, s):
    assert mode in ['gbm', 'cc', 'gae']

    if mode == 'gbm':
        x = states[:, :, :, 0][..., np.newaxis]
        y = states[:, :, :, 1][..., np.newaxis]
        x = split(x, w, s)
        y = split(y, w, s)
        x, y = unison_shuffled_copies(x, y)
        return x, y
    elif mode == 'cc' or mode == 'gae':
        x = split(states, w, s)
        x = shuffle(x)
        return x, x

# Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py, which is
# based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

def log(x):
    return tf.log(tf.maximum(x, 1e-6))

def lrelu(x, alpha=.2):
  return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

def sample_z(batch_size, latent_size):
    #return np.random.uniform(-1., 1., [batch_size, latent_size])
    return np.random.normal(0., 1., [batch_size, latent_size])

def dispims(M, height, width, border=0, bordercolor=0.0, **kwargs):
    """ Display the columns of matrix M in a montage. """
    numimages = M.shape[1]
    n0 = np.int(np.ceil(np.sqrt(numimages)))
    n1 = np.int(np.ceil(np.sqrt(numimages)))
    im = bordercolor*\
         np.ones(((height+border)*n1+border,(width+border)*n0+border),dtype='<f8')
    for i in range(n0):
        for j in range(n1):
            if i*n1+j < M.shape[1]:
                im[j*(height+border)+border:(j+1)*(height+border)+border,\
                   i*(width+border)+border:(i+1)*(width+border)+border] = \
                np.vstack((\
                  np.hstack((np.reshape(M[:,i*n1+j],(width,height)).T,\
                         bordercolor*np.ones((height,border),dtype=float))),\
                  bordercolor*np.ones((border,width+border),dtype=float)\
                  ))
    pylab.imshow(im.T,cmap=pylab.cm.gray,interpolation='nearest', **kwargs)
    pylab.show()

def get_random_string():
    return str(uuid.uuid4().get_hex().upper()[0:6])

def str2list(string):
    l = string.split(',')
    assert len(l) == 4

    import ast
    l[0] = ast.literal_eval(l[0])
    for i in range(1, len(l)):
        l[i] = int(l[i])

    return l

def gather_data(env, epochs, unpack=False):
    data = []
    for epoch in range(epochs):
        state = env.reset()
        while True:
            action = np.random.uniform(low=env.action_space.low, high=env.action_space.high)
            next_state, reward, done, _ = env.step(action)
            data.append([state, action, reward, next_state, done])
            state = np.copy(next_state)
            if done:
                break
    if unpack == False:
        return data
    else:
        states, actions, rewards, next_states = [np.stack(ele, axis=0) for ele in zip(*data)[:-1]]
        return states, actions, rewards[..., np.newaxis], next_states

def gather_data2(env, no_samples=1000):
    assert env.spec.id in ['Pendulum-v0', 'MountainCarContinuous-v0']
    if env.spec.id == 'Pendulum-v0':
        from prototype8.dmlac.real_env_pendulum import real_env_pendulum_state
        state_func = real_env_pendulum_state()
        high = np.array([np.pi, 8.])
        rand = np.random.uniform(low=-high, high=high, size=[no_samples, len(high)])
        pos = rand[:, 0]
        vel = rand[:, 1]
        states = np.stack([np.cos(pos), np.sin(pos), vel], axis=-1)
    elif env.spec.id == 'MountainCarContinuous-v0':
        # Warning: This function is incorrect; probably should not be used.
        from custom_environments.environment_state_functions import mountain_car_continuous_state_function
        state_func = mountain_car_continuous_state_function()
        states = np.random.uniform(low=env.observation_space.low, high=env.observation_space.high, size=[no_samples, len(env.observation_space.low)])

    actions = np.random.uniform(low=env.action_space.low, high=env.action_space.high, size=[no_samples, len(env.action_space.low)])

    next_states = state_func.step_np(states, actions)

    return states, actions, next_states

def get_mcc_policy(env, hit_wall=True, reach_goal=True, train=True):
    from custom_environments.environment_state_functions import mountain_car_continuous_state_function
    from custom_environments.environment_reward_functions import mountain_car_continuous_reward_function

    state_function = mountain_car_continuous_state_function()
    reward_function = mountain_car_continuous_reward_function()

    seed_state = np.concatenate([np.random.uniform(low=-.6, high=-.4, size=1), np.zeros(1)])[np.newaxis, ...]
    i = 0
    while True:
        print 'Finding... iteration:', i
        i += 1
        states = []
        rewards = []
        next_states = []
        state = np.copy(seed_state)
        policy = np.random.uniform(env.action_space.low, env.action_space.high, env._max_episode_steps)
        has_hit_wall = 0
        has_reach_goal = 0
        found_length = 0
        length = 0

        for a in policy:
            states.append(np.copy(state))
            action = np.atleast_2d(a)
            reward = reward_function.step_np(state, action)
            rewards.append(reward)
            next_state = state_function.step_np(state, action)
            next_states.append(np.copy(next_state))
            state = np.copy(next_state)
            length += 1

            if next_state[0, 0] == -1.2 and next_state[0, 1] == 0.: has_hit_wall = length
            if reward[0] > 50.: has_reach_goal = length
            if hit_wall == bool(has_hit_wall) and reach_goal == bool(has_reach_goal) and found_length == 0:
                if hit_wall and reach_goal: assert has_reach_goal > hit_wall
                found_length = length

        if found_length > 0: break

    if not(train == True and reach_goal == True): found_length = None
    print 'Found! Length:', found_length

    states = np.concatenate(states, axis=0)[:found_length, ...]
    actions = np.copy(policy[..., np.newaxis])[:found_length, ...]
    rewards = np.concatenate(rewards, axis=0)[:found_length, ...]
    next_states = np.concatenate(next_states, axis=0)[:found_length, ...]

    return states, actions, rewards, next_states
