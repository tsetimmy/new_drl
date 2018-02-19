import tensorflow as tf
import numpy as np

class environment_modeler_gated:
    def __init__(self, s_shape, a_size, out_shape, a_type, numfactors):

        #Declare the input placeholders
        assert a_type in ['discrete', 'continuous']
        self.states = tf.placeholder(shape=s_shape, dtype=tf.float32)
        self.states_ = tf.placeholder(shape=out_shape, dtype=tf.float32)
        if a_type == 'discrete':
            self.actions_placeholder = tf.placeholder(shape=[None], dtype=tf.int32)
            self.actions = tf.one_hot(self.actions_placeholder, a_size, dtype=tf.float32)
        elif a_type == 'continuous':
            self.actions_placeholder = tf.placeholder(shape=[None, a_size], dtype=tf.float32)
            self.actions = self.actions_placeholder

        #Parameters
        self.s_shape = s_shape
        self.out_shape = out_shape
        self.a_shape = self.actions.shape.as_list()
        self.a_type = a_type
        self.numfactors = numfactors

        #Asserts
        assert len(self.s_shape) == 2
        assert len(self.out_shape) == 2
        assert len(self.a_shape) == 2

        #Declare the variables
        self.declare_variables()

        #Build the computational graph
        recon_s, recon_s_, recon_a = self.build_computational_graph(self.states, self.states_, self.actions)

        #Define the reconstruction loss
        '''
        self.loss_s = tf.reduce_mean(tf.reduce_sum(tf.square(recon_s - self.states), axis=-1))
        self.loss_s_ = tf.reduce_mean(tf.reduce_sum(tf.square(recon_s_ - self.states_), axis=-1))
        if a_type == 'discrete':
            loss_a = tf.multiply(-tf.log(tf.nn.softmax(recon_a, dim=-1) + 1e-10), self.actions)
        elif a_type == 'continuous':
            loss_a = tf.square(recon_a - self.actions)
        self.loss_a = tf.reduce_mean(tf.reduce_sum(loss_a, axis=-1))

        self.loss = self.loss_s + self.loss_s_ + self.loss_a
        '''

        self.loss_s, self.loss_s_, self.loss_a = self.get_recon_losses(recon_s, recon_s_, recon_a, self.states, self.states_, self.actions)
        self.loss = self.loss_s + self.loss_s_ + self.loss_a

        #Optimizers
        self.update_model = tf.train.AdamOptimizer(learning_rate=.001).minimize(self.loss)

    def declare_variables(self):
        #Declare initializer
        self.xavier_init = tf.contrib.layers.xavier_initializer()

        #Variables for states
        self.wfs = tf.Variable(self.xavier_init([self.s_shape[-1], self.numfactors]))
        self.bfs = tf.Variable(tf.zeros([self.s_shape[-1]]))

        #Variables for actions
        self.wfa = tf.Variable(self.xavier_init([self.a_shape[-1], self.numfactors]))
        self.bfa = tf.Variable(tf.zeros([self.a_shape[-1]]))

        #Variables for states_
        self.wfs_ = tf.Variable(self.xavier_init([self.out_shape[-1], self.numfactors]))
        self.bfs_ = tf.Variable(tf.zeros([self.out_shape[-1]]))

    def build_recon_s_(self, states, actions):
        #Compute the factors for the inputs (states and actions)
        fs = tf.matmul(states, self.wfs)
        fa = tf.matmul(actions, self.wfa)

        #Reconstruct states_
        recon_s_ = tf.matmul(tf.multiply(fs, fa), tf.transpose(self.wfs_)) + self.bfs_

        return recon_s_

    def build_computational_graph(self, states, states_, actions):
        #Compute the factors for the inputs (states and actions)
        fs = tf.matmul(states, self.wfs)
        fa = tf.matmul(actions, self.wfa)

        #Reconstruct states_
        recon_s_ = tf.matmul(tf.multiply(fs, fa), tf.transpose(self.wfs_)) + self.bfs_

        #Compute the factors for states_
        fs_ = tf.matmul(recon_s_, self.wfs_)

        #Reconstruct states
        recon_s = tf.matmul(tf.multiply(fs_, fa), tf.transpose(self.wfs)) + self.bfs

        #Reconstruct actions
        recon_a = tf.matmul(tf.multiply(fs, fs_), tf.transpose(self.wfa)) + self.bfa

        return recon_s, recon_s_, recon_a

    def get_recon_losses(self, recon_s, recon_s_, recon_a, states, states_, actions):
        loss_s = tf.reduce_mean(tf.reduce_sum(tf.square(recon_s - states), axis=-1))
        loss_s_ = tf.reduce_mean(tf.reduce_sum(tf.square(recon_s_ - states_), axis=-1))
        if self.a_type == 'discrete':
            loss_a_tmp = tf.multiply(-tf.log(tf.nn.softmax(recon_a, dim=-1) + 1e-10), actions)
        elif self.a_type == 'continuous':
            loss_a_tmp = tf.square(recon_a - actions)
        loss_a = tf.reduce_mean(tf.reduce_sum(loss_a_tmp, axis=-1))

        return loss_s, loss_s_, loss_a







def random_action(a_size, a_type):
    if a_type == 'discrete':
        return np.random.randint(a_size)
    elif a_type == 'continuous':
        return np.random.uniform(size=a_size)

def main():
    import gym
    import sys
    import copy
    sys.path.append('../..')
    from utils import Memory

    #env = gym.make('LunarLander-v2')
    env = gym.make('MountainCar-v0')
    #env = gym.make('CartPole-v0')
    mem = Memory(1000000)
    batch_size = 32
    try:
        a_size = env.action_space.n
        a_type = 'discrete'
    except:
        try:
            a_size = env.action_space.shape[0]
            a_type = 'continuous'
        except:
            raise ValueError('Cannot find action size.')
    emg = environment_modeler_gated(s_shape=[None, env.observation_space.shape[0]], a_size=a_size, out_shape=[None, env.observation_space.shape[0]], a_type=a_type, numfactors=256)
    #emg = environment_modeler_gated(s_shape=[None, env.observation_space.shape[0]], a_size=a_size, out_shape=[None, 1], a_type=a_type, numfactors=256)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        while True:
            s = env.reset()

            done = False
            while done == False:
                env.render()
                #a = np.random.randint(a_size)
                a = random_action(a_size, a_type)
                s_, r, done, _ = env.step(a)

                mem.add([s, a, r, s_, done])
                batch = mem.sample(batch_size)
                if len(batch) == batch_size:
                    states = []
                    actions = []
                    rewards = []
                    states_ = []
                    for i in range(batch_size):
                        states.append(batch[i][0])
                        actions.append(batch[i][1])
                        rewards.append(batch[i][2])
                        states_.append(batch[i][3])

                    states = np.stack(states, axis=0)
                    actions = np.stack(actions, axis=0)
                    rewards = np.stack(rewards, axis=0)
                    states_ = np.stack(states_, axis=0)


                    #_, loss_s, loss_a, loss_s_, loss = sess.run([emg.update_model, emg.loss_s, emg.loss_a, emg.loss_s_, emg.loss], feed_dict={emg.states:states, emg.states_:rewards[..., np.newaxis], emg.actions_placeholder:actions})
                    _, loss_s, loss_a, loss_s_, loss = sess.run([emg.update_model, emg.loss_s, emg.loss_a, emg.loss_s_, emg.loss], feed_dict={emg.states:states, emg.states_:states_, emg.actions_placeholder:actions})
                    print 'loss_s', loss_s, 'loss_a', loss_a, 'loss_s_', loss_s_, 'loss', loss

                s = copy.deepcopy(s_)
                if done == True:
                    break


if __name__ == '__main__':
    main()
