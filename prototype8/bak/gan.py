import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from utils import log
from utils import lrelu
from utils import sample_z

class CGAN():
    def __init__(self, input_shape=[None, 4], action_size=4, latent_size=4, gen_input_shape=[None, 4]):
        # Define Prior
        self.states = tf.placeholder(shape=input_shape, dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, action_size, dtype=tf.float32)
        self.prior = tf.concat([self.states, self.actions_onehot], axis=1)

        # Define noise inputs and true inputs
        self.Z = tf.placeholder(shape=[None, latent_size], dtype=tf.float32)
        self.X = tf.placeholder(shape=gen_input_shape, dtype=tf.float32)

        # Xavier initializer
        xavier_init = tf.contrib.layers.xavier_initializer()

        # Generator variables
        self.G_W1 = tf.Variable(tf.random_normal([latent_size + int(self.prior.shape[1]), 128], stddev=.02))
        self.G_b1 = tf.Variable(tf.zeros(shape=[128]))

        self.G_W2 = tf.Variable(tf.random_normal([128, gen_input_shape[1]], stddev=.02))
        self.G_b2 = tf.Variable(tf.zeros(shape=[gen_input_shape[1]]))

        '''
        self.G_W3 = tf.Variable(tf.random_normal([256, 128], stddev=.02))
        self.G_b3 = tf.Variable(tf.zeros(shape=[128]))

        self.G_W4 = tf.Variable(tf.random_normal([128, 64], stddev=.02))
        self.G_b4 = tf.Variable(tf.zeros(shape=[64]))

        self.G_W5 = tf.Variable(tf.random_normal([64, 32], stddev=.02))
        self.G_b5 = tf.Variable(tf.zeros(shape=[32]))

        self.G_W6 = tf.Variable(tf.random_normal([32, gen_input_shape[1]], stddev=.02))
        self.G_b6 = tf.Variable(tf.zeros(shape=[gen_input_shape[1]]))
        '''

        self.theta_G = [self.G_W1, self.G_b1, self.G_W2, self.G_b2]

        # Discriminator variables
        self.D_W1 = tf.Variable(tf.random_normal([gen_input_shape[1] + int(self.prior.shape[1]), 128], stddev=.02))
        self.D_b1 = tf.Variable(tf.zeros(shape=[128]))

        self.D_W2 = tf.Variable(tf.random_normal([128, 1], stddev=.02))
        self.D_b2 = tf.Variable(tf.zeros(shape=[1]))

        '''
        self.D_W3 = tf.Variable(tf.random_normal([256, 128], stddev=.02))
        self.D_b3 = tf.Variable(tf.zeros(shape=[128]))

        self.D_W4 = tf.Variable(tf.random_normal([128, 64], stddev=.02))
        self.D_b4 = tf.Variable(tf.zeros(shape=[64]))

        self.D_W5 = tf.Variable(tf.random_normal([64, 32], stddev=.02))
        self.D_b5 = tf.Variable(tf.zeros(shape=[32]))

        self.D_W6 = tf.Variable(tf.random_normal([32, 1], stddev=.02))
        self.D_b6 = tf.Variable(tf.zeros(shape=[1]))
        '''

        self.theta_D = [self.D_W1, self.D_b1, self.D_W2, self.D_b2]

        # Build the GAN
        self.G_sample = self.generator(self.Z)
        self.D_real = self.discriminator(self.X)
        self.D_fake = self.discriminator(self.G_sample)

        # Loss
        self.D_loss = -tf.reduce_mean(log(self.D_real) + log(1. - self.D_fake))
        self.G_loss = -tf.reduce_mean(log(self.D_fake))

        # Optimizers
        self.D_solver = tf.train.AdamOptimizer().minimize(self.D_loss, var_list=self.theta_D)
        self.G_solver = tf.train.AdamOptimizer().minimize(self.G_loss, var_list=self.theta_G)

    def generator(self, z):
        inputs = tf.concat([z, self.prior], axis=1)
        self.G_h1 = lrelu(tf.matmul(inputs, self.G_W1) + self.G_b1)
        self.G_h2 = tf.matmul(self.G_h1, self.G_W2) + self.G_b2
        '''
        self.G_h3 = lrelu(tf.matmul(self.G_h2, self.G_W3) + self.G_b3)
        self.G_h4 = lrelu(tf.matmul(self.G_h3, self.G_W4) + self.G_b4)
        self.G_h5 = lrelu(tf.matmul(self.G_h4, self.G_W5) + self.G_b5)
        self.G_h6 = tf.matmul(self.G_h5, self.G_W6) + self.G_b6
        '''
        return self.G_h2

    def discriminator(self, x):
        inputs = tf.concat([x, self.prior], axis=1)
        self.D_h1 = lrelu(tf.matmul(inputs, self.D_W1) + self.D_b1)
        self.D_h2 = tf.nn.sigmoid(tf.matmul(self.D_h1, self.D_W2) + self.D_b2)
        '''
        self.D_h3 = lrelu(tf.matmul(self.D_h2, self.D_W3) + self.D_b3)
        self.D_h4 = lrelu(tf.matmul(self.D_h3, self.D_W4) + self.D_b4)
        self.D_h5 = lrelu(tf.matmul(self.D_h4, self.D_W5) + self.D_b5)
        self.D_h6 = tf.nn.sigmoid(tf.matmul(self.D_h5, self.D_W6) + self.D_b6)
        '''
        return self.D_h2

#def main():
#    cgan = CGAN()
#    #cgan_rewards = CGAN(gen_input_shape=[None, 1])
#    env = gym.make('CartPole-v0')
#    memory = Memory(1000000)
#    batch_size = 32
#
#    init = tf.initialize_all_variables()
#    with tf.Session() as sess:
#        sess.run(init)
#        for epoch in range(100000):
#            state = env.reset()
#            done = False
#            while True:
#                action = np.random.randint(env.action_space.n)
#                state1, reward, done, _ = env.step(action)
#
#                memory.add([state[np.newaxis, ...], action, reward, state1[np.newaxis, ...], done])
#
#                batch = np.array(memory.sample(batch_size))
#                if len(batch) > 0:
#                    states = np.concatenate(batch[:, 0], axis=0).astype(np.float32)
#                    actions = batch[:, 1]
#                    rewards = batch[:, 2]
#                    states1 = np.concatenate(batch[:, 3], axis=0).astype(np.float32)
#                    dones = batch[:, 4]
#
#                    '''
#                    _, D_loss_reward = sess.run([cgan_rewards.D_solver, cgan_rewards.D_loss], feed_dict={cgan_rewards.states:states, cgan_rewards.actions:actions, cgan_rewards.Z:sample_z(len(batch)), cgan_rewards.X:rewards[..., np.newaxis]})
#                    _, G_loss_rewards = sess.run([cgan_rewards.G_solver, cgan_rewards.G_loss], feed_dict={cgan_rewards.states:states, cgan_rewards.actions:actions, cgan_rewards.Z:sample_z(len(batch))})
#                    print D_loss_reward, G_loss_rewards
#                    '''
#                    _, D_loss = sess.run([cgan.D_solver, cgan.D_loss], feed_dict={cgan.states:states, cgan.actions:actions, cgan.Z:sample_z(len(batch)), cgan.X:states1})
#                    _, G_loss = sess.run([cgan.G_solver, cgan.G_loss], feed_dict={cgan.states:states, cgan.actions:actions, cgan.Z:sample_z(len(batch))})
#                    print D_loss, G_loss
#
#
#                state = np.copy(state1)
#
#
#                # Do a test here
#                batch = np.array(memory.sample(batch_size))
#                if len(batch) > 0:
#                    states = np.concatenate(batch[:, 0], axis=0).astype(np.float32)
#                    actions = batch[:, 1]
#                    rewards = batch[:, 2]
#                    states1 = np.concatenate(batch[:, 3], axis=0).astype(np.float32)
#                    dones = batch[:, 4]
#
#                    D_real, D_fake, G_sample = sess.run([cgan.D_real, cgan.D_fake, cgan.G_sample], feed_dict={cgan.states:states, cgan.actions:actions, cgan.Z:sample_z(len(batch)), cgan.X:states1})
#                    correct = 0
#                    assert len(D_real) == len(D_fake)
#                    for i in range(len(D_real)):
#                        if D_real[i, 0] >= .5:
#                            correct += 1
#                        if D_fake[i, 0] <= .5:
#                            correct += 1
#                    print 'accuracy:', float(correct) / float(2 * len(D_real))
#                    print G_sample
#                    print "----"
#                    print states1
#                    print "((("
#
#
#                    '''
#                    D_real, D_fake, G_sample = sess.run([cgan_rewards.D_real, cgan_rewards.D_fake, cgan_rewards.G_sample], feed_dict={cgan_rewards.states:states, cgan_rewards.actions:actions, cgan_rewards.Z:sample_z(len(batch)), cgan_rewards.X:rewards[..., np.newaxis]})
#                    correct = 0
#                    assert len(D_real) == len(D_fake)
#                    for i in range(len(D_real)):
#                        if D_real[i, 0] >= .5:
#                            correct += 1
#                        if D_fake[i, 0] <= .5:
#                            correct += 1
#                    print 'accuracy:', float(correct) / float(2 * len(D_real))
#                    print G_sample
#                    print "----"
#                    print states1
#                    print "((("
#                '''
#                # Do a test here -- END --
#
#                if done == True:
#                    print 'epoch:', epoch
#                    break
#
#if __name__ == '__main__':
#    main()
