import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

class network():
    def __init__(self, input_shape, a_size=3, scope=None, padding='VALID'):
        with tf.variable_scope(scope, default_name='default_scope'):
            self.image_in = tf.placeholder(shape=input_shape, dtype=tf.uint8)
            self.image_in_normalized = tf.to_float(self.image_in) / 255.0
            self.conv1 = slim.conv2d(activation_fn=tf.nn.relu, inputs=self.image_in_normalized, num_outputs=32, kernel_size=[8, 8], stride=[4, 4], padding=padding)
            self.conv2 = slim.conv2d(activation_fn=tf.nn.relu, inputs=self.conv1, num_outputs=64, kernel_size=[4, 4], stride=[2, 2], padding=padding)
            self.conv3 = slim.conv2d(activation_fn=tf.nn.relu, inputs=self.conv2, num_outputs=64, kernel_size=[3, 3], stride=[1, 1], padding=padding)
            self.fc1 = slim.fully_connected(slim.flatten(self.conv3), 512, activation_fn=tf.nn.relu)
            self.Q = slim.fully_connected(self.fc1, a_size, activation_fn=None)

            #Define loss function
            self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
            self.actions_onehot = tf.one_hot(self.actions, a_size, dtype=tf.float32)
            self.targetQ = tf.placeholder(shape=[None, 1], dtype=tf.float32)
            #self.loss = tf.reduce_sum(tf.square(self.targetQ - self.responsible_output))

            #self.responsible_output, self.loss, self.update_model = self.get_qloss(self.Q, self.actions_onehot)

            self.responsible_output = tf.reduce_sum(tf.multiply(self.Q, self.actions_onehot), axis=1, keep_dims=True)
            difference = tf.abs(self.responsible_output - self.targetQ)
            quadratic_part = tf.clip_by_value(difference, 0.0, 1.0)
            linear_part = difference - quadratic_part
            errors = (0.5 * tf.square(quadratic_part)) + linear_part
            self.loss = tf.reduce_sum(errors)

            #self.update_model = tf.train.AdamOptimizer(learning_rate=2.5e-4).minimize(self.loss)
            self.update_model = tf.train.RMSPropOptimizer(learning_rate=.00025, decay=.95, epsilon=.01).minimize(self.loss)

            self.string = 'network'

    def get_action(self, sess, image_in):
        Q = sess.run(self.Q, feed_dict={self.image_in:image_in})
        action = np.argmax(Q)
        return action

    def train(self, sess, states, actions, targetQ):
        _, l = sess.run([self.update_model, self.loss], feed_dict={self.image_in:states, self.actions:actions, self.targetQ:targetQ})
        return l

    def get_Q1(self, sess, states1, tnet):
        return sess.run(tnet.Q, feed_dict={tnet.image_in:states1})

class network2:
    def __init__(self, input_shape, a_size=3, scope=None, padding='VALID'):
        with tf.variable_scope(scope, default_name='default_scope'):
            self.image_in = tf.placeholder(shape=input_shape, dtype=tf.uint8)
            self.image_in_normalized = tf.to_float(self.image_in) / 255.0
            self.conv1 = slim.conv2d(inputs=self.image_in_normalized, num_outputs=64, kernel_size=[8, 8], stride=[4, 4], padding=padding, biases_initializer=None, activation_fn=None)
            self.conv2 = slim.conv2d(inputs=self.image_in_normalized, num_outputs=64, kernel_size=[8, 8], stride=[4, 4], padding=padding, biases_initializer=None, activation_fn=None)
            mult = tf.multiply(self.conv1, self.conv2)
            print self.conv1.shape
            self.hidden = slim.conv2d(inputs=mult, num_outputs=32, kernel_size=[1, 1], stride=[1, 1], padding=padding, activation_fn=tf.nn.sigmoid)
            print self.hidden.shape
            self.Q = slim.fully_connected(slim.flatten(self.hidden), a_size, activation_fn=None)

            #Define loss function
            self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
            self.actions_onehot = tf.one_hot(self.actions, a_size, dtype=tf.float32)
            self.targetQ = tf.placeholder(shape=[None, 1], dtype=tf.float32)
            #self.loss = tf.reduce_sum(tf.square(self.targetQ - self.responsible_output))

            #self.responsible_output, self.loss, self.update_model = self.get_qloss(self.Q, self.actions_onehot)

            self.responsible_output = tf.reduce_sum(tf.multiply(self.Q, self.actions_onehot), axis=1, keep_dims=True)
            difference = tf.abs(self.responsible_output - self.targetQ)
            quadratic_part = tf.clip_by_value(difference, 0.0, 1.0)
            linear_part = difference - quadratic_part
            errors = (0.5 * tf.square(quadratic_part)) + linear_part
            self.loss = tf.reduce_sum(errors)

            #self.update_model = tf.train.AdamOptimizer(learning_rate=2.5e-4).minimize(self.loss)
            self.update_model = tf.train.RMSPropOptimizer(learning_rate=.00025, decay=.95, epsilon=.01).minimize(self.loss)

            self.string = 'network2'

    def get_action(self, sess, image_in):
        Q = sess.run(self.Q, feed_dict={self.image_in:image_in})
        action = np.argmax(Q)
        return action

    def train(self, sess, states, actions, targetQ):
        _, l = sess.run([self.update_model, self.loss], feed_dict={self.image_in:states, self.actions:actions, self.targetQ:targetQ})
        return l

    def get_Q1(self, sess, states1, tnet):
        return sess.run(tnet.Q, feed_dict={tnet.image_in:states1})

class network3:
    def __init__(self, input_shape, a_size=3, scope=None, padding='VALID'):
        with tf.variable_scope(scope, default_name='default_scope'):
            self.image_in = tf.placeholder(shape=input_shape, dtype=tf.uint8)
            self.image_in_normalized = tf.to_float(self.image_in) / 255.0
            self.conv1 = slim.conv2d(inputs=self.image_in_normalized, num_outputs=64, kernel_size=[8, 8], stride=[4, 4], padding=padding, activation_fn=tf.nn.relu)
            self.conv2 = slim.conv2d(inputs=self.image_in_normalized, num_outputs=64, kernel_size=[8, 8], stride=[4, 4], padding=padding, activation_fn=tf.nn.relu)
            mult = tf.multiply(self.conv1, self.conv2)
            self.hidden = slim.conv2d(inputs=mult, num_outputs=32, kernel_size=[1, 1], stride=[1, 1], padding=padding, activation_fn=tf.nn.sigmoid)
            self.Q = slim.fully_connected(slim.flatten(self.hidden), a_size, activation_fn=None)

            #Define loss function
            self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
            self.actions_onehot = tf.one_hot(self.actions, a_size, dtype=tf.float32)
            self.targetQ = tf.placeholder(shape=[None, 1], dtype=tf.float32)
            #self.loss = tf.reduce_sum(tf.square(self.targetQ - self.responsible_output))

            #self.responsible_output, self.loss, self.update_model = self.get_qloss(self.Q, self.actions_onehot)

            self.responsible_output = tf.reduce_sum(tf.multiply(self.Q, self.actions_onehot), axis=1, keep_dims=True)
            difference = tf.abs(self.responsible_output - self.targetQ)
            quadratic_part = tf.clip_by_value(difference, 0.0, 1.0)
            linear_part = difference - quadratic_part
            errors = (0.5 * tf.square(quadratic_part)) + linear_part
            self.loss = tf.reduce_sum(errors)

            #self.update_model = tf.train.AdamOptimizer(learning_rate=2.5e-4).minimize(self.loss)
            self.update_model = tf.train.RMSPropOptimizer(learning_rate=.00025, decay=.95, epsilon=.01).minimize(self.loss)

            self.string = 'network3'


    def get_action(self, sess, image_in):
        Q = sess.run(self.Q, feed_dict={self.image_in:image_in})
        action = np.argmax(Q)
        return action

    def train(self, sess, states, actions, targetQ):
        _, l = sess.run([self.update_model, self.loss], feed_dict={self.image_in:states, self.actions:actions, self.targetQ:targetQ})
        return l

    def get_Q1(self, sess, states1, tnet):
        return sess.run(tnet.Q, feed_dict={tnet.image_in:states1})


class network4:
    def __init__(self, input_shape, a_size=3, scope=None, padding='VALID'):
        with tf.variable_scope(scope, default_name='default_scope'):
            self.image_in = tf.placeholder(shape=input_shape, dtype=tf.uint8)
            self.image_in_normalized = tf.to_float(self.image_in) / 255.0
            self.conv1 = slim.conv2d(inputs=self.image_in_normalized, num_outputs=64, kernel_size=[8, 8], stride=[4, 4], padding=padding, activation_fn=tf.nn.relu)
            self.conv2 = slim.conv2d(inputs=self.image_in_normalized, num_outputs=64, kernel_size=[8, 8], stride=[4, 4], padding=padding, activation_fn=tf.nn.relu)
            mult = tf.multiply(self.conv1, self.conv2)
            self.hidden = slim.conv2d(inputs=mult, num_outputs=32, kernel_size=[1, 1], stride=[1, 1], padding=padding, activation_fn=tf.nn.relu)
            self.Q = slim.fully_connected(slim.flatten(self.hidden), a_size, activation_fn=None)

            #Define loss function
            self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
            self.actions_onehot = tf.one_hot(self.actions, a_size, dtype=tf.float32)
            self.targetQ = tf.placeholder(shape=[None, 1], dtype=tf.float32)
            #self.loss = tf.reduce_sum(tf.square(self.targetQ - self.responsible_output))

            #self.responsible_output, self.loss, self.update_model = self.get_qloss(self.Q, self.actions_onehot)

            self.responsible_output = tf.reduce_sum(tf.multiply(self.Q, self.actions_onehot), axis=1, keep_dims=True)
            difference = tf.abs(self.responsible_output - self.targetQ)
            quadratic_part = tf.clip_by_value(difference, 0.0, 1.0)
            linear_part = difference - quadratic_part
            errors = (0.5 * tf.square(quadratic_part)) + linear_part
            self.loss = tf.reduce_sum(errors)

            #self.update_model = tf.train.AdamOptimizer(learning_rate=2.5e-4).minimize(self.loss)
            self.update_model = tf.train.RMSPropOptimizer(learning_rate=.00025, decay=.95, epsilon=.01).minimize(self.loss)

            self.string = 'network4'


    def get_action(self, sess, image_in):
        Q = sess.run(self.Q, feed_dict={self.image_in:image_in})
        action = np.argmax(Q)
        return action

    def train(self, sess, states, actions, targetQ):
        _, l = sess.run([self.update_model, self.loss], feed_dict={self.image_in:states, self.actions:actions, self.targetQ:targetQ})
        return l

    def get_Q1(self, sess, states1, tnet):
        return sess.run(tnet.Q, feed_dict={tnet.image_in:states1})

class network5():
    def __init__(self, input_shape, a_size=3, scope=None, padding='VALID'):
        with tf.variable_scope(scope, default_name='default_scope'):
            self.image_in = tf.placeholder(shape=input_shape, dtype=tf.uint8)
            self.image_in_normalized = tf.to_float(self.image_in) / 255.0
            self.conv0 = slim.conv2d(activation_fn=tf.nn.relu, inputs=self.image_in_normalized, num_outputs=32, kernel_size=[8, 8], stride=[4, 4], padding=padding)
            self.conv1 = slim.conv2d(activation_fn=tf.nn.relu, inputs=self.image_in_normalized, num_outputs=32, kernel_size=[8, 8], stride=[4, 4], padding=padding)
            mult = tf.multiply(self.conv0, self.conv1)
            self.hidden = slim.conv2d(inputs=mult, num_outputs=32, kernel_size=[1, 1], stride=[1, 1], padding=padding, activation_fn=tf.nn.relu)


            self.conv2 = slim.conv2d(activation_fn=tf.nn.relu, inputs=self.hidden, num_outputs=64, kernel_size=[4, 4], stride=[2, 2], padding=padding)
            self.conv3 = slim.conv2d(activation_fn=tf.nn.relu, inputs=self.conv2, num_outputs=64, kernel_size=[3, 3], stride=[1, 1], padding=padding)
            self.fc1 = slim.fully_connected(slim.flatten(self.conv3), 512, activation_fn=tf.nn.relu)
            self.Q = slim.fully_connected(self.fc1, a_size, activation_fn=None)

            #Define loss function
            self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
            self.actions_onehot = tf.one_hot(self.actions, a_size, dtype=tf.float32)
            self.targetQ = tf.placeholder(shape=[None, 1], dtype=tf.float32)
            #self.loss = tf.reduce_sum(tf.square(self.targetQ - self.responsible_output))

            #self.responsible_output, self.loss, self.update_model = self.get_qloss(self.Q, self.actions_onehot)

            self.responsible_output = tf.reduce_sum(tf.multiply(self.Q, self.actions_onehot), axis=1, keep_dims=True)
            difference = tf.abs(self.responsible_output - self.targetQ)
            quadratic_part = tf.clip_by_value(difference, 0.0, 1.0)
            linear_part = difference - quadratic_part
            errors = (0.5 * tf.square(quadratic_part)) + linear_part
            self.loss = tf.reduce_sum(errors)

            #self.update_model = tf.train.AdamOptimizer(learning_rate=2.5e-4).minimize(self.loss)
            self.update_model = tf.train.RMSPropOptimizer(learning_rate=.00025, decay=.95, epsilon=.01).minimize(self.loss)

            self.string = 'network5'

    def get_action(self, sess, image_in):
        Q = sess.run(self.Q, feed_dict={self.image_in:image_in})
        action = np.argmax(Q)
        return action

    def train(self, sess, states, actions, targetQ):
        _, l = sess.run([self.update_model, self.loss], feed_dict={self.image_in:states, self.actions:actions, self.targetQ:targetQ})
        return l

    def get_Q1(self, sess, states1, tnet):
        return sess.run(tnet.Q, feed_dict={tnet.image_in:states1})

class network6():
    def __init__(self, input_shape, a_size=3, scope=None, padding='VALID'):
        with tf.variable_scope(scope, default_name='default_scope'):
            self.image_in = tf.placeholder(shape=input_shape, dtype=tf.uint8)
            self.image_in_normalized = tf.to_float(self.image_in) / 255.0
            self.conv0 = slim.conv2d(activation_fn=None, inputs=self.image_in_normalized, num_outputs=32, kernel_size=[8, 8], stride=[4, 4], padding=padding, biases_initializer=None)
            self.conv1 = slim.conv2d(activation_fn=None, inputs=self.image_in_normalized, num_outputs=32, kernel_size=[8, 8], stride=[4, 4], padding=padding, biases_initializer=None)
            mult = tf.multiply(self.conv0, self.conv1)
            self.hidden = slim.conv2d(inputs=mult, num_outputs=32, kernel_size=[1, 1], stride=[1, 1], padding=padding, activation_fn=tf.nn.sigmoid)


            self.conv2 = slim.conv2d(activation_fn=tf.nn.relu, inputs=self.hidden, num_outputs=64, kernel_size=[4, 4], stride=[2, 2], padding=padding)
            self.conv3 = slim.conv2d(activation_fn=tf.nn.relu, inputs=self.conv2, num_outputs=64, kernel_size=[3, 3], stride=[1, 1], padding=padding)
            self.fc1 = slim.fully_connected(slim.flatten(self.conv3), 512, activation_fn=tf.nn.relu)
            self.Q = slim.fully_connected(self.fc1, a_size, activation_fn=None)

            #Define loss function
            self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
            self.actions_onehot = tf.one_hot(self.actions, a_size, dtype=tf.float32)
            self.targetQ = tf.placeholder(shape=[None, 1], dtype=tf.float32)
            #self.loss = tf.reduce_sum(tf.square(self.targetQ - self.responsible_output))

            #self.responsible_output, self.loss, self.update_model = self.get_qloss(self.Q, self.actions_onehot)

            self.responsible_output = tf.reduce_sum(tf.multiply(self.Q, self.actions_onehot), axis=1, keep_dims=True)
            difference = tf.abs(self.responsible_output - self.targetQ)
            quadratic_part = tf.clip_by_value(difference, 0.0, 1.0)
            linear_part = difference - quadratic_part
            errors = (0.5 * tf.square(quadratic_part)) + linear_part
            self.loss = tf.reduce_sum(errors)

            #self.update_model = tf.train.AdamOptimizer(learning_rate=2.5e-4).minimize(self.loss)
            self.update_model = tf.train.RMSPropOptimizer(learning_rate=.00025, decay=.95, epsilon=.01).minimize(self.loss)

            self.string = 'network6'

    def get_action(self, sess, image_in):
        Q = sess.run(self.Q, feed_dict={self.image_in:image_in})
        action = np.argmax(Q)
        return action

    def train(self, sess, states, actions, targetQ):
        _, l = sess.run([self.update_model, self.loss], feed_dict={self.image_in:states, self.actions:actions, self.targetQ:targetQ})
        return l

    def get_Q1(self, sess, states1, tnet):
        return sess.run(tnet.Q, feed_dict={tnet.image_in:states1})


class network7():
    def __init__(self, input_shape, a_size=3, scope=None, padding='VALID'):
        with tf.variable_scope(scope, default_name='default_scope'):
            self.image_in = tf.placeholder(shape=input_shape, dtype=tf.uint8)
            self.image_in_normalized = tf.to_float(self.image_in) / 255.0
            self.conv0 = slim.conv2d(activation_fn=None, inputs=self.image_in_normalized, num_outputs=32, kernel_size=[8, 8], stride=[4, 4], padding=padding, biases_initializer=None)
            self.conv1 = slim.conv2d(activation_fn=None, inputs=self.image_in_normalized, num_outputs=32, kernel_size=[8, 8], stride=[4, 4], padding=padding, biases_initializer=None)
            mult = tf.multiply(self.conv0, self.conv1)

            self.xavier_init = tf.contrib.layers.xavier_initializer()
            w = tf.Variable(self.xavier_init([32,32]))
            b = tf.Variable(tf.zeros([32]))


            mult_flat = tf.reshape(mult, shape=(-1, 32))
            hidden_flat = tf.nn.sigmoid(tf.matmul(mult_flat, w) + b)
            hidden = tf.reshape(hidden_flat, shape=(-1, int(mult.shape[1]), int(mult.shape[2]), int(mult.shape[3])))

            self.hidden = hidden
            #self.hidden = slim.conv2d(inputs=mult, num_outputs=32, kernel_size=[1, 1], stride=[1, 1], padding=padding, activation_fn=tf.nn.sigmoid)


            self.conv2 = slim.conv2d(activation_fn=tf.nn.relu, inputs=self.hidden, num_outputs=64, kernel_size=[4, 4], stride=[2, 2], padding=padding)
            self.conv3 = slim.conv2d(activation_fn=tf.nn.relu, inputs=self.conv2, num_outputs=64, kernel_size=[3, 3], stride=[1, 1], padding=padding)
            self.fc1 = slim.fully_connected(slim.flatten(self.conv3), 512, activation_fn=tf.nn.relu)
            self.Q = slim.fully_connected(self.fc1, a_size, activation_fn=None)

            #Define loss function
            self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
            self.actions_onehot = tf.one_hot(self.actions, a_size, dtype=tf.float32)
            self.targetQ = tf.placeholder(shape=[None, 1], dtype=tf.float32)
            #self.loss = tf.reduce_sum(tf.square(self.targetQ - self.responsible_output))

            #self.responsible_output, self.loss, self.update_model = self.get_qloss(self.Q, self.actions_onehot)

            self.responsible_output = tf.reduce_sum(tf.multiply(self.Q, self.actions_onehot), axis=1, keep_dims=True)
            difference = tf.abs(self.responsible_output - self.targetQ)
            quadratic_part = tf.clip_by_value(difference, 0.0, 1.0)
            linear_part = difference - quadratic_part
            errors = (0.5 * tf.square(quadratic_part)) + linear_part
            self.loss = tf.reduce_sum(errors)

            #self.update_model = tf.train.AdamOptimizer(learning_rate=2.5e-4).minimize(self.loss)
            self.update_model = tf.train.RMSPropOptimizer(learning_rate=.00025, decay=.95, epsilon=.01).minimize(self.loss)

            self.string = 'network7'

    def get_action(self, sess, image_in):
        Q = sess.run(self.Q, feed_dict={self.image_in:image_in})
        action = np.argmax(Q)
        return action

    def train(self, sess, states, actions, targetQ):
        _, l = sess.run([self.update_model, self.loss], feed_dict={self.image_in:states, self.actions:actions, self.targetQ:targetQ})
        return l

    def get_Q1(self, sess, states1, tnet):
        return sess.run(tnet.Q, feed_dict={tnet.image_in:states1})














class convnet_feature_extractor(network):
    def __init__(self, input_shape, a_size=3, padding='VALID', stop_gradient=True, scope=None):
        network.__init__(self, input_shape, a_size, scope, padding)
        self.stop_gradient = stop_gradient

        #...#
        #self.hidden = self.fc1
        self.hidden = slim.flatten(self.conv3)
        if self.stop_gradient:
            self.hidden = tf.stop_gradient(self.hidden)

        #deconv1 = slim.fully_connected(self.fc1, int(slim.flatten(self.conv3).shape[-1]), activation_fn=tf.nn.relu)
        #self.deconv1 = tf.reshape(deconv1, shape=self.get_shape(self.conv3))
        self.deconv2 = slim.conv2d_transpose(activation_fn=tf.nn.relu, inputs=self.conv3, num_outputs=64, kernel_size=[3, 3], stride=[1, 1], padding=padding)
        self.deconv3 = slim.conv2d_transpose(activation_fn=tf.nn.relu, inputs=self.deconv2, num_outputs=32, kernel_size=[4, 4], stride=[2, 2], padding=padding)
        self.deconvim = slim.conv2d_transpose(activation_fn=tf.nn.relu, inputs=self.deconv3, num_outputs=input_shape[-1], kernel_size=[8, 8], stride=[4, 4], padding=padding)

        #Define the reconstruction loss
        self.recon_loss = tf.reduce_mean(tf.reduce_sum(tf.square(slim.flatten(self.image_in_normalized - self.deconvim)), axis=-1))
        self.update_recon = tf.train.AdamOptimizer(). minimize(self.recon_loss)

    def get_shape(self, tensor):
        shape = (-1,)
        for i in range(1, len(tensor.shape)):
            shape += (int(tensor.shape[i]),)
        return shape

class cnn2:
    def __init__(self, input_shape, a_size=3, padding='VALID', stop_gradient=True):
        self.feature_extractor = convnet_feature_extractor(input_shape, a_size, padding, stop_gradient)
        self.a_size = a_size
        self.image_in = self.feature_extractor.image_in

        self.Q_source = self.get_Q('qnet')
        self.Q_target = self.get_Q('tnet')

        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, a_size, dtype=tf.float32)
        self.targetQ = tf.placeholder(shape=[None, 1], dtype=tf.float32)

        self.responsible_output = tf.reduce_sum(tf.multiply(self.Q_source, self.actions_onehot), axis=1, keep_dims=True)
        difference = tf.abs(self.responsible_output - self.targetQ)
        quadratic_part = tf.clip_by_value(difference, 0.0, 1.0)
        linear_part = difference - quadratic_part
        errors = (0.5 * tf.square(quadratic_part)) + linear_part
        self.loss = tf.reduce_sum(errors)

        self.update_model = tf.train.RMSPropOptimizer(learning_rate=.00025, decay=.95, epsilon=.01).minimize(self.loss)

    def get_action(self, sess, image_in):
        Q = sess.run(self.Q_source, feed_dict={self.image_in:image_in})
        action = np.argmax(Q)
        return action

    def train(self, sess, states, actions, targetQ):
        _, l = sess.run([self.update_model, self.loss], feed_dict={self.image_in:states, self.actions:actions, self.targetQ:targetQ})
        return l

    def get_Q(self, scope):
        with tf.variable_scope(scope):
            fc1 = slim.fully_connected(self.feature_extractor.hidden, 512, activation_fn=tf.nn.relu)
            Q = slim.fully_connected(fc1, self.a_size, activation_fn=None)
        return Q

    def get_Q1(self, sess, states1, _):
        return sess.run(self.Q_target, feed_dict={self.image_in:states1})

    def train_fe(self, sess, states):
        _ = sess.run(self.feature_extractor.update_recon, feed_dict={self.image_in:states})

def main():
    #test = cnn2([None, 36, 36, 4], 3, 'VALID')
    #test = network5([None, 84, 84, 4])
    test = network7([None, 84, 84, 4])

    for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        print v

if __name__ == '__main__':
    main()
