import numpy as np
import tensorflow as tf
import pickle


def original():
    from gp_original import GaussianProcess
    data = pickle.load( open( "error_data.p", "rb" ) )
    states = data[0]
    actions = data[1]
    rewards = data[2]
    next_states = data[3]
    states_actions = np.concatenate([states, actions], axis=-1)

    X = states_actions
    y = rewards[..., np.newaxis]
    dim = states_actions.shape[-1]
    ls = [.1] * dim
    sess = tf.Session()
    gp = GaussianProcess(dim=dim, sess=sess, ls=ls, noise=0.1)
    sess.run(tf.initialize_all_variables())
    gp.solve(X, y, epochs=100, batch_size=len(X), train=True)

    y_, var_ = gp.predict(X)

def mine():
    from gp_mine import GaussianProcessMine

    data = pickle.load( open( "error_data.p", "rb" ) )
    states = data[0]
    actions = data[1]
    rewards = data[2]
    next_states = data[3]
    states_actions = np.concatenate([states, actions], axis=-1)

    X = states_actions
    y = rewards[..., np.newaxis]
    dim = states_actions.shape[-1]
    ls = [.1] * dim

    xp = tf.placeholder(tf.float32, [None, dim])
    yp = tf.placeholder(tf.float32, [None, 1])
    x_p = tf.placeholder(tf.float32, [None, dim])
    sess = tf.Session()

    gp = GaussianProcessMine()
    L, y_, var_, opt_op, grad_ls, grad_amp = gp.build(xp, yp, x_p)
    gp.L = L
    gp.xp = xp
    gp.yp = yp
    gp.opt_op = opt_op
    gp.sess = sess
    sess.run(tf.initialize_all_variables())
    gp.solve(X, y, epochs=100, batch_size=len(X), train=True)


    pass

if __name__ == "__main__":
  #original()
  mine()
