import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import sys
sys.path.append('..')
from prototype8.dmlac.real_env_pendulum import get_next, get_next_state
from bayesian_neural_network_edward.bayesian_neural_network\
     import bayesian_dynamics_model

class dynamics_model:
    def __init__(self, input_size, output_size, iterations):
        self.iterations = iterations

        self.input_size = input_size
        self.output_size = output_size

        # Initialize the Bayesian neural network.
        self.bnn = bayesian_dynamics_model(self.input_size, self.output_size)
        self.bnn.initialize_inference(n_iter=self.iterations, n_samples=10)

    def train(self, x, y):
        info_dict = self.bnn.inference.update({self.bnn.x:x, self.bnn.y_ph:y})
        self.bnn.inference.print_progress(info_dict)

    def random_batch(self, batch_size=32):
        '''Gets a random batch from Pendulum-v0 environment.'''
        # theta, thetadot, action.
        bounds_low = np.array([-np.pi, -8., -2.])
        bounds_high = np.array([np.pi, 8., 2.])
        
        states = np.stack([np.random.uniform(bounds_low[i], bounds_high[i], batch_size)
                           for i in range(len(bounds_low))], axis=-1)

        next_states = get_next(states[:, 0], states[:, 1], states[:, 2])

        next_states = np.stack(next_states, axis=-1)
        actions = np.copy(states[:, 2])
        states = np.stack([np.cos(states[:, 0]), np.sin(states[:, 0]), states[:, 1]], axis=-1)

        return states, actions, next_states

    def random_policy(self, T=100):
        '''Gets a random policy for the Pendulum-v0 environment.'''
        action_bound_high = 2.
        action_bound_low = -2.

        policy = np.random.uniform(action_bound_low, action_bound_high, T)

        return policy

    def random_seed_state(self):
        theta = np.random.uniform(-np.pi, np.pi)
        thetadot = np.random.uniform(-8., 8.)

        return np.array([np.cos(theta), np.sin(theta), thetadot])

def visualize_trajectory():
    pass

    thetas = []
    state = np.copy(seed_state)
    for action in policy:
        state = get_next_state(state, action)
        thetas.append(state[0, 0])

    plt.plot(np.arange(len(thetas)), thetas)
    plt.grid()
    plt.show()
    exit()


def main():
    batch_size = 32
    iterations = 10000
    model = dynamics_model(4, 3, iterations)

    policy = model.random_policy()
    seed_state = model.random_seed_state()

    # Train ANN
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # Training step
        for _ in range(iterations):
            batch = model.random_batch(batch_size=batch_size)
            model.train(x=np.concatenate([batch[0], batch[1][..., np.newaxis]], axis=-1), y=batch[2])

        # Evaluation step
        for _ in range(1):
            state = np.copy(seed_state)
            for action in policy:

                next_state = sess.run(model)


    





if __name__ == '__main__':
    main()
