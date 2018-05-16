import numpy as np

import sys
sys.path.append('..')
from prototype8.dmlac.real_env_pendulum import get_next

class dynamics_model:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

    def random_batch(self, batch_size=32):
        '''Gets a random batch from Pendulum-v0 environment'''
        #theta, thetadot, action
        bounds_low = np.array([0., -8., -2.])
        bounds_high = np.array([2.*np.pi, 8., 2.])
        
        states = np.stack([np.random.uniform(bounds_low[i], bounds_high[i], batch_size)
                           for i in range(len(bounds_low))], axis=-1)

        next_states = get_next(states[:, 0], states[:, 1], states[:, 2])

        next_states = np.stack(next_states, axis=-1)
        actions = np.copy(states[:, 2])
        states = np.stack([np.cos(states[:, 0]), np.sin(states[:, 0]), states[:, 1]], axis=-1)

        return states, actions, next_states

def main():
    model = dynamics_model()
    model.random_batch()

if __name__ == '__main__':
    main()
