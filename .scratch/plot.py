import numpy as np
import matplotlib.pyplot as plt
import pickle

import uuid


def main():
    filenames = ['data_halfcheetah.p', 'data_hopper.p', 'data_humanoid.p', 'data_walker2d.p']


    for filename in filenames:
        data = pickle.load(open(filename, 'rb'))
        print type(data)
        for datum in data:
            state, action, reward, next_state = datum

            uid = str(uuid.uuid4())
            for i in range(next_state.shape[-1]):
                plt.figure()
                plt.plot(np.arange(len(next_state[:, i])), next_state[:, i])
                plt.grid()
                plt.title(filename+' next_state, dim='+str(i))
                plt.savefig(filename+'next_state'+str(i)+'_'+uid+'.pdf')
            
            R = np.squeeze(reward)
            plt.figure()
            plt.plot(np.arange(len(R)), R)
            plt.grid()
            plt.title(filename+' reward')
            plt.savefig(filename+'reward_'+uid+'.pdf')


if __name__ == '__main__':
    main()
