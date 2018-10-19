import numpy as np
from lwpr import LWPR
from gd import gather_data, gather_data_epoch
import matplotlib.pyplot as plt

def main():
    no_data = 1000*10
    environment = 'AntBulletEnv-v0'
    state_action, state, reward, next_state = gather_data(no_data, environment)
    assert len(state_action) == len(next_state)
    assert len(state_action) == len(reward)
    no_data = len(state_action)

#    model_state = LWPR(state_action.shape[-1], next_state.shape[-1])
#    model_state.init_D = 1. * np.eye(state_action.shape[-1])
#    model_state.update_D = True
#    model_state.init_alpha = 20. * np.eye(state_action.shape[-1])
#    model_state.meta = False

    model_reward = LWPR(state_action.shape[-1], reward.shape[-1])
    model_reward.init_D = 1. * np.eye(state_action.shape[-1])
    model_reward.update_D = True
    model_reward.init_alpha = 20. * np.eye(state_action.shape[-1])
    model_reward.meta = False

    #for k in range(20):
    for k in range(1):
        ind = np.random.permutation(no_data)
        for i in range(no_data):
            print (k, i)
#            model_state.update(state_action[ind[i]], next_state[ind[i]] - state[ind[i]])
            model_reward.update(state_action[ind[i]], reward[ind[i]])

    for k in range(10):
        state_action_test, state_test, reward_test, next_state_test = gather_data_epoch(1, environment)
#        Y = []
#        confs = []
        Y_r = []
        confs_r = []
        for i in range(len(state_action_test)):
#            y, conf = model_state.predict_conf(state_action_test[i])
#            Y.append(y)
#            confs.append(conf)
            y_r, conf_r = model_reward.predict_conf(state_action_test[i])
            Y_r.append(y_r)
            confs_r.append(conf_r)
#        Y = np.stack(Y, axis=0)
#        confs = np.stack(confs, axis=0)
        Y_r = np.stack(Y_r, axis=0)
        confs_r = np.stack(confs_r, axis=0)

#        for i in range(next_state.shape[-1]):
#            plt.figure()
#            plt.plot(np.arange(len(next_state_test[:, i:i+1])), next_state_test[:, i:i+1])
#            plt.errorbar(np.arange(len(Y[:, i:i+1])), Y[:, i:i+1] + state_test[:, i:i+1], yerr=confs[:, i:i+1], color='m', ecolor='g')
#            plt.grid()

        plt.figure()
        plt.plot(np.arange(len(reward_test)), reward_test)
        plt.errorbar(np.arange(len(Y_r)), Y_r, yerr=confs_r, color='r', ecolor='y')

        plt.grid()
        plt.show()


if __name__ == '__main__':
    main()
