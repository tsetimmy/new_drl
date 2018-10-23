import numpy as np
from lwpr import LWPR
from gd import gather_data, gather_data_epoch
import matplotlib.pyplot as plt
import argparse
import uuid
import pickle

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default='AntBulletEnv-v0')
    parser.add_argument("--no_data", type=int, default=10000)
    args = parser.parse_args()

    state_action, state, reward, next_state = gather_data(args.no_data, args.environment)
    assert len(state_action) == len(next_state)
    assert len(state_action) == len(reward)

    '''
    data = pickle.load(open('data.p'))
    state2, action2, reward2, next_state2 = data[0]
    state_action2 = np.concatenate([state2, action2], axis=-1)
    state_action = np.concatenate([state_action, state_action2], axis=0)
    reward = np.concatenate([reward, reward2], axis=0)
    next_state = np.concatenate([next_state, next_state2], axis=0)
    '''

    no_data = len(state_action)

    model_state = LWPR(state_action.shape[-1], next_state.shape[-1])
    model_state.init_D = 1. * np.eye(state_action.shape[-1])
    model_state.update_D = True
    model_state.init_alpha = 20. * np.eye(state_action.shape[-1])
    model_state.meta = True

    model_reward = LWPR(state_action.shape[-1], reward.shape[-1])
    model_reward.init_D = 1. * np.eye(state_action.shape[-1])
    model_reward.update_D = True
    model_reward.init_alpha = 20. * np.eye(state_action.shape[-1])
    model_reward.meta = True

    #for k in range(20):
    for k in range(1):
        ind = np.random.permutation(no_data)
        for i in range(no_data):
            #print (k, i)
            #print next_state[ind[i]] - state[ind[i]]
            model_state.update(state_action[ind[i]], next_state[ind[i]] - state[ind[i]])
            model_reward.update(state_action[ind[i]], reward[ind[i]])

    uid = str(uuid.uuid4())
    for k in range(10):
        state_action_test, state_test, reward_test, next_state_test = gather_data_epoch(1, args.environment)
        '''
        if k % 2 == 0:
            state_action_test, state_test, reward_test, next_state_test = gather_data_epoch(1, args.environment)
        else:
            idx = np.random.randint(1, len(data))
            state_test, action_test, reward_test, next_state_test = data[idx]
            state_action_test = np.concatenate([state_test, action_test], axis=-1)
        '''
        Y = []
        confs = []
        Y_r = []
        confs_r = []
        for i in range(len(state_action_test)):
            y, conf = model_state.predict_conf(state_action_test[i])
            #Y.append(y + state_test[i])
            Y.append(y)
            confs.append(conf)
            y_r, conf_r = model_reward.predict_conf(state_action_test[i])
            Y_r.append(y_r)
            confs_r.append(conf_r)
        Y = np.stack(Y, axis=0)
        confs = np.stack(confs, axis=0)
        Y_r = np.stack(Y_r, axis=0)
        confs_r = np.stack(confs_r, axis=0)

        for i in range(next_state.shape[-1]):
            plt.figure()
            assert len(next_state_test[:, i:i+1]) == len(Y[:, i:i+1])
            plt.plot(np.arange(len(next_state_test[:, i:i+1])), next_state_test[:, i:i+1] - state_test[:, i:i+1])
            plt.errorbar(np.arange(len(Y[:, i:i+1])), Y[:, i:i+1], yerr=confs[:, i:i+1], color='r', ecolor='y')
            plt.grid()
            #plt.savefig(args.environment+'_'+'k:'+str(k)+'_'+'dim:'+str(i)+'_'+uid+'.pdf')

        plt.figure()
        plt.plot(np.arange(len(reward_test)), reward_test)
        plt.errorbar(np.arange(len(Y_r)), Y_r, yerr=confs_r, color='r', ecolor='g')
        plt.grid()
        plt.show()
        #plt.savefig(args.environment+'_'+'k:'+str(k)+'_'+'dim:'+'reward'+'_'+uid+'.pdf')

if __name__ == '__main__':
    main()
