import numpy as np
import gym
import pybullet_envs
from lwpr import LWPR

def gather_data(no_data):
    env = gym.make('AntBulletEnv-v0')

    data = []

    while True:
        state = env.reset()
        done = False
        while True:
            action = np.random.uniform(low=env.action_space.low, high=env.action_space.high)
            next_state, reward, done, _ = env.step(action)

            data.append([state.astype(np.float64), action, reward, next_state.astype(np.float64)])

            if done:
                break

        if len(data) >= no_data:
            break

    state, action, reward, next_state = zip(*data)
    state = np.stack(state, axis=0)
    action = np.stack(action, axis=0)
    reward = np.array(reward)[..., np.newaxis]
    next_state = np.stack(next_state, axis=0)

    state_action = np.concatenate([state, action], axis=-1)

    return state_action, next_state

def main():
    no_data = 100
    state_action, next_state = gather_data(no_data)

    model = LWPR(state_action.shape[-1], next_state.shape[-1])
    model.init_D = 20. * np.eye(state_action.shape[-1])
    model.update_D = True
    model.init_alpha = 40. * np.eye(state_action.shape[-1])
    model.meta = False

    for k in range(20):
        ind = np.random.permutation(no_data)
        mse = 0
         
        for i in range(no_data):
            yp = model.update(state_action[ind[i]], next_state[ind[i]])
            mse = mse + (next_state[ind[i], :] - yp)**2
     
        nMSE = mse/no_data/np.var(next_state)
        print nMSE
        exit()
        print "#Data: %5i  #RFs: %3i  nMSE=%5.3f"%(model.n_data, model.num_rfs, nMSE)


    

if __name__ == '__main__':
    main()
