import numpy as np
import gym
import pybullet_envs

def gather_data_epoch(no_epochs, environment):
    env = gym.make(environment)

    data = []
    epochs = 0

    while True:
        state = env.reset()
        done = False
        while True:
            action = np.random.uniform(low=env.action_space.low, high=env.action_space.high)
            next_state, reward, done, _ = env.step(action)

            data.append([state.astype(np.float64), action, reward, next_state.astype(np.float64)])

            state = next_state.copy()

            if done:
                epochs += 1
                break

        if epochs == no_epochs:
            break

    state, action, reward, next_state = zip(*data)
    state = np.stack(state, axis=0)
    action = np.stack(action, axis=0)
    reward = np.array(reward)[..., np.newaxis]
    next_state = np.stack(next_state, axis=0)

    state_action = np.concatenate([state, action], axis=-1)

    return state_action, state, reward, next_state

def gather_data(no_data, environment):
    env = gym.make(environment)

    data = []

    while True:
        state = env.reset()
        done = False
        while True:
            action = np.random.uniform(low=env.action_space.low, high=env.action_space.high)
            next_state, reward, done, _ = env.step(action)

            data.append([state.astype(np.float64), action, reward, next_state.astype(np.float64)])

            state = next_state.copy()

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

    return state_action, state, reward, next_state
