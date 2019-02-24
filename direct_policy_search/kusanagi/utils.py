import numpy as np
from choldate import cholupdate

def get_data(env, max_steps, action_high):
    action_low = -action_high

    states = []
    actions = []
    costs = []
    next_states = []
    state = env.reset()
    steps = 0
    while True:
        #env.render()
        action = np.random.random(env.action_space.shape) * (action_high - action_low) + action_low

        next_state, cost, done, _ = env.step(action)
        steps += 1
        #print (cost, env.loss_func(env.state_noiseless[None, ...]))

        states.append(state)
        actions.append(action)
        costs.append(cost)
        next_states.append(next_state)

        state = next_state.copy()
        if done or steps >= max_steps:
            break

    states = np.stack(states, axis=0)
    actions = np.stack(actions, axis=0)
    costs = np.stack(costs, axis=0)
    next_states = np.stack(next_states, axis=0)
    return states, actions, -costs, next_states

def get_data2(env, trials=1, max_steps=25, maxA=10):
    return [np.concatenate(ele, axis=0) for ele in list(zip(*[get_data(env, max_steps, maxA) for _ in range(trials)]))]

def get_data3(env, trials=1, max_steps=25, maxA=10):
    states, actions, rewards, next_states = get_data2(env, trials, max_steps, maxA)
    data = [[state, action, reward, next_state] for state, action, reward, next_state in zip(states, actions, rewards, next_states)]
    return data
