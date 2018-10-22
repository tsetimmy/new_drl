import numpy as np

import argparse
import gym
import pybullet_envs

import sys
sys.path.append('..')
from utils import gather_data3
from lwpr import LWPR
import cma

class AGENT:
    def __init__(self, observation_space_dim, action_space_dim, action_space_low, action_space_high, unroll_steps):
        self.observation_space_dim = observation_space_dim
        self.action_space_dim = action_space_dim
        self.action_space_low = action_space_low
        self.action_space_high = action_space_high
        self.unroll_steps = unroll_steps

        #TODO: change this.
        self.thetas = np.random.normal(size=[10])

    def _forward(self, thetas, X):
        #TODO: implment this.
        return np.random.normal(size=[len(X), self.action_space_dim])

    def _loss(self, thetas, model, X):
        rng_state = np.random.get_state()
        np.random.seed(2)

        rewards = []
        state = X.copy()
        for unroll_step in range(self.unroll_steps):
            action = self._forward(thetas, state)
            state_action = np.concatenate([state, action], axis=-1)
            mu, sd = [np.stack(ele, axis=0) for ele in zip(*[model.predict_conf(sa) for sa in state_action])]
            print mu
            print '----------'
            print sd
            exit()
            state_reward = np.stack([np.random.multivariate_normal(MU, np.diag(np.square(SD))) for MU, SD in zip(mu, sd)], axis=0)
            state += state_reward[:, :-1]
            reward = state_reward[:, -1:].copy()
            rewards.append(reward)

        rewards = np.concatenate(rewards, axis=-1)
        rewards = np.sum(rewards, axis=-1)
        loss = -np.mean(rewards)
        print loss
        exit()
        np.random.set_state(rng_state)
        return loss

    def _fit(self, model, init_states, cma_maxiter):
        options = {'maxiter': cma_maxiter, 'verb_disp': 1, 'verb_log': 0}
        res = cma.fmin(self._loss, self.thetas, 2., args=(model, init_states.copy()), options=options)
        self.thetas = np.copy(res[0])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default='AntBulletEnv-v0')
    parser.add_argument("--no_data_start", type=int, default=10000)
    parser.add_argument("--train_policy_batch_size", type=int, default=30)
    parser.add_argument("--cma_maxiter", type=int, default=1000)
    parser.add_argument("--unroll_steps", type=int, default=200)
    args = parser.parse_args()

    print args

    env = gym.make(args.environment)

    input_dim = env.observation_space.shape[0]+env.action_space.shape[0]
    output_dim = env.observation_space.shape[0] + 1
    model = LWPR(input_dim, output_dim)
    model.init_D = 1. * np.eye(input_dim)
    model.update_D = True
    model.init_alpha = 20. * np.eye(input_dim)
    model.meta = True

    agent = AGENT(env.observation_space.shape[0],
                  env.action_space.shape[0],
                  action_space_low=env.action_space.low,
                  action_space_high=env.action_space.high,
                  unroll_steps=args.unroll_steps)

    init_states = np.stack([env.reset() for _ in range(args.train_policy_batch_size)], axis=0)

    #Train the dynamics model the intial data.
    data_buffer = gather_data3(env, args.no_data_start)
    states, actions, rewards, next_states, _ = zip(*data_buffer)
    states = np.stack(states, axis=0)
    actions = np.stack(actions, axis=0)
    rewards = np.array(rewards)[..., np.newaxis]
    next_states = np.stack(next_states, axis=0)

    state_actions = np.concatenate([states, actions], axis=-1)
    state_diff = next_states - states
    targets = np.concatenate([state_diff, rewards], axis=-1)

    assert len(state_actions) == len(targets)
    ind = np.random.permutation(len(state_actions))
    for i in range(len(state_actions)):
        model.update(state_actions[ind[i]], targets[ind[i]])

    for epoch in range(1000):
        agent._fit(model, init_states, args.cma_maxiter)
        


        exit()









if __name__ == '__main__':
    main()
