import numpy as np

import argparse
import gym
import pybullet_envs

from utils import gather_data3
from lwpr import LWPR

class AGENT:
    def __init__(self, observation_space_dim, action_space_dim):
        self.observation_space_dim = observation_space_dim
        self.action_space_dim = action_space_dim

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default='AntBulletEnv-v0')
    parser.add_argument("--no_data_start", type=int, default=10000)
    args = parser.parse_args()

    print args

    env = gym.make(args.environment)

    lwpr = LWPR(env.observation_space.shape[0]+env.action_space.shape[0], env.observation_space.shape[0]+1)


    agent = AGENT(env.observation_space.shape[0], env.action_space.shape[0])




if __name__ == '__main__':
    main()
