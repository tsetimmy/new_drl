'''
from ple.games.monsterkong import MonsterKong
#from ple.games.catcher import Catcher
from ple import PLE

game = FlappyBird()
#game = Catcher()
p = PLE(game, fps=30, display_screen=True)
#agent = myAgentHere(allowed_actions=p.getActionSet())

print p.getActionSet()

p.init()
reward = 0.0

while True:
   if p.game_over():
           p.reset_game()

   observation = p.getScreenRGB()
   #action = agent.pickAction(reward, observation)
   reward = p.act()

'''

import numpy as np
class ple_wrapper:
    def __init__(self, game, display_screen=False):
        from ple import PLE
        assert game in ['catcher',
                        'monsterkong',
                        'flappybird',
                        'pixelcopter',
                        'pong',
                        'puckworld',
                        'raycastmaze',
                        'snake',
                        'waterworld']
        if game == 'catcher':
            from ple.games.catcher import Catcher
            env = Catcher()
        elif game == 'monsterkong':
            from ple.games.monsterkong import MonsterKong
            env = MonsterKong()
        elif game == 'flappybird':
            from ple.games.flappybird import FlappyBird
            env = FlappyBird()
        elif game == 'pixelcopter':
            from ple.games.pixelcopter import Pixelcopter
            env = Pixelcopter()
        elif game == 'pong':
            from ple.games.pong import Pong
            env = Pong()
        elif game == 'puckworld':
            from ple.games.puckworld import PuckWorld
            env = PuckWorld()
        elif game == 'raycastmaze':
            from ple.games.raycastmaze import RaycastMaze
            env = RaycastMaze()
        elif game == 'snake':
            from ple.games.snake import Snake
            env = Snake()
        elif game == 'waterworld':
            from ple.games.waterworld import WaterWorld
            env = WaterWorld()

        self.p = PLE(env, fps=30, display_screen=display_screen)
        self.action_set = self.p.getActionSet()
        self.action_size = len(self.action_set)
        self.screen_dims = self.p.getScreenDims()
        self.p.init()

    def gray_scale(self, frame):
        gray_scale_frame = np.dot(frame, np.array([.299, .587, .114])).astype(np.uint8)
        assert gray_scale_frame.shape == frame.shape[:-1]
        return gray_scale_frame

    def get_screen(self):
        return np.transpose(self.gray_scale(self.p.getScreenRGB()))

    def reset(self):
        self.p.reset_game()
        state, _, done = self.step(-1)
        assert done == False
        return state
        #return self.get_screen()

    def step(self, action):
        reward = self.p.act(self.action_set[action])
        state_ = self.get_screen()
        done = self.p.game_over()

        return state_, reward, done
        

def main():
    '''
    for game in ['catcher', 'monsterkong', 'flappybird', 'pixelcopter', 'pong', 'puckworld', 'raycastmaze', 'snake', 'waterworld']:
        p = ple_wrapper(game)
    '''
    import copy
    game = 'flappybird'
    env = ple_wrapper(game, True)
    while True:
        state = env.reset()
        done = False
        while done == False:
            action = np.random.randint(env.action_size)
            state_, reward, done = env.step(action)
            state = copy.deepcopy(state_)

if __name__ == "__main__":
    main()
