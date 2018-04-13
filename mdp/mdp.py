import numpy as np

class mdp:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols

        self.pos = [0, 0]

        self.toint = {'left':0, 'right':2, 'up':1, 'down':3}
        self.tostring = {0:'left', 2:'right', 1:'up', 3:'down'}

    def step(self, action):
        assert action in ['left', 'right', 'up', 'down']
        action = self.toint[action]
        rand = np.random.uniform()

        if rand < .1:
            action -= 1
        elif rand >= .1 and rand < .2:
            action += 1
        action %= 4

        action = self.tostring[action]

        if action == 'left':
            self.pos[1] -= 1
        elif action == 'right':
            self.pos[1] += 1
        elif action == 'up':
            self.pos[0] -= 1
        elif action == 'down':
            self.pos[0] += 1

        self.pos[0] = max(0, self.pos[0])
        self.pos[0] = min(self.rows-1, self.pos[0])
        self.pos[1] = max(0, self.pos[1])
        self.pos[1] = min(self.cols-1, self.pos[1])

        print self.pos





def main():
    env = mdp(100, 100)

    env.step('left')
    env.step('right')
    env.step('right')
    env.step('right')
    env.step('right')
    env.step('down')
    env.step('down')
    env.step('down')
    env.step('down')
    env.step('down')
    env.step('up')
    
if __name__ == '__main__':
    main()
