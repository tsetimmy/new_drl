#!/usr/bin/env python

from Box2D import b2
import math

import pygame
from pygame.locals import K_RIGHT, K_LEFT, KEYDOWN, KEYUP

import numpy as np

# original code from here (please use README for details):
# http://www.cs.colostate.edu/~anderson/cs645/index.html/lib/exe/fetch.php?media=notes:cartpole.py
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"


class CartPole:
    def __init__(self, pixelFeature=True, maxTimeSteps=400, render=True, screenSize=84):
        self.pixelFeature = pixelFeature
        self.max_timesteps = maxTimeSteps
        self.render = render
        self.numActions = 3

        self.timestep_count = 0
        self.trackWidth = 5.0
        self.cartWidth = 0.3
        self.cartHeight = 0.2
        self.cartMass = 0.5
        self.poleMass = 0.1
        self.force = .2
        self.trackThickness = self.cartHeight
        self.poleLength = self.cartHeight*6
        self.poleThickness = 0.04

        #self.screenSize = (640,480) #origin upper left
        #self.screenSize = (128,96) #origin upper left
        self.screenSize = (screenSize, screenSize)
        self.worldSize = (float(self.trackWidth),float(self.trackWidth)) #origin at center

        self.world = b2.world(gravity=(0,-10),doSleep=True)
        self.framesPerSecond = 30 # used for dynamics update and for graphics update
        self.velocityIterations = 8
        self.positionIterations = 6

        # Make track bodies and fixtures
        self.trackColor = (100,100,100)
        poleCategory = 0x0002

        f = b2.fixtureDef(shape=b2.polygonShape(box=(self.trackWidth/2,self.trackThickness/2)),
                          friction=0.001, categoryBits=0x0001, maskBits=(0xFFFF & ~poleCategory))
        self.track = self.world.CreateStaticBody(position = (0,0), 
                                                 fixtures=f, userData={'color': self.trackColor})
        self.trackTop = self.world.CreateStaticBody(position = (0,self.trackThickness+self.cartHeight*1.1),
                                                    fixtures = f)

        f = b2.fixtureDef(shape=b2.polygonShape(box=(self.trackThickness/2,self.trackThickness/2)),
                          friction=0.001, categoryBits=0x0001, maskBits=(0xFFFF & ~poleCategory))
        self.wallLeft = self.world.CreateStaticBody(position = (-self.trackWidth/2+self.trackThickness/2, self.trackThickness),
                                               fixtures=f, userData={'color': self.trackColor})
        self.wallRight = self.world.CreateStaticBody(position = (self.trackWidth/2-self.trackThickness/2, self.trackThickness),
                                                fixtures=f,userData={'color': self.trackColor})

        # Make cart body and fixture
        f = b2.fixtureDef(shape=b2.polygonShape(box=(self.cartWidth/2,self.cartHeight/2)),
                          density=self.cartMass, friction=0.001, restitution=0.5, categoryBits=0x0001,
                          maskBits=(0xFFFF & ~poleCategory))
        self.cart = self.world.CreateDynamicBody(position=(0,self.trackThickness),
                                            fixtures=f, userData={'color':(20,200,0)})

        # Make pole pody and fixture
        # Initially pole is hanging down, which defines the zero angle.
        f = b2.fixtureDef(shape=b2.polygonShape(box=(self.poleThickness/2, self.poleLength/2)),
                          density=self.poleMass, categoryBits=poleCategory)
        self.pole = self.world.CreateDynamicBody(position=(0,self.trackThickness+self.cartHeight/2+self.poleThickness-self.poleLength/2),
                                            fixtures=f, userData={'color':(200,20,0)})

        # Make pole-cart joint
        self.world.CreateRevoluteJoint(bodyA = self.pole, bodyB = self.cart,
                                  anchor=(0,self.trackThickness+self.cartHeight/2 + self.poleThickness))

        self.initDisplay()
        self.draw()

    def getCurrentState(self):
        if self.pixelFeature == True:
            string_image = pygame.image.tostring(self.screen, 'RGB')
            temp_surf = pygame.image.fromstring(string_image,self.screenSize,'RGB' )
            tmp_arr = pygame.surfarray.array3d(temp_surf)
            tmp_gray = np.dot(tmp_arr, np.array([.299, .587, .114])).astype(np.uint8)
            return np.transpose(tmp_gray)
        else:
            return np.asarray([self.cart.position[0],
                self.cart.linearVelocity[0],
                self.pole.angle,
                self.pole.angularVelocity])

    def act(self,action):

        """CartPole.act(action): action is -1, 0 or 1"""
        self.action = action
        f = (self.force*action, 0)
        p = self.cart.GetWorldPoint(localPoint=(0.0, self.cartHeight/2))
        self.cart.ApplyForce(f, p, True)
        timeStep = 1.0/self.framesPerSecond
        self.world.Step(timeStep, self.velocityIterations, self.positionIterations)
        self.world.ClearForces()

        currentState = self.getCurrentState()
        angle = self.pole.angle
        if angle > 0.:
            while not (-np.pi <= angle <= np.pi):
                angle -= 2 * np.pi
        else:
            while not (-np.pi <= angle <= np.pi):
                angle += 2 * np.pi
        if np.abs(angle) > .75 * np.pi:
            reward = 1.
        elif np.abs(angle) < .25 * np.pi:
            reward = -1.
        else:
            reward = 0.

        done = False
        self.timestep_count += 1
        if self.timestep_count == self.max_timesteps:
            done = True
        if self.render:
            self.draw()
        return currentState, reward, done

    def initDisplay(self):

        self.screen = pygame.display.set_mode(self.screenSize, 0, 32)
        pygame.display.set_caption('Cart')
        self.clock = pygame.time.Clock()
        
    def draw(self):
        # Clear screen
        self.screen.fill((250,250,250))
        # Draw circle for joint. Do before cart, so will appear as half circle.
        jointCoord = self.w2p(self.cart.GetWorldPoint((0,self.cartHeight/2)))
        junk,radius = self.dw2dp((0,2*self.poleThickness))
        pygame.draw.circle(self.screen, self.cart.userData['color'], jointCoord, radius, 0)
        # Draw other bodies
        for body in (self.track,self.wallLeft,self.wallRight,self.cart,self.pole): # or world.bodies
            for fixture in body.fixtures:
                shape = fixture.shape
                # Assume polygon shapes!!!
                vertices = [self.w2p((body.transform * v)) for v in shape.vertices]
                pygame.draw.polygon(self.screen, body.userData['color'], vertices)

        #print self.cart.position,self.cart.linearVelocity,self.pole.angle,self.pole.angularVelocity
        # Draw arrow showing force    
        '''
        if self.action != 0:
            cartCenter = self.w2p(self.cart.GetWorldPoint((0,0)))
            arrowEnd = (cartCenter[0]+self.action*20, cartCenter[1])
            pygame.draw.line(self.screen, (250,250,0), cartCenter, arrowEnd, 3)
            pygame.draw.line(self.screen, (250,250,0), arrowEnd,
                             (arrowEnd[0]-self.action*5, arrowEnd[1]+5), 3)
            pygame.draw.line(self.screen, (250,250,0), arrowEnd,
                             (arrowEnd[0]-self.action*5, arrowEnd[1]-5), 3)
        '''

        pygame.display.flip()
        self.clock.tick(self.framesPerSecond)

    # Now print the position and angle of the body.

    def w2p(self,(x,y)):
        """ Convert world coordinates to screen (pixel) coordinates"""
        return (int(0.5+(x+self.worldSize[0]/2) / self.worldSize[0] * self.screenSize[0]),
                int(0.5+self.screenSize[1] - (y+self.worldSize[1]/2) / self.worldSize[1] * self.screenSize[1]))
    def p2w(self,(x,y)):
        """ Convert screen (pixel) coordinates to world coordinates"""
        return (x / self.screenSize[0] * self.worldSize[0] - self.worldSize[0]/2,
                (self.screenSize[1]-y) / self.screenSize[1] * self.worldSize[1] - self.worldSize[1]/2)
    def dw2dp(self,(dx,dy)):
        """ Convert delta world coordinates to delta screen (pixel) coordinates"""
        return (int(0.5+dx/self.worldSize[0] * self.screenSize[0]),
                int(0.5+dy/self.worldSize[1] * self.screenSize[1]))


def run():
    #def run(screen_width=400., screen_height=400.):
    cartpole = CartPole()

    action = 0
    running = True
    reps = 0
    while running:
        reps += 1

        # Set action to -1, 1, or 0 by pressing lef or right arrow or nothing.
        for event in pygame.event.get():
            if event.type == pygame.QUIT: 
                running = False
            elif event.type == KEYDOWN:
                if event.key == K_RIGHT:
                    action = 1
                elif event.key == K_LEFT:
                    action = -1
            elif event.type == KEYUP:
                if event.key == K_RIGHT or event.key == K_LEFT:
                    action = 0

        # Apply action to cartpole simulation
        cartpole.act(action)
        # Redraw cartpole in new state
        cartpole.draw()

    pygame.quit()

if __name__ == "__main__":
    run()
