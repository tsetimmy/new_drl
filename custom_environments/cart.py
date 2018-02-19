#!/usr/bin/env python

from Box2D import b2
import math
import numpy as np

import pygame
from pygame.locals import K_RIGHT, K_LEFT, KEYDOWN, KEYUP

import pickle

import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

class Cart:
    def __init__(self, pixelFeature=True, maxTimeSteps=400, render=True, screenSize=36):
        self.pixelFeature = pixelFeature
        self.maxTimeSteps = maxTimeSteps
        self.render = render
        self.actionSpace = 'discrete'
        self.numActions = 3

        self.timeStepCounter = 0

        self.trackWidth = 5.0
        self.cartWidth = 0.3
        self.cartHeight = 0.2
        self.cartMass = 0.5
        self.force = .2
        self.trackThickness = self.cartHeight

        #Reward zone - start and length
        self.rewardZoneStart = -1.6
        #self.rewardZoneStart = np.random.uniform(-2.5+.6, 2.5-.6)
        self.rewardZoneLength = .6
        self.rewardZoneColor = (204, 0, 0)
        self.rewardZoneCategory = 0x0002

        #self.screenSize = (128*3, 96*3) #origin upper left
        #self.screenSize = (32, 32)
        self.screenSize = (screenSize, screenSize)
        self.worldSize = (float(self.trackWidth), float(self.trackWidth)) #origin at center

        self.world = b2.world(gravity=(0, -10), doSleep=True)
        self.framesPerSecond = 30 # used for dynamics update and for graphics update
        self.velocityIterations = 8
        self.positionIterations = 6

        #Make track bodies and fixtures
        self.trackColor = (100, 100, 100)

        #Make the track
        f = b2.fixtureDef(shape=b2.polygonShape(box=(self.trackWidth/2,self.trackThickness/2)),
                          friction=0.001, categoryBits=0x0001, maskBits=(0xFFFF & ~self.rewardZoneCategory))
        self.track = self.world.CreateStaticBody(position = (0,0), 
                                                 fixtures=f, userData={'color': self.trackColor})

        #Make the reward zone
        f = b2.fixtureDef(shape=b2.polygonShape(box=(self.rewardZoneLength/2,self.trackThickness/2)),
                          friction=0.001, categoryBits=self.rewardZoneCategory)
        self.rewardZone = self.world.CreateStaticBody(position = (self.rewardZoneStart,0), 
                                                 fixtures=f, userData={'color': self.rewardZoneColor})

        #Make the left and right wall
        f = b2.fixtureDef(shape=b2.polygonShape(box=(self.trackThickness/2,self.trackThickness/2)),
                          friction=0.001, categoryBits=0x0001, maskBits=(0xFFFF & ~self.rewardZoneCategory))
        self.wallLeft = self.world.CreateStaticBody(position = (-self.trackWidth/2+self.trackThickness/2, self.trackThickness),
                                               fixtures=f, userData={'color': self.trackColor})
        self.wallRight = self.world.CreateStaticBody(position = (self.trackWidth/2-self.trackThickness/2, self.trackThickness),
                                                fixtures=f,userData={'color': self.trackColor})

        # Make cart body and fixture
        f = b2.fixtureDef(shape=b2.polygonShape(box=(self.cartWidth/2,self.cartHeight/2)),
                          density=self.cartMass, friction=0.001, restitution=0.5, categoryBits=0x0001, maskBits=(0xFFFF & ~self.rewardZoneCategory))
        self.cart = self.world.CreateDynamicBody(position=(0,self.trackThickness), fixtures=f, userData={'color':(20,200,0)}) 

        self.initDisplay()
        self.draw()

    def draw(self):
        # Clear screen
        self.screen.fill((250, 250, 250))

        # Draw other bodies
        for body in (self.track,self.rewardZone,self.wallLeft,self.wallRight,self.cart): # or world.bodies
            for fixture in body.fixtures:
                shape = fixture.shape
                # Assume polygon shapes!!!
                vertices = [self.w2p((body.transform * v)) for v in shape.vertices]
                pygame.draw.polygon(self.screen, body.userData['color'], vertices)

        pygame.display.flip()
        self.clock.tick(self.framesPerSecond)

    def act(self, action):
        """CartPole.act(action): action is -1, 0 or 1"""
        self.action = action
        f = (self.force*action, 0)
        p = self.cart.GetWorldPoint(localPoint=(0.0, self.cartHeight/2))
        self.cart.ApplyForce(f, p, True)
        timeStep = 1.0/self.framesPerSecond
        self.world.Step(timeStep, self.velocityIterations, self.positionIterations)
        self.world.ClearForces()

        currentState = self.getCurrentState()
        #get the reward
        reward = 0.
        if self.cart.position[0] <= self.rewardZoneStart + self.rewardZoneLength/2 and\
            self.cart.position[0] >= self.rewardZoneStart - self.rewardZoneLength/2 and\
            abs(self.cart.linearVelocity[0]) <= .8:
            reward = 1.
        #check if we're done
        done = False
        self.timeStepCounter += 1
        if self.timeStepCounter >= self.maxTimeSteps:
            done = True
        if self.render:
            self.draw()
        return currentState, reward, done

    def getCurrentState(self):
        if self.pixelFeature == True:
            string_image = pygame.image.tostring(self.screen, 'RGB')
            temp_surf = pygame.image.fromstring(string_image,self.screenSize,'RGB' )
            tmp_arr = pygame.surfarray.array3d(temp_surf)
            tmp_gray = np.dot(tmp_arr, np.array([.299, .587, .114])).astype(np.uint8)
            return np.transpose(tmp_gray)
        else:
            return np.asarray([self.cart.position[0],\
                self.cart.linearVelocity[0]])

    def initDisplay(self):
        self.screen = pygame.display.set_mode(self.screenSize, 0, 32)
        pygame.display.set_caption('Cart')
        self.clock = pygame.time.Clock()

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
    cart = Cart(pixelFeature=True, render=True)
    #print cart.getCurrentState()

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

        # Apply action to cart simulation
        state, reward, done = cart.act(action)

        #print state, reward, done
        # Redraw cart in new state
        cart.draw()

    pygame.quit()

if __name__ == "__main__":
    run()
