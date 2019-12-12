# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 13:16:34 2019

@author: Arnau
"""
from sklearn.linear_model import LogisticRegression
from game_env import GameEnv
from pygame.locals import *
import pygame as pyg
from snake import *
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

class ai_classification(GameEnv):
    
    def __init__(self, grid: Grid, logreg: LogisticRegression):
        super().__init__(grid)
        self.logreg = logreg
        
    
    def start(self):
        self.snake: Snake = Snake(self.grid)
        self.apple = self.apple_spawn()
        pyg.init()
        self.screen = pyg.display.set_mode((self.grid.x * self.grid.scale, self.grid.y * self.grid.scale))
        pyg.display.set_caption("Snake")
        self.draw()
    
    def state(self):
        df = pd.DataFrame({'Headx': [self.snake.head()[0]], 'Heady': [self.snake.head()[1]], 'Applex' : [self.apple[0]], 'Appley' : [self.apple[1]]})
        
        #comment to not consider the body
        '''
        for i in range(20):
            for j in range(20):
                df[str(i)+"&"+str(j)] = pd.Series(0)
        for (x,y) in self.snake.body:
            df[str(x)+"&"+str(y)] = pd.Series(1)
        '''
        
        df['x+'] = pd.Series(max(self.snake.direction[0],0))
        df['x-'] = pd.Series(max(-self.snake.direction[0],0))
        df['y+'] = pd.Series(max(self.snake.direction[1],0))
        df['y-'] = pd.Series(max(-self.snake.direction[1],0))
        
        
        #uncomment to not use polynomial features

        X_cols = df.copy()
        X = X_cols.values
        X = X.reshape(len(X_cols),-1)
    
        #To add the dummy x_0 and potentially features’ high-order
        poly = PolynomialFeatures(2)  
        X = poly.fit_transform(X)
        df = pd.DataFrame(X)
        
    
        return df
    
    def act(self, action : int):
        if action==0:
            self.snake.turn_right()
        elif action == 1:
            self.snake.turn_left()
        elif action == 2:
            pass
        else:
            raise ValueError
        
    def play(self):
        """
            function to be called to launch a game as a human
            """

        self.start()

        clock = pyg.time.Clock()

        starting = False
        while not starting:

            for event in pyg.event.get():
                if event.type == QUIT:
                    pyg.quit()
                    sys.exit("Quit game")

                if event.type == KEYDOWN:
                    starting = True

        while True:

            clock.tick(10)  # FPS --> speed of the game for a human user

            df = self.state()
            action = self.logreg.predict([df.iloc[0]])
            self.act(action[0])

            if self.snake.next_box() == self.apple:
                # We need to make the snake grow before moving to the apple
                # Otherwise the growth appears with a delay on the screen
                # This is due to the implementation of the growth --> see Snake.grow()
                self.snake.grow()
                self.apple = self.apple_spawn()  # spawn a new apple

            self.snake.move()

            if self.snake.check_death():  # if it dies, we need to go outside
                break

            # update the graphic elements
            self.draw()
        print("Game Over")
        pyg.quit()

'''
ai_class = ai_classification(Grid(20,20,20), clf)
ai_class.play()
'''