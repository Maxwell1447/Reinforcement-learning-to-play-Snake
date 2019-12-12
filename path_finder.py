from game_env import GameEnv
from pygame.locals import *
import pygame as pyg
from snake import *
import numpy as np
import pandas as pd



class PathFinder(GameEnv):
    
     
    def __init__(self, grid: Grid):
        super().__init__(grid)
        self.tab = pd.DataFrame()
        self.FPS = -1
         
    def update_tab(self, action: str):
        df = pd.DataFrame({'Headx': [self.snake.head()[0]], 'Heady': [self.snake.head()[1]], 'Applex' : [self.apple[0]], 'Appley' : [self.apple[1]]})
        for i in range(20):
            for j in range(20):
                df[str(i)+"&"+str(j)] = pd.Series(0)
        for (x,y) in self.snake.body:
            df[str(x)+"&"+str(y)] = pd.Series(1)
        df['x+'] = pd.Series(max(self.snake.direction[0],0))
        df['x-'] = pd.Series(max(-self.snake.direction[0],0))
        df['y+'] = pd.Series(max(self.snake.direction[1],0))
        df['y-'] = pd.Series(max(-self.snake.direction[1],0))
        if action=='Right':
            df['Action'] = pd.Series(0)
        if action=='Left':
            df['Action'] = pd.Series(1)
        if action=='Forward':
            df['Action'] = pd.Series(2)
        self.tab = self.tab.append(df)
        
    def act(self, action : int, count=0):
        dir_ = self.snake.direction[:]
        if count > 2:
            return 0
        if action==0:
            dir_[1] *= -1
            dir_.reverse()
            x_next = self.snake.head()[0] + dir_[0]
            y_next = self.snake.head()[1] + dir_[1]
            if [x_next,y_next] in self.snake.body:
                self.act(1,count = count+1)
            else:
                print("Right")
                self.update_tab('Right')
                self.snake.turn_right()
        elif action == 1:
            dir_[0] *= -1
            dir_.reverse()
            x_next = self.snake.head()[0] + dir_[0]
            y_next = self.snake.head()[1] + dir_[1]
            if [x_next,y_next] in self.snake.body:
                self.act(2,count = count+1)             
            else:
                print("Left")
                self.update_tab('Left')
                self.snake.turn_left()
        elif action == 2:
            x_next = self.snake.head()[0] + dir_[0]
            y_next = self.snake.head()[1] + dir_[1]
            if [x_next,y_next] in self.snake.body:
                self.act(0,count = count+1)
            else:
                print("Forward")
                self.update_tab('Forward')
        else:
            raise ValueError
    
    def update_path(self):
        dx = self.apple[0] - self.snake.head()[0]
        dy = self.apple[1] - self.snake.head()[1]
        dir_ = self.snake.direction[:]
        #print("dx = ",dx,"dy = ",dy,"dir = ",dir_)
        
        # right = 0
        # left = 1
        
        #snake is going up
        if dir_[1] < 0:
        #apple is on the right, turn right
            if dx > 0:
                return 0
            #apple is on the left, turn left
            if dx < 0:
                return 1
            #the apple is not in front of him
            if dy > 0:
                return 0
            
        #snake is going down
        if dir_[1] > 0:
            #apple is on the right, turn left
            if dx > 0:
               return 1
            #apple is on the left, turn right
            if dx < 0:
                return 0
            #the apple is not in front of him
            if dy < 0:
                return 1
            
        #snake is going to the right
        if dir_[0] > 0:
            #apple is above, turn left
            if dy <  0:
                return 1
            #apple is below, turn right
            if dy > 0:
                return 0
            #the apple is not in front of him
            if dx < 0:
                return 0
            
        #snake is going to the left
        if dir_[0] < 0:
            #apple is above, turn right
            if dy < 0:
                return 0
            #apple is below, turn left
            if dy > 0:
                return 1
            #the apple is not in front of him
            if dx > 0:
                return 1
        
        #the apple is in front of him
        return 2
        
            
        
        
    
    def start(self):
        self.snake: Snake = Snake(self.grid)
        self.apple = self.apple_spawn()
        pyg.init()
        self.screen = pyg.display.set_mode((self.grid.x * self.grid.scale, self.grid.y * self.grid.scale))
        pyg.display.set_caption("Snake")
        self.draw()
        
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

            clock.tick(60)  # FPS --> speed of the game for a human user

            # path  finder
            action = self.update_path()
            self.act(action)

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
        self.tab.iloc[:len(self.tab)-10].to_csv('data.csv', mode='a',header=False)
        pyg.quit()
        
        
        
pf = PathFinder(Grid(20,20,20))
pf.play()