import os

from env.game_env import GameEnv
from pygame.locals import *
import pygame as pyg
import pandas as pd
from snake import *
import sys


class DataEnv(GameEnv):

    def __init__(self, grid: Grid):
        super().__init__(grid)
        self.tab = pd.DataFrame()
        self.FPS = -1
        self.name = None

    def start(self):
        super().start()
        self.pygame_init()

    def update_tab(self, action: str):
        df = pd.DataFrame({'Headx': [self.snake.head()[0]], 'Heady': [self.snake.head()[1]], 'Applex': [self.apple[0]],
                           'Appley': [self.apple[1]]})
        for i in range(self.grid.x):
            for j in range(self.grid.y):
                df[str(i) + "&" + str(j)] = pd.Series(0)
        for (x, y) in self.snake.body:
            df[str(x) + "&" + str(y)] = pd.Series(1)
        df['x+'] = pd.Series(max(self.snake.direction[0], 0))
        df['x-'] = pd.Series(max(-self.snake.direction[0], 0))
        df['y+'] = pd.Series(max(self.snake.direction[1], 0))
        df['y-'] = pd.Series(max(-self.snake.direction[1], 0))
        if action == 'Right':
            df['Action'] = pd.Series(0)
        if action == 'Left':
            df['Action'] = pd.Series(1)
        if action == 'Forward':
            df['Action'] = pd.Series(2)
        self.tab = self.tab.append(df)

    def act(self, action: str, update=True):
        if update:
            self.update_tab(action)
        if action == "Right":
            self.snake.turn_right()
        elif action == "Left":
            self.snake.turn_left()
        elif action == "Forward":
            pass
        else:
            raise ValueError("action has to be 'Right', 'Left' or 'Forward'")

    def predict_death(self, action: str):
        initial_direction = self.snake.direction.copy()
        self.act(action, False)
        prediction = self.snake.occupied_box(self.snake.next_box())
        self.snake.direction = initial_direction
        return prediction

    def choose_action(self):
        """abstract method to be implemented"""

        raise NotImplemented("please implement choose action")

    def survive_action(self):
        """choose respectively Left, Forward and Right action in this order until one allows to survive"""

        initial_dir = self.snake.direction.copy()
        for action in ["Left", "Forward", "Right"]:
            self.snake.direction = initial_dir.copy()
            self.act(action)
            if not self.snake.occupied_box(self.snake.next_box()):
                break

        self.snake.direction = initial_dir.copy()
        return action

    def play(self, data_feeding=True, wait=True):
        """ lauch snake played by a path_finder """

        self.tab = pd.DataFrame()
        
        self.start()

        clock = pyg.time.Clock()

        while wait:

            for event in pyg.event.get():
                if event.type == QUIT:
                    pyg.quit()
                    sys.exit("Quit game")

                if event.type == KEYDOWN:
                    wait = False

        while True:
            for event in pyg.event.get():
                if event.type == QUIT:
                    pyg.quit()
                    sys.exit("Quit game")

            if self.FPS > 0:
                clock.tick(self.FPS)  # FPS --> speed of the game for a human user

            # path  finder
            action = self.choose_action()
            self.act(action, data_feeding)

            if self.snake.next_box() == self.apple:
                # We need to make the snake grow before moving to the apple
                # Otherwise the growth appears with a delay on the screen
                # This is due to the implementation of the growth --> see Snake.grow()
                self.snake.grow()
                self.snake.move()
                self.step_num += 1
                self.apple = self.apple_spawn()  # spawn a new apple
            else:
                self.snake.move()
                self.step_num += 1

            if self.snake.check_death():  # if it dies, we need to go outside
                break

            # update the graphic elements
            self.draw()
        print("Game Over")
        path = './data/data_{}_{}.csv'.format(self.name, self.grid.x)
        header = not os.path.exists(path)
        #self.tab.iloc[:len(self.tab) - 10].to_csv(path, mode='a', header=header)
        pyg.quit()

        return self.step_num, self.apple_score

