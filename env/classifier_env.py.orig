from env.game_env import GameEnv
from pygame.locals import *

from snake import *
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import pygame as pyg
import sys


class ClassifierEnv(GameEnv):

    def __init__(self, grid: Grid, clf, all_data: bool, poly_features: bool):
        super().__init__(grid)
        self.clf = clf
        self.all_data = all_data
        self.poly_features = poly_features

    def start(self):
        self.snake: Snake = Snake(self.grid)
        self.apple = self.apple_spawn()
        pyg.init()
        self.screen = pyg.display.set_mode((self.grid.x * self.grid.scale, self.grid.y * self.grid.scale))
        pyg.display.set_caption("Snake")
        self.draw()

    def state(self):
        df = pd.DataFrame({'Headx': [self.snake.head()[0]], 'Heady': [self.snake.head()[1]], 'Applex': [self.apple[0]],
                           'Appley': [self.apple[1]]})

        if self.all_data:
            for i in range(self.grid.x):
                for j in range(self.grid.y):
                    df[str(i) + "&" + str(j)] = pd.Series(0)
            for (x, y) in self.snake.body:
                df[str(x) + "&" + str(y)] = pd.Series(1)

        df['x+'] = pd.Series(max(self.snake.direction[0], 0))
        df['x-'] = pd.Series(max(-self.snake.direction[0], 0))
        df['y+'] = pd.Series(max(self.snake.direction[1], 0))
        df['y-'] = pd.Series(max(-self.snake.direction[1], 0))

        if self.poly_features:
            X_cols = df.copy()
            X = X_cols.values
            X = X.reshape(len(X_cols), -1)
            # To add the dummy x_0 and features’ high-order
            poly = PolynomialFeatures(2)
            X = poly.fit_transform(X)
            df = pd.DataFrame(X)

        return df

    def act(self, action: int):
        if action == 0:
            self.snake.turn_right()
        elif action == 1:
            self.snake.turn_left()
        elif action == 2:
            pass
        else:
            raise ValueError

    def play(self, wait=True):
        """
        function to be called to launch a game
        """

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

            clock.tick(10)  # FPS --> speed of the game

            df = self.state()
            action = self.clf.predict([df.iloc[0]])
            self.act(action[0])

            if self.snake.next_box() == self.apple:
                # We need to make the snake grow before moving to the apple
                # Otherwise the growth appears with a delay on the screen
                # This is due to the implementation of the growth --> see Snake.grow()
                self.snake.grow()
                self.apple = self.apple_spawn()  # spawn a new apple

            self.snake.move()
            self.step_num += 1

            if self.snake.check_death():  # if it dies, we need to go outside
                break

            self.step_num += 1
            # update the graphic elements
            self.draw()
        print("Game Over")
        pyg.quit()

        return self.step_num, self.apple_score
