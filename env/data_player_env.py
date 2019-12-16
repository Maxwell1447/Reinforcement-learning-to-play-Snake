import pygame as pyg
from pygame.locals import *
from env.player_env import PlayerEnv
from snake import *
import sys
import pandas as pd


class DataPlayerEnv(PlayerEnv):

    def __init__(self, grid: Grid):
        super().__init__(grid)
        self.tab = pd.DataFrame()

    def update(self, action: str):
        df = pd.DataFrame({'Headx': [self.snake.head()[0]], 'Heady': [self.snake.head()[1]], 'Applex': [self.apple[0]],
                           'Appley': [self.apple[1]]})
        for i in range(20):
            for j in range(20):
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

    def keyboard_action(self, i_dir):
        """
        Modifies the direction of the snake according to the key pressed by the human player
        :param i_dir: the direction chosen by the human player. This is a [dx, dy] like list.
        """
        dir_ = self.snake.direction
        if dir_[0] * i_dir[0] + dir_[1] * i_dir[1] == 0:
            if dir_[0] * i_dir[1] - dir_[1] * i_dir[0] < 0:
                self.update('Left')
                self.snake.turn_left()
            else:
                self.update('Right')
                self.snake.turn_right()

    def play(self, data_feeding=True):
        """
        function to be called to launch a game
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

            # Human events --> change the snake direction
            # Need to be replaced by the machine decision
            events = pyg.event.get()

            for event in events:
                if event.type == QUIT:
                    pyg.quit()
                    sys.exit("Quit game")

                if event.type == KEYDOWN:
                    if event.key == K_LEFT:
                        self.keyboard_action([-1, 0])
                        break
                    if event.key == K_RIGHT:
                        self.keyboard_action([1, 0])
                        break
                    if event.key == K_UP:
                        self.keyboard_action([0, -1])
                        break
                    if event.key == K_DOWN:
                        self.keyboard_action([0, 1])
                        break
            else:
                self.update('Forward')

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
        self.tab.iloc[:len(self.tab) - 10].to_csv('data.csv', mode='a', header=False)
        pyg.quit()

        return self.step_num, self.apple_score
