import pygame as pyg
from pygame.locals import *
from game_env import GameEnv
from snake import *
import sys


class PlayerEnv(GameEnv):

    def __init__(self, grid: Grid):
        super().__init__(grid)

    def start(self):
        self.snake: Snake = Snake(self.grid)
        self.apple = self.apple_spawn()
        pyg.init()
        self.screen = pyg.display.set_mode((self.grid.x * self.grid.scale, self.grid.y * self.grid.scale))
        pyg.display.set_caption("Snake")
        self.draw()

    def keyboard_action(self, i_dir):
        """
            Modifies the direction of the snake according to the key pressed by the human player
            :param i_dir: the direction chosen by the human player. This is a [dx, dy] like list.
            """
        dir_ = self.snake.direction
        if dir_[0] * i_dir[0] + dir_[1] * i_dir[1] == 0:
            if dir_[0] * i_dir[1] - dir_[1] * i_dir[0] < 0:
                self.snake.turn_left()
            else:
                self.snake.turn_right()

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

            # Human events --> change the snake direction
            # Need to be replaced by the machine decision
            for event in pyg.event.get():
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
