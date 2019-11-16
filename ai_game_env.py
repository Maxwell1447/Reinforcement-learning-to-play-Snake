from gym import Env
from gym import spaces
from game_env import GameEnv
from snake import *
import pygame as pyg
import numpy as np


DEATH_REWARD = -100
APPLE_REWARD = 50
STEP_REWARD = -1


class IAGameEnv(Env, GameEnv):
    """Custom Environment that follows gym interface"""

    metadata = {'render.modes': ['human']}

    def __init__(self, grid: Grid):

        Env.__init__(self)
        GameEnv.__init__(self, grid)

        # Define action and observation space
        # They must be gym.spaces objects
        # Using 3 discrete actions:
        self.action_space = spaces.Discrete(3)
        # Using image as input:
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(grid.x * grid.scale, grid.y * grid.scale, 3), dtype=np.uint8)
        self.reset()

    def step(self, action_number):
        self.snake.update_direction(action_number)

        if self.snake.check_death():
            return self.observation(), DEATH_REWARD, True, None

        if self.snake.next_box() == self.apple:
            # We need to make the snake grow before moving to the apple
            # Otherwise the growth appears with a delay on the screen
            # This is due to the implementation of the growth --> see Snake.grow()
            self.snake.grow()
            self.apple = self.apple_spawn()  # spawn a new apple
            self.snake.move()
            self.render()
            return self.observation(), APPLE_REWARD, False, None

        self.snake.move()
        self.render()
        return self.observation(), STEP_REWARD, False, None

    def reset(self):
        self.start()

    def render(self, mode='human', close=False):
        self.draw()

    def observation(self):
        pxl_array = np.array(pyg.PixelArray(self.screen))
        return self.hex_to_rgb(pxl_array).T.astype(np.uint8)

    @staticmethod
    def hex_to_rgb(value):
        return np.array([value >> 16, (value >> 8) & 0xff, value & 0xff])