from gym import Env
from gym import spaces
from env.game_env import GameEnv
from snake import *
import pygame as pyg
from pygame.locals import *
import sys
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
        self.FPS = -1
        self.clock = None

        # Define action and observation space
        # They must be gym.spaces objects
        # Using 3 discrete actions:
        self.action_space = spaces.Discrete(3)
        # Using image as input:
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(grid.x * grid.scale, grid.y * grid.scale, 3), dtype=np.uint8)
        self.reset()

    def step(self, action_number):
        events = pyg.event.get()
        for event in events:
            if event.type == QUIT:
                pyg.quit()
                sys.exit("Quit game")
        if self.FPS > 0:
            self.clock.tick(self.FPS)

        self.snake.update_direction(action_number)

        if self.snake.check_death():
            return self.observation(), DEATH_REWARD, True, {}

        if self.snake.next_box() == self.apple:
            # We need to make the snake grow before moving to the apple
            # Otherwise the growth appears with a delay on the screen
            # This is due to the implementation of the growth --> see Snake.grow()
            self.snake.grow()
            self.apple = self.apple_spawn()  # spawn a new apple
            self.snake.move()
            self.render()
            return self.observation(), APPLE_REWARD, False, {}

        self.snake.move()
        self.render()
        return self.observation(), STEP_REWARD, False, {}

    def reset(self):
        self.start()
        return self.observation()

    def render(self, mode='human', close=False):
        self.draw()

    def observation_(self):

        pxl_array = np.array(pyg.PixelArray(self.screen), dtype=np.uint8)
        # return self.hex_to_rgb(pxl_array)
        return np.stack(self.hex_to_rgb(pxl_array), axis=2)

    def observation(self):

        state = np.zeros(self.grid.shape())

        for box in self.snake.body:
            try:
                state[box[0], box[1]] = 256 * 2 / 3
            except IndexError:
                pass

        head = self.snake.head()
        try:
            state[head[0], head[1]] = 255
        except IndexError:
            pass

        state[self.apple[0], self.apple[1]] = 256 * 1 / 3

        return state

    def set_fps(self, fps: int):
        if fps > 0:
            if self.FPS < 0:
                self.clock = pyg.time.Clock()
            self.FPS = fps

    @staticmethod
    def hex_to_rgb(value):
        return np.array([value >> 16, (value >> 8) & 0xff, value & 0xff])
