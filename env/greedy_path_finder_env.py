from env.data_game_env import DataEnv
from snake import *
from utils import cross_product_z, scalar_product


class GreedyEnv(DataEnv):

    def __init__(self, grid: Grid):
        super().__init__(grid)
        self.name = "greedy"

    def overall_direction_action(self):
        dir_ = self.snake.direction
        delta = [self.apple[0] - self.snake.head()[0], self.apple[1] - self.snake.head()[1]]

        if cross_product_z(dir_, delta) > 0:
            return "Right"
        if cross_product_z(dir_, delta) < 0:
            return "Left"
        if scalar_product(dir_, delta) > 0:
            return "Forward"
        if scalar_product(dir_, delta) < 0:
            return "None"

        raise ValueError()

    def choose_action(self):

        action = self.overall_direction_action()

        if action == "None" or self.predict_death(action):
            action = self.survive_action()

        return action
