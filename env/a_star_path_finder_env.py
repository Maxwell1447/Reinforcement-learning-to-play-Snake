from env.data_game_env import DataEnv
from snake import *
from supervised.a_star import AStar
from utils import scalar_product, cross_product_z


class AStarEnv(DataEnv):

    def __init__(self, grid: Grid):
        super().__init__(grid)
        self.name = "a_star"
        self.path = []

    # Override
    def choose_action(self):

        print(self.snake.head())

        if len(self.path) == 0:
            a_star = AStar(self.snake)
            self.path = a_star.find(self.apple[0], self.apple[1])
            del a_star
            print(self.path)

        if self.path is None:
            self.path = []
            return self.survive_action()

        try:
            action = self.translate_to_action(self.path.pop(0))
        except IndexError:
            print(self.apple, self.snake.head())
            raise IndexError()

        if action == "None":
            self.path = []
            return "Left"

        print("action = ", action)

        return action

    def translate_to_action(self, coord):
        delta = [coord[0] - self.snake.head()[0], coord[1] - self.snake.head()[1]]
        dir_ = self.snake.direction

        print(delta, dir_)

        if scalar_product(dir_, delta) < 0:
            return "None"
        elif scalar_product(dir_, delta) > 0:
            return "Forward"
        elif cross_product_z(dir_, delta) > 0:
            return "Right"
        elif cross_product_z(dir_, delta) < 0:
            return "Left"
        else:
            raise ValueError("Impossible translation")
