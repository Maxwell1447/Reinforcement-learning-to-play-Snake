from env.data_game_env import DataEnv
from snake import *
from supervised.a_star import AStar
from utils import scalar_product, cross_product_z


class AStarEnv(DataEnv):

    def __init__(self, grid: Grid):
        super().__init__(grid)
        self.name = "a_star"
        self.path = []
        self.a_star = AStar(self.snake)

    def start(self):
        super().start()
        self.a_star = AStar(self.snake)

    # Override
    def choose_action(self):

        if len(self.path) == 0:
            self.path = self.a_star.find(self.apple[0], self.apple[1])

        if self.path is None or self.path == []:
            self.path = []
            return self.survive_action()

        action = self.translate_to_action(self.path.pop(0))

        if action == "None":
            self.path = []
            return "Left"

        return action

    def translate_to_action(self, coord):
        delta = [coord[0] - self.snake.head()[0], coord[1] - self.snake.head()[1]]
        dir_ = self.snake.direction

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
