from snake import *
from math import sqrt


class Node:
    """
    implements a node in the graph corresponding to the box at position (x, y)
    self.g is the number of steps necessary to reach this box from the starting point
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.g = 0
        self.parent = None

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __str__(self):
        return "Node: x={}  y={}\n".format(self.x, self.y)


class AStar:
    """
    Implement the A* algorithm for path finding applied for snake

    """

    def __init__(self, snake: Snake):
        self.snake = snake
        self.open_list = []
        self.closed_list = []
        self.start = None
        self.end = None
        self.nodes = [[Node(x, y) for y in range(self.snake.grid.y)] for x in range(self.snake.grid.x)]

    """
    :return the list of reachable neighbours of the given node
    """
    def give_neighbours(self, node: Node):
        neighbours = []
        x = node.x
        y = node.y
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            if self.snake.grid.is_in(x+dx, y+dy):
                if not [x+dx, y+dy] in self.snake.body:
                    neighbours.append((x+dx, y+dy))

        return neighbours

    """
    :param x_end   x position of the goal point
    :param y_end   y position of the goal point
    
    :return a list of coordinates to reach the goal according to the A* algorithm
    """
    def find(self, x_end, y_end):
        self.start = self.nodes[self.snake.head()[0]][self.snake.head()[1]]
        self.end = self.nodes[x_end][y_end]

        self.open_list.append(self.start)

        while len(self.open_list) > 0:
            current_node = self.get_min_f_node()
            self.closed_list.append(current_node)

            if current_node == self.end:
                path = []
                current = current_node
                while current is not None:
                    # print(current.g, " + ", self.h(current), " = ", self.f(current))
                    path.append((current.x, current.y))
                    current = current.parent
                return path[::-1]

            for x, y in self.give_neighbours(current_node):
                child = self.nodes[x][y]

                if child in self.closed_list:
                    continue

                child.g = current_node.g + 1

                for open_node in self.open_list:
                    if child == open_node and child.g > open_node.g:
                        continue

                child.parent = current_node
                self.open_list.append(child)

    """
    :return the Euclidean distance to the goal point
    """
    def h(self, node: Node):
        return sqrt((node.x - self.end.x)**2 + (node.y - self.end.y)**2)

    """
    :return the greedy parameter f to be minimized according to the rule F = G + H
    """
    def f(self, node: Node):
        assert self.snake.grid.is_in(node.x, node.y)
        return node.g + self.h(node)

    """
    :return the node in the open list minimizing f
    """
    def get_min_f_node(self):
        j = 0
        for i in range(len(self.open_list)):
            if self.f(self.open_list[j]) > self.f(self.open_list[i]):
                j = i

        return self.open_list.pop(j)


def swap(l, a, b):
    temp = l[a]
    l[a] = l[b]
    l[b] = temp
