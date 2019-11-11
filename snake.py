import random as rd
import pygame as pyg
import numpy as np
from pygame.locals import *
import sys


def list_sum(a, b):
    """
    :param a: list
    :param b: list
    :return: the term by term sum of a and b
    """
    return [x+y for x, y in zip(a, b)]


class Grid:
    """
    Just the info of the size of the ground
    x -> width
    y -> height
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y


class Snake:
    """
    This class represents the behaviour of the snake
    """

    def __init__(self, grid_):
        """
        grid -> ground
        direction -> current direction as a [dx, dy] list
        body -> list of coordinates corresponding to the body of the snake
            body[0] is the tail
            body[-1] is the head
        """
        self.grid = grid_
        self.direction = [1, 0]
        self.body = [[grid_.x//2, grid_.y//2]]  # starts in the middle of the grid

    def head(self):
        return self.body[-1]

    def turn_right(self):
        self.direction[1] *= -1
        self.direction.reverse()

    def turn_left(self):
        self.direction[0] *= -1
        self.direction.reverse()

    def move(self):
        """
        move the head snake in the current direction
        + make the body move to follow
        """
        self.body.append(self.next_box())
        self.body.pop(0)

    def next_box(self):
        """
        :return: the box in front of the head according to the current direction
        """
        return list_sum(self.head(), self.direction)

    def grow(self):
        """
        increments the length of the snake.
        needs to be called when an apple is eaten
        """
        tail = self.body[0]
        self.body.insert(0, tail)  # duplicates the tail

    def check_death(self):
        """
        :return: whether the snake dies or not
        """
        head = self.head()
        if head in self.body[:-1]:  # the head collides with the rest of the body
            return True

        if head[0] < 0 or head[1] < 0 \
                or head[0] >= self.grid.x or head[1] >= self.grid.y:  # out of the edge of the grid
            return True

        return False

    def length(self):
        """
        :return: the length of the body
        """
        return len(self.body)


def action(snake, i_dir):
    """
    Modifies the direction of the snake according to the key pressed by the human player
    :param snake: the snake
    :param i_dir: the direction chosen by the human player. This is a [dx, dy] like list.
    """
    dir_ = snake.direction
    if dir_[0] * i_dir[0] + dir_[1] * i_dir[1] == 0:
        if dir_[0] * i_dir[1] - dir_[1] * i_dir[0] < 0:
            snake.turn_left()
        else:
            snake.turn_right()


def apple_spawn(grid_, snake):
    """
    Choose an empty place to spawn the new apple
    :return: the choice
    """
    choice = [rd.randint(0, grid_.x-1), rd.randint(0, grid_.y-1)]
    while choice in snake.body:
        choice = [rd.randint(0, grid_.x - 1), rd.randint(0, grid_.y - 1)]
    return choice


def draw(snake, apple, scale, screen):
    """
    Draw
    :param snake: snake that must be drawn
    :param apple: apple that must be drawn
    :param scale: scale of a box compared to a pixel
    :param screen: the entire surface to be drawn
    :return:
    """

    screen.fill((20, 20, 20))  # Overlay the screen with a black-gray surface
    apple_color = (230, 0, 0)
    head_color = (0, 200, 0)
    body_color = (20, 20, 200)

    # draw the apple
    pyg.draw.rect(screen, apple_color, [scale * apple[0], scale * apple[1], scale - 1, scale - 1], 0)

    # draw the body (except the head)
    for box in snake.body[:-1]:
        pyg.draw.rect(screen, body_color, [scale * box[0], scale * box[1], scale - 1, scale - 1], 0)

    head = snake.head()
    # draw the head
    pyg.draw.rect(screen, head_color, [scale * head[0], scale * head[1], scale - 1, scale - 1], 0)

    # updates the screen --> not necessary for the reinforcement learning
    pyg.display.flip()


def play():
    """
    function to be called to launch a game as a human
    """

    grid = Grid(40, 30)  # define the dimensions of the grid
    snake = Snake(grid)  # create the snake
    apple = apple_spawn(grid, snake)  # spawn the apple

    scale = 20
    width = grid.x*scale
    height = grid.y*scale
    screen_size = (width, height)

    clock = pyg.time.Clock()

    pyg.init()
    screen = pyg.display.set_mode(screen_size)
    pyg.display.set_caption("Snake")

    draw(snake, apple, scale, screen)

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
                    action(snake, [-1, 0])
                    break
                if event.key == K_RIGHT:
                    action(snake, [1, 0])
                    break
                if event.key == K_UP:
                    action(snake, [0, -1])
                    break
                if event.key == K_DOWN:
                    action(snake, [0, 1])
                    break

        if snake.next_box() == apple:
            # We need to make the snake grow before moving to the apple
            # Otherwise the growth appears with a delay on the screen
            # This is due to the implementation of the growth --> see Snake.grow()
            snake.grow()
            apple = apple_spawn(grid, snake)  # spawn a new apple

        snake.move()

        if snake.check_death():  # if it dies, we need to go outside
            break

        # update the graphic elements
        draw(snake, apple, scale, screen)

        # Possibility to get the image on the screen
        pxl_array = pyg.PixelArray(screen)
        # values in hex
        # head = 1,315,860
        # body = 1,316,040
        # empty = 51,200
        np.array(pxl_array)  # can be converted to numpy.array


play()
