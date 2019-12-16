

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

    def __init__(self, x, y, scale):
        self.x = x
        self.y = y
        self.scale = scale

    def is_in(self, x, y):
        return (0 <= x < self.x) and (0 <= y < self.y)


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

    def update_direction(self, action_number):
        if action_number == 1:
            self.turn_left()
        if action_number == 2:
            self.turn_right()

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
