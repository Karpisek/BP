import numpy as np


class Coordinates:
    def __init__(self, x, y, info=None):
        self.x = x
        self.y = y

        if info is not None:
            self.x *= info.width
            self.y *= info.height

    def distance(self, next_point):

        dx = next_point.x - self.x
        dy = next_point.y - self.y

        return np.sqrt(dx * dx + dy * dy)

    def update(self, x, y):
        self.x = x
        self.y = y

    def tuple(self) -> (int, int):
        return int(self.x), int(self.y)

    def __str__(self):
        return f"x:{self.x} y:{self.y}"
