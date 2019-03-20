import numpy as np


class Coordinates:
    def __init__(self, x, y):
        self.x = x
        self.y = y

        self.is_relative = True

    def convert_to_fixed(self, info):
        if self.is_relative:
            self.x *= info.width
            self.y *= info.height

            self.is_relative = False

    def distance(self, next_point):

        dx = next_point.x - self.x
        dy = next_point.y - self.y

        return np.sqrt(dx * dx + dy * dy)

    def update(self, x, y):
        self.x = x
        self.y = y

    def tuple(self):
        return self.x, self.y

    @staticmethod
    def fixed_to_relative(coordinate, dimension):
        return coordinate / dimension

    @staticmethod
    def relative_to_fixed(coordinate, dimension):
        return coordinate * dimension

    def get_fixed(self, info):
        return self.x * info.width, self.y * info.height

    def __str__(self):
        return f"x:{self.x} y:{self.y}"
