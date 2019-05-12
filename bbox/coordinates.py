import numpy as np


class Coordinates:
    """
    Alternative representation of coordinates, relative coordinates are converted to real coordinates
    if InputInfo is defined
    """

    def __init__(self, x, y, info=None):
        """
        :param x: x coordinate
        :param y: y coordinate
        :param info: InputInfo instace - to convert coordinate from relative to real coordinates
        """

        self.x = x
        self.y = y

        if info is not None:
            self.x *= info.width
            self.y *= info.height

    def distance(self, next_point):
        """
        Calculates distance between two points

        :param next_point: point to mesure sistance
        :return: distance between two points
        """

        dx = next_point.x - self.x
        dy = next_point.y - self.y

        return np.sqrt(dx * dx + dy * dy)

    def update(self, x, y):
        """
        Updates the x and y coordinate

        :param x: new x coordinate
        :param y: new y coordinate
        """

        self.x = x
        self.y = y

    def tuple(self) -> (int, int):
        """
        Converts coordinate into int tuple representation

        :return: int tuple representation of coordinates
        """

        return int(self.x), int(self.y)

    def __str__(self):
        return f"x:{self.x} y:{self.y}"
