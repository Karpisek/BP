import numpy as np

import params
from bbox import Coordinates
from pc_lines.line import Line


class VanishingPointError(Exception):
    pass


class VanishingPoint:
    def __init__(self, point=None, direction=None):

        print("VP:", point)
        if point is None and direction is None:
            raise VanishingPointError

        # in case vanishing point is defined
        if point is not None:
            self.infinity = False
            x, y = point
            self._point = int(np.clip(x, params.UINT_MIN, params.UINT_MAX)), int(np.clip(y, params.UINT_MIN, params.UINT_MAX))

        # in case vanishing point is near infinity
        if direction is not None:
            self.infinity = True
            self._direction = direction

    @property
    def point(self):
        if not self.infinity:
            return self._point
        else:
            raise VanishingPointError

    @property
    def coordinates(self):
        return Coordinates(*self.point)

    @property
    def direction(self):
        if not self.infinity:
            raise VanishingPointError
        else:
            return self._direction

    def draw_line(self, image, point, color, thickness):
        if self.infinity:
            line = Line(point1=point,
                        direction=self.direction)
        else:
            line = Line(point1=point,
                        point2=self.point)

        line.draw(image=image,
                  color=color,
                  thickness=thickness)

    def __str__(self):
        return f"Vanishing Point - x: {self._point[0]} y: {self._point[1]}"

