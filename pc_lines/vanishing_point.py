import numpy as np

import params
from bbox import Coordinates
from pc_lines.line import Line


class VanishingPointError(Exception):
    pass


class VanishingPoint:
    """
    Class representing vanishing points.
    In most cases it is a point in scene (or outside the scene)
    if vanishing point is detected near infinity angle is stored instead.
    """

    def __init__(self, point=None, direction=None):
        """
        :param point: point to store
        :param direction: direction to store (if VP near infinity)
        :raise VanishingPointError if no parameter is passed
        """

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
        """
        :return: Vanishing point
        :raise VanishingPointError if vanishing point is defined by angle rather then point
        """

        if not self.infinity:
            return self._point
        else:
            raise VanishingPointError

    @property
    def coordinates(self):
        """
        :return: Coordinates of vanishing point
        :raise VanishingPointError if vanishing point is defined by angle rather then point
        """

        return Coordinates(*self.point)

    @property
    def direction(self):
        """
        :return: direction to vanishing point (used when vanishing point near infinity)
        :raise VanishingPointError if vanishing point is defined by point
        """

        if not self.infinity:
            raise VanishingPointError
        else:
            return self._direction

    def draw_line(self, image, point, color, thickness):
        """
        Helper function to dra line from point to this vanishing point. Works in both cases - when vanishing point
        is defined by a point or when defined by an angle

        :param image: selected image to draw on
        :param point: selected point as origin
        :param color: color of line
        :param thickness: thickness of line
        """

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

    def serialize(self):
        """
        Serializes vanishing point

        :return: serialized vanishing point in form of dictionary
        """

        try:
            return {"point": self.point, "direction": None}
        except VanishingPointError:
            return {"point": None, "direction": self.direction}
