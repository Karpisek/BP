import cv2
import numpy as np
import sys

import params


class VanishingPointError(Exception):
    pass


class VanishingPoint:
    def __init__(self, point=None, angle=None):

        print("VP:", point, angle)

        if point is None and angle is None:
            raise VanishingPointError

        x, y = point
        self._point = np.clip(x, params.UINT_MIN, params.UINT_MAX), np.clip(y, params.UINT_MIN, params.UINT_MAX)
        self.angle = angle

        # in case vanishing point is defined
        if point is not None:
            self.infinity = False

        # in case vanishing point is near infinity
        if angle is not None:
            self.infinity = True

    @property
    def point(self):
        if self._point is not None:
            return self._point
        else:
            raise VanishingPointError

    def found(self):
        return self._point is not None or self.angle is not None

    def __str__(self):
        return f"Vanishing Point - x: {self._point[0]} y: {self._point[1]}"

