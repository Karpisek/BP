import cv2
import numpy as np


class SamePointError(Exception):
    pass


class NoIntersectionError(Exception):
    pass


class NotOnLineError(Exception):
    pass


class Line:
    def __init__(self, point1, point2=None, direction=None):

        if point1 == point2:
            raise SamePointError

        if point2 is None and direction is None:
            raise SamePointError
        
        x1, y1 = point1

        if point2 is not None:
            x2, y2 = point2

            dx = x2 - x1
            dy = y2 - y1

        else:
            dx, dy = direction

        self._origin = [x1, y1]
        self._direction = [dx, dy]

        if self.direction[1] == 0:
            coef = (self.direction[1] / self.direction[0])
            self.a = round(-coef, 3)
            self.b = 1
            self.c = round(-(self.origin[1] - self.origin[0] * coef), 3)

        else:
            coef = (self.direction[0] / self.direction[1])
            self.a = 1
            self.b = round(-coef, 3)
            self.c = round(-(self.origin[0] - self.origin[1] * coef), 3)

    @property
    def origin(self) -> [float, float]:
        return self._origin

    @property
    def direction(self) -> [float, float]:
        return self._direction

    @property
    def horizontal(self) -> bool:
        return self.direction[1] == 0

    @property
    def vertical(self) -> bool:
        return self.direction[0] == 0

    @property
    def magnitude(self) -> float:
        return np.sqrt(self.direction[0] ** 2 + self.direction[1] ** 2)

    def normal_direction(self) -> (float, float):
        return self.a, self.b

    def angle(self, line2) -> float:
        if np.dot(self.direction, line2.direction) == 0:
            angle = 90.

        else:
            cos_alpha = np.dot(self.direction, line2.direction) / (self.magnitude * line2.magnitude)

            if cos_alpha > 1:
                cos_alpha = 1

            elif cos_alpha < -1:
                cos_alpha = -1

            angle = np.degrees(np.arccos(cos_alpha))

        return angle

    def parallel(self, line2) -> bool:
        return self.angle(line2) == 0 or self.angle(line2) == 180

    def intersection(self, line2) -> (float, float):
        a = np.array([
            [self.a, self.b],
            [line2.a, line2.b]
        ])

        b = np.array([
            [-self.c],
            [-line2.c]
        ])

        try:
            solved = np.linalg.solve(a, b)
        except np.linalg.LinAlgError:
            raise NoIntersectionError

        return solved[0][0], solved[1][0]

    def line_distance(self, line) -> float:
        point = self.origin[0], self.origin[1]
        return line.point_distance(point)

    def point_distance(self, point) -> float:
        x, y = point

        return np.abs(self.a * x + self.b * y + self.c)/np.sqrt(self.a ** 2 + self.b ** 2)

    def general_equation(self)-> (float, float, float):
        # ax + by + c = 0
        return self.a, self.b, self.c

    def on_line(self, point) -> bool:
        x, y = point

        return self.a * x + self.b * y + self.c == 0

    def find_coordinate(self, x=None, y=None) -> (float, float):
        if y is None and x is None:
            return None

        if x is not None:
            if self.b == 0:
                return -self.c/self.a, x
            return x, ((-self.a) * x - self.c) / self.b
        else:
            if self.a == 0:
                return y, -self.c/self.b
            return ((-self.b) * y - self.c) / self.a, y

    def draw(self, image, color, thickness) -> None:
        h, _, _ = image.shape

        p1 = [int(cord) for cord in self.find_coordinate(y=0)]
        p2 = [int(cord) for cord in self.find_coordinate(y=h)]

        cv2.line(image, tuple(p1), tuple(p2), color, thickness)

    def __str__(self):
        return f'{self.a}x + {self.b}y + {self.c} = 0'
