import numpy as np


class SamePointError(Exception):
    pass


class NoIntersectionError(Exception):
    pass


class Line:
    def __init__(self, point1, point2):
        
        if point1 == point2:
            raise SamePointError
        
        x1, y1 = point1
        x2, y2 = point2

        dx = x2 - x1
        dy = y2 - y1

        self.origin = [x1, y1]
        self.direction = [dx, dy]

        if self.direction[1] == 0:
            coef = (self.direction[1] / self.direction[0])
            self.a = -coef
            self.b = 1
            self.c = -(self.origin[1] - self.origin[0] * coef)

        else:
            coef = (self.direction[0] / self.direction[1])
            self.a = 1
            self.b = -coef
            self.c = -(self.origin[0] - self.origin[1] * coef)

    @property
    def horizontal(self) -> bool:
        return self.direction[1] == 0

    @property
    def vertical(self) -> bool:
        return self.direction[0] == 0

    @property
    def magnitude(self) -> float:
        return np.sqrt(self.direction[0] ** 2 + self.direction[1] ** 2)

    def angle(self, line2) -> int:
        if np.dot(self.direction, line2.direction) == 0:
            angle = 90.

        else:
            cos_alpha = np.dot(self.direction, line2.direction) / (self.magnitude * line2.magnitude)
            angle = np.degrees(np.arccos(cos_alpha))

        return int(angle)

    def parallel(self, line2) -> bool:
        return self.angle(line2) == 0 or self.angle(line2) == 180

    def intersection(self, line2) -> (float, float):
        if self.parallel(line2):
            raise NoIntersectionError

        a = np.array([
            [self.a, self.b],
            [line2.a, line2.b]
        ])

        b = np.array([
            [-self.c],
            [-line2.c]
        ])

        solved = np.linalg.solve(a, b)

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


