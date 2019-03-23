class VanishingPointError(Exception):
    pass


class VanishingPoint:
    def __init__(self, point=None, angle=None):

        if point is not None and angle is not None:
            raise VanishingPointError

        # in case vanishing point is defined
        if point is not None:
            self.point = point
            self.infinity = False

        # in case vanishing point is near infinity
        if angle is not None:
            self.angle = angle
            self.infinity = True

    def found(self):
        return False

    def __str__(self):
        return f"Vanishing Point - x: {self.point[0]} y: {self.point[1]}"

