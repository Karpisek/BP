from .line import Line, NoIntersectionError, SamePointError, ransac, NotOnLineError
import constants

from matplotlib import pyplot


class ParametersNotDefinedError(Exception):
    pass


class PcLines:
    """
    Implementation of parallel coordinate system.
    Allows to convert cartesian point into PC line and cartesian line to PC point.
    Implements straight and twisted space and transfer of coordinates between them suggested by Dubska et al.
    """

    def __init__(self, width):
        self.delta = width
        self.t_space = []
        self.s_space = []

    @property
    def count(self) -> float:
        """
        :return: count of points present in straight and twisted space
        """
        return len(self.t_space) + len(self.s_space)

    @property
    def s_points(self):
        """
        :return: points in straight space, if there are any points in twisted space - they are converted to
        straight space using equation suggested by Dubska et al.
        """

        s_points = [point[0] for point in self.s_space]

        for t_point in [point[0] for point in self.t_space]:
            u, v = t_point
            try:
                coefficient = (self.delta + 2 * u) / self.delta
                s_point = u / coefficient, v / coefficient
                s_points.append(s_point)
            except ZeroDivisionError:
                continue

        return s_points

    @property
    def t_points(self) -> [(float, float)]:
        """
        :return: points in twisted space, if there are any points in straight space - they are converted to
        twisted space using equation suggested by Dubska et al.
        """

        t_points = [point[0] for point in self.t_space]

        for s_point in [point[0] for point in self.s_space]:
            u, v = s_point
            try:
                coefficient = (self.delta - 2 * u)/self.delta
                t_point = u/coefficient, v/coefficient
                t_points.append(t_point)
            except ZeroDivisionError:
                continue

        return t_points

    def clear(self) -> None:
        """
        Clears point in both twisted and straight space
        """

        self.t_space = []
        self.s_space = []

    def find_most_lines_cross(self, write=False):
        """
        Finds intersection on lines given to PC space -> lines are represented as points so it uses
        RANSAC algorithm for line aproximation, this line is then converted to cartesian coordinate system.
        If the cross is near infinity an angle is returned instead.

        :param write: if debug info should be writen to file
        :return: detected line cross or angle if intersection near infinity
        """

        # pyplot.xlim((-2 * self.delta, 2 * self.delta))
        # pyplot.ylim((-2 * self.delta, 2 * self.delta))

        line, ratio = ransac(creator_points=self.s_points,
                             voting_points=self.s_points,
                             ransac_threshold=constants.CALIBRATOR_RANSAC_THRESHOLD_RATIO * self.delta,
                             write=write)

        # print(line)

        try:
            u1, v1 = line.find_coordinate(x=0)
            u2, v2 = line.find_coordinate(x=self.delta)

            # pyplot.plot([line.find_coordinate(x=-2 * self.delta)[0], line.find_coordinate(x=2 * self.delta)[0]], [line.find_coordinate(x=-2 * self.delta)[1], line.find_coordinate(x=2 * self.delta)[1]])
            # self.plot()

            # pyplot.show()

            return v1, v2

        except NotOnLineError:
            u1, v1 = line.find_coordinate(y=0)
            angle = (u1 + self.delta) * 180 / (2 * self.delta)

            # pyplot.plot([line.find_coordinate(y=-2 * self.delta)[0], line.find_coordinate(y=2 * self.delta)[0]], [line.find_coordinate(y=-2 * self.delta)[1], line.find_coordinate(y=2 * self.delta)[1]])
            # self.plot()
            #
            # pyplot.show()

            return angle, None

    def add_to_pc_space(self, point1=None, point2=None, line=None):
        """
        Adds line to parallel coordinate space. Line can be represented by two points or a single line.
        Tries to put them in straight space if possible. If there is not an option they are put in twisted space
        instead.

        :param point1: first point
        :param point2: second point
        :param line: line
        :raise SamePointError if given points are the same
        :raise ParametersNotDefined if not enough parameters are given. For example: only one point
        """

        if line is not None:
            point1 = line.find_coordinate(x=0)
            point2 = line.find_coordinate(x=1000)
        elif point1 is None and point2 is None:
            raise ParametersNotDefinedError

        x1, y1 = point1
        x2, y2 = point2

        try:
            magnitude = Line(point1, point2).magnitude
        except SamePointError:
            return

        l1_s = Line((0, x1), (self.delta, y1))
        l2_s = Line((0, x2), (self.delta, y2))

        l1_t = Line((-self.delta, -y1), (0, x1))
        l2_t = Line((-self.delta, -y2), (0, x2))

        try:
            u, v = l1_s.intersection(l2_s)
            self.s_space.append(((u, v), magnitude))
            return
        except NoIntersectionError:
            pass

        try:
            u, v = l1_t.intersection(l2_t)
            self.t_space.append(((u, v), magnitude))
            return
        except NoIntersectionError:
            pass

    def plot(self) -> None:
        """
        Plots PC space debug information using matplotlib
        """

        x_val = [x[0] for x in self.s_points]
        y_val = [x[1] for x in self.s_points]

        pyplot.plot(x_val, y_val, 'ro')

        x_val = [x[0][0] for x in self.s_space]
        y_val = [x[0][1] for x in self.s_space]

        pyplot.plot(x_val, y_val, 'bo')

        # x_val = [x[0][0] for x in self.t_space]
        # y_val = [x[0][1] for x in self.t_space]
        #
        # pyplot.plot(x_val, y_val, 'bo')
        pyplot.show()
