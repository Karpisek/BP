import numpy as np
from matplotlib import pyplot

from pc_lines.vanishing_point import VanishingPoint
from .line import Line, NoIntersectionError, SamePointError, NotOnLineError
import params


class PcLines:
    def __init__(self, width):
        self.delta = width
        self.t_space = []
        self.s_space = []

    def find_most_line_cross(self):
        s_line, s_ratio = self.ransac(self.s_space)
        t_line, t_ratio = self.ransac(self.t_space)

        try:
            point2 = s_line.find_coordinate(x=0)
            point3 = s_line.find_coordinate(x=self.delta)
        except NotOnLineError:
            point2 = (s_line.b, 1000)
            point3 = (s_line.b, -1000)

        # plt.xlim((-self._delta, self._delta))
        # plt.ylim((-900, 900))

        self.plot()
        print(point2, point3)
        # plt.plot([point2[0], point3[0]], [point2[1], point3[1]])

        try:
            point2 = t_line.find_coordinate(x=-self.delta)
            point3 = t_line.find_coordinate(x=0)
        except NotOnLineError:
            point2 = (t_line.b, 1000)
            point3 = (t_line.b, -1000)

        # plt.plot([point2[0], point3[0]], [point2[1], point3[1]])

        # plt.show()

        if s_ratio > t_ratio:
            try:
                x = s_line.find_coordinate(x=0)[1]
                y = s_line.find_coordinate(x=self.delta)[1]

                vp = VanishingPoint(point=(int(np.round(x)), int(np.round(y))))

            except NotOnLineError:
                x = s_line.b

                distance_from_zero = np.abs(x) / self.delta
                angle = 90 - (90 * distance_from_zero)

                print(angle)

                vp = VanishingPoint(angle=angle)
                print("infinity in S")

            self.plot()

        else:
            try:
                x = t_line.find_coordinate(x=0)[1]
                y = -t_line.find_coordinate(x=-self.delta)[1]

                vp = VanishingPoint(point=(int(np.round(x)), int(np.round(y))))
            except NotOnLineError:

                x = t_line.b

                distance_from_zero = np.abs(x) / self.delta
                angle = 90 - (90 * distance_from_zero)

                print(-angle)

                vp = VanishingPoint(angle=-angle)
                print("infinity in T")
            self.plot()

        print(s_ratio, t_ratio)
        print(vp)
        return vp

    def ransac(self, points_with_magnitude):
        best_line_ratio = 0
        best_line = None

        best_rated_points = self.find_best_rated(points_with_magnitude)
        all_points = [point[0] for point in points_with_magnitude]

        for point1 in best_rated_points:
            for point2 in best_rated_points:

                # np.random.shuffle(points)
                # [point1, point2] = points[:2]
                # testing_points = points[2:]

                try:
                    line = Line(point1, point2)
                except SamePointError:
                    continue

                participate = []
                num = 0

                ransac_threshold = self.delta * params.CALIBRATOR_RANSAC_THRESHOLD_RATIO
                for point in all_points:
                    distance = line.point_distance(point)

                    if distance < ransac_threshold:
                        participate.append(point)
                        num += 1

                if num > best_line_ratio:
                    best_line_ratio = num
                    best_line = line

        return best_line, best_line_ratio

    def pc_line_from_points(self, point1, point2):
        x1, y1 = point1
        x2, y2 = point2

        magnitude = Line(point1, point2).magnitude

        if magnitude < params.CALIBRATOR_FLOW_THRESHOLD:
            return

        l1_s = Line((0, x1), (self.delta, y1))
        l1_t = Line((-self.delta, -y1), (0, x1))

        l2_s = Line((0, x2), (self.delta, y2))
        l2_t = Line((-self.delta, -y2), (0, x2))

        u = None
        v = None

        try:
            u, v = l1_s.intersection(l2_s)
        except NoIntersectionError:
            pass

        try:
            u, v = l1_t.intersection(l2_t)
        except NoIntersectionError:
            pass

        if u is not None and v is not None:
            if u < 0:
                self.t_space.append(((u, v), magnitude))
            else:
                self.s_space.append(((u, v), magnitude))

    def plot(self):
        x_val = [x[0][0] for x in self.s_space]
        y_val = [x[0][1] for x in self.s_space]

        pyplot.plot(x_val, y_val, 'ro')

        x_val = [x[0][0] for x in self.t_space]
        y_val = [x[0][1] for x in self.t_space]

        pyplot.plot(x_val, y_val, 'ro')
        pyplot.show()

    @staticmethod
    def find_best_rated(points_with_magnitude):
        points_with_magnitude.sort(key=lambda p: p[1], reverse=True)
        ordered_space = [point[0] for point in points_with_magnitude[:params.CALIBRATOR_RANSAC_STEP_POINTS_COUNT]]

        return ordered_space
