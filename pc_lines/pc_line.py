import cv2
import numpy as np
from matplotlib import pyplot

from pc_lines.vanishing_point import VanishingPoint
from .line import Line, NoIntersectionError, SamePointError
import params


class PcLines:
    def __init__(self, width):
        self.delta = width
        self.t_space = []
        self.s_space = []

    @property
    def count(self) -> float:
        return len(self.t_space) + len(self.s_space)

    def clear(self) -> None:
        self.t_space = []
        self.s_space = []

    def find_most_line_cross(self, preset_points=None) -> object:

        pyplot.xlim((-self.delta, self.delta))
        pyplot.ylim((-self.delta, self.delta))

        if preset_points is not None:
            s_line, s_ratio = self.ransac_from_preset(preset_points=preset_points)
        else:
            s_line, s_ratio = self.ransac()

        x = s_line.find_coordinate(x=0)[1]
        y = s_line.find_coordinate(x=self.delta)[1]

        vp_s = VanishingPoint(point=(int(np.round(x)), int(np.round(y))))

        pyplot.plot([0, self.delta], [x, y])
        pyplot.plot([-self.delta, 0], [-y, x])

        self.plot()
        pyplot.show()
        print(s_ratio)
        print(vp_s)
        self.debug_spaces_print(s_line)

        return vp_s

    def ransac_from_preset(self, preset_points) -> (object, int):
        preset_x_coordinates, preset_y_coordinates = preset_points
        # preset_x_coordinates = [int(x * params.CALIBRATOR_GRID_DENSITY - info.width / 2) for x in range(int((2 * info.width) / params.CALIBRATOR_GRID_DENSITY))]
        # preset_y_coordinates = [int(y * params.CALIBRATOR_GRID_DENSITY - 9 * info.height / 10) for y in range(int(info.height / params.CALIBRATOR_GRID_DENSITY))]

        best_line_ratio = 0
        best_line = None

        s_points = [point[0] for point in self.s_space]
        t_points = [point[0] for point in self.t_space]

        print(len(s_points) + len(t_points))

        for x in preset_x_coordinates:
            for y in preset_y_coordinates:
                try:
                    line = Line((self.delta, y), (0, x))
                except SamePointError:
                    continue

                num = 0
                ransac_threshold = self.delta * params.CALIBRATOR_RANSAC_THRESHOLD_RATIO

                for point in s_points:
                    distance = line.point_distance(point)

                    if distance < ransac_threshold:
                        num += 1

                point2 = line.find_coordinate(x=0)
                y = line.find_coordinate(x=self.delta)[1]
                twisted_line = Line((-self.delta, -y), point2)

                for point in t_points:
                    distance = twisted_line.point_distance(point)

                    if distance < ransac_threshold:
                        num += 1

                best_y = np.inf
                if best_line is not None:
                    best_y = best_line.find_coordinate(x=self.delta)[1]

                if num >= best_line_ratio:
                    if best_line is not None:
                        if num == best_line_ratio:
                            if line.find_coordinate(x=self.delta)[1] > best_y:
                                continue

                    best_line_ratio = num
                    best_line = line

        return best_line, best_line_ratio

    def ransac(self):

        best_line_ratio = 0
        best_line = None

        s_points = [point[0] for point in self.s_space]
        t_points = [point[0] for point in self.t_space]

        for point1 in s_points:
            for point2 in s_points:
                try:
                    line = Line(point1, point2)
                except SamePointError:
                    continue

                num = 0
                ransac_threshold = self.delta * params.CALIBRATOR_RANSAC_THRESHOLD_RATIO
                for point in s_points:
                    distance = line.point_distance(point)

                    if distance < ransac_threshold:
                        num += 1

                self.debug_spaces_print(line)
                if num > best_line_ratio:
                    best_line_ratio = num
                    best_line = line

        for point1 in t_points:
            for point2 in t_points:
                try:
                    line = Line(point1, point2)
                except SamePointError:
                    continue

                num = 0
                ransac_threshold = self.delta * params.CALIBRATOR_RANSAC_THRESHOLD_RATIO
                for point in t_points:
                    distance = line.point_distance(point)

                    if distance < ransac_threshold:
                        num += 1

                self.debug_spaces_print(line)
                if num > best_line_ratio:
                    best_line_ratio = num
                    best_line = line

        return best_line, best_line_ratio

    def pc_line_from_angle(self, angle):
        u = (2 * self.delta / 180) * angle
        print(angle, u)

    def pc_line_from_points(self, point1, point2):
        x1, y1 = point1
        x2, y2 = point2

        try:
            magnitude = Line(point1, point2).magnitude
        except SamePointError:
            return

        if magnitude < params.CALIBRATOR_FLOW_THRESHOLD:
            return

        l1_s = Line((0, x1), (self.delta, y1))
        l2_s = Line((0, x2), (self.delta, y2))

        l1_t = Line((-self.delta, -y1), (0, x1))
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
            if 0 > u > -self.delta:
                self.t_space.append(((u, v), magnitude))
            elif 0 < u < self.delta:
                self.s_space.append(((u, v), magnitude))

    def plot(self) -> None:
        x_val = [x[0][0] for x in self.s_space]
        y_val = [x[0][1] for x in self.s_space]

        pyplot.plot(x_val, y_val, 'ro')

        x_val = [x[0][0] for x in self.t_space]
        y_val = [x[0][1] for x in self.t_space]

        pyplot.plot(x_val, y_val, 'bo')
        pyplot.show()

    def debug_spaces_print(self, line) -> None:
        image = np.zeros(shape=(2 * self.delta, 2 * self.delta, 3))

        cv2.line(image, (int(self.delta), 0), (int(self.delta), 2 * self.delta), (255, 255, 255), 1)
        cv2.line(image, (0, int(self.delta)), (2*self.delta, int(self.delta)), (255, 255, 255), 1)

        y1 = int(self.delta - line.find_coordinate(x=0)[1])
        y2 = int(self.delta - line.find_coordinate(x=self.delta)[1])

        x1 = self.delta
        x2 = 2 * self.delta

        #
        # print(line)
        # print((x1, y1), (x2, y2))
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), int(params.CALIBRATOR_RANSAC_THRESHOLD_RATIO * self.delta * 2))

        x1 = 0
        x2 = self.delta

        y2 = int(self.delta + line.find_coordinate(x=self.delta)[1])

        cv2.line(image, (x1, y2), (x2, y1), (0, 0, 255), int(params.CALIBRATOR_RANSAC_THRESHOLD_RATIO * self.delta * 2))

        for point in self.s_space:
            x, y = point[0]

            x = self.delta + x
            y = self.delta - y

            cv2.circle(image, (int(x), int(y)), 1, (255, 0, 0), 2)

        for point in self.t_space:
            x, y = point[0]

            x = self.delta + x
            y = self.delta - y

            cv2.circle(image, (int(x), int(y)), 1, (0, 255, 0), 2)

        cv2.imwrite("ransac.jpg", image)
