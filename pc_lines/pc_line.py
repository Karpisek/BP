import cv2
import numpy as np
from matplotlib import pyplot

from .line import Line, NoIntersectionError, SamePointError, ransac, NotOnLineError
import params


class ParametersNotDefinedError(Exception):
    pass


class PcLines:
    def __init__(self, width):
        self.delta = width
        self.t_space = []
        self.s_space = []

    @property
    def count(self) -> float:
        return len(self.t_space) + len(self.s_space)

    @property
    def s_points(self):
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
        self.t_space = []
        self.s_space = []

    def find_most_lines_cross(self):

        # pyplot.xlim((-2 * self.delta, 2 * self.delta))
        # pyplot.ylim((-2 * self.delta, 2 * self.delta))

        line, ratio = ransac(creator_points=self.s_points,
                             voting_points=self.s_points,
                             ransac_threshold=params.CALIBRATOR_RANSAC_THRESHOLD_RATIO * self.delta)

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

            # pyplot.show()

            return angle, None

    def pc_points(self, points=None, angles=None) -> [Line]:
        created_lines = []

        if points is not None:
            for point in points:
                x, y = point
                try:
                    created_lines.append(Line((self.delta, y), (0, x)))
                except SamePointError:
                    continue

        if angles is not None:
            for angle in angles:
                u = (2 * self.delta / 180) * angle
                try:
                    created_lines.append(Line((u, 0), (u, 10)))
                except SamePointError:
                    continue

        return created_lines

    def ransac_from_preset(self, preset_pc_points: [Line]) -> (object, int):
        # preset_x_coordinates, preset_y_coordinates = preset_points
        # preset_x_coordinates = [int(x * params.CALIBRATOR_GRID_DENSITY - info.width / 2) for x in range(int((2 * info.width) / params.CALIBRATOR_GRID_DENSITY))]
        # preset_y_coordinates = [int(y * params.CALIBRATOR_GRID_DENSITY - 9 * info.height / 10) for y in range(int(info.height / params.CALIBRATOR_GRID_DENSITY))]

        best_line_ratio = 0
        best_line = None

        # print(len(self.s_points))

        for line in preset_pc_points:
            num = 0
            ransac_threshold = self.delta * params.CALIBRATOR_RANSAC_THRESHOLD_RATIO

            for point in self.s_points:
                distance = line.point_distance(point)

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

    def add_to_pc_space(self, point1=None, point2=None, line=None):

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
            print("tady2")
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

        print("tady")

    def plot(self) -> None:
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

    def debug_spaces_print(self, line, text=None) -> None:
        image = np.zeros(shape=(2 * self.delta, 2 * self.delta, 3))

        cv2.line(image, (int(self.delta), 0), (int(self.delta), 2 * self.delta), (255, 255, 255), 1)
        cv2.line(image, (0, int(self.delta)), (2*self.delta, int(self.delta)), (255, 255, 255), 1)
        #
        # y1 = int(self.delta - line.find_coordinate(x=0)[1])
        # y2 = int(self.delta - line.find_coordinate(x=self.delta)[1])
        #
        # x1 = self.delta
        # x2 = 2 * self.delta
        #
        # #
        # # print(line)
        # # print((x1, y1), (x2, y2))
        # # cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), int(params.CALIBRATOR_RANSAC_THRESHOLD_RATIO * self.delta * 2))
        #
        # x1 = 0
        # x2 = self.delta

        point2 = int(self.delta + line.find_coordinate(x=0)[0]), int(self.delta - line.find_coordinate(x=0)[1])
        point1 = int(self.delta + line.find_coordinate(x=-self.delta)[0]), int(self.delta - line.find_coordinate(x=-self.delta)[1])

        cv2.line(image, point1, point2, (0, 0, 255), int(params.CALIBRATOR_RANSAC_THRESHOLD_RATIO * self.delta * 2))

        # for point in self.s_space:
        #     x, y = point[0]
        #
        #     x = self.delta + x
        #     y = self.delta - y
        #
        #     cv2.circle(image, (int(x), int(y)), 1, (255, 0, 0), 2)

        for point in self.t_space:
            x, y = point[0]

            x = self.delta + x
            y = self.delta - y

            cv2.circle(image, (int(x), int(y)), 1, (0, 255, 0), 2)

        if text is not None:
            cv2.imwrite(f"ransac_{str(text)}.jpg", image)
        else:
            cv2.imwrite("ransac.jpg", image)
