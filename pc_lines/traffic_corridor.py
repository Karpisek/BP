import cv2
import numpy as np

import params
from pc_lines.line import Line, SamePointError
from pc_lines.pc_line import PcLines


class TrafficCorridorRepository:
    def __init__(self, info):
        self._corridors_found = False
        self._stopline_found = False

        self._info = info
        self._corridors = {}
        self._corridors_count = 0
        self._corridor_mask = np.zeros(shape=(info.height, info.width), dtype=np.uint8)
        self._stop_line = None

        self._stop_places = []

        self._pc_lines = PcLines(info.width)

    @property
    def count(self):
        return self._corridors_count

    @property
    def ready(self):
        return self._corridors_found and self._stopline_found

    @property
    def corridor_mask(self):
        return self._corridor_mask

    def red_line_crossed(self, coordinates):
        if not self.ready:
            return False

        red_line_point = self._stop_line.find_coordinate(x=coordinates.x)
        return red_line_point[1] > coordinates.y

    def get_corridor(self, coordinates) -> int:
        if 0 < coordinates.x < self._info.width and 0 < coordinates.y < self._info.height:
            return self._corridor_mask[int(coordinates.y)][int(coordinates.x)] - 1
        else:
            return -1

    def get_mask(self, fill) -> np.ndarray:

        image = np.zeros(shape=(self._info.height, self._info.width, 3),
                         dtype=np.uint8)

        if self._corridors_found:
            for key, corridor in self._corridors.items():
                corridor.draw_corridor(image=image,
                                       info=self._info,
                                       color=params.RANDOM_COLOR[key],
                                       fill=fill,
                                       thickness=params.CORRIDORS_VISUALIZATION_THICKNESS)

        image = cv2.bitwise_and(image, image, mask=self._corridor_mask)

        if self._stopline_found:
            self._stop_line.draw(image=image,
                                 color=params.COLOR_RED,
                                 thickness=5)

        return image

    def create_new_corridor(self, left_point, right_point):
        self._corridors_count += 1
        new_corridor = TrafficCorridor(index=self._corridors_count,
                                       left_point=left_point,
                                       right_point=right_point)

        self._corridors[new_corridor.id] = new_corridor

        new_corridor.draw_corridor(image=self._corridor_mask,
                                   info=self._info,
                                   color=self._corridors_count)

    def find_corridors(self, lifelines_mask):

        # extract left-bot-right 1px border around lifeline image and make "1D array"
        left_edge = lifelines_mask[:, :1]
        bottom_edge = lifelines_mask[-1:, :].transpose(1, 0, 2)
        right_edge = np.flip(lifelines_mask[:, -1:])
        frame_edge = np.concatenate((left_edge, bottom_edge, right_edge))

        # greyscale image and reduce noise by multiple blur and threshold
        edge_grey = cv2.cvtColor(frame_edge, cv2.COLOR_RGB2GRAY)

        edge_grey_blured = cv2.blur(edge_grey, (1, 21))
        _, threshold = cv2.threshold(edge_grey_blured, 50, 255, cv2.THRESH_BINARY)

        # edge_grey_blured = cv2.blur(threshold, (1, 31))
        # _, edge_grey_sharp = cv2.threshold(edge_grey_blured, 20, 255, cv2.THRESH_BINARY)

        height, width = edge_grey_blured.shape

        edge_grey_canny = cv2.Canny(threshold, 50, 150)

        coordinates = []

        # border image with 0
        edge_grey_canny[0][0] = 0
        edge_grey_canny[-1][0] = 0

        for i in range(height):
            if edge_grey_canny[i][0] == 255:
                coordinates.append(i)

        points = np.array(coordinates).reshape(-1, 2)

        for index, point in enumerate(points):
            try:
                half_diff = int((points[index + 1][0] - point[1]) / 2)
            except IndexError:
                break

            point[1] += half_diff
            points[index + 1][0] -= half_diff

        for point in points:
            points = []

            for coordinate in list(point):
                if coordinate < self._info.height:
                    points.append((0, coordinate))
                elif coordinate < self._info.height + self._info.width:
                    points.append((coordinate - self._info.height, self._info.height - 1))
                else:
                    points.append((self._info.width - 1, self._info.width + 2 * self._info.height - coordinate))

            self.create_new_corridor(tuple(points[0]), tuple(points[1]))

        # cv2.imwrite("mask.jpg", self.get_mask())
        # cv2.imwrite("lifeline.jpg", frame_edge)
        # cv2.imwrite("lifeline_before.jpg", lifelines_mask)

        self._corridor_mask = cv2.bitwise_and(self._corridor_mask, self._corridor_mask, mask=self._info.update_area.mask())
        self._corridors_found = True

    def add_stop_point(self, coordinates):
        self._stop_places.append(coordinates)

        if len(self._stop_places) > params.CORRIDORS_STOP_POINTS_MINIMAL and not self._stopline_found:
            self.find_stop_line()

    def find_stop_line(self):

        best_line_ratio = 0
        best_line = None

        vp2 = self._info.vanishing_points[1]

        for point2 in self._stop_places:
            try:
                line = Line(vp2.point, point2.tuple())
            except SamePointError:
                continue

            num = 0
            ransac_threshold = params.CORRIDORS_RANSAC_THRESHOLD
            for point in self._stop_places:
                distance = line.point_distance(point.tuple())

                if distance < ransac_threshold:
                    num += 1

            # self.debug_spaces_print(line)
            if num > best_line_ratio:
                best_line_ratio = num
                best_line = line

        self._stop_line = best_line
        self._stopline_found = True


class TrafficCorridor:
    def __init__(self, index, left_point, right_point):
        self._id = index
        self.left_point = left_point
        self.right_point = right_point

        # print(left_point, right_point)

    @property
    def id(self):
        return self._id

    def draw_corridor(self, image, info, color=None, fill=True, thickness=params.DEFAULT_THICKNESS):
        mask = np.zeros(shape=(info.height, info.width),
                        dtype=np.uint8)

        if color is None:
            color = self.id

        cv2.line(img=image,
                 pt1=self.left_point,
                 pt2=info.vanishing_points[0].point,
                 color=color,
                 thickness=thickness)

        cv2.line(img=mask,
                 pt1=self.left_point,
                 pt2=info.vanishing_points[0].point,
                 color=color,
                 thickness=thickness)

        cv2.line(img=image,
                 pt1=self.right_point,
                 pt2=info.vanishing_points[0].point,
                 color=color,
                 thickness=thickness)

        cv2.line(img=mask,
                 pt1=self.right_point,
                 pt2=info.vanishing_points[0].point,
                 color=color,
                 thickness=thickness)

        if fill:
            middle_point = int((self.left_point[0] + self.right_point[0]) / 2), int((self.left_point[1] + self.right_point[1]) / 2)

            mask_with_border = np.pad(mask, 1, 'constant', constant_values=255)
            cv2.floodFill(image=image,
                          mask=mask_with_border,
                          seedPoint=middle_point,
                          newVal=color)

    def __str__(self):
        return f"coridor: id [{self.id}], coord ({self.left_point}, {self.right_point})"
