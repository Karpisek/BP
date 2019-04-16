import cv2
import numpy as np

import params
from pc_lines.line import Line, SamePointError
from pc_lines.pc_line import PcLines
from pc_lines.vanishing_point import VanishingPoint


def _mouse_callback(event, x, y, _, param):
    corridor_maker = param

    if event == cv2.EVENT_LBUTTONDOWN:
        corridor_maker.click(point=(x, y))

    corridor_maker.move(point=(x, y))


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

    @property
    def count(self):
        return self._corridors_count

    @property
    def corridors_found(self):
        return self._corridors_found

    @property
    def corridor_ids(self):
        return self._corridors.keys()

    @property
    def stopline_found(self):
        return self._stopline_found

    @property
    def ready(self):
        return self.corridors_found and self._stopline_found

    @property
    def corridor_mask(self):
        return self._corridor_mask

    def select_manually(self, image):
        corridor_maker = CorridorMaker(image)
        stop_line_maker = StopLineMaker(image)

        selected_corridors = corridor_maker.run(info=self._info)
        for index, line in enumerate(selected_corridors):
            line1 = line

            try:
                line2 = selected_corridors[index + 1]
            except IndexError:
                break

            self.create_new_corridor(left_line=line1,
                                     right_line=line2)

        most_left_selection = selected_corridors[0]
        most_right_selection = selected_corridors[-1]

        self._info.vanishing_points.append(VanishingPoint(point=most_left_selection.intersection(most_right_selection)))

        self._stop_line = stop_line_maker.run(info=self._info)

        self._stopline_found = True
        self._corridors_found = True

    def line_crossed(self, previous_coordinates, coordinates):
        if not self.ready:
            return False

        red_line_point = self._stop_line.find_coordinate(x=coordinates.x)
        return previous_coordinates.y > red_line_point[1] > coordinates.y

    def behind_line(self, coordinates):
        if not self.ready:
            return False

        red_line_point = self._stop_line.find_coordinate(x=coordinates.x)
        return red_line_point[1] > coordinates.y

    def get_corridor(self, coordinates) -> int:
        if not self._corridors_found:
            return -1

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
                                       fill=fill)

        image = cv2.bitwise_and(image, image, mask=self._corridor_mask)

        if self._stopline_found:
            self._stop_line.draw(image=image,
                                 color=params.COLOR_RED,
                                 thickness=5)

        return image

    def create_new_corridor(self, left_line, right_line):
        self._corridors_count += 1

        left_edge_points = left_line.edge_points(self._info)
        right_edge_points = right_line.edge_points(self._info)

        middle_point = None
        for point1 in left_edge_points:
            if middle_point is not None:
                break

            for point2 in right_edge_points:
                if point1[0] > point2[0]:
                    continue
                else:
                    middle_point = int((point1[0] + point2[0])/2), int((point1[1] + point2[1])/2)
                    break

        new_corridor = TrafficCorridor(index=self._corridors_count,
                                       left_line=left_line,
                                       right_line=right_line,
                                       middle_point=middle_point)

        self._corridors[new_corridor.id] = new_corridor

        new_corridor.draw_corridor(image=self._corridor_mask,
                                   info=self._info,
                                   color=self._corridors_count)

    def find_corridors(self, lifelines_mask, vp1):

        # extract left-bot-right 1px border around lifeline image and make "1D array"
        left_edge = lifelines_mask[:, :1]
        bottom_edge = lifelines_mask[-1:, :].transpose(1, 0, 2)
        right_edge = np.flip(lifelines_mask[:, -1:])
        frame_edge = np.concatenate((left_edge, bottom_edge, right_edge))

        # greyscale image and reduce noise by multiple blur and threshold
        edge_grey = cv2.cvtColor(frame_edge, cv2.COLOR_RGB2GRAY)

        edge_grey_blured = cv2.medianBlur(edge_grey, 21)
        _, threshold = cv2.threshold(edge_grey_blured, 20, 255, cv2.THRESH_BINARY)

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

            left_bottom = tuple(points[0])
            right_bottom = tuple(points[1])
            left_top = right_top = vp1.point

            self.create_new_corridor(left_line=Line(left_bottom, left_top),
                                     right_line=Line(right_bottom, right_top))

        # cv2.imwrite("mask.jpg", self.get_mask())
        cv2.imwrite("lifeline.jpg", frame_edge)
        cv2.imwrite("lifeline_before.jpg", edge_grey_blured)

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

        test_image = np.zeros(shape=(self._info.height, self._info.width))

        for point in self._stop_places:
            cv2.circle(test_image, point.tuple(), radius=5, color=255, thickness=5)

        cv2.imwrite("stopky.png", test_image)

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

            if num > best_line_ratio:
                best_line_ratio = num
                best_line = line

        self._stop_line = best_line
        self._stopline_found = True


class TrafficCorridor:
    def __init__(self, index, left_line, right_line, middle_point):
        self._id = index

        self.left_line = left_line
        self.right_line = right_line
        self.middle_point = middle_point

    @property
    def id(self):
        return self._id

    def draw_corridor(self, image, info, color=None, fill=True, thickness=params.DEFAULT_THICKNESS):
        mask = np.zeros(shape=(info.height, info.width),
                        dtype=np.uint8)

        if color is None:
            color = self.id

        self.left_line.draw(image, color, thickness)
        self.left_line.draw(mask, color, thickness)

        self.right_line.draw(image, color, thickness)
        self.right_line.draw(mask, color, thickness)

        if fill:
            mask_with_border = np.pad(mask, 1, 'constant', constant_values=255)

            cv2.floodFill(image=image,
                          mask=mask_with_border,
                          seedPoint=self.middle_point,
                          newVal=color)

    def __str__(self):
        return f"corridor: id [{self.id}]"


class LineDrag:
    def __init__(self, image):
        self.selected = []
        self.base_image = image

        self.point1 = None
        self.point2 = None

    def click(self, point):
        if self.point1 is None:
            self.point1 = point

        else:
            self.__add_line()

    def move(self, point):
        self.point2 = point

    def __add_line(self):
        self.add_line()

        self.point1 = None
        self.point2 = None

    def draw(self, image):
        if self.point1 is not None:
            cv2.line(img=image,
                     pt1=self.point1,
                     pt2=self.point2,
                     color=params.COLOR_GREEN,
                     thickness=params.CORRIDORS_LINE_SELECTOR_THICKNESS)

    def clear(self):
        self.point1 = None
        self.point2 = None

    def run(self, info) -> []:
        raise NotImplementedError

    def add_line(self):
        raise NotImplementedError

    def erase_last(self):
        raise NotImplementedError


class StopLineMaker(LineDrag):
    def __init__(self, image):
        super().__init__(image)

        self.line = None

    def add_line(self):
        try:
            self.line = Line(self.point1, self.point2)
        except SamePointError:
            pass

    def draw(self, image):
        super().draw(image)

        if self.line is not None:
            self.line.draw(image, color=params.COLOR_RED, thickness=params.CORRIDORS_LINE_SELECTOR_THICKNESS)

    def run(self, info) -> []:
        cv2.namedWindow("select_stop_line")
        cv2.setMouseCallback("select_stop_line", _mouse_callback, self)

        while True:
            image_copy = np.copy(self.base_image)

            self.draw(image_copy)
            cv2.imshow("select_stop_line", image_copy)

            key = cv2.waitKey(1)

            if key == 13:
                self.line.draw(image_copy, params.COLOR_BLUE, params.CORRIDORS_LINE_SELECTOR_THICKNESS)
                cv2.imshow("select_stop_line", image_copy)

                key = cv2.waitKey(0)
                if key == 13:
                    cv2.destroyWindow("select_stop_line")
                    return self.line

                if key == 8:
                    self.clear()

            if key == 8:
                self.erase_last()

    def erase_last(self):
        self.line = None


class CorridorMaker(LineDrag):
    def __init__(self, image):
        super().__init__(image)
        self.selected = []
        self.base_image = image

    def add_line(self):
        try:
            new_line = Line(self.point1, self.point2)

            if new_line.angle(Line.horizontal_line()) != 0:
                self.selected.append(Line(self.point1, self.point2))

        except SamePointError:
            pass

    def draw(self, image):
        super().draw(image)

        for line in self.selected:
            line.draw(image, color=params.COLOR_RED, thickness=params.CORRIDORS_LINE_SELECTOR_THICKNESS)

    def __iter__(self):
        return iter(self.selected)

    def clear(self):
        self.selected = []
        super().clear()

    def erase_last(self):
        try:
            self.selected.pop()
        except IndexError:
            pass

    def run(self, info) -> []:
        cv2.namedWindow("select_corridors")
        cv2.setMouseCallback("select_corridors", _mouse_callback, self)

        while True:
            image_copy = np.copy(self.base_image)

            self.draw(image_copy)
            cv2.imshow("select_corridors", image_copy)

            key = cv2.waitKey(1)

            if key == 13:
                for line in self:
                    line.draw(image_copy, params.COLOR_BLUE, params.CORRIDORS_LINE_SELECTOR_THICKNESS)
                    cv2.imshow("select_corridors", image_copy)

                key = cv2.waitKey(0)
                if key == 13:
                    cv2.destroyWindow("select_corridors")

                    if len(self.selected) < 2:
                        print("Not minimal number of lines selected. Please select at least 2 lines")
                        continue
                    else:
                        return sorted(self.selected, key=lambda l: l.find_coordinate(y=info.height)[0])

                if key == 8:
                    self.clear()

            if key == 8:
                self.erase_last()

