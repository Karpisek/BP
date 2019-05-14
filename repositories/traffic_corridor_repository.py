import cv2
import numpy as np
from primitives import constants

from primitives.line import Line, SamePointError
from primitives.vanishing_point import VanishingPoint, VanishingPointError
from repositories.models.corridor import TrafficCorridor
from user_interaction.corridor_marker import CorridorMaker
from user_interaction.stopline_marker import StopLineMaker


class TrafficCorridorRepository:
    """
    Repository used for creating and storing traffic corridors. Holds information of stop line as well
    """

    def __init__(self, info):
        """
        :param info: instance of InputInfo
        """
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
        """
        :return: number of detected corridors
        """

        return self._corridors_count

    @property
    def corridors_found(self):
        """
        :return: if corridors were found
        """

        return self._corridors_found

    @property
    def corridor_ids(self):
        """
        :return: IDs of corridors
        """

        return self._corridors.keys()

    @property
    def stopline_found(self):
        """
        :return: If stopline was found
        """

        return self._stopline_found

    @property
    def stopline(self):
        """
        :return: detected stop line
        """

        return self._stop_line

    @stopline.setter
    def stopline(self, value):
        """
        Sets the stopline with specified offset, assuming that first vanishing point is in the top part of frame
        """

        self._stop_line = value

        left_edge_point = self.stopline.edge_points(info=self._info)[0]
        right_edge_point = self.stopline.edge_points(info=self._info)[1]

        self._stop_line = Line(point1=(left_edge_point[0], left_edge_point[1] - constants.CORRIDORS_STOP_LINE_OFFSET),
                               point2=(right_edge_point[0], right_edge_point[1] - constants.CORRIDORS_STOP_LINE_OFFSET))

        left_edge_point = self.stopline.edge_points(info=self._info)[0]
        right_edge_point = self.stopline.edge_points(info=self._info)[1]

        top_line = Line(point1=(left_edge_point[0], 3 * left_edge_point[1] / 5),
                        point2=(right_edge_point[0], 3 * right_edge_point[1] / 5))

        self._info.update_area.change_area(top_line=top_line)

    @property
    def ready(self):
        """
        :return: if both corridors and stop line is found
        """

        return self.corridors_found and self._stopline_found

    @property
    def corridor_mask(self):
        """
        :return: mask of corridors
        """

        return self._corridor_mask

    def select_manually(self, image):
        """
        Allows user to select corridors and stop line manually

        :param image: image to select on
        """

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

        self._corridor_mask = cv2.bitwise_and(self._corridor_mask, self._corridor_mask,
                                              mask=self._info.update_area.mask())

        most_left_selection = selected_corridors[0]
        most_right_selection = selected_corridors[-1]

        self._info.vanishing_points.append(VanishingPoint(point=most_left_selection.intersection(most_right_selection)))

        self.stopline = stop_line_maker.run(info=self._info)

        self._stopline_found = True
        self._corridors_found = True

    def line_crossed(self, previous_coordinates, coordinates):
        """
        :param previous_coordinates: previous coordinates of object
        :param coordinates: new coordinates of object
        :return: if line is between these two specified positions (while object is moving the right direction)
        """

        if not self.ready:
            return False

        red_line_point = self.stopline.find_coordinate(x=coordinates.x)
        return previous_coordinates.y >= red_line_point[1] > coordinates.y

    def behind_line(self, coordinates):
        """
        :param coordinates: coordinates
        :return: if coordinates are behind stop line (closer to vanishing point)
        """

        if not self.ready:
            return False

        red_line_point = self.stopline.find_coordinate(x=coordinates.x)
        return red_line_point[1] > coordinates.y

    def __contains__(self, coordinates):
        return self.get_corridor(coordinates) > 0 or not self._corridors_found

    def get_corridor(self, coordinates) -> int:
        """
        :param coordinates: object coordinates
        :return: corridor ID where these coordinates belong to. -1 if no corridors were found, 0 when outside corridors
        """

        if not self._corridors_found:
            return -1

        if 0 < coordinates.x < self._info.width and 0 < coordinates.y < self._info.height:
            return self._corridor_mask[int(coordinates.y)][int(coordinates.x)]
        else:
            return -1

    def get_mask(self, fill) -> np.ndarray:
        """
        :param fill: if mask should be filled
        :return: mask of corridors
        """

        image = np.zeros(shape=(self._info.height, self._info.width, 3),
                         dtype=np.uint8)

        if self._corridors_found:
            for key, corridor in self._corridors.items():
                corridor.draw_corridor(image=image,
                                       info=self._info,
                                       color=constants.RANDOM_COLOR[key],
                                       fill=fill)

        image = cv2.bitwise_and(image, image, mask=self._corridor_mask)

        if self._stopline_found:
            self.stopline.draw(image=image,
                               color=constants.COLOR_RED,
                               thickness=5)

        return image

    def create_new_corridor(self, left_line, right_line):
        """
        Creates new corridor from two specified lines
        """

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
                if point1[1] == 0 or point2[1] == 0:
                    continue

                else:
                    middle_point = int((point1[0] + point2[0]) / 2), int((point1[1] + point2[1]) / 2)
                    break

        new_corridor = TrafficCorridor(index=self._corridors_count,
                                       left_line=left_line,
                                       right_line=right_line,
                                       middle_point=middle_point)

        self._corridors[new_corridor.id] = new_corridor

        # image = np.zeros(shape=(self._info.height, self._info.width), dtype=np.uint8)

        # left_line.draw(image, thickness=1, color=255)
        # right_line.draw(image, thickness=1, color=255)

        # cv2.imwrite("test.jpg", image)
        new_corridor.draw_corridor(image=self._corridor_mask,
                                   info=self._info,
                                   color=self._corridors_count)

    def find_corridors(self, lifelines_mask, vp1):
        """
        Finds corridors on frame using first positions of cars and detected first vanishing point to construct
        mask on which thresholding is used to detect separate corridors.
        By computation is used only 1px wide age around left, bottom and right edge of frame.

        :param lifelines_mask: mask of lifelines
        :param vp1: detected first vanishing point
        """

        # extract left-bot-right 1px border around lifeline image and make "1D array"
        left_edge = lifelines_mask[:, :1]
        bottom_edge = lifelines_mask[-1:, :].transpose(1, 0, 2)
        right_edge = np.flip(lifelines_mask[:, -1:])
        frame_edge = np.concatenate((left_edge, bottom_edge, right_edge))

        # greyscale image and reduce noise by multiple blur and threshold
        edge_grey = cv2.cvtColor(frame_edge, cv2.COLOR_RGB2GRAY)

        edge_grey_blured = cv2.medianBlur(edge_grey, 11)
        _, threshold = cv2.threshold(edge_grey_blured, 50, 255, cv2.THRESH_BINARY)
        threshold = cv2.dilate(threshold, (5, 5), iterations=5)

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

        self._corridor_mask = cv2.bitwise_and(self._corridor_mask, self._corridor_mask,
                                              mask=self._info.update_area.mask())
        self._corridors_found = True

    def add_stop_point(self, coordinates):
        """
        Adds selected stop point to accumulation of stop points. If enough poits are accumulated
        stop line computation is done.

        :param coordinates: coordinates to add
        """

        self._stop_places.append(coordinates)

        if len(self._stop_places) >= constants.CORRIDORS_STOP_POINTS_MINIMAL and not self._stopline_found:
            self.find_stop_line()

    def find_stop_line(self):
        """
        Finds stop line from stored stop-points.
        For stop line is used second vanishing point which is used as anchor for all detected stop line points.
        Stop line is detected using RANSAC algorithm.
        """

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
            except VanishingPointError:
                line = Line(point1=point2.tuple(), direction=vp2.direction)

            num = 0
            ransac_threshold = constants.CORRIDORS_RANSAC_THRESHOLD
            for point in self._stop_places:
                distance = line.point_distance(point.tuple())

                if distance < ransac_threshold:
                    num += 1

            if num > best_line_ratio:
                best_line_ratio = num
                best_line = line

        self.stopline = best_line
        self._stopline_found = True

    def serialize(self):
        """
        Serialize corridors repository in form of dictionary.

        :return: dictionary of serialized corridors
        """

        return {"corridors": [corridor.serialize() for _, corridor in self._corridors.items()]}
