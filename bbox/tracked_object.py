import cv2
import numpy as np
import constants

from bbox.size import ObjectSize
from bbox.coordinates import Coordinates
from pc_lines.line import Line, SamePointError
from pipeline.base.pipeline import Mode

BOX_THICKNESS = 2
CENTER_POINT_RADIUS = 2

KALMAN_TRANSITION_MATRIX = np.array([
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
], np.float32)

KALMAN_MESUREMENT_POSITION_MATRIX = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
], np.float32)

KALMAN_MESUREMENT_FLOW_MATRIX = np.array([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
], np.float32)

KALMAN_PROCESS_NOISE_COV = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
], np.float32) * 0.5

KALMAN_MESUREMENT_NOISE_COV = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
], np.float32) * 1


class TooManyCarsError(Exception):
    pass


class TrackedObjectsRepository:
    """
    Repository for tracked object. Allows to create, update and remove tracked objects.
    Removes tracked objects if their position is outside specified area.
    """

    def __init__(self, info):
        """
        :param info: instance of InputInfo
        """

        self._id_counter = 0
        self._lifelines = []
        self._collected_lifelines_id = []
        self._tracked_objects = []
        self._info = info

    @property
    def list(self):
        """
        :return: list of tracked objects
        """

        return self._tracked_objects

    @property
    def lifelines(self):
        """
        :return: list of trajectories of tracked objects
        """

        return self._lifelines

    @property
    def flows(self):
        return [tracked_object.flow for tracked_object in self._tracked_objects]

    def new_tracked_object(self, coordinates, size, confident_score, _):
        """
        Creates new tracked object. If it found collision with existing tracked object these objects are marget
        together. Colision is defined by percentage of overlap

        :param coordinates: new coordinates
        :param size: size of new object
        :param confident_score: how certain we are about this object
        :param _: ANY
        """

        new_object = TrackedObject(coordinates=coordinates,
                                   size=size,
                                   confident_score=confident_score,
                                   info=self._info,
                                   object_id=self._id_counter)

        collision = False
        for index, tracked_object in enumerate(self._tracked_objects[:]):
            if tracked_object.overlap(new_object) > constants.TRACKER_MAX_OVERLAP:
                new_object.id = tracked_object.id
                self._tracked_objects[index] = new_object
                collision = True
                break

        if not collision:
            self._tracked_objects.append(new_object)
            self._id_counter += 1

    def count(self) -> int:
        """
        :return: count of currently tracked objects
        """

        return len(self._tracked_objects)

    def all_boxes_mask(self, area_size="inner"):
        """
        :param area_size: specified area of boxes.
        :return: created mask
        """

        height = self._info.height
        width = self._info.width
        global_mask = np.zeros(shape=(height, width),
                               dtype=np.uint8)

        for index, tracked_object in enumerate(self._tracked_objects[::-1]):
            global_mask = np.maximum(global_mask, tracked_object.mask(width=width,
                                                                      height=height,
                                                                      area_size=area_size,
                                                                      color=constants.COLOR_WHITE_MONO))

        return global_mask

    def predict(self) -> None:
        """
        Predicts positions on all tracked objects
        """

        for tracked_object in self._tracked_objects:
            tracked_object.predict()

    def control_boxes(self, mode) -> None:
        """
        checks every tracked object if they satisfy all condition to be tracked.
        Different workmodes have different conditions to be tracked.

        :param mode: current work mode
        """

        for tracked_object in self._tracked_objects:
            if tracked_object.id not in self._collected_lifelines_id:
                if tracked_object.tracker_point not in self._info.update_area or (mode == Mode.CALIBRATION_VP and tracked_object.center not in self._info.update_area):
                    self.lifelines.append(tracked_object.history)
                    self._collected_lifelines_id.append(tracked_object.id)
                else:
                    tracked_object.update_history()

            if tracked_object.tracker_point not in self._info.corridors_repository:
                if tracked_object.center not in self._info.corridors_repository:
                    self._tracked_objects.remove(tracked_object)

            elif tracked_object.tracker_point not in self._info.update_area:
                self._tracked_objects.remove(tracked_object)

            elif self._info.corridors_repository.behind_line(tracked_object.tracker_point):
                tracked_object.lifetime -= 1
                if tracked_object.lifetime <= 0:
                    self._tracked_objects.remove(tracked_object)

    def serialize(self):
        """
        Serializes all tracked object in list
        :return: serialized tracked objects
        """

        return [tracked_object.serialize() for tracked_object in self._tracked_objects]

    def restart(self):
        """
        Clears this repository
        """

        self._id_counter = 0
        self._lifelines = []
        self._tracked_objects = []


class TrackedObject:
    """
    Represents one tracked object
    Uses Kalman Filter to predict his position. Two possible mesurement are allowed -> position and velocity.
    Stores history of center points to construct trajectories.
    """

    @staticmethod
    def draw_flow(image, flows):
        for flow in flows:
            center, x, y = flow
            cv2.arrowedLine(image, center, (center[0] + int(x), center[1] + int(y)), color=constants.COLOR_RED, thickness=2, tipLength=0.2)

        return image

    @staticmethod
    def filter_lifelines(lifelines, vp1):
        """
        Filters given trajectories by those which have direction ti the first vanishing point.

        :param lifelines: given trajectories
        :param vp1: detected vanishing point
        :return: filtered trajectories
        """

        filtered = []

        for lifeline in lifelines:
            line_to_vp = Line(lifeline[0], vp1.point)

            try:
                lifeline_line = Line(lifeline[0], lifeline[-1])
            except SamePointError:
                continue

            if constants.TRACKER_TRAJECTORY_MIN_ANGLE < lifeline_line.angle(line_to_vp) < constants.TRACKER_TRAJECTORY_MAX_ANGLE:
                continue

            elif lifeline[0][1] < lifeline[-1][1]:
                continue

            else:
                filtered.append(lifeline)

        return filtered

    def __init__(self, coordinates, size, confident_score, info, object_id):

        super().__init__()

        self._id = object_id
        self._velocity = 0

        self._reference_object_size = self._current_size = size
        self._reference_coordinates = coordinates

        self.score = confident_score

        self._info = info
        self._kalman = cv2.KalmanFilter(4, 4)

        self.lifetime = constants.TRACKER_LIFETIME

        self._kalman.transitionMatrix = KALMAN_TRANSITION_MATRIX
        self._kalman.measurementMatrix = KALMAN_MESUREMENT_POSITION_MATRIX
        self._kalman.processNoiseCov = KALMAN_PROCESS_NOISE_COV

        self._kalman.measurementNoiseCov = KALMAN_MESUREMENT_NOISE_COV

        self._kalman.statePost = np.array([
            [np.float32(coordinates.x)],  # x-coord
            [np.float32(coordinates.y)],  # y-coord
            [np.float32(0)],  # dx
            [np.float32(0)],  # dy
        ])

        self._history = [self.center.tuple()]
        self._flow = None

    @property
    def flow(self):
        """
        :return: pixel flow of tracked object from last mesurement
        """

        return self.center.tuple(), self._kalman.statePost[2][0], self._kalman.statePost[3][0]

    @property
    def history(self):
        """
        :return: list of history center points
        """
        return self._history

    @property
    def tracker_point(self):
        """
        :return: middle point on bottom edge of tracked object
        """

        return Coordinates(x=self.center.x, y=int(self.center.y + self.size.height/2))

    @property
    def id(self) -> int:
        """
        :return: id of tracked object
        """

        return self._id

    @id.setter
    def id(self, value):
        """
        sets ID of tracked object
        """

        self._id = value

    @property
    def center(self):
        """
        :return: center Coordinates of trakced object
        """

        return Coordinates(int(self._kalman.statePost[0][0]), int(self._kalman.statePost[1][0]))

    @property
    def size(self) -> (int, int):
        """
        :return: size of tracked object
        """

        return self._current_size

    @property
    def velocity(self) -> int:
        """
        :return: velocity of tracked object rounded to nearest integer
        """

        return int(self._velocity)

    @property
    def left_top_anchor(self):
        """
        :return: top left anchor of tracked object
        """

        x = int((self.center.x - self.size.width / 2))
        y = int((self.center.y - self.size.height / 2))

        return Coordinates(x, y)

    @property
    def left_bot_anchor(self):
        """
        :return: left bot anchor of tracked object
        """

        x = int((self.center.x - self.size.width / 2))
        y = int((self.center.y + self.size.height / 2))

        return Coordinates(x, y)

    @property
    def right_top_anchor(self):
        """
        :return: top right anchor of tracked object
        """

        x = int((self.center.x + self.size.width / 2))
        y = int((self.center.y - self.size.height / 2))

        return Coordinates(x, y)

    @property
    def right_bot_anchor(self):
        """
        :return: right bot anchor of tracked object
        """

        x = int((self.center.x + self.size.width / 2))
        y = int((self.center.y + self.size.height / 2))

        return Coordinates(x, y)

    @property
    def car_info(self) -> str:
        """
        :return: info about tracked object - ID of tracked object
        """

        return str(self._id)

    @property
    def radius(self):
        """
        :return: radius around tracked object specified by diagonal line
        """

        return int(np.sqrt(self.size.width * self.size.width + self.size.height * self.size.height) / 2)

    def update_history(self):
        """
        Signal to update history of this tracked object. It controls if enough Y coordinate diference was made.
        If so it saves current position to history.
        """

        if np.abs(self.center.y - self._history[-1][1]) > constants.TRACKER_HISTORY_DIFFERENCE:
            self._history.append(self.center.tuple())

    def overlap(self, overlapping_object):
        """
        Checks if this tracked objects overlaps with selected object.

        :param overlapping_object: tracked object to compare
        :return: percentage of overlap
        """

        x_min = np.max((self.left_top_anchor.x, overlapping_object.left_top_anchor.x))
        y_min = np.max((self.left_top_anchor.y, overlapping_object.left_top_anchor.y))
        x_max = np.min((self.right_bot_anchor.x, overlapping_object.right_bot_anchor.x))
        y_max = np.min((self.right_bot_anchor.y, overlapping_object.right_bot_anchor.y))

        image = np.zeros(shape=(self._info.height, self._info.width))

        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), 200, 2)

        x_size = (x_max - x_min)
        y_size = (y_max - y_min)

        # no overlapping
        if x_size < 0 or y_size < 0:
            return 0

        overlapped_square_size = x_size * y_size

        return overlapped_square_size / self.square_size()

    def square_size(self):
        """
        :return: square size of tracked object
        """

        return self.size.width * self.size.height

    def area(self, area_size) -> int:
        """
        Defining multiple areas around object.

        :param area_size: selected area size
        :return: radius of selected area
        """

        if area_size == "inner":
            return self.radius if self.radius < 10 else 10

        if area_size == "outer":
            return self.radius

        if area_size == "small-outer":
            return self.radius - 3

        else:
            return 0

    def anchors(self) -> ((int, int), (int, int), (int, int)):
        """
        :return: 2 anchors describing this tracked object and center point.
        """

        return self.left_top_anchor.tuple(), self.right_bot_anchor.tuple(), self.center.tuple()

    def predict(self) -> ((float, float), (float, float)) or None:
        """
        Predicts new position of tracked object using Kalman Filter.
        Adjusts size of object depending on distance to vanishing point.
        """

        self._kalman.predict()

        if self._info.vp1 is not None:
            vp1 = self._info.vanishing_points[0]
            dist_diff = self.center.distance(vp1.coordinates) / self._reference_coordinates.distance(vp1.coordinates)

            current_width = self._reference_object_size.width * dist_diff
            current_height = self._reference_object_size.height * dist_diff

            self._current_size = ObjectSize(current_width, current_height)

        else:
            self._current_size = self._reference_object_size

    def update_position(self, size, score, new_coordinates):
        """
        Updates Kalman filter by mesurement of new position of tracked object.

        :param size: measured object size
        :param score: measured object score
        :param new_coordinates: coordinates of measured object
        :param mode: current mode
        :return:
        """

        if not self._info.corridors_repository.behind_line(new_coordinates):
            mesurement = np.array([
                [np.float32(new_coordinates.x)],
                [np.float32(new_coordinates.y)],
                [np.float32(0)],
                [np.float32(0)]
            ])

            self._kalman.measurementMatrix = KALMAN_MESUREMENT_POSITION_MATRIX
            self._kalman.correct(mesurement)

            self.score = score

            self._reference_object_size = size
            self._reference_coordinates = new_coordinates

    def update_flow(self, old_positions, new_positions) -> None:
        """
        Updates Kalman filter by mesurement of optical flow in scene.
        Tracked object position difference is extracted from complete optical flow of the scene using selected
        area around this tracked object.

        :param old_positions: old positions
        :param new_positions: new positions
        """

        x_flow, y_flow = self.extract_flow(old_positions, new_positions)

        mesurement = np.array([
            [np.float32(0)],
            [np.float32(0)],
            [np.float32(x_flow / constants.TRACKER_OPTICAL_FLOW_FREQUENCY)],
            [np.float32(y_flow / constants.TRACKER_OPTICAL_FLOW_FREQUENCY)]
        ])

        self._velocity = np.sqrt(x_flow ** 2 + y_flow ** 2)
        self._kalman.measurementMatrix = KALMAN_MESUREMENT_FLOW_MATRIX
        self._kalman.correct(mesurement)
        self._flow = x_flow, y_flow

    def in_area(self, new_coordinates):
        """
        :param new_coordinates: selected coordinates coordinates
        :return: checks if selected coordinates are in area of tracked object
        """

        return self.left_top_anchor.x < new_coordinates.x < self.right_bot_anchor.x and self.left_top_anchor.y < new_coordinates.y < self.right_bot_anchor.y

    def in_radius(self, new_coordinates) -> int:
        """
        :param new_coordinates: selected coordinates
        :return: checks if selected coordinates are in certain radius around center of tracked object
        """

        width = self.size.width
        height = self.size.height

        diagonal = np.sqrt(width * width + height * height)

        max_pixels = diagonal / 2

        return self.center.distance(new_coordinates) < max_pixels

    def extract_flow(self, old_positions, new_positions) -> (float, float):
        """
        Extracts optical flow for this tracked object. Optical flow is selected using an area around center point of
        this tracked object.

        :return: extracted flow corresponding to this tracked object
        """

        global_dx = 0
        global_dy = 0

        number_of_flows = 0
        for flow in zip(new_positions, old_positions):
            new_pos, old_pos = flow

            if self.in_area(Coordinates(*new_pos)):

                new_x, new_y = new_pos
                old_x, old_y = old_pos
                dx = new_x - old_x
                dy = new_y - old_y

                global_dx += dx
                global_dy += dy

                number_of_flows += 1

        if number_of_flows:
            return global_dx/number_of_flows, global_dy/number_of_flows
        else:
            return 0, 0

    def mask(self, width, height, area_size="inner", color=constants.COLOR_WHITE_MONO) -> np.ndarray:
        """
        Creates (binary) mask of circle area around this tracked object in scene

        :param width: width of image
        :param height: height of image
        :param area_size: size of selected area
        :param color: color of mask
        :return: generated mask
        """

        mask = np.zeros(shape=(height, width), dtype=np.uint8)
        cv2.circle(img=mask,
                   center=self.center.tuple(),
                   radius=self.area(area_size),
                   color=color,
                   thickness=constants.FILL)

        return mask

    def serialize(self):
        """
        :return: serialized tracked object: anchors, area, car info and velocity
        """

        return self.anchors(), self.area(area_size="outer"), self.car_info, self.velocity
