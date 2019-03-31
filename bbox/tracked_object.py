import cv2
import numpy as np
from munkres import Munkres

import params
from bbox.coordinates import Coordinates
from pc_lines.line import Line, SamePointError, NotOnLineError

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


class TrackedObjectsRepository:
    def __init__(self, info):
        self._id_counter = 0
        self._lifelines = []
        self._tracked_objects = []
        self._info = info

    @property
    def list(self):
        return self._tracked_objects

    @property
    def lifelines(self):
        return self._lifelines

    def new_tracked_object(self, coordinates, size, confident_score):
        new_object = TrackedObject(coordinates=coordinates,
                                   size=size,
                                   confident_score=confident_score,
                                   info=self._info,
                                   object_id=self._id_counter)

        self._tracked_objects.append(new_object)
        self._id_counter += 1

    def count(self) -> int:
        return len(self._tracked_objects)

    def all_boxes_mask(self, area_size="inner"):
        height = self._info.height
        width = self._info.width
        global_mask = np.zeros(shape=(height, width),
                               dtype=np.uint8)

        for tracked_object in self._tracked_objects:
            global_mask = cv2.bitwise_or(global_mask, tracked_object.mask(width=width,
                                                                          height=height,
                                                                          area_size=area_size))

        return global_mask

    def predict(self) -> None:
        for tracked_object in self._tracked_objects:
            tracked_object.predict()

            # if new_lifeline is not None:
            #     self._lifelines.append(new_lifeline)

    def control_boxes(self) -> None:
        for tracked_object in self._tracked_objects:
            if not self._info.update_area.contains(tracked_object.center):
                starting_coordinates = tracked_object.history
                end_coordinates = tracked_object.center

                if end_coordinates.y < starting_coordinates.y:
                    self._lifelines.append((starting_coordinates.tuple(), end_coordinates.tuple()))

                self._tracked_objects.remove(tracked_object)

        # [self._tracked_objects.remove(tracked_object) for tracked_object in self._tracked_objects if not self._info.update_area.contains(tracked_object.center)]

    def serialize(self):
        return [tracked_object.serialize() for tracked_object in self._tracked_objects]


class TrackedObject:

    @staticmethod
    def draw_lifelines(image, lifelines=None, color=params.COLOR_RED, thickness=1) -> np.ndarray:
        if lifelines is not None:
            for line in lifelines:

                mask = np.zeros_like(image)

                p1, p2 = line

                try:
                    l = Line(p1, p2)
                    l.draw(mask, color, thickness)
                    image = cv2.add(image, mask)
                except SamePointError:
                    continue
                except NotOnLineError:
                    print(p1, p2)
                    raise

        return image

    @staticmethod
    def lifeline_convex_hull(info, lifelines=None) -> np.ndarray:
        if lifelines is not None:

            contours = []
            for line in lifelines:
                try:
                    l = Line(*line)
                except SamePointError:
                    continue

                p1 = tuple([int(cord) for cord in l.find_coordinate(y=0)])
                p2 = tuple([int(cord) for cord in l.find_coordinate(y=info.height)])

                if p2[0] < 0 or p2[0] > info.width:
                    continue

                contours.append([p1])
                contours.append([p2])

            hull = cv2.convexHull(np.array(contours))

            return hull

    @staticmethod
    def draw(image, boxes, lifelines=None) -> np.ndarray:

        for box in boxes:
            anchors, area_of_interest, car_info = box

            top_left, bot_right, center_point = anchors

            cv2.circle(image, center_point, area_of_interest, params.COLOR_BLUE, BOX_THICKNESS)
            cv2.putText(image, car_info, top_left, 1, 1, params.COLOR_WHITE, 2)

        if lifelines is not None:
            return TrackedObject.draw_lifelines(image, lifelines)
        else:
            return image

    def __init__(self, coordinates, size, confident_score, info, object_id):

        super().__init__()

        coordinates.convert_to_fixed(info)
        size.convert_to_fixed(info)

        self._id = object_id
        self._object_size = size

        self.score = confident_score

        self._info = info
        self._kalman = cv2.KalmanFilter(4, 4)

        self.lifetime = 0

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

        self._start_coordinates = coordinates

    @property
    def history(self):
        return self._start_coordinates

    @property
    def id(self) -> int:
        return self._id

    @property
    def center(self):
        return Coordinates(int(self._kalman.statePost[0][0]), int(self._kalman.statePost[1][0]))

    @property
    def size(self) -> (int, int):
        return self._object_size

    @property
    def velocity(self) -> (int, int):
        return int(np.abs(self._kalman.statePost[2][0])), int(np.abs(self._kalman.statePost[3][0]))

    @property
    def left_anchor(self) -> (int, int):
        x_min = int((self.center.x - self.size.width / 2))
        y_min = int((self.center.y - self.size.height / 2))

        return x_min, y_min

    @property
    def right_anchor(self) -> (int, int):
        x_max = int((self.center.x + self.size.width / 2))
        y_max = int((self.center.y + self.size.height / 2))

        return x_max, y_max

    @property
    def car_info(self) -> str:
        return str(self._id)

    def area(self, area_size) -> int:
        width = self.size.width
        height = self.size.height

        diagonal = np.sqrt(width * width + height * height)

        if area_size == "inner":
            return int(diagonal/2) if int(diagonal/2) < 20 else 20
        if area_size == "outer":
            return int(diagonal / 2)
        if area_size == "small-outer":
            return int((diagonal - 3) / 2)
        else:
            return 0

    def anchors(self) -> ((int, int), (int, int), (int, int), (int, int)):
        return self.left_anchor, self.right_anchor, self.center.tuple()

    def predict(self) -> ((float, float), (float, float)) or None:
        self._kalman.predict()
        self.lifetime += 1
        # if self.center.y < self.history.y:
        #     if self.lifetime == 20:
        #         return self.history.tuple(), self.center.tuple()

    def update_position(self, size, score, new_coordinates) -> None:

        # self._history = new_coordinates

        size.convert_to_fixed(self._info)
        new_coordinates.convert_to_fixed(info=self._info)

        mesurement = np.array([
            [np.float32(new_coordinates.x)],
            [np.float32(new_coordinates.y)],
            [np.float32(0)],
            [np.float32(0)]
        ])

        self._kalman.measurementMatrix = KALMAN_MESUREMENT_POSITION_MATRIX
        self._kalman.correct(mesurement)

        self.score = score
        self._object_size = size

    def update_flow(self, old_positions, new_positions) -> None:
        x_flow, y_flow = self.extract_flow(old_positions, new_positions)

        mesurement = np.array([
            [np.float32(0)],
            [np.float32(0)],
            [np.float32(x_flow/params.TRACKER_OPTICAL_FLOW_FREQUENCY)],
            [np.float32(y_flow/params.TRACKER_OPTICAL_FLOW_FREQUENCY)]
        ])

        self._kalman.measurementMatrix = KALMAN_MESUREMENT_FLOW_MATRIX
        self._kalman.correct(mesurement)

    def in_radius(self, new_coordinates) -> int:

        width = self.size.width
        height = self.size.height

        diagonal = np.sqrt(width * width + height * height)

        max_pixels = diagonal/2

        return self.center.distance(new_coordinates) < max_pixels

    def extract_flow(self, old_positions, new_positions) -> (float, float):

        global_dx = 0
        global_dy = 0

        number_of_flows = 0
        for flow in zip(new_positions, old_positions):
            new_pos, old_pos = flow

            if self.in_radius(Coordinates(*new_pos)):

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

    def mask(self, width, height, area_size="inner") -> np.ndarray:
        mask = np.zeros(shape=(height, width), dtype=np.uint8)
        cv2.circle(img=mask,
                   center=self.center.tuple(),
                   radius=self.area(area_size),
                   color=params.COLOR_WHITE_MONO,
                   thickness=params.FILL)

        return mask

    def serialize(self):
        return self.anchors(), self.area(area_size="outer"), self.car_info

    def __del__(self):
        pass
        # only lifelines from bot to top
        # if self.center.y < self.history.y:
        #     Box2D._lifelines.append((self.history.tuple(), self.center.tuple()))

