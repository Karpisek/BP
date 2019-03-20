import cv2
import numpy as np

from bbox.coordinates import Coordinates

COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (255, 0, 0)
COLOR_RED = (0, 0, 255)

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


class Box2D:

    id_counter = 0
    boxes = []

    LIFETIME = 25
    MINIMAL_SCORE_CORRECTION = 0.3
    MINIMAL_SCORE_NEW = 0.7

    # SIMILARITY_THRESHOLD = 50  # smaller side

    def __init__(self, coordinates, size, confident_score, info, tracker):

        super().__init__()

        coordinates.convert_to_fixed(info)
        size.convert_to_fixed(info)

        self._parent_tracker = tracker
        self._object_size = size

        self.score = confident_score

        self._corrected = False
        self._info = info
        self._kalman = cv2.KalmanFilter(4, 4)

        print(self._kalman.measurementNoiseCov)
        self.lifetime = Box2D.LIFETIME

        # TODO finetune parametrs

        self._kalman.transitionMatrix = KALMAN_TRANSITION_MATRIX
        self._kalman.measurementMatrix = KALMAN_MESUREMENT_POSITION_MATRIX
        self._kalman.processNoiseCov = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], np.float32) * 0.5

        self._kalman.measurementNoiseCov = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0.1, 0],
            [0, 0, 0, 0.1],
        ], np.float32) * 30

        # self._kalman.errorCovPost = np.array([
        #     [1, 1, 1, 1],
        #     [1, 1, 1, 1],
        #     [1, 1, 1, 1],
        #     [1, 1, 1, 1],
        # ], np.float32) * 0.1

        self._kalman.statePost = np.array([
            [np.float32(coordinates.x)],  # x-coord
            [np.float32(coordinates.y)],  # y-coord
            [np.float32(0)],  # dx
            [np.float32(0)],  # dy
        ])

        self.id = Box2D.id_counter

        Box2D.id_counter += 1

    @property
    def center(self):
        return Coordinates(int(self._kalman.statePost[0][0]), int(self._kalman.statePost[1][0]))

    @property
    def size(self):
        return self._object_size

    @property
    def velocity(self):
        return int(np.abs(self._kalman.statePost[2][0])), int(np.abs(self._kalman.statePost[3][0]))

    @property
    def left_anchor(self):
        x_min = int((self.center.x - self.size.width / 2))
        y_min = int((self.center.y - self.size.height / 2))

        return x_min, y_min

    @property
    def right_anchor(self):
        x_max = int((self.center.x + self.size.width / 2))
        y_max = int((self.center.y + self.size.height / 2))

        return x_max, y_max

    @property
    def tracker(self):
        x_tracker = int(self.center.x)
        y_tracker = int((self.center.y + self.size.height / 2))

        return x_tracker, y_tracker

    @property
    def car_info(self):
        return str(self.id)

    @staticmethod
    def draw(image, anchors, area_of_interest, car_info, flow_diff, color=COLOR_GREEN, center=False):
        """
        prints 2D box on given image
        :param image: selected image to plot on
        :param center: if should print the midle point of box
        :return:
        """

        top_left, bot_right, center_point, tracker = anchors
        # cv2.rectangle(image, top_left, bot_right, color, BOX_THICKNESS)

        cv2.circle(image, center_point, area_of_interest, COLOR_BLUE, BOX_THICKNESS)

        # cv2.circle(image, tracker, CENTER_POINT_RADIUS, COLOR_RED, BOX_THICKNESS)

        # if center:
            # cv2.circle(image, center_point, CENTER_POINT_RADIUS, color, BOX_THICKNESS)

        # cv2.putText(image, "{0:.3f}".format(dx), top_left, 1, 1, COLOR_BLUE, 2)
        cv2.putText(image, car_info, top_left, 1, 1, COLOR_BLUE, 2)

        dx, dy = flow_diff
        x, y = center_point

        cv2.line(image, center_point, (int(x+dx), int(y+dy)), COLOR_RED, 1)

    def area_of_interest(self):
        width = self.size.width
        height = self.size.height

        diagonal = np.sqrt(width * width + height * height)
        return int(diagonal/2)

    def anchors(self):
        return self.left_anchor, self.right_anchor, self.center.tuple(), self.tracker

    def predict(self):
        self._kalman.predict()
        self.lifetime -= 1

    def update_position(self, size, score, new_coordinates):

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
        self._corrected = True

        self.score = score
        self._object_size = size

        self.lifetime = Box2D.LIFETIME

    def update_flow(self):
        x_flow, y_flow = self.global_flow()

        mesurement = np.array([
            [np.float32(0)],
            [np.float32(0)],
            [np.float32(x_flow/3)],
            [np.float32(y_flow/3)]
        ])

        self._kalman.measurementMatrix = KALMAN_MESUREMENT_FLOW_MATRIX
        self._kalman.correct(mesurement)
        self._corrected = True

        self.lifetime = Box2D.LIFETIME

    def in_radius(self, new_coordinates):

        width = self.size.width
        height = self.size.height

        diagonal = np.sqrt(width * width + height * height)

        max_pixels = diagonal/2

        return self.center.distance(new_coordinates) < max_pixels

    def global_flow(self):

        global_dx = 0
        global_dy = 0

        number_of_flows = 0
        for flow in zip(self._parent_tracker.new_positions, self._parent_tracker.old_positions):
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

    def serialize(self):
        return self.anchors(), self.area_of_interest(), self.car_info, self.global_flow()

    def __del__(self):
        print(f"deleting {self.id}, {self.center}")
        pass

