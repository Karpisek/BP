from copy import deepcopy
from enum import Enum, IntEnum

import cv2

import params
import numpy as np
from detectors import Coordinates
from pipeline import ThreadedPipeBlock, is_frequency
from pipeline.base.pipeline import Mode
from pipeline.traffic_light_observer import Color


class CarBehaviourMode(IntEnum):
    NORMAL = 1
    LINE_CROSSED = 2
    RED_DRIVER = 3
    ORANGE_DRIVER = 4


class Box2D:
    def __init__(self, car_id):
        self._top_left = None
        self._bottom_right = None
        self._car_id = car_id
        self._red_distance_traveled = 0

        self._behaviour = CarBehaviourMode.NORMAL

        self._lifetime = 1

    @property
    def behaviour(self):
        return self._behaviour

    @property
    def top_left(self):
        return self._top_left

    @property
    def bottom_right(self):
        return self._bottom_right

    @property
    def car_id(self):
        return self._car_id

    @property
    def initialized(self):
        return self._top_left is not None

    @property
    def lifetime(self):
        return self._lifetime

    @property
    def tracker_point(self):
        return Coordinates((self._bottom_right[0] + self._top_left[0]) / 2, self._bottom_right[1])

    @property
    def front_point(self):
        return Coordinates((self._bottom_right[0] + self._top_left[0]) / 2, self._top_left[1])

    @property
    def center_point(self):
        return Coordinates((self._bottom_right[0] + self._top_left[0]) / 2, (self._bottom_right[1] + self._top_left[1]) / 2)

    def get_corridor(self, info):
        return info.corridors_repository.get_corridor(self.tracker_point)

    def distance_from_vanishing_point(self, info):
        return info.vp1.coordinates.distance(self.tracker_point)

    def tic(self):
        self._lifetime -= 1

    def update(self, anchors, lights_state, info, velocity):
        previous_coordinates = None

        if self.initialized:
            previous_coordinates = self.tracker_point

        self._top_left, self._bottom_right, _ = anchors
        self._lifetime += 1

        if previous_coordinates is not None:
            if lights_state == Color.RED or lights_state == Color.RED_ORANGE:
                if info.corridors_repository.line_crossed(previous_coordinates, self.tracker_point):
                    self._behaviour = CarBehaviourMode.RED_DRIVER

                if self._behaviour == CarBehaviourMode.LINE_CROSSED:
                    self._red_distance_traveled += velocity

            if lights_state == Color.ORANGE:
                if info.corridors_repository.line_crossed(previous_coordinates, self.tracker_point):
                    self._behaviour = CarBehaviourMode.ORANGE_DRIVER

            if lights_state == Color.GREEN and self._behaviour not in [CarBehaviourMode.ORANGE_DRIVER, CarBehaviourMode.RED_DRIVER]:
                if info.corridors_repository.behind_line(self.tracker_point):
                    self._behaviour = CarBehaviourMode.LINE_CROSSED

        if self._red_distance_traveled > params.OBSERVER_RED_STANDER_MAX_TRAVEL:
            self._behaviour = CarBehaviourMode.RED_DRIVER

        return self._behaviour

    def draw(self, image):
        if self._behaviour == CarBehaviourMode.RED_DRIVER:
            color = params.COLOR_RED
        elif self._behaviour == CarBehaviourMode.ORANGE_DRIVER:
            color = params.COLOR_ORANGE
        else:
            color = params.COLOR_GREEN

        cv2.rectangle(img=image,
                      pt1=self._top_left,
                      pt2=self._bottom_right,
                      color=color,
                      thickness=params.OBSERVER_BOX_THICKNESS)

        cv2.rectangle(img=image,
                      pt1=self._top_left,
                      pt2=(self._top_left[0] + 30, self._top_left[1] - 15),
                      color=color,
                      thickness=params.FILL)

        cv2.circle(img=image,
                   center=self.tracker_point.tuple(),
                   color=params.COLOR_RED,
                   radius=5,
                   thickness=params.FILL)

        cv2.putText(img=image,
                    text=self.car_id,
                    org=self._top_left,
                    fontFace=1,
                    fontScale=1,
                    color=params.COLOR_BLACK,
                    thickness=2)

    def __str__(self):
        return f"[Box id: {self._car_id}]"


class BBoxRepository:
    def __init__(self):
        self._boxes = {}
        self._red_riders = {}
        self._orange_riders = {}
        self._all_cars = {}

    @property
    def boxes(self):
        return self._boxes

    @property
    def red_riders(self):
        return self._red_riders

    @property
    def orange_riders(self):
        return self._orange_riders

    @property
    def car_count(self):
        return len(self._all_cars)

    @property
    def red_drivers_count(self):
        return len(self._red_riders)

    @property
    def orange_drivers_count(self):
        return len(self._orange_riders)

    def get_boxes_in_corridors(self, info):
        if info.vp1 is None or not info.corridors_repository.corridors_found:
            return {}

        corridor_ids = info.corridors_repository.corridor_ids
        sorted_boxes = sorted(self._boxes.values(), key=lambda b: b.distance_from_vanishing_point(info))

        return {corridor: [box for box in sorted_boxes if box.get_corridor(info) == corridor]for corridor in corridor_ids}

    def insert_or_update(self, anchors, car_id, velocity, lights_state, info, seq):
        if car_id not in self._boxes:
            self._boxes[car_id] = Box2D(car_id)

        behaviour = self._boxes[car_id].update(anchors, lights_state, info, velocity)

        if behaviour in [CarBehaviourMode.ORANGE_DRIVER, CarBehaviourMode.RED_DRIVER, CarBehaviourMode.LINE_CROSSED]:
            if car_id not in self._all_cars:
                self._all_cars[car_id] = seq

        if behaviour == CarBehaviourMode.RED_DRIVER:
            try:
                self._orange_riders.pop(car_id)
            except KeyError:
                pass

            if car_id not in self._red_riders:
                self._red_riders[car_id] = seq

        if behaviour == CarBehaviourMode.ORANGE_DRIVER:
            if car_id not in self._orange_riders:
                self._orange_riders[car_id] = seq

    def check_lifetime(self):
        for key, box in self._boxes.copy().items():
            box.tic()
            if box.lifetime < 0:
                self._boxes.pop(key)

    def draw(self, image):
        for key, box in self._boxes.items():
            box.draw(image)

        return image

    def get_box_by_id(self, car_id):
        return self._boxes[car_id]

    def draw_statistics(self, image, info):
        statistics_panel = np.full(shape=(30, info.width, 3),
                                   dtype=np.uint8,
                                   fill_value=params.COLOR_WHITE)

        cv2.putText(img=statistics_panel,
                    text=f"Total car count: {self.car_count}",
                    org=(10, 20),
                    fontFace=1,
                    fontScale=1,
                    color=params.COLOR_BLACK,
                    thickness=1)

        cv2.putText(img=statistics_panel,
                    text=f"Red drivers: {self.red_drivers_count}",
                    org=(300, 20),
                    fontFace=1,
                    fontScale=1,
                    color=params.COLOR_BLACK,
                    thickness=1)

        cv2.putText(img=statistics_panel,
                    text=f"Orange drivers: {self.orange_drivers_count}",
                    org=(500, 20),
                    fontFace=1,
                    fontScale=1,
                    color=params.COLOR_BLACK,
                    thickness=1)

        return np.concatenate((statistics_panel, image), axis=0)

    def get_statistics(self):
        return {
            "total_cars_count": self.car_count,
            "red_drivers_count": self.red_drivers_count,
            "orange_drivers_count": self.orange_drivers_count,
            "red_drivers": self.red_riders,
            "orange_drivers": self.orange_riders,
            "all_drivers": self._all_cars
        }

    def restart(self):
        self._boxes = {}
        self._red_riders = {}
        self._orange_riders = {}
        self._all_cars = {}


class Observer(ThreadedPipeBlock):
    def __init__(self, info, output, pipe_id=params.OBSERVER_ID):

        super().__init__(pipe_id=pipe_id,
                         output=output)

        self._info = info
        self._previous_lights_state = None
        self._bounding_boxes_repository = BBoxRepository()

    def _mode_changed(self, new_mode):
        if new_mode == Mode.DETECTION:
            self._bounding_boxes_repository.restart()

    def _step(self, seq):
        tracker_seq, tracked_objects = self.receive(pipe_id=params.TRACKER_ID)
        lights_seq, current_lights_state = self.receive(pipe_id=params.TRAFFIC_LIGHT_OBSERVER_ID)

        for tracked_object in tracked_objects:
            anchors, _, car_info, car_velocity = tracked_object
            self._bounding_boxes_repository.insert_or_update(anchors, car_info, car_velocity, current_lights_state, self._info, seq)

        self._bounding_boxes_repository.check_lifetime()

        if self._previous_lights_state == Color.RED_ORANGE and current_lights_state == Color.GREEN:
            if not self._info.corridors_repository.stopline_found:
                boxes_in_corridors = self._bounding_boxes_repository.get_boxes_in_corridors(info=self._info)

                print(boxes_in_corridors)
                for corridors in boxes_in_corridors.values():
                    try:
                        first_car = corridors[0]
                        self._info.corridors_repository.add_stop_point(first_car.center_point)
                    except IndexError:
                        continue

        if is_frequency(seq, params.VIDEO_PLAYER_FREQUENCY):
            message = seq, deepcopy(self._bounding_boxes_repository), current_lights_state
            self.send(message, pipe_id=params.VIDEO_PLAYER_ID, block=False)

        if is_frequency(seq, params.VIOLATION_WRITER_FREQUENCY):
            message = seq, deepcopy(self._bounding_boxes_repository), current_lights_state
            self.send(message, pipe_id=params.VIOLATION_WRITER_ID)

        self._previous_lights_state = current_lights_state
