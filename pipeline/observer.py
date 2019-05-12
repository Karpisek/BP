import cv2
import params
import numpy as np

from copy import deepcopy
from enum import IntEnum
from detectors import Coordinates
from pc_lines.line import ransac, Line, SamePointError
from pipeline import ThreadedPipeBlock, is_frequency
from pipeline.base.pipeline import Mode
from pipeline.traffic_light_observer import Color


class CarBehaviourMode(IntEnum):
    NORMAL = 1
    LINE_CROSSED = 2
    RED_DRIVER = 3
    ORANGE_DRIVER = 4


class Box2D:
    """
    Bounding box around car.
    Holds printable information about car and its behavior classification in video.
    """

    def __init__(self, car_id):
        self._top_left = None
        self._bottom_right = None
        self._car_id = car_id
        self._red_distance_traveled = 0

        self._behaviour = CarBehaviourMode.NORMAL

        self._lifetime = 1
        self._history = []

    @property
    def behaviour(self):
        """
        :return: behaviour classification
        """
        return self._behaviour

    @property
    def top_left(self):
        """
        :return: coordinates top left corner of bounding box
        """

        return self._top_left

    @property
    def bottom_right(self):
        """
        :return: coordinates of bottom right corner of bounding box
        """

        return self._bottom_right

    @property
    def car_id(self):
        """
        :return: car id observed by this bounding box
        """

        return self._car_id

    @property
    def initialized(self):
        """
        :return: if bounding box is set
        """

        return self._top_left is not None

    @property
    def lifetime(self):
        """
        :return: lifetime of this bounding box.
        """

        return self._lifetime

    @property
    def tracker_point(self):
        """
        :return: coordinates of tracker point - center of bounding box bottom line
        """

        return Coordinates((self._bottom_right[0] + self._top_left[0]) / 2, self._bottom_right[1])

    @property
    def front_point(self):
        """
        :return: coordinates of front point - center of bounding box top line
        """

        return Coordinates((self._bottom_right[0] + self._top_left[0]) / 2, self._top_left[1])

    @property
    def center_point(self):
        """
        :return: center point of bounding box
        """

        return Coordinates((self._bottom_right[0] + self._top_left[0]) / 2, (self._bottom_right[1] + self._top_left[1]) / 2)

    def get_corridor(self, info):
        """
        Returns ID of corridor in which the tracked point of this bounding box lays

        :param info: instance of InputInfo used for getting corridor repository references
        :return: id of corridor
        """

        return info.corridors_repository.get_corridor(self.tracker_point)

    def distance_from_vanishing_point(self, info):
        """
        :param info: instance of InputInfo used for getting vanishing point references
        :return: distance from vanishing point in pixels
        """
        return info.vp1.coordinates.distance(self.tracker_point)

    def tic(self):
        """
        On each frame is 1 subtracted from bounding box lifetime
        """
        self._lifetime -= 1

    def update(self, anchors, lights_state, info, velocity):
        """
        Updates position, velocity and behavior depending on passed light state.
        On each update history of center points is saved for trajectory printing.

        :param anchors: new anchor position
        :param lights_state: current light state
        :param info: instance of InputInfo
        :param velocity: velocity of observed car
        :return: classification of behaviour of car
        """
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
                if info.corridors_repository.line_crossed(previous_coordinates, self.tracker_point):
                    self._behaviour = CarBehaviourMode.LINE_CROSSED

        if not len(self._history) or np.abs(self.center_point.y - self._history[-1].y) > 20:
            self._history.append(self.center_point)

        return self._behaviour

    def draw_boxes(self, image):
        """
        Helper method for draw bounding box with color corresponding to behaviour

        :param image: selected image to draw on
        :return: updated image
        """

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

        return image

    def draw_trajectories(self, image, method="second"):
        """
        Helper method for drawing car trajectories (trajectories of center point)

        :param image: selected image
        :param method: method of trajectory obtaining to be printed
        :return: updated image
        """

        if method == "second":
            for index, point in enumerate(self._history):
                cv2.circle(img=image,
                           center=point.tuple(),
                           color=params.COLOR_BLUE,
                           radius=5,
                           thickness=params.FILL)

                try:
                    cv2.line(img=image,
                             pt1=point.tuple(),
                             pt2=self._history[index + 1].tuple(),
                             color=params.COLOR_BLUE,
                             thickness=2)

                except IndexError:
                    pass

        if method == "first":
            try:
                Line(self._history[0].tuple(), self._history[-1].tuple()).draw(image, params.COLOR_RED, 1)
            except SamePointError:
                return

            cv2.circle(img=image,
                       center=self._history[0].tuple(),
                       color=params.COLOR_BLUE,
                       radius=5,
                       thickness=params.FILL)

            cv2.circle(img=image,
                       center=self._history[-1].tuple(),
                       color=params.COLOR_BLUE,
                       radius=5,
                       thickness=params.FILL)

            cv2.line(img=image,
                     pt1=self._history[0].tuple(),
                     pt2=self._history[-1].tuple(),
                     color=params.COLOR_BLUE,
                     thickness=2)

        position_history = [coordinates.tuple() for coordinates in self._history]
        line, value = ransac(position_history, position_history, 1)

        if line is not None and value > 5:
            line.draw(image, params.COLOR_RED, 2)

    def __str__(self):
        return f"[Box id: {self._car_id}]"


class BBoxRepository:
    """
    Bounding boxes repository.
    Holds all instances of bounding boxes in separate dictionaries
    """

    def __init__(self):
        self._boxes = {}
        self._red_riders = {}
        self._orange_riders = {}
        self._all_cars = {}

    @property
    def boxes(self):
        """
        :return: all bouniding boxes
        """

        return self._boxes

    @property
    def red_riders(self):
        """
        :return: all red drivers
        """

        return self._red_riders

    @property
    def orange_riders(self):
        """
        :return: all orange drivers
        """

        return self._orange_riders

    @property
    def car_count(self):
        """
        :return: total car count
        """

        return len(self._all_cars)

    @property
    def red_drivers_count(self):
        """
        :return: number of red rivers
        """

        return len(self._red_riders)

    @property
    def orange_drivers_count(self):
        """
        :return: orange drivers count
        """

        return len(self._orange_riders)

    def get_boxes_in_corridors(self, info):
        """
        :param info: instance of InputInfo used for corridor assigning
        :return: dictionary of assigned bounding boxes to corresponding traffic corridors
        """

        if info.vp1 is None or not info.corridors_repository.corridors_found:
            return {}

        corridor_ids = info.corridors_repository.corridor_ids
        sorted_boxes = sorted(self._boxes.values(), key=lambda b: b.distance_from_vanishing_point(info))

        return {corridor: [box for box in sorted_boxes if box.get_corridor(info) == corridor]for corridor in corridor_ids}

    def insert_or_update(self, anchors, car_id, velocity, lights_state, info, seq):
        """
        Updates or creates new instance of bounding box identified by car ID.
        If red or orange driver is detected, then it instance is being saved into corresponding dictionary

        :param anchors: anchor points of observed car box
        :param car_id: unique car id
        :param velocity: velocity of car
        :param lights_state: current light state
        :param info: instance of InputInfo for behaviour classification
        :param seq: current sequence number
        """

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
        """
        Control lifetime of all bounding boxes, if any of them has lifetime less then 0 it is not updated for a while
         -> it is removed from observed bouning boxes
        """

        for key, box in self._boxes.copy().items():
            box.tic()
            if box.lifetime < 0:
                self._boxes.pop(key)

    def draw_boxes(self, image):
        """
        Helper function for drawing all present boxes on image

        :param image: selected image to draw on
        :return: updated image
        """

        for key, box in self._boxes.items():
            box.draw_boxes(image)

        return image

    def draw_trajectories(self, image):
        """
        Helper function for drawing all trajectories of all center points of observed bounding boxes

        :param image: selected image
        :return: updated image
        """

        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        for key, box in self._boxes.items():
            box.draw_trajectories(image)

        return image

    def get_box_by_id(self, car_id):
        """
        :param car_id: desired car ID
        :return: instance of bounding box with desired ID
        """

        return self._boxes[car_id]

    def draw_statistics(self, image, info):
        """
        Helper function to draw statistics panel about passed cars on selected image

        :param image: selected image
        :param info: instance of InputInfo
        :return: updated image
        """

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
        """
        Provides serialized statistics in form of dictionary.

        :return: dictionary of serialized statistics
        """

        return {
            "total_cars_count": self.car_count,
            "red_drivers_count": self.red_drivers_count,
            "orange_drivers_count": self.orange_drivers_count,
            "red_drivers": self.red_riders,
            "orange_drivers": self.orange_riders,
            "all_drivers": self._all_cars
        }

    def restart(self):
        """
        Clears all dictionaries containing instances of bounding boxes.
        """

        self._boxes = {}
        self._red_riders = {}
        self._orange_riders = {}
        self._all_cars = {}


class Observer(ThreadedPipeBlock):
    """
    Observes the scene and combines obtained information about car and light objects.
    On every step deepcopy of BBoxRepository is send to output PipeBocks.

    While working in calibration mode it helps with detecting stop line by examining car behaviour on certain light
    states.
    """

    def __init__(self, info, output, pipe_id=params.OBSERVER_ID):
        """
        :param info: instance of InputInfo
        :param output: list of output instances of PipeBlock
        :param pipe_id: unique ID of this PipeBlock subclass
        """

        super().__init__(pipe_id=pipe_id,
                         output=output,
                         info=info)

        self._previous_lights_state = None
        self._bounding_boxes_repository = BBoxRepository()

    def _mode_changed(self, new_mode):
        """
        When detection mode starts BBoxRepositories are cleared.

        :param new_mode: new mode
        """

        super()._mode_changed(new_mode)

        if new_mode == Mode.DETECTION:
            self._bounding_boxes_repository.restart()

    def _step(self, seq):
        """
        Each step serialized tracked objects are received from Tracker.
        light states are received from TrafficLightObserver.
        For ever serialized tracked object is updated or inserted new bounding box using BBoxRepository

        If current light status is certain value and stop line is not found, then
        position of first car in every corridor is being used for it approximation.

        :param seq: current sequnece number
        """

        tracker_seq, tracked_objects = self.receive(pipe_id=params.TRACKER_ID)
        lights_seq, current_lights_state = self.receive(pipe_id=params.TRAFFIC_LIGHT_OBSERVER_ID)

        for tracked_object in tracked_objects:
            anchors, _, car_info, car_velocity = tracked_object
            self._bounding_boxes_repository.insert_or_update(anchors, car_info, car_velocity, current_lights_state, self._info, seq)

        self._bounding_boxes_repository.check_lifetime()

        if self._previous_lights_state in [Color.RED_ORANGE, Color.RED] and current_lights_state == Color.GREEN:
            if not self._info.corridors_repository.stopline_found:
                boxes_in_corridors = self._bounding_boxes_repository.get_boxes_in_corridors(info=self._info)

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
