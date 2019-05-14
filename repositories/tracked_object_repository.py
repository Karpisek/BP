import numpy as np

from primitives import constants
from repositories.models.tracked_object import TrackedObject
from primitives.enums import Mode


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
