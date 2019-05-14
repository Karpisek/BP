"""
VideoWriter class definition
"""

__author__ = "Miroslav Karpisek"
__email__ = "xkarpi05@stud.fit.vutbr.cz"
__date__ = "14.5.2019"

import json
import cv2


class VideoWriter:
    """
    Writes frames and annotations for one certain Violation.
    Each VideoWriter is being identified by ID of examined car
    """

    def __init__(self, info, car_id, history, path):
        """
        :param info: instance of InputInfo containing all information about examined scene.
        :param car_id: id of current car for searching in history
        :param history: history of frames and another saved information
        :param path: path of directory where should output be put
        """

        self._path = f"{path}/{car_id}"
        self._car_id = car_id
        self._info = info
        self._output = cv2.VideoWriter(self._path + ".avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), self._info.fps,
                                       (self._info.width, self._info.height))
        self._lifetime = 2 * len(history)
        self._annotation_output = {"top_left": [],
                                   "bottom_right": [],
                                   "behaviour": []}

        for package in history:
            self.add_package(package)

    @property
    def car_id(self):
        """
        :return: ID of examined car
        """

        return self._car_id

    @property
    def lifetime(self):
        """
        :return: lifetime of this VideoWriter
        """

        return self._lifetime

    def add_package(self, package):
        """
        Writes new information to files from given package

        :param package: package with mew frame and information about object on it.
        """

        image, boxes_repository, lights_state = package

        try:
            car_box = boxes_repository.get_box_by_id(self.car_id)
            self._annotation_output["top_left"].append(car_box.top_left)
            self._annotation_output["bottom_right"].append(car_box.bottom_right)
            self._annotation_output["behaviour"].append(car_box.behaviour)

        except KeyError:
            self._annotation_output["top_left"].append(None)
            self._annotation_output["bottom_right"].append(None)
            self._annotation_output["behaviour"].append(None)

        self._output.write(image)
        self._lifetime -= 1

    def close(self):
        """
        Releases video output file and writes annotation to .json file
        """

        self._output.release()
        with open(self._path + ".json", "w") as file:
            json.dump(self._annotation_output, file)
