import shutil

import cv2
import os

import numpy as np

import params

from collections import deque
from copy import deepcopy

from pipeline import ThreadedPipeBlock


class ViolationWriter(ThreadedPipeBlock):

    def _mode_changed(self, new_mode):
        pass

    def __init__(self, info, program_arguments):
        super().__init__(pipe_id=params.VIOLATION_WRITER_ID, work_modes=params.VIOLATION_WRITER_WORKMODES, deamon=False)
        self._info = info
        self._video_writers = {}
        self._last_boxes_repository = None
        self._path = f"{program_arguments.output_dir}/{self._info.filename}"
        self._history = deque(maxlen=21)

    def _before(self):
        if os.path.exists(self._path):
            shutil.rmtree(self._path)

        os.makedirs(self._path)

    def _step(self, seq):
        self.receive(pipe_id=params.VIDEO_PLAYER_ID)

        loader_seq, image = self.receive(pipe_id=params.FRAME_LOADER_ID)
        observer_seq, boxes_repository, lights_state = self.receive(pipe_id=params.OBSERVER_ID)
        package = image, boxes_repository, lights_state

        for car_id in boxes_repository.red_riders:
            if car_id not in self._video_writers.keys():
                self._video_writers[car_id] = VideoWriter(info=self._info,
                                                          car_id=car_id,
                                                          image_history=deepcopy(self._history),
                                                          path=self._path)

        for car_id in boxes_repository.orange_riders:
            if car_id not in self._video_writers.keys():
                self._video_writers[car_id] = VideoWriter(info=self._info,
                                                          car_id=car_id,
                                                          image_history=deepcopy(self._history),
                                                          path=self._path)

        for _, writer in list(self._video_writers.items()):
            writer.add_package(package)

            if writer.lifetime == 0:
                writer.close()
                del self._video_writers[writer.car_id]

        self._append_to_history(package)
        self._last_boxes_repository = boxes_repository

    def _after(self):
        self._write_statistics()
        self._write_calibration()

        for _, writer in list(self._video_writers.items()):
            writer.close()
            del self._video_writers[writer.car_id]

    def _append_to_history(self, package):
        if len(self._history) == 20:
            self._history.popleft()

        self._history.append(package)

    def _write_statistics(self):
        if self._last_boxes_repository is not None:
            path = f"{self._path}/{params.STATISTICS_LOG_FILENAME}"

            with open(path, "w") as file:
                self._last_boxes_repository.write_statistics(file)

    def _write_calibration(self):
        path = f"{self._path}/{params.CALIBRATION_FILENAME}"

        with open(path, "w") as file:
            self._info.write_calibration(file)


class VideoWriter:
    def __init__(self, info, car_id, image_history, path):
        self._path = f"{path}/{car_id}.avi"
        self._car_id = car_id
        self._info = info
        self._output = cv2.VideoWriter(self._path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), self._info.fps, (self._info.width, self._info.height))
        self._lifetime = len(image_history)
        for package in image_history:
            self.add_package(package)

    @property
    def car_id(self):
        return self._car_id

    @property
    def lifetime(self):
        return self._lifetime

    def add_package(self, package):
        image, boxes_repository, lights_state = package

        image = np.copy(image)

        try:
            boxes_repository.get_box_by_id(self.car_id).draw(image)
        except KeyError:
            pass

        image = self._info.draw_syntetic_traffic_lights(image, lights_state)

        self._output.write(image)
        self._lifetime -= 1

    def close(self):
        self._output.release()
