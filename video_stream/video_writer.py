import json
import shutil

import cv2
import os

import numpy as np

import params

from collections import deque
from copy import deepcopy

from pipeline import ThreadedPipeBlock
from pipeline.base.pipeline import Mode
from repositories.traffic_light_repository import Color


class ViolationWriter(ThreadedPipeBlock):

    def _mode_changed(self, new_mode):
        super()._mode_changed(new_mode)

    def __init__(self, info, program_arguments):
        super().__init__(info=info, pipe_id=params.VIOLATION_WRITER_ID, work_modes=[Mode.DETECTION], deamon=False)
        self._video_writers = {}
        self._captured_ids = []
        self._light_states = {"red": [],
                              "orange": [],
                              "red_orange": [],
                              "green": [],
                              "none": []}

        self._current_light_state = None
        self._last_boxes_repository = None
        self._path = f"{program_arguments.output_dir}/{self._info.filename}"
        self._history = deque(maxlen=params.VIOLATION_WRITER_SEQUENCE_LENGTH)

    def _before(self):
        if os.path.exists(self._path):
            shutil.rmtree(self._path)

        os.makedirs(self._path)

    def _step(self, seq):
        self.receive(pipe_id=params.VIDEO_PLAYER_ID)

        loader_seq, image = self.receive(pipe_id=params.FRAME_LOADER_ID)
        observer_seq, boxes_repository, lights_state = self.receive(pipe_id=params.OBSERVER_ID)
        package = image, boxes_repository, lights_state

        self._save_light_state(lights_state, seq)

        for car_id in boxes_repository.red_riders.keys():
            if car_id not in self._captured_ids:
                self._video_writers[car_id] = VideoWriter(info=self._info,
                                                          car_id=car_id,
                                                          image_history=deepcopy(self._history),
                                                          path=self._path)
                self._captured_ids.append(car_id)

        for car_id in boxes_repository.orange_riders.keys():
            if car_id not in self._captured_ids:
                self._video_writers[car_id] = VideoWriter(info=self._info,
                                                          car_id=car_id,
                                                          image_history=deepcopy(self._history),
                                                          path=self._path)
                self._captured_ids.append(car_id)

        for _, writer in list(self._video_writers.items()):
            writer.add_package(package)

            if writer.lifetime < 0:
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
            path = f"{self._path}/{str(self._info.calibration_mode)}_{params.STATISTICS_LOG_FILENAME}"

            with open(path, "w") as file:
                data = self._last_boxes_repository.get_statistics()
                data.update({"light_states": self._light_states})

                json.dump(data, file)

    def _write_calibration(self):
        path = f"{self._path}/{str(self._info.calibration_mode)}_{params.CALIBRATION_FILENAME}"

        with open(path, "w") as file:
            data = self._info.get_calibration()

            print(data)
            json.dump(data, file)

    def _save_light_state(self, lights_state, seq):
        if lights_state != self._current_light_state:
            if lights_state == Color.GREEN:
                self._light_states["green"].append(seq)

            elif lights_state == Color.ORANGE:
                self._light_states["orange"].append(seq)

            elif lights_state == Color.RED:
                self._light_states["red"].append(seq)

            elif lights_state == Color.RED_ORANGE:
                self._light_states["red_orange"].append(seq)

            else:
                self._light_states["none"].append(seq)

            self._current_light_state = lights_state


class VideoWriter:
    def __init__(self, info, car_id, image_history, path):
        self._path = f"{path}/{car_id}"
        self._car_id = car_id
        self._info = info
        self._output = cv2.VideoWriter(self._path + ".avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), self._info.fps,
                                       (self._info.width, self._info.height))
        self._lifetime = 2 * len(image_history)
        self._annotation_output = {"top_left": [],
                                   "bottom_right": [],
                                   "behaviour": []}

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
        self._output.release()
        with open(self._path + ".json", "w") as file:
            json.dump(self._annotation_output, file)
