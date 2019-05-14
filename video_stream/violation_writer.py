import json
import os
import shutil
from collections import deque
from copy import deepcopy

from primitives import constants
from primitives.enums import Color, Mode
from pipeline.base.pipeline import ThreadedPipeBlock
from video_stream.video_writer import VideoWriter


class ViolationWriter(ThreadedPipeBlock):
    """
    Writes detected violations to separate file. Each violation is saved in short video and corresponding
    annotation file containing annotation of car for every frame.

    Each violation car is assignet to instance of VideoWriter.

    After computation is done, statistics about examined scene are saved as well. New folder is generated for all
    files about examined video. Containing video files, annotation files and statistic file.

    Holds history of frames for new VideoWrite to start on.
    """

    def _mode_changed(self, new_mode):
        super()._mode_changed(new_mode)

    def __init__(self, info, program_arguments):
        """
        :param info: instance of InputInfo containing all information of examined video
        :param program_arguments: instance of Parser used for setting root directory of output directory
        """

        super().__init__(info=info, pipe_id=constants.VIOLATION_WRITER_ID, work_modes=[Mode.DETECTION], deamon=False)
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
        self._history = deque(maxlen=constants.VIOLATION_WRITER_SEQUENCE_LENGTH)

    def _before(self):
        """
        Cleans output directory with no warning, if no such directory is being found it creates new one.
        """

        if os.path.exists(self._path):
            shutil.rmtree(self._path)

        os.makedirs(self._path)

    def _step(self, seq):
        """
        Receives new message from video player - just for check there is no user interaction closing the program.
        Gets new frame from FrameLoader and all information about objects in scene from Observer

        Delegates information about violations to VideoWriter instances for each of violation car.
        When new car makes violation, new instance of VideoWriter is generated.
        If lifetime of any VideoWriter is over specified threshold - it is being closed

        :param seq: current sequence number
        """

        self.receive(pipe_id=constants.VIDEO_PLAYER_ID)

        loader_seq, image = self.receive(pipe_id=constants.FRAME_LOADER_ID)
        observer_seq, boxes_repository, lights_state = self.receive(pipe_id=constants.OBSERVER_ID)
        package = image, boxes_repository, lights_state

        self._save_light_state(lights_state, seq)

        for car_id in boxes_repository.red_riders.keys():
            if car_id not in self._captured_ids:
                self._video_writers[car_id] = VideoWriter(info=self._info,
                                                          car_id=car_id,
                                                          history=deepcopy(self._history),
                                                          path=self._path)
                self._captured_ids.append(car_id)

        for car_id in boxes_repository.orange_riders.keys():
            if car_id not in self._captured_ids:
                self._video_writers[car_id] = VideoWriter(info=self._info,
                                                          car_id=car_id,
                                                          history=deepcopy(self._history),
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
        """
        Writes statistics and calibration information in selected file.
        Closes VideoWriter instances for violations.
        """

        self._write_statistics()
        self._write_calibration()

        for _, writer in list(self._video_writers.items()):
            writer.close()
            del self._video_writers[writer.car_id]

    def _append_to_history(self, package):
        """
        Appends new package to history. Only last x frames are saved
        :param package: package which should be added to history
        """

        if len(self._history) == constants.VIDEO_WRITER_HISTORY:
            self._history.popleft()

        self._history.append(package)

    def _write_statistics(self):
        """
        Writes statistics to selected file
        """

        if self._last_boxes_repository is not None:
            path = f"{self._path}/{str(self._info.calibration_mode)}_{constants.STATISTICS_LOG_FILENAME}"

            with open(path, "w") as file:
                data = self._last_boxes_repository.get_statistics()
                data.update({"light_states": self._light_states})

                json.dump(data, file)

    def _write_calibration(self):
        """
        Write calibration information to selected file
        """

        path = f"{self._path}/{str(self._info.calibration_mode)}_{constants.CALIBRATION_FILENAME}"

        with open(path, "w") as file:
            data = self._info.get_calibration()

            print(data)
            json.dump(data, file)

    def _save_light_state(self, lights_state, seq):
        """
        Saves current light state to history

        :param lights_state: current light state
        :param seq: current sequence number
        """

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

