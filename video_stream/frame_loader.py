"""
FrameLoader class definition
"""

__author__ = "Miroslav Karpisek"
__email__ = "xkarpi05@stud.fit.vutbr.cz"
__date__ = "14.5.2019"

import numpy as np
import constants

from pipeline import ThreadedPipeBlock
from pipeline.base.pipeline import is_frequency, Mode


class FrameLoader(ThreadedPipeBlock):
    """
    A class used to loading images from selected video and delegation frames to connected PipeBlocks.
    It handles cases when the input file has to be re-opened.
    Checks if mode of computation has to be changed.
    """

    def __init__(self, output, info):
        """
        :param output: list of PipeBlock instances used for current frame delegation
        :param info: instance of informations about the current video
        """
        super().__init__(info=info, pipe_id=constants.FRAME_LOADER_ID, output=output)

    def _before(self):
        """
        Before computation is done, where no traffic light is selected by user. T
        he first 200th frame is take and calls traffic lights repository for automatic
        detection of traffic light on that frame.
        """

        if not self._info.traffic_lights_repository.ready:

            for _ in range(200):
                self._info.read()

            image = self._info.read(constants.DETECTOR_IMAGE_WIDTH)

            self._info.traffic_lights_repository.find(image=image)
            self._info.reopen()

    def _mode_changed(self, new_mode):
        """
        Reopens video file when mode is changed

        :param new_mode: new mode
        """

        super()._mode_changed(new_mode)

        if new_mode in [Mode.DETECTION, Mode.CALIBRATION_CORRIDORS]:
            self._info.reopen()

    def _step(self, seq):
        """
        Reads new image from input video.
        Creates copies of it and sends them to all outputs if current sequence number satisfies desired frequency
        for certain receiver.

        :param seq: current sequence number
        """

        if self._mode == Mode.CALIBRATION_VP and self._info.corridors_repository.corridors_found:
            self._update_mode(Mode.CALIBRATION_CORRIDORS)

        if self._mode == Mode.CALIBRATION_CORRIDORS and self._info.calibrated:
            self._update_mode(Mode.DETECTION)

        image = self._info.read()

        if is_frequency(seq, constants.VIDEO_PLAYER_FREQUENCY):
            message = (seq, np.copy(image))
            self.send(message, pipe_id=constants.VIDEO_PLAYER_ID)

        if is_frequency(seq, constants.TRAFFIC_LIGHT_OBSERVER_FREQUENCY):
            message = (seq, np.copy(image))
            self.send(message, pipe_id=constants.TRAFFIC_LIGHT_OBSERVER_ID)

        if is_frequency(seq, constants.CALIBRATOR_FREQUENCY):
            message = (seq, np.copy(image))
            self.send(message, pipe_id=constants.CALIBRATOR_ID, block=False)

        if is_frequency(seq, constants.TRACKER_OPTICAL_FLOW_FREQUENCY):
            message = (seq, np.copy(image))
            self.send(message, pipe_id=constants.TRACKER_ID)

        if is_frequency(seq, constants.DETECTOR_CAR_FREQUENCY):
            message = (seq, np.copy(image))
            self.send(message, pipe_id=constants.DETECTOR_CAR_ID)

        if is_frequency(seq, constants.VIOLATION_WRITER_FREQUENCY):
            message = (seq, np.copy(image))
            self.send(message, pipe_id=constants.VIOLATION_WRITER_ID)

    def _after(self):
        """
        Delegates message containing EOFError class used for signalization for the end of input video.
        """

        self.send(EOFError, pipe_id=constants.VIDEO_PLAYER_ID)
