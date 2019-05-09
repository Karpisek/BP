import cv2
import numpy as np

import params

from pipeline import ThreadedPipeBlock
from pipeline.base.pipeline import is_frequency, Mode


class FrameLoader(ThreadedPipeBlock):

    def __init__(self, output, info):
        super().__init__(info=info, pipe_id=params.FRAME_LOADER_ID, output=output)

    def _before(self):
        if not self._info.traffic_lights_repository.ready:
            image = self._info.read(params.DETECTOR_IMAGE_WIDTH)

            self._info.traffic_lights_repository.find(image=image)
            self._info.reopen()

    def _mode_changed(self, new_mode):
        super()._mode_changed(new_mode)

        if new_mode in [Mode.DETECTION, Mode.CALIBRATION_CORRIDORS]:
            self._info.reopen()

    def _step(self, seq):
        if self._mode == Mode.CALIBRATION_VP and self._info.calibrated:
            self._update_mode(Mode.DETECTION)

        image = self._info.read()

        if is_frequency(seq, params.VIDEO_PLAYER_FREQUENCY):
            message = (seq, np.copy(image))
            self.send(message, pipe_id=params.VIDEO_PLAYER_ID)

        if is_frequency(seq, params.TRAFFIC_LIGHT_OBSERVER_FREQUENCY):
            message = (seq, np.copy(image))
            self.send(message, pipe_id=params.TRAFFIC_LIGHT_OBSERVER_ID)

        if is_frequency(seq, params.CALIBRATOR_FREQUENCY):
            message = (seq, np.copy(image))
            self.send(message, pipe_id=params.CALIBRATOR_ID, block=False)

        if is_frequency(seq, params.TRACKER_OPTICAL_FLOW_FREQUENCY):
            message = (seq, np.copy(image))
            self.send(message, pipe_id=params.TRACKER_ID)

        if is_frequency(seq, params.DETECTOR_CAR_FREQUENCY):
            message = (seq, np.copy(image))
            self.send(message, pipe_id=params.DETECTOR_CAR_ID)

        if is_frequency(seq, params.VIOLATION_WRITER_FREQUENCY):
            message = (seq, np.copy(image))
            self.send(message, pipe_id=params.VIOLATION_WRITER_ID)

    def _after(self):
        self.send(EOFError, pipe_id=params.VIDEO_PLAYER_ID)
