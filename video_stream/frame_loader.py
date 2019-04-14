import cv2
import numpy as np

import params

from pipeline import ThreadedPipeBlock
from pipeline.base.pipeline import is_frequency


class FrameLoader(ThreadedPipeBlock):
    def __init__(self, output, info):
        super().__init__(pipe_id=params.FRAME_LOADER_ID, output=output)
        self._info = info
        self._subtractor = cv2.createBackgroundSubtractorMOG2(history=params.FRAME_LOADER_SUBTRACTOR_HISTORY,
                                                              varThreshold=params.FRAME_LOADER_THRESHOLD)

        self._uncalibrated_phase = True

    def _start(self):
        if not self._info.traffic_lights_repository.ready:
            final_image = self._info.read(params.DETECTOR_IMAGE_WIDTH)

            for _ in range(params.DETECTOR_LIGHT_IMAGE_ROW):
                image = self._info.read(params.DETECTOR_IMAGE_WIDTH)

                final_image = np.maximum(final_image, image)

            self._info.traffic_lights_repository.find(image=final_image)
            self._info.reopen()

    def _step(self, seq):
        if self._uncalibrated_phase and self._info.calibrated:
            self._info.reopen()
            self._uncalibrated_phase = False

        try:
            image = self._info.read()
        except EOFError:
            image = None

        for _ in range(int(self._info.fps / 20)):
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
