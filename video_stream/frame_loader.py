import time

import cv2
import numpy as np

import params

from pipeline import ThreadedPipeBlock
from pipeline.pipeline import is_frequency

IMAGE_WIDTH_FOR_CNN = 900


class FrameLoader(ThreadedPipeBlock):
    def __init__(self, output, info):
        """
        Loads frames from given path and saves them in Queue
        async using multiprocessing
        :param path: given path for input
        """

        super().__init__(pipe_id=params.FRAME_LOADER_ID, output=output)
        self._info = info
        self._subtractor = cv2.createBackgroundSubtractorMOG2(history=params.FRAME_LOADER_SUBTRACTOR_HISTORY,
                                                              varThreshold=params.FRAME_LOADER_THRESHOLD)

    def _step(self, seq):
        """
        runs until are images in input stream
        saves them to queue
        :return: none
        """

        seq += 1
        status, image = self._info.input.read()

        for _ in range(int(self._info.fps / 20)):
            status, image = self._info.input.read()

        if status is False:
            return

        # foreground = self._subtractor.apply(image)

        if is_frequency(seq, params.VIDEO_PLAYER_FREQUENCY):
            message = (seq, np.copy(image))
            self.send(message, pipe_id=params.VIDEO_PLAYER_ID)

        if is_frequency(seq, params.CALIBRATOR_FREQUENCY):
            message = (seq, np.copy(image))
            self.send(message, pipe_id=params.CALIBRATOR_ID, block=False)

        if is_frequency(seq, params.TRACKER_OPTICAL_FLOW_FREQUENCY):
            message = (seq, np.copy(image))
            self.send(message, pipe_id=params.TRACKER_ID)

        if is_frequency(seq, params.DETECTOR_CAR_FREQUENCY):

            height, width, _ = image.shape
            scale = height / width
            image = cv2.resize(image, (IMAGE_WIDTH_FOR_CNN, int(IMAGE_WIDTH_FOR_CNN * scale)))

            message = (seq, np.copy(image))
            self.send(message, pipe_id=params.DETECTOR_CAR_ID)

        if is_frequency(seq, params.DETECTOR_LIGHT_FREQUENCY):
            height, width, _ = image.shape
            scale = height / width
            image = cv2.resize(image, (IMAGE_WIDTH_FOR_CNN, int(IMAGE_WIDTH_FOR_CNN * scale)))

            message = (seq, np.copy(image))
            self.send(message, pipe_id=params.DETECTOR_LIGHT_ID)
