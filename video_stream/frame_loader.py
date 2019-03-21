import cv2

from params import VIDEO_PLAYER_ID, TRACKER_ID, DETECTOR_ID, FRAME_LOADER_ID
from pipeline import ThreadedPipeBlock
from .tracker import APROXIMATION_FRAME_COUNT, OPTICAL_FLOW_PAUSE

IMAGE_WIDTH_FOR_CNN = 900


class FrameLoader(ThreadedPipeBlock):
    def __init__(self, path, output, input_info):
        """
        Loads frames from given path and saves them in Queue
        async using multiprocessing
        :param path: given path for input
        """

        super().__init__(pipe_id=FRAME_LOADER_ID, output=output)
        self._frame_rate = 0
        self._tape = cv2.VideoCapture(path)
        self.set_info(input_info)

    def _step(self, seq):
        """
        runs until are images in input stream
        saves them to queue
        :return: none
        """

        seq += 1
        _, image = self._tape.read()

        message = (seq, image)
        self.send(message, pipe_id=VIDEO_PLAYER_ID)

        if seq % OPTICAL_FLOW_PAUSE == 0:
            self.send(message, pipe_id=TRACKER_ID)

        if seq % APROXIMATION_FRAME_COUNT == 0:

            height, width, _ = image.shape
            scale = height / width
            image = cv2.resize(image, (IMAGE_WIDTH_FOR_CNN, int(IMAGE_WIDTH_FOR_CNN * scale)))

            self.send((seq, image), pipe_id=DETECTOR_ID)



            # time.sleep(0.1)

    @property
    def frame_rate(self):
        """
        :return: fps of current video sequence
        """
        return self._frame_rate

    def set_info(self, input_info):
        fps = self._tape.get(cv2.CAP_PROP_FPS)
        height = self._tape.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width = self._tape.get(cv2.CAP_PROP_FRAME_WIDTH)

        input_info.set_info(fps=fps, height=height, width=width)
