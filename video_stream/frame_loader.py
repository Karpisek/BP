import cv2
from threading import Thread

from pipeline import PipeBlock
from .tracker import APROXIMATION_FRAME_COUNT, OPTICAL_FLOW_PAUSE

IMAGE_WIDTH_FOR_CNN = 900


class FrameLoader(PipeBlock):
    def __init__(self, path, output, input_info):
        """
        Loads frames from given path and saves them in Queue
        async using multiprocessing
        :param path: given path for input
        """

        super().__init__(output)
        self._frame_rate = 0
        self._tape = cv2.VideoCapture(path)
        self.set_info(input_info)

        self._thread = Thread(target=self._run, args=(path, ))
        self._thread.daemon = True
        self._thread.start()

        pass

    def _run(self, path):
        """
        runs until are images in input stream
        saves them to queue
        :return: none
        """

        seq = 0
        while True:
            seq += 1
            _, image = self._tape.read()

            self.send_to((seq, image), out_pipe=1)

            if seq % OPTICAL_FLOW_PAUSE == 0:
                self.send_to((seq, image), out_pipe=2, in_pipe=1)

            if seq % APROXIMATION_FRAME_COUNT == 0:

                height, width, _ = image.shape
                scale = height / width
                image = cv2.resize(image, (IMAGE_WIDTH_FOR_CNN, int(IMAGE_WIDTH_FOR_CNN * scale)))

                self.send_to((seq, image), out_pipe=0)



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
