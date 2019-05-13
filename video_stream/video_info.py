"""
InputInfo class definition
"""
import cv2

import constants


class VideoInfo:
    """
    Class handles operations on opened file. It encapsulates API around opened video-stream.
    """

    def __init__(self, video_path):
        """
        :param video_path: path of input video stream
        """

        self._input = cv2.VideoCapture(video_path)
        filename_with_extension = video_path.rsplit('/', 1)[1]
        self._file_name = filename_with_extension.rsplit('.', 1)[0]

        self._fps = self._input.get(cv2.CAP_PROP_FPS)
        self._height = self._input.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self._width = self._input.get(cv2.CAP_PROP_FRAME_WIDTH)
        self._resize = False

        self._frame_count = int(self._input.get(cv2.CAP_PROP_FRAME_COUNT) / (int(self._fps / constants.FRAME_LOADER_MAX_FPS) + 1))
        self._ratio = self._height / self._width

        if self._width > constants.FRAME_LOADER_MAX_WIDTH:
            self._width = int(constants.FRAME_LOADER_MAX_WIDTH)
            self._height = int(constants.FRAME_LOADER_MAX_WIDTH * self._ratio)
            self._resize = True

    @property
    def filename(self):
        """
        :return: opened input video filename
        """

        return self._file_name

    @property
    def ratio(self):
        """
        :return: width vs. height ratio of input video
        """
        return self._ratio

    @property
    def frame_count(self):
        """
        :return: frame count of opened video
        """

        return self._frame_count

    @property
    def height(self):
        """
        :return: height of frame in opened video
        """

        return int(self._height)

    @property
    def width(self) -> int:
        """
        :return: width of video frames
        """

        return int(self._width)

    @property
    def fps(self) -> int:
        """
        :return: frames per second of opened video
        """

        return int(self._fps)

    def read(self, width=None):
        """
        Reads new frame from opened video.
        If number of frames per second of video is higher then
        specified by constant, it throws a number of them away.

        :raise EOFError when end of input
        :return: new frame
        """

        status, frame = self._input.read()

        for _ in range(int(self.fps / constants.FRAME_LOADER_MAX_FPS)):
            status, frame = self._input.read()

        if not status:
            raise EOFError

        if width is not None:
            return cv2.resize(frame, (width, int(width * self.ratio)))
        elif self._resize:
            return cv2.resize(frame, (self._width, self._height))
        else:
            return frame

    def reopen(self):
        """
        Sets the recording head to the first frame in input video
        """

        self._input.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def resize(self, width, height):
        """
        Re-sizes input frame size. Does not control if aspect ratio is same.

        :param width: selected new width
        :param height: selected new height
        """

        self._width = width
        self._width = height

