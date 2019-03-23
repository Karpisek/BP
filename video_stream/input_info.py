import cv2


class Info:
    def __init__(self, path):

        self.input = cv2.VideoCapture(path)
        self._fps = self.input.get(cv2.CAP_PROP_FPS)
        self._height = self.input.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self._width = self.input.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.track_boxes = True

    def set_info(self, fps=None, height=None, width=None, track_boxes=True):
        self._fps = fps
        self._height = height
        self._width = width
        self.track_boxes = track_boxes

        print(f"fps: {self.fps}, height: {self.height}, width: {self.width}, track boxes: {self.track_boxes}")

    @property
    def width(self) -> int:
        return int(self._width)

    @width.setter
    def width(self, value: float):
        self._width = value

    @property
    def height(self):
        return int(self._height)

    @height.setter
    def height(self, value: float):
        self._height = value

    @property
    def fps(self) -> int:
        return int(self._fps)

    @fps.setter
    def fps(self, value: float):
        self._fps = value
