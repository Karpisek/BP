import cv2
import numpy as np

import params
from pc_lines import TrafficCorridorRepository


class Info:
    def __init__(self, path):

        self.input = cv2.VideoCapture(path)
        self._fps = self.input.get(cv2.CAP_PROP_FPS)
        self._height = self.input.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self._width = self.input.get(cv2.CAP_PROP_FRAME_WIDTH)
        self._vanishing_points = []

        self.track_boxes = True

        self._corridors_repository = TrafficCorridorRepository(self)

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

    @property
    def vanishing_points(self) -> []:
        return self._vanishing_points

    @property
    def corridors_repository(self):
        return self._corridors_repository

    def calibrated(self) -> bool:
        return len(self.vanishing_points) > 1

    def draw_vanishing_points(self, image) -> None:

        if not self.calibrated:
            return

        p6 = 0, int(self.height)
        p7 = 1 * int(self.width / 4), int(3 * self.height / 4)
        p8 = 2 * int(self.width / 4), int(3 * self.height / 4)
        p9 = 3 * int(self.width / 4), int(3 * self.height / 4)
        p10 = 4 * int(self.width / 4), int(3 * self.height / 4)

        for i in range(len(self.vanishing_points)):
            cv2.circle(image, self.vanishing_points[i].point, 2, params.COLOR_RED, 1)
            cv2.line(image, p6, self.vanishing_points[i].point, params.COLOR_GREEN, 1)
            cv2.line(image, p7, self.vanishing_points[i].point, params.COLOR_GREEN, 1)
            cv2.line(image, p8, self.vanishing_points[i].point, params.COLOR_GREEN, 1)
            cv2.line(image, p9, self.vanishing_points[i].point, params.COLOR_GREEN, 1)
            cv2.line(image, p10, self.vanishing_points[i].point, params.COLOR_GREEN, 1)

        return image

    def draw_corridors(self, image) -> np.ndarray:
        corridor_mask = self.corridors_repository.get_mask()
        inverse_corridor_mask = cv2.bitwise_not(corridor_mask)

        # removing area from image
        image = cv2.bitwise_and(image, image, inverse_corridor_mask)

        # add corridors to image
        return cv2.add(image, corridor_mask)
