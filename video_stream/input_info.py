import cv2
import numpy as np

import params
from bbox import Area, Coordinates
from detectors import TrafficLightsRepository
from pc_lines import TrafficCorridorRepository


class Info:
    def __init__(self, path):

        self.input = cv2.VideoCapture(path)
        self._fps = self.input.get(cv2.CAP_PROP_FPS)
        self._height = self.input.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self._width = self.input.get(cv2.CAP_PROP_FRAME_WIDTH)

        self._vanishing_points = []
        self._traffic_lights = []

        self.track_boxes = True

        self._corridors_repository = TrafficCorridorRepository(self)
        # self._traffic_lights_repository = TrafficLightsRepository()

        self._tracker_start_area = Area(info=self,
                                        top_left=Coordinates(0, self.height/3),
                                        bottom_right=Coordinates(self.width, self.height))

        self._tracker_update_area = Area(info=self,
                                         top_left=Coordinates(0, self.height/4),
                                         bottom_right=Coordinates(self.width, self.height))

    @property
    def principal_point(self) -> Coordinates:
        return Coordinates(self.width / 2, self.height / 2)

    @property
    def start_area(self):
        return self._tracker_start_area

    @property
    def update_area(self):
        return self._tracker_update_area

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

        points = [(0, int(self.height)),
                  (1 * int(self.width / 4), int(3 * self.height / 4)),
                  (2 * int(self.width / 4), int(3 * self.height / 4)),
                  (3 * int(self.width / 4), int(3 * self.height / 4)),
                  (4 * int(self.width / 4), int(3 * self.height / 4))]

        for i in range(len(self.vanishing_points)):
            for p in points:

                self.vanishing_points[i].draw_line(image=image,
                                                   point=p,
                                                   color=params.RANDOM_COLOR[i],
                                                   thickness=1)

        return image

    def draw_corridors(self, image) -> np.ndarray:
        corridor_mask = self.corridors_repository.get_mask()
        inverse_corridor_mask = cv2.bitwise_not(corridor_mask)

        # removing area from image
        image = cv2.bitwise_and(image, image, inverse_corridor_mask)

        # add corridors to image
        return cv2.add(image, corridor_mask)

    def vp1_preset_points(self) -> ([(int, int)]):
        points = []
        for x in range(int((2 * self.width) / params.CALIBRATOR_GRID_DENSITY)):
            for y in range(int(self.height / params.CALIBRATOR_GRID_DENSITY)):
                new_x = int(x * params.CALIBRATOR_GRID_DENSITY - self.width / 2)
                new_y = int(y * params.CALIBRATOR_GRID_DENSITY - 9 * self.height / 10)

                points.append((new_x, new_y))

        return points
