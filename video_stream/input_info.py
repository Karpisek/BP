import cv2
import numpy as np

import params
from bbox import Area, Coordinates
from detectors import Color
from pc_lines import TrafficCorridorRepository


class Info:
    def __init__(self, path):

        self.input = cv2.VideoCapture(path)
        self._fps = self.input.get(cv2.CAP_PROP_FPS)
        self._height = self.input.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self._width = self.input.get(cv2.CAP_PROP_FRAME_WIDTH)

        if self._width > params.FRAME_LOADER_MAX_WIDTH:
            ratio = self._height / self._width
            self._width = params.FRAME_LOADER_MAX_WIDTH
            self._height = self._width * ratio
            self._resize = True
        else:
            self._resize = False

        self._frame_count = self.input.get(cv2.CAP_PROP_FRAME_COUNT)

        self._path = path

        self._vanishing_points = []
        self._traffic_lights = []

        self.track_boxes = True

        self._corridors_repository = TrafficCorridorRepository(self)

        self._tracker_start_area = Area(info=self,
                                        top_left=Coordinates(0, self.height/2),
                                        bottom_right=Coordinates(self.width, self.height))

        self._tracker_update_area = Area(info=self,
                                         top_left=Coordinates(0, self.height/2),
                                         bottom_right=Coordinates(self.width, self.height))

        print(f"INFO: fps: {self.fps}, height: {self.height}, width: {self.width}, frame count: {self.frame_count}")

    @property
    def resize(self):
        return self._resize

    @property
    def frame_count(self):
        return self._frame_count

    @property
    def calibrated(self):
        return self._corridors_repository.ready

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

    def reopen(self):
        self.input.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def draw_vanishing_points(self, image) -> None:
        points = [(0, int(self.height)),
                  (1 * int(self.width / 4), int(3 * self.height / 4)),
                  (2 * int(self.width / 4), int(3 * self.height / 4)),
                  (3 * int(self.width / 4), int(3 * self.height / 4)),
                  (4 * int(self.width / 4), int(3 * self.height / 4))]

        for i in range(len(self.vanishing_points)):
            for p in points:

                self.vanishing_points[i].draw_line(image=image,
                                                   point=p,
                                                   color=params.COLOR_YELLOW,
                                                   thickness=2)

        return image

    def draw_corridors(self, image) -> np.ndarray:
        corridor_mask = self.corridors_repository.get_mask(fill=False)
        return cv2.add(image, corridor_mask)

    def draw(self, image, lights_status):
        color1 = params.COLOR_GRAY
        color2 = params.COLOR_GRAY
        color3 = params.COLOR_GRAY

        if lights_status == Color.RED or lights_status == Color.RED_ORANGE:
            color1 = params.COLOR_RED

        if lights_status == Color.ORANGE or lights_status == Color.RED_ORANGE:
            color2 = params.COLOR_ORANGE

        if lights_status == Color.GREEN:
            color3 = params.COLOR_GREEN

        cv2.rectangle(img=image,
                      pt1=(10, 10),
                      pt2=(30, 70),
                      color=params.COLOR_BLACK,
                      thickness=params.FILL)

        cv2.circle(img=image,
                   center=(20, 20),
                   radius=5,
                   color=color1,
                   thickness=params.FILL)

        cv2.circle(img=image,
                   center=(20, 40),
                   radius=5,
                   color=color2,
                   thickness=params.FILL)

        cv2.circle(img=image,
                   center=(20, 60),
                   radius=5,
                   color=color3,
                   thickness=params.FILL)

    def resize(self, width, height):
        self._width = width
        self._width = height

