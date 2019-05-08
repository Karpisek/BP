import json
from enum import Enum

import cv2
import numpy as np

import params
from bbox import Area, Coordinates
from repositories.traffic_corridor_repository import TrafficCorridorRepository
from repositories.traffic_light_repository import TrafficLightsRepository
from pipeline.traffic_light_observer import Color


class CalibrationMode(Enum):
    AUTOMATIC = 0
    LIGHTS_MANUAL = 1
    CORRIDORS_MANUAL = 2
    MANUAL = 3

    def __str__(self):
        return self.name.lower()


class VideoInfo:
    def __init__(self, video_path):
        self._input = cv2.VideoCapture(video_path)
        filename_with_extension = video_path.rsplit('/', 1)[1]
        self._file_name = filename_with_extension.rsplit('.', 1)[0]

        self._fps = self._input.get(cv2.CAP_PROP_FPS)
        self._height = self._input.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self._width = self._input.get(cv2.CAP_PROP_FRAME_WIDTH)
        self._resize = False

        self._frame_count = int(self._input.get(cv2.CAP_PROP_FRAME_COUNT) / (int(self._fps / params.FRAME_LOADER_MAX_FPS) + 1))

        self._ratio = self._height / self._width

        if self._width > params.FRAME_LOADER_MAX_WIDTH:
            self._width = int(params.FRAME_LOADER_MAX_WIDTH)
            self._height = int(params.FRAME_LOADER_MAX_WIDTH * self._ratio)
            self._resize = True

    @property
    def filename(self):
        return self._file_name

    @property
    def ratio(self):
        return self._ratio

    @property
    def frame_count(self):
        return self._frame_count

    @property
    def height(self):
        return int(self._height)

    @property
    def fps(self) -> int:
        return int(self._fps)

    def read(self, width=None):
        """
        :raise EOFError when end of input
        :return:
        """

        status, frame = self._input.read()

        for _ in range(int(self.fps / params.FRAME_LOADER_MAX_FPS)):
            status, frame = self._input.read()

        if not status:
            raise EOFError

        if width is not None:
            print(frame.shape)
            print(width, int(width*self.ratio))
            return cv2.resize(frame, (width, int(width * self.ratio)))
        elif self._resize:
            return cv2.resize(frame, (self._width, self._height))
        else:
            return frame

    def reopen(self):
        self._input.set(cv2.CAP_PROP_POS_FRAMES, 0)


class Info(VideoInfo):
    def __init__(self, video_path, light_detection_model, program_arguments):

        super().__init__(video_path)

        self._vanishing_points = []
        self._traffic_lights_repository = TrafficLightsRepository(model=light_detection_model, info=self)
        self._corridors_repository = TrafficCorridorRepository(self)

        self._detect_vanishing_points = True
        self._calibration_mode = CalibrationMode.AUTOMATIC

        # solve given program arguments
        self._solve_program_arguments(program_arguments)

        self._tracker_start_area = Area(info=self,
                                        top_left=Coordinates(0, self.height/2),
                                        bottom_right=Coordinates(self.width, self.height))

        self._tracker_update_area = Area(info=self,
                                         top_left=Coordinates(0, self.height/4),
                                         bottom_right=Coordinates(self.width, self.height))

        print(f"INFO: fps: {self.fps}, height: {self.height}, width: {self.width}, frame count: {self.frame_count}")

    def _solve_program_arguments(self, program_arguments):

        image = self.read()
        self.reopen()

        if program_arguments.insert_corridors:
            print("Please select edges of corridors in frame")

            self._corridors_repository.select_manually(image)
            self._detect_vanishing_points = False
            self._calibration_mode = CalibrationMode.CORRIDORS_MANUAL

        if program_arguments.insert_light:
            print("Please select as accurate as possible rectangle containing traffic light")

            self._traffic_lights_repository.select_manually(image)

            if self._calibration_mode == CalibrationMode.AUTOMATIC:
                self._calibration_mode = CalibrationMode.LIGHTS_MANUAL
            else:
                self._calibration_mode = CalibrationMode.MANUAL

    @property
    def calibration_mode(self):
        return self._calibration_mode

    @property
    def vp1(self):
        if not self.vanishing_points:
            return None
        else:
            return self.vanishing_points[0]

    @property
    def traffic_lights_repository(self):
        return self._traffic_lights_repository

    @property
    def calibrated(self):
        return True if not self._detect_vanishing_points else self._corridors_repository.ready

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

    @property
    def vanishing_points(self) -> []:
        return self._vanishing_points

    @property
    def corridors_repository(self):
        return self._corridors_repository

    def draw_vanishing_points(self, image) -> np.ndarray:
        points = [(0, int(self.height)),
                  (1 * int(self.width / 4), int(3 * self.height / 4)),
                  (2 * int(self.width / 4), int(3 * self.height / 4)),
                  (3 * int(self.width / 4), int(3 * self.height / 4)),
                  (4 * int(self.width / 4), int(3 * self.height / 4))]

        for i in range(len(self.vanishing_points)):
            for p in points:
                self.vanishing_points[i].draw_line(image=image,
                                                   point=p,
                                                   color=params.COLOR_VANISHING_DIRECTIONS[i],
                                                   thickness=2)

        return image

    def draw_corridors(self, image) -> np.ndarray:
        corridor_mask = self.corridors_repository.get_mask(fill=True)

        # grayscale_mask = cv2.cvtColor(corridor_mask, cv2.COLOR_RGB2GRAY)
        # _, thresholded_mask = cv2.threshold(grayscale_mask, 1, 255, cv2.THRESH_BINARY)
        # mask = cv2.cvtColor(thresholded_mask, cv2.COLOR_GRAY2RGB)
        #
        # image = cv2.subtract(image, mask)
        # image += corridor_mask

        return cv2.add(corridor_mask, image)

    def draw_detected_traffic_lights(self, image) -> np.ndarray:
        return self.traffic_lights_repository.draw(image)

    def draw_syntetic_traffic_lights(self, image, lights_status):
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

        return image

    def resize(self, width, height):
        self._width = width
        self._width = height

    def get_calibration(self):
        data = {"vanishing points": [vp.serialize() for vp in self.vanishing_points]}
        data.update(self.traffic_lights_repository.serialize())
        data.update(self.corridors_repository.serialize())

        return data






