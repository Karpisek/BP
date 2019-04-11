import cv2
import numpy as np

import params
from bbox import Area, Coordinates
from repositories.traffic_corridor_repository import TrafficCorridorRepository
from repositories.traffic_light_repository import TrafficLightsRepository
from pipeline.traffic_light_observer import Color


class Info:
    def __init__(self, video_path, light_detection_model, program_arguments):

        self._input = cv2.VideoCapture(video_path)

        self._fps = self._input.get(cv2.CAP_PROP_FPS)
        self._height = self._input.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self._width = self._input.get(cv2.CAP_PROP_FRAME_WIDTH)
        self._frame_count = self._input.get(cv2.CAP_PROP_FRAME_COUNT)
        self._ratio = self._height / self._width

        self._vanishing_points = []
        self._traffic_lights_repository = TrafficLightsRepository(model=light_detection_model, info=self)
        self._corridors_repository = TrafficCorridorRepository(self)

        print("taday")
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

        if program_arguments.insert_light:
            print("Please select as accurate as possible rectangle containing traffic light")

            self._traffic_lights_repository.select_manually(image)

        if program_arguments.insert_stop_line:
            print("Please select stop line")

    @property
    def traffic_lights_repository(self):
        return self._traffic_lights_repository

    @property
    def ratio(self):
        return self._ratio

    @property
    def traffic_lights_repository(self):
        return self._traffic_lights_repository

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

    @property
    def height(self):
        return int(self._height)

    @property
    def fps(self) -> int:
        return int(self._fps)

    @property
    def vanishing_points(self) -> []:
        return self._vanishing_points

    @property
    def corridors_repository(self):
        return self._corridors_repository

    def read(self, width=None):
        _, frame = self._input.read()

        if width is None:
            return frame
        else:
            return cv2.resize(frame, (width, int(width * self.ratio)))

    def reopen(self):
        self._input.set(cv2.CAP_PROP_POS_FRAMES, 0)

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
        return cv2.add(image, corridor_mask)

    def draw_detected_traffic_lights(self, image) -> np.ndarray:
        return self.traffic_lights_repository.draw(image)

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

