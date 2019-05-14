"""
InputInfo class definition
"""
from primitives.area import Area
from primitives.coordinates import Coordinates
from primitives.enums import CalibrationMode, Color
from video_stream.video_info import VideoInfo

__author__ = "Miroslav Karpisek"
__email__ = "xkarpi05@stud.fit.vutbr.cz"
__date__ = "14.5.2019"

import cv2
import numpy as np
from primitives import constants

from repositories.traffic_corridor_repository import TrafficCorridorRepository
from repositories.traffic_light_repository import TrafficLightsRepository


class Info(VideoInfo):
    """
    Subclass of VideoInfo providing informations about examined video.
    These informations are specific for red light violation detection solving.

    This class holds informations for example about detected vanishing points, corridors etc.
    """

    def __init__(self, video_path, light_detection_model, program_arguments):
        """
        :param video_path: input video path
        :param light_detection_model: path to the light detection model
        :param program_arguments: instance of Parser class containing program arguments
        """

        super().__init__(video_path)

        self._vanishing_points = []
        self._traffic_lights_repository = TrafficLightsRepository(model=light_detection_model, info=self)
        self._corridors_repository = TrafficCorridorRepository(self)

        self._detect_vanishing_points = True
        self._calibration_mode = CalibrationMode.AUTOMATIC

        self._tracker_start_area = Area(info=self,
                                        top_left=Coordinates(0, self.height / 3),
                                        top_right=Coordinates(self.width, self.height / 2),
                                        bottom_right=Coordinates(self.width, self.height),
                                        bottom_left=Coordinates(0, self.height))

        self._tracker_update_area = Area(info=self,
                                         top_left=Coordinates(0, self.height / 4),
                                         top_right=Coordinates(self.width, self.height / 4),
                                         bottom_right=Coordinates(self.width, self.height),
                                         bottom_left=Coordinates(0, self.height))

        # solve given program arguments
        self._solve_program_arguments(program_arguments)

        print(f"INFO: fps: {self.fps}, height: {self.height}, width: {self.width}, frame count: {self.frame_count}")

    @property
    def calibration_mode(self):
        """
        :return: mode of calibration
        """

        return self._calibration_mode

    @property
    def vp1(self):
        """
        :return: first vanishing point, if detected. Else None
        """

        if not self.vanishing_points:
            return None
        else:
            return self.vanishing_points[0]

    @property
    def vp2(self):
        """
        :return: second vanishing point, if detected. Else None
        """

        if not self.vanishing_points:
            return None
        else:
            return self.vanishing_points[1]

    @property
    def traffic_lights_repository(self):
        """
        :return: traffic lights repository
        """

        return self._traffic_lights_repository

    @property
    def calibrated(self):
        """
        :return: True if all informations for detection are satisfied
        """

        return True if not self._detect_vanishing_points else self._corridors_repository.ready

    @property
    def principal_point(self) -> Coordinates:
        """
        :return: principal point coordinates of input video, Assuming that the principal point is in middle of image
        """

        return Coordinates(self.width / 2, self.height / 2)

    @property
    def start_area(self):
        """
        :return: area used for creating new instances of detected cars
        """

        return self._tracker_start_area

    @property
    def update_area(self):
        """
        :return: area inside which is car tracking done
        """

        return self._tracker_update_area

    @property
    def vanishing_points(self) -> []:
        """
        :return: list of vanishing points
        """

        return self._vanishing_points

    @property
    def corridors_repository(self):
        """
        :return: repository of corridors
        """

        return self._corridors_repository

    def _solve_program_arguments(self, program_arguments):
        """
        Depending on specified program arguments it provides possibility to annotate traffic lights location and
        corridors on video.

        Prints on console for guiding user interaction.

        :param program_arguments: program arguments used for parsing
        """

        for _ in range(200):
            self.read()

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

    def draw_vanishing_points(self, image) -> np.ndarray:
        """
        Helper method for drawing detected vanishing points found on image.
        Converts passed image into grayscale and draw lines heading to all detected vanishing points.

        :param image: selected image
        :return: updated image
        """

        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        points = [(0, int(self.height)),
                  (1 * int(self.width / 4), int(3 * self.height / 4)),
                  (2 * int(self.width / 4), int(3 * self.height / 4)),
                  (3 * int(self.width / 4), int(3 * self.height / 4)),
                  (4 * int(self.width / 4), int(3 * self.height / 4))]

        # for i in range(len(self.vanishing_points)):
        #     for p in points:
        #         self.vanishing_points[i].draw_line(image=image,
        #                                            point=p,
        #                                            color=params.COLOR_VANISHING_DIRECTIONS[i],
        #                                            thickness=2)
        try:
            for p in points:
                self.vanishing_points[0].draw_line(image=image,
                                                   point=p,
                                                   color=constants.COLOR_VANISHING_DIRECTIONS[0],
                                                   thickness=5)
        except IndexError:
            pass

        return image

    def draw_corridors(self, image) -> np.ndarray:
        """
        Helper method for drawing corridors on passed image.
        Converts passed image into grayscale and draws detected corridors on passed image.

        :param image: selected image
        :return: updated image
        """

        corridor_mask = self.corridors_repository.get_mask(fill=True)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        return cv2.add(corridor_mask, image)

    def draw_detected_traffic_lights(self, image) -> np.ndarray:
        """
        Helper function for drawing detected traffic light position.

        :param image: passed image
        :return: image containing bounding box around detected traffic lights
        """

        return self.traffic_lights_repository.draw(image)

    @staticmethod
    def draw_synthetic_traffic_lights(image, lights_status):
        """
        Helper function for painting current detected light status on passed image

        :param image: selected image to draw on
        :param lights_status: current light status
        :return: updated image
        """

        color1 = constants.COLOR_GRAY
        color2 = constants.COLOR_GRAY
        color3 = constants.COLOR_GRAY

        if lights_status == Color.RED or lights_status == Color.RED_ORANGE:
            color1 = constants.COLOR_RED

        if lights_status == Color.ORANGE or lights_status == Color.RED_ORANGE:
            color2 = constants.COLOR_ORANGE

        if lights_status == Color.GREEN:
            color3 = constants.COLOR_GREEN

        cv2.rectangle(img=image,
                      pt1=(10, 10),
                      pt2=(60, 145),
                      color=constants.COLOR_BLACK,
                      thickness=constants.FILL)

        cv2.rectangle(img=image,
                      pt1=(10, 10),
                      pt2=(60, 145),
                      color=constants.COLOR_WHITE,
                      thickness=3)

        cv2.circle(img=image,
                   center=(35, 35),
                   radius=15,
                   color=color1,
                   thickness=constants.FILL)

        cv2.circle(img=image,
                   center=(35, 75),
                   radius=15,
                   color=color2,
                   thickness=constants.FILL)

        cv2.circle(img=image,
                   center=(35, 120),
                   radius=15,
                   color=color3,
                   thickness=constants.FILL)

        return image

    def get_calibration(self) -> {}:
        """
        Provides serialized information of detected vanishing points, corridors and traffic lights.

        :return: dictionary of serialized data
        """

        data = {"vanishing points": [vp.serialize() for vp in self.vanishing_points]}
        data.update(self.traffic_lights_repository.serialize())
        data.update(self.corridors_repository.serialize())

        return data






