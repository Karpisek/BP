import random

import cv2
import numpy as np

import params
import time

from queue import Full

from pc_lines.line import Line, SamePointError
from pipeline import ThreadedPipeBlock


colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(100)]


class SyncError(Exception):
    pass


class Calibrator(ThreadedPipeBlock):
    def __init__(self, output=None):
        super().__init__(pipe_id=params.CALIBRATOR_ID, output=output)

        self._vanishing_points = [VanishingPoint() for _ in range(3)]

    def _step(self, seq):
        seq_loader, new_frame = self.receive(pipe_id=params.FRAME_LOADER_ID)
        seq_tracker, boxes_mask, boxes_mask_no_border = self.receive(pipe_id=params.TRACKER_ID)

        if seq_loader != seq_tracker:
            raise SyncError

        if not self._vanishing_points[0].found():
            self.detect_first_vanishing_point(new_frame, boxes_mask, boxes_mask_no_border)
        elif not self._vanishing_points[1].found():
            self.detect_second_vanishing_point()

        # print(seq_tracker, seq_loader)

    def find_vanishing_point(self):
        time.sleep(2)
        self._vanishing_points.append(VanishingPoint())

    def detect_first_vanishing_point(self, new_frame, boxes_mask, boxes_mask_no_border):
        selected_areas = cv2.bitwise_and(new_frame, cv2.cvtColor(boxes_mask, cv2.COLOR_GRAY2BGR))

        blured = cv2.blur(selected_areas, (5, 5))

        canny = cv2.Canny(blured, 50, 150, apertureSize=3)
        no_border_canny = cv2.bitwise_and(canny, boxes_mask_no_border)

        lines = cv2.HoughLinesP(no_border_canny, 1, np.pi / 350, 20, minLineLength=30, maxLineGap=5)

        helper_image = cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB)

        if lines is not None:
            for (x1, y1, x2, y2), in lines:
                point1 = x1, y1
                point2 = x2, y2

                try:
                    # detected_line = Line(point1, point2)

                    cv2.line(helper_image, point1, point2, colors[random.randint(0, 99)], 1, 1)

                except SamePointError:
                    continue

        cv2.imwrite("test.jpg", helper_image)

        print("tady")

    def detect_second_vanishing_point(self):
        pass


class VanishingPointError(Exception):
    pass


class VanishingPoint:
    def __init__(self, point=None, angle=None):

        if point is not None and angle is not None:
            raise VanishingPointError

        # in case vanishing point is defined
        if point is not None:
            self.point = point
            self.infinity = False

        # in case vanishing point is near infinity
        if angle is not None:
            self.angle = angle
            self.infinity = True

    def found(self):
        return False


