import random

import cv2
import numpy as np

import params
import time

from queue import Full

from pc_lines.line import Line, SamePointError
from pc_lines.pc_line import PcLines
from pc_lines.vanishing_point import VanishingPoint
from pipeline import ThreadedPipeBlock


colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(100)]


class SyncError(Exception):
    pass


class Calibrator(ThreadedPipeBlock):
    def __init__(self, output=None, info=None):
        super().__init__(pipe_id=params.CALIBRATOR_ID, output=output)

        self._vanishing_points = [VanishingPoint() for _ in range(3)]
        self._pc_lines = PcLines(info.width)

    def _step(self, seq):
        seq_loader, new_frame = self.receive(pipe_id=params.FRAME_LOADER_ID)
        seq_tracker, boxes_mask, boxes_mask_no_border = self.receive(pipe_id=params.TRACKER_ID)

        # if seq_loader != seq_tracker:
        #     raise SyncError
        #
        # if not self._vanishing_points[0].found():
        #     self.detect_first_vanishing_point(new_frame, boxes_mask, boxes_mask_no_border)
        # elif not self._vanishing_points[1].found():
        #     self.detect_second_vanishing_point()
        #
        # print(seq_tracker, seq_loader)

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
                    self._pc_lines.pc_line_from_points(point1, point2)
                except SamePointError:
                    continue

            vp = self._pc_lines.find_most_line_cross()
            print(vp)
            cv2.circle(new_frame, vp.point, 2, (0,255,0),2)
            cv2.imwrite("test.jpg", new_frame)

        print("tady")

    def detect_second_vanishing_point(self):
        pass