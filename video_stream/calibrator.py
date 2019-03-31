import random

import cv2
import numpy as np

import params
from bbox import Box2D, Coordinates

from pc_lines.line import SamePointError, Line
from pc_lines.pc_line import PcLines
from pc_lines.vanishing_point import VanishingPoint
from pipeline import ThreadedPipeBlock


colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(100)]


class SyncError(Exception):
    pass


class Calibrator(ThreadedPipeBlock):
    def __init__(self, output=None, info=None):
        super().__init__(pipe_id=params.CALIBRATOR_ID, output=output)

        self._pc_lines = PcLines(info.width)
        self._info = info

        self._info.vanishing_points.append(VanishingPoint((548, -56)))

    def _step(self, seq):
        seq_loader, new_frame = self.receive(pipe_id=params.FRAME_LOADER_ID)
        seq_tracker, boxes_mask, boxes_mask_no_border, optical_flow_serialized = self.receive(pipe_id=params.TRACKER_ID)

        if seq_loader != seq_tracker:
            raise SyncError

        if len(self._info.vanishing_points) < 1:
            # self.detect_first_vanishing_point(new_frame, boxes_mask, boxes_mask_no_border)
            self.detect_first_vp()

        elif len(self._info.vanishing_points) < 2:
            self.detect_second_vanishing_point(new_frame, boxes_mask, boxes_mask_no_border)

        else:
            self.find_corridors()
            exit()

    # def detect_first_vanishing_point(self, new_frame, boxes_mask, boxes_mask_no_border) -> None:
    #     selected_areas = cv2.bitwise_and(new_frame, cv2.cvtColor(boxes_mask, cv2.COLOR_GRAY2BGR))
    #     blured = cv2.blur(selected_areas, (3, 3))
    #
    #     canny = cv2.Canny(blured, 50, 150, apertureSize=3)
    #     no_border_canny = cv2.bitwise_and(canny, boxes_mask_no_border)
    #     lines = cv2.HoughLinesP(image=no_border_canny,
    #                             rho=1,
    #                             theta=np.pi / 350,
    #                             threshold=params.CALIBRATOR_HLP_THRESHOLD,
    #                             minLineLength=params.CALIBRATOR_MIN_LINE_LENGTH,
    #                             maxLineGap=params.CALIBRATOR_MAX_LINE_GAP)
    #
    #     if lines is not None:
    #         for (x1, y1, x2, y2), in lines:
    #             point1 = x1, y1
    #             point2 = x2, y2
    #
    #             try:
    #                 self._pc_lines.pc_line_from_points(point1, point2)
    #                 cv2.line(selected_areas, point1, point2, params.COLOR_BLUE, 1)
    #             except SamePointError:
    #                 continue
    #
    #     print(self._pc_lines.count)
    #     if len(Box2D.lifelines()) > params.CALIBRATOR_TRACK_MINIMUM:
    #
    #         for lifeline in Box2D.lifelines():
    #             old_position, new_position = lifeline
    #             self._pc_lines.pc_line_from_points(tuple(old_position), tuple(new_position))
    #
    #         new_vanishing_point = self._pc_lines.find_most_line_cross(self._info)
    #         self._info.vanishing_points.append(new_vanishing_point)
    #         self._pc_lines.clear()
    #
    #     cv2.imwrite("test.jpg", selected_areas)

    def detect_first_vp(self):
        if len(Box2D.lifelines()) > params.CALIBRATOR_TRACK_MINIMUM:

            for lifeline in Box2D.lifelines():
                old_position, new_position = lifeline
                self._pc_lines.pc_line_from_points(tuple(old_position), tuple(new_position))

            new_vanishing_point = self._pc_lines.find_most_line_cross(preset_points=self._info.vp1_preset_points())
            self._info.vanishing_points.append(new_vanishing_point)
            self._pc_lines.clear()

    def detect_second_vanishing_point(self, new_frame, boxes_mask, boxes_mask_no_border) -> None:

        selected_areas = cv2.bitwise_and(new_frame, cv2.cvtColor(boxes_mask, cv2.COLOR_GRAY2BGR))
        blured = cv2.blur(selected_areas, (3, 3))

        canny = cv2.Canny(blured, 50, 150, apertureSize=3)
        no_border_canny = cv2.bitwise_and(canny, boxes_mask_no_border)
        lines = cv2.HoughLinesP(image=no_border_canny,
                                rho=1,
                                theta=np.pi / 350,
                                threshold=params.CALIBRATOR_HLP_THRESHOLD,
                                minLineLength=params.CALIBRATOR_MIN_LINE_LENGTH,
                                maxLineGap=params.CALIBRATOR_MAX_LINE_GAP)

        vp1 = self._info.vanishing_points[0]
        if lines is not None:
            for (x1, y1, x2, y2), in lines:
                point1 = x1, y1
                point2 = x2, y2

                try:
                    if vp1.coordinates.distance(Coordinates(x1, y1)) > vp1.coordinates.distance(Coordinates(x2, y2)):
                        line_to_vp = Line(point1, vp1.point)
                    else:
                        line_to_vp = Line(point2, vp1.point)

                    print(Line(point1, point2).angle(line_to_vp))
                    if Line(point1, point2).angle(line_to_vp) < 50 or Line(point1, point2).angle(line_to_vp) > 130:
                        cv2.line(selected_areas, point1, point2, params.COLOR_WHITE, 1)
                        continue

                    self._pc_lines.pc_line_from_points(point1, point2)
                    cv2.line(selected_areas, point1, point2, params.COLOR_BLUE, 1)
                except SamePointError:
                    continue

        print(self._pc_lines.count)
        if self._pc_lines.count > params.CALIBRATOR_TRACK_MINIMUM:

            for lifeline in Box2D.lifelines():
                old_position, new_position = lifeline
                self._pc_lines.pc_line_from_points(tuple(old_position), tuple(new_position))

            new_vanishing_point = self._pc_lines.find_most_line_cross()
            self._info.vanishing_points.append(new_vanishing_point)
            self._pc_lines.clear()

        cv2.imwrite("test.jpg", selected_areas)

    def find_corridors(self):
        mask = np.zeros(shape=(self._info.height, self._info.width, 3), dtype=np.uint8)
        mask = Box2D.draw_lifelines(image=mask,
                                    lifelines=Box2D.lifelines(),
                                    color=params.COLOR_LIFELINE,
                                    thickness=params.CALIBRATOR_LIFELINE_THICKNESS)

        self._info.corridors_repository.find_corridors(mask)
