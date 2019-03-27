import random

import cv2
import numpy as np

import params
from bbox import Box2D

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

        self._vanishing_points = []
        self._pc_lines = PcLines(info.width)
        self.info = info

    def draw_vanishing_points(self, image, info) -> None:

        p1 = 0, int(info.height/2)
        p2 = 1 * int(info.width/4), int(info.height/2)
        p3 = 2 * int(info.width/4), int(info.height/2)
        p4 = 3 * int(info.width/4), int(info.height/2)
        p5 = 4 * int(info.width/4), int(info.height/2)

        p6 = 0, int(info.height)
        p7 = 1 * int(info.width / 4), int(3 * info.height / 4)
        p8 = 2 * int(info.width / 4), int(3 * info.height / 4)
        p9 = 3 * int(info.width / 4), int(3 * info.height / 4)
        p10 = 4 * int(info.width / 4), int(3 * info.height / 4)

        for i in range(len(self._vanishing_points)):
            cv2.circle(image, self._vanishing_points[i].point, 2, params.COLOR_RED, 1)
            # cv2.line(image, p1, self._vanishing_points[i].point, params.COLOR_GREEN, 1)
            # cv2.line(image, p2, self._vanishing_points[i].point, params.COLOR_GREEN, 1)
            # cv2.line(image, p3, self._vanishing_points[i].point, params.COLOR_GREEN, 1)
            # cv2.line(image, p4, self._vanishing_points[i].point, params.COLOR_GREEN, 1)
            # cv2.line(image, p5, self._vanishing_points[i].point, params.COLOR_GREEN, 1)

            cv2.line(image, p6, self._vanishing_points[i].point, params.COLOR_GREEN, 1)
            cv2.line(image, p7, self._vanishing_points[i].point, params.COLOR_GREEN, 1)
            cv2.line(image, p8, self._vanishing_points[i].point, params.COLOR_GREEN, 1)
            cv2.line(image, p9, self._vanishing_points[i].point, params.COLOR_GREEN, 1)
            cv2.line(image, p10, self._vanishing_points[i].point, params.COLOR_GREEN, 1)
        return image

    @property
    def calibrated(self):
        return len(self._vanishing_points) > 0

    def _step(self, seq):
        seq_loader, new_frame = self.receive(pipe_id=params.FRAME_LOADER_ID)
        seq_tracker, boxes_mask, boxes_mask_no_border, optical_flow_serialized = self.receive(pipe_id=params.TRACKER_ID)

        if seq_loader != seq_tracker:
            raise SyncError

        if len(self._vanishing_points) < 1:
            self.detect_first_vanishing_point()

        # elif len(self._vanishing_points) < 2:
        #     self.detect_second_vanishing_point(new_frame, boxes_mask, boxes_mask_no_border)
        else:
            exit()

    def detect_first_vanishing_point(self) -> None:
        if len(Box2D.lifelines()) > params.CALIBRATOR_TRACK_MINIMUM:
            # image = np.zeros(shape=(self.info.height, self.info.width, 3), dtype=np.uint8)
            #
            # convex_hull = Box2D.lifeline_convex_hull(self.info, Box2D.lifelines())
            #
            # p1 = tuple(convex_hull[1][0])
            # p2 = tuple(convex_hull[2][0])
            # p3 = tuple(convex_hull[3][0])
            # p4 = tuple(convex_hull[0][0])
            #
            # line1 = Line(p1, p2)
            # line2 = Line(p3, p4)
            #
            # self._vanishing_points.append(VanishingPoint(point=line1.intersection(line2)))

            # cv2.imwrite("lifelines.jpg", image[:][int(self.info.height/2):])

            for lifeline in Box2D.lifelines():
                old_position, new_position = lifeline
                self._pc_lines.pc_line_from_points(tuple(old_position), tuple(new_position))
            self._vanishing_points.append(self._pc_lines.find_most_line_cross(self.info))
            self._pc_lines.clear()

    def detect_second_vanishing_point(self, new_frame, boxes_mask, boxes_mask_no_border) -> None:
        mask = np.zeros_like(new_frame)

        selected_areas = cv2.bitwise_and(new_frame, cv2.cvtColor(boxes_mask, cv2.COLOR_GRAY2BGR))
        blured = cv2.blur(selected_areas, (5, 5))

        canny = cv2.Canny(blured, 50, 150, apertureSize=3)
        no_border_canny = cv2.bitwise_and(canny, boxes_mask_no_border)
        lines = cv2.HoughLinesP(no_border_canny, 1, np.pi / 350, 20, minLineLength=30, maxLineGap=3)

        print("second")
        if lines is not None:

            vp_1 = self._vanishing_points[0]

            for (x1, y1, x2, y2), in lines:
                point1 = x1, y1
                point2 = x2, y2

                try:
                    track = Line(point1, point2)
                    a_vp = Line(point1, vp_1.point)
                    b_vp = Line(point2, vp_1.point)

                    # select point which is further from first vanishing point
                    if a_vp.magnitude > b_vp.magnitude:
                        to_vp = a_vp
                    else:
                        to_vp = b_vp

                    if track.angle(to_vp) > params.CALIBRATOR_ANGLE_MIN:
                        self._pc_lines.pc_line_from_points(point1, point2)
                        cv2.line(mask, point1, point2, params.COLOR_BLUE, 1)
                except SamePointError:
                    continue

        if self._pc_lines.count > params.CALIBRATOR_TRACK_MINIMUM:
            self._vanishing_points.append(self._pc_lines.find_most_line_cross(self.info))

        cv2.imwrite("mask.jpg", mask)
        # cv2.imwrite("boxes_mask_no_border.jpg", boxes_mask_no_border)
        cv2.imwrite("test.jpg", no_border_canny)
