from bbox.coordinates import Coordinates
from bbox.tracked_object import TrackedObject
from pc_lines.line import Line, SamePointError
from pc_lines.pc_line import PcLines
from pc_lines.vanishing_point import VanishingPoint, VanishingPointError
from pipeline import ThreadedPipeBlock

import cv2
import numpy as np
import params
from pipeline.base.pipeline import Mode
from repositories.traffic_light_repository import Color


class SyncError(Exception):
    pass


class Calibrator(ThreadedPipeBlock):

    def __init__(self, output=None, info=None):
        super().__init__(info=info, pipe_id=params.CALIBRATOR_ID, output=output)

        self._pc_lines = PcLines(info.width)

        self._detected_lines = []

    def _mode_changed(self, new_mode):
        super()._mode_changed(new_mode)

        if new_mode == Mode.DETECTION:
            raise EOFError

    def _step(self, seq):
        seq_loader, new_frame = self.receive(pipe_id=params.FRAME_LOADER_ID)
        seq_lights, light_status = self.receive(pipe_id=params.TRAFFIC_LIGHT_OBSERVER_ID)
        seq_tracker, boxes_mask, boxes_mask_no_border, lifelines = self.receive(pipe_id=params.TRACKER_ID)

        if not self._info.corridors_repository.corridors_found:
            if len(self._info.vanishing_points) < 1:
                self.detect_first_vp(lifelines)

            elif len(self._info.vanishing_points) < 2:
                self.detect_second_vanishing_point(new_frame, boxes_mask, boxes_mask_no_border, light_status)

            # elif len(self._info.vanishing_points) < 3:
            #     self.calculate_third_vp()

            elif len(lifelines) > params.CORRIDORS_MINIMUM_LIFELINES:
                self.find_corridors(lifelines)

    def detect_first_vp(self, lifelines):
        if len(lifelines) > params.CALIBRATOR_VP1_TRACK_MINIMUM:

            for lifeline in lifelines:
                old_position, new_position = lifeline
                self._pc_lines.add_to_pc_space(tuple(old_position), tuple(new_position))

            new_vanishing_point = self._pc_lines.find_most_lines_cross()
            self._info.vanishing_points.append(VanishingPoint(point=new_vanishing_point))
            self._pc_lines.clear()

    def detect_second_vanishing_point(self, new_frame, boxes_mask, boxes_mask_no_border, light_status) -> None:
        if light_status in [Color.RED, Color.RED_ORANGE]:
            return

        selected_areas = cv2.bitwise_and(new_frame, cv2.cvtColor(boxes_mask, cv2.COLOR_GRAY2RGB))
        # blured = cv2.blur(selected_areas, (3, 3))
        blured = cv2.GaussianBlur(selected_areas, (7, 7), 0)

        canny = cv2.Canny(blured, 50, 150, apertureSize=3)
        no_border_canny = cv2.bitwise_and(canny, boxes_mask_no_border)
        lines = cv2.HoughLinesP(image=no_border_canny,
                                rho=1,
                                theta=np.pi / 350,
                                threshold=params.CALIBRATOR_HLP_THRESHOLD,
                                minLineLength=params.CALIBRATOR_MIN_LINE_LENGTH,
                                maxLineGap=params.CALIBRATOR_MAX_LINE_GAP)

        vp1 = self._info.vanishing_points[0]

        canny = cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB)

        if lines is not None:
            for (x1, y1, x2, y2), in lines:
                point1 = x1, y1
                point2 = x2, y2

                try:
                    if vp1.coordinates.distance(Coordinates(x1, y1)) > vp1.coordinates.distance(Coordinates(x2, y2)):
                        line_to_vp = Line(point1, vp1.point)
                    else:
                        line_to_vp = Line(point2, vp1.point)

                    if Line(point1, point2).angle(line_to_vp) < 30 or Line(point1, point2).angle(line_to_vp) > 150:
                        cv2.line(canny, point1, point2, params.COLOR_RED, 1)
                        continue

                    self._pc_lines.add_to_pc_space(point1, point2)
                    cv2.line(canny, point1, point2, params.COLOR_BLUE, 1)
                except SamePointError:
                    continue

            cv2.imwrite("test.jpg", canny)

        if self._pc_lines.count > params.CALIBRATOR_VP2_TRACK_MINIMUM:
            new_vanishing_point = self._pc_lines.find_most_lines_cross()

            x, y = new_vanishing_point
            if y is not None:
                self._info.vanishing_points.append(VanishingPoint(point=new_vanishing_point))
            else:
                dx = np.cos(np.deg2rad(x))
                dy = np.sin(np.deg2rad(x))
                direction = dx, dy
                self._info.vanishing_points.append(VanishingPoint(direction=direction))

            self._pc_lines.clear()

    # def calculate_third_vp(self):
    #     vp1 = self._info.vanishing_points[0].point
    #
    #     try:
    #         vp2 = self._info.vanishing_points[1].point
    #         vp1_to_vp2 = Line(point1=vp1,
    #                           point2=vp2)
    #
    #     except VanishingPointError:
    #         vp1_to_vp2 = Line(point1=vp1,
    #                           direction=self._info.vanishing_points[1].direction)
    #
    #     self._info.vanishing_points.append(VanishingPoint(direction=vp1_to_vp2.normal_direction()))

    def find_corridors(self, lifelines):
        filtered_lifelines = TrackedObject.filter_lifelines(lifelines, self._info.vp1)

        mask = np.zeros(shape=(self._info.height, self._info.width, 3), dtype=np.uint8)
        mask = TrackedObject.draw_lifelines(image=mask,
                                            lifelines=filtered_lifelines,
                                            color=params.COLOR_LIFELINE,
                                            thickness=100)

        self._info.corridors_repository.find_corridors(lifelines_mask=mask,
                                                       vp1=self._info.vanishing_points[0])
