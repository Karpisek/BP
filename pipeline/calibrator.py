from bbox import TrackedObject, Coordinates
from pc_lines.line import SamePointError, Line
from pc_lines.pc_line import PcLines
from pc_lines.vanishing_point import VanishingPoint
from pipeline import ThreadedPipeBlock

import cv2
import numpy as np
import params


class SyncError(Exception):
    pass


class Calibrator(ThreadedPipeBlock):
    def __init__(self, output=None, info=None):
        super().__init__(pipe_id=params.CALIBRATOR_ID, output=output)

        self._pc_lines = PcLines(info.width)
        self._info = info

        # self._info.vanishing_points.append(VanishingPoint((548, -86)))

    def _step(self, seq):
        seq_loader, new_frame = self.receive(pipe_id=params.FRAME_LOADER_ID)
        seq_tracker, boxes_mask, boxes_mask_no_border, lifelines = self.receive(pipe_id=params.TRACKER_ID)

        if len(self._info.vanishing_points) < 1:
            self.detect_first_vp(lifelines)

        elif len(self._info.vanishing_points) < 2:
            self.detect_second_vanishing_point(new_frame, boxes_mask, boxes_mask_no_border)

        else:
            self.calculate_third_vp()
            self.find_corridors(lifelines)
            exit()

    def detect_first_vp(self, lifelines):
        if len(lifelines) > params.CALIBRATOR_VP1_TRACK_MINIMUM:

            for lifeline in lifelines:
                old_position, new_position = lifeline
                self._pc_lines.pc_line_from_points(tuple(old_position), tuple(new_position))

            preset_pc_points = self._pc_lines.pc_points(points=self._info.vp1_preset_points())

            new_vanishing_point = self._pc_lines.find_most_line_cross(preset_points=preset_pc_points)
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
            detected_point_pairs = []

            for (x1, y1, x2, y2), in lines:
                point1 = x1, y1
                point2 = x2, y2

                try:
                    if Coordinates(*vp1.point).distance(Coordinates(x1, y1)) > Coordinates(*vp1.point).distance(Coordinates(x2, y2)):
                        line_to_vp = Line(point1, vp1.point)
                    else:
                        line_to_vp = Line(point2, vp1.point)

                    if Line(point1, point2).angle(line_to_vp) < 30 or Line(point1, point2).angle(line_to_vp) > 150:
                        cv2.line(selected_areas, point1, point2, params.COLOR_WHITE, 1)
                        continue

                    detected_point_pairs.append((Coordinates(*point1), Coordinates(*point2)))
                    # self._pc_lines.pc_line_from_points(point1, point2)
                    cv2.line(selected_areas, point1, point2, params.COLOR_BLUE, 1)
                except SamePointError:
                    continue

            detected_point_pairs.sort(key=lambda point_pair: point_pair[0].distance(point_pair[1]),
                                      reverse=True)

            for pair in detected_point_pairs[:params.CALIBRATOR_VP2_TRACK_MAX_PER_RUN]:
                coordinates1, coordinates2 = pair
                self._pc_lines.pc_line_from_points(coordinates1.tuple(), coordinates2.tuple())

        if self._pc_lines.count > params.CALIBRATOR_VP2_TRACK_MINIMUM:
            new_vanishing_point = self._pc_lines.find_most_line_cross(only_preset=False)
            self._info.vanishing_points.append(new_vanishing_point)
            self._pc_lines.clear()

    def calculate_third_vp(self):
        vp1 = self._info.vanishing_points[0].point
        vp2 = self._info.vanishing_points[1].point

        vp1_to_vp2 = Line(vp1, vp2)
        # normal = vp1_to_vp2.direction

        # vp1_to_principal = Line(vp1, self._info.principal_point.tuple())
        # vp2_to_principal = Line(vp2, self._info.principal_point.tuple())
        #
        # vp2_to_vp3 = Line(vp2, direction=vp1_to_principal.normal_direction())
        # vp1_to_vp3 = Line(vp1, direction=vp2_to_principal.normal_direction())

        self._info.vanishing_points.append(VanishingPoint(direction=vp1_to_vp2.normal_direction()))
        # print(vp3)

    def find_corridors(self, lifelines):
        mask = np.zeros(shape=(self._info.height, self._info.width, 3), dtype=np.uint8)
        mask = TrackedObject.draw_lifelines(image=mask,
                                            lifelines=lifelines,
                                            color=params.COLOR_LIFELINE,
                                            thickness=params.CALIBRATOR_LIFELINE_THICKNESS)

        self._info.corridors_repository.find_corridors(mask)
