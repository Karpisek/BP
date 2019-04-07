from pc_lines.line import Line
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

        self._detected_lines = []

    def _step(self, seq):
        seq_loader, new_frame = self.receive(pipe_id=params.FRAME_LOADER_ID)
        seq_tracker, boxes_mask, boxes_mask_no_border, lifelines = self.receive(pipe_id=params.TRACKER_ID)

        if len(self._info.vanishing_points) < 1:
            self.detect_first_vp(lifelines)

        # elif len(self._info.vanishing_points) < 2:
        #     self.detect_second_vanishing_point(new_frame, boxes_mask, boxes_mask_no_border)
        #
        # else:
        #     self.calculate_third_vp()
        #     self.find_corridors(lifelines)
        #     exit()

    def detect_first_vp(self, lifelines):
        print(len(lifelines))
        if len(lifelines) > params.CALIBRATOR_VP1_TRACK_MINIMUM:

            for lifeline in lifelines:
                old_position, new_position = lifeline
                self._pc_lines.add_to_pc_space(tuple(old_position), tuple(new_position))

            new_vanishing_point = VanishingPoint(point=self._pc_lines.find_most_lines_cross())
            self._info.vanishing_points.append(new_vanishing_point)

        self._pc_lines.clear()

    # def detect_first_vp(self, new_frame: np.ndarray):
    #     blured_frame = cv2.blur(src=new_frame,
    #                             ksize=params.CALIBRATOR_BLUR_KERNEL)
    #
    #     median_blured_image = cv2.medianBlur(src=blured_frame,
    #                                          ksize=params.CALIBRATOR_MEDIAN_BLUR_KERNEL_SIZE)
    #
    #     vertical_offset = int(self._info.height * params.CALIBRATOR_FOCUS_AREA_COEFFICIENT)
    #
    #     area_of_focus = median_blured_image[vertical_offset:, :]
    #     extracted_edges = cv2.Canny(image=area_of_focus,
    #                                 threshold1=50,
    #                                 threshold2=200)
    #
    #     dilate_edges = cv2.dilate(src=extracted_edges,
    #                               kernel=params.CALIBRATOR_DILATATION_KERNEL,
    #                               iterations=5)
    #
    #     detected_lines = cv2.HoughLinesP(image=dilate_edges,
    #                                      rho=1,
    #                                      theta=np.pi / 180,
    #                                      threshold=20,
    #                                      minLineLength=params.CALIBRATOR_HOUGH_MINIMAL_LINE_LENGTH,
    #                                      maxLineGap=params.CALIBRATOR_MAX_LINE_GAP)
    #
    #     if detected_lines is not None:
    #         best_line = None
    #         horizontal_line = Line(point1=(0, 0),
    #                                point2=(10, 0))
    #
    #         for line in detected_lines:
    #             x1, y1, x2, y2 = line[0]
    #
    #             y1 += vertical_offset
    #             y2 += vertical_offset
    #
    #             new_line = Line(point1=(x1, y1),
    #                             point2=(x2, y2))
    #
    #             new_x_base = new_line.find_coordinate(y=self._info.height)[0]
    #
    #             if new_line.angle(horizontal_line) > 30:
    #                 if best_line is None or new_line.magnitude > best_line.magnitude:
    #                     no_collision = True
    #
    #                     for old_line in self._detected_lines:
    #                         old_x_base = old_line.find_coordinate(y=self._info.height)[0]
    #
    #                         if abs(old_x_base - new_x_base) < params.CALIBRATOR_BASE_MIN_DISTANCE:
    #                             no_collision = False
    #
    #                     if no_collision:
    #                         best_line = new_line
    #
    #         if best_line is not None:
    #             self._detected_lines.append(best_line)
    #
    #     print(len(self._detected_lines))
    #     if len(self._detected_lines) > params.CALIBRATOR_VP1_MINIMUM:
    #         for line in self._detected_lines:
    #             self._pc_lines.add_to_pc_space(line=line)
    #             line.draw(new_frame, params.COLOR_RED, params.DEFAULT_THICKNESS)
    #
    #         detected_vanishing_point = VanishingPoint(point=self._pc_lines.find_most_lines_cross(preset=self._info.vp1_preset_points()))
    #         cv2.imwrite("test.jpg", new_frame)
    #         self._info.vanishing_points.append(detected_vanishing_point)
    #         print(detected_vanishing_point)
    #         exit()

    # def calculate_third_vp(self):
    #     vp1 = self._info.vanishing_points[0].point
    #     vp2 = self._info.vanishing_points[1].point
    #
    #     vp1_to_vp2 = Line(vp1, vp2)
    #     # normal = vp1_to_vp2.direction
    #
    #     # vp1_to_principal = Line(vp1, self._info.principal_point.tuple())
    #     # vp2_to_principal = Line(vp2, self._info.principal_point.tuple())
    #     #
    #     # vp2_to_vp3 = Line(vp2, direction=vp1_to_principal.normal_direction())
    #     # vp1_to_vp3 = Line(vp1, direction=vp2_to_principal.normal_direction())
    #
    #     self._info.vanishing_points.append(VanishingPoint(direction=vp1_to_vp2.normal_direction()))
    #     # print(vp3)
    #
    # def find_corridors(self, lifelines):
    #     mask = np.zeros(shape=(self._info.height, self._info.width, 3), dtype=np.uint8)
    #     mask = TrackedObject.draw_lifelines(image=mask,
    #                                         lifelines=lifelines,
    #                                         color=params.COLOR_LIFELINE,
    #                                         thickness=params.CALIBRATOR_LIFELINE_THICKNESS)
    #
    #     self._info.corridors_repository.find_corridors(mask)
