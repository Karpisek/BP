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

        self.parts = [[] for _ in range(params.CALIBRATOR_HORIZONTAL_PARTS)]

    def _step(self, seq):
        seq_loader, new_frame = self.receive(pipe_id=params.FRAME_LOADER_ID)

        if len(self._info.vanishing_points) < 1:
            self.detect_first_vp(new_frame)

        # elif len(self._info.vanishing_points) < 2:
        #     self.detect_second_vanishing_point(new_frame, boxes_mask, boxes_mask_no_border)
        #
        # else:
        #     self.calculate_third_vp()
        #     self.find_corridors(lifelines)
        #     exit()

    def detect_first_vp(self, new_frame: np.ndarray):
        blured_frame = cv2.blur(src=new_frame,
                                ksize=params.CALIBRATOR_BLUR_KERNEL)

        median_blured_image = cv2.medianBlur(src=blured_frame,
                                             ksize=params.CALIBRATOR_MEDIAN_BLUR_KERNEL_SIZE)

        vertical_offset = int(self._info.height * params.CALIBRATOR_FOCUS_AREA_COEFFICIENT)

        area_of_focus = median_blured_image[vertical_offset:, :]
        extracted_edges = cv2.Canny(image=area_of_focus,
                                    threshold1=50,
                                    threshold2=200)

        dilate_edges = cv2.dilate(src=extracted_edges,
                                  kernel=params.CALIBRATOR_DILATATION_KERNEL,
                                  iterations=5)

        part_width = int(self._info.width/params.CALIBRATOR_HORIZONTAL_PARTS)

        detected_lines = cv2.HoughLinesP(image=dilate_edges,
                                         rho=1,
                                         theta=np.pi / 180,
                                         threshold=20,
                                         minLineLength=params.CALIBRATOR_HOUGH_MINIMAL_LINE_LENGTH,
                                         maxLineGap=params.CALIBRATOR_MAX_LINE_GAP)

        if detected_lines is not None:
            best_lines = [[] for _ in range(params.CALIBRATOR_HORIZONTAL_PARTS)]
            horizontal_line = Line(point1=(0, 0),
                                   point2=(10, 0))

            for line in detected_lines:
                x1, y1, x2, y2 = line[0]

                y1 += vertical_offset
                y2 += vertical_offset

                new_line = Line(point1=(x1, y1),
                                point2=(x2, y2))

                base_intersection = new_line.find_coordinate(y=self._info.height)

                if base_intersection[0] < self._info.width/3:
                    section_index = 0
                elif base_intersection[0] > 2*self._info.width/3:
                    section_index = 1
                else:
                    continue

                if new_line.angle(horizontal_line) > 30:
                    best_lines[section_index].append(new_line)

            for index, section in enumerate(best_lines):
                if len(section):
                    best_line_in_section = sorted(section, key=lambda x: x.magnitude, reverse=True)[0]
                    self.parts[index].append(best_line_in_section)

        minimal_line_count = np.inf
        print(":::::")
        for part in self.parts:
            if len(part) < minimal_line_count:
                minimal_line_count = len(part)
            print(len(part))

        if minimal_line_count > params.CALIBRATOR_VP1_MINIMUM:
            for section in self.parts:
                section.sort(key=lambda x: x.magnitude, reverse=True)

            intersection = self.parts[0][0].intersection(self.parts[1][0])

            self.parts[0][0].draw(new_frame, params.COLOR_RED, params.DEFAULT_THICKNESS)
            self.parts[1][0].draw(new_frame, params.COLOR_RED, params.DEFAULT_THICKNESS)

            detected_vanishing_point = VanishingPoint(point=intersection)
            cv2.imwrite("test.jpg", new_frame)
            self._info.vanishing_points.append(detected_vanishing_point)
            print(detected_vanishing_point)
            exit()

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
