import cv2
import numpy as np
from primitives import constants

from primitives.coordinates import Coordinates
from repositories.models.tracked_object import TrackedObject
from primitives.enums import Color, Mode
from primitives.line import Line, SamePointError, ransac
from primitives.pc_space import ParallelCoordinateSpace
from primitives.vanishing_point import VanishingPoint, VanishingPointError
from pipeline.base.pipeline import ThreadedPipeBlock


class SyncError(Exception):
    pass


class Calibrator(ThreadedPipeBlock):
    """
    Makes calibration of current scene - searches for dominant Vanishing points. Third one is being computed from the
    previous ones. Works only in calibration work mode, after then is this Block closed
    """

    def __init__(self, output=None, info=None):
        """
        :param output: output PipeBlocks
        :param info: instance of InputInfo containing all information about examined video. Holds information
        about founded vanishing point.
        """
        super().__init__(info=info, pipe_id=constants.CALIBRATOR_ID, output=output)

        self._pc_lines = ParallelCoordinateSpace(info.width)
        self._detected_lines = []

    def _mode_changed(self, new_mode):
        """
        If new mode is detection this thread is closed

        :raise EOFError to stop computing
        :param new_mode: new mode
        """
        super()._mode_changed(new_mode)

        if new_mode == Mode.DETECTION:
            raise EOFError

    def _step(self, seq):
        """
        On each step it tries to compute new vanishing point if it has enough information to do so.
        Detects first two dominant vanishing points in scene. Third is not beeing used in this project but can be
        easily implemented.

        :param seq: current sequnce number
        """

        seq_loader, new_frame = self.receive(pipe_id=constants.FRAME_LOADER_ID)
        seq_lights, light_status = self.receive(pipe_id=constants.TRAFFIC_LIGHT_OBSERVER_ID)
        seq_tracker, boxes_mask, boxes_mask_no_border, lifelines = self.receive(pipe_id=constants.TRACKER_ID)

        if not self._info.corridors_repository.corridors_found:
            if len(self._info.vanishing_points) < 1:
                self._detect_first_vp(lifelines)

            elif len(self._info.vanishing_points) < 2:
                self._detect_second_vanishing_point(new_frame, boxes_mask, boxes_mask_no_border, light_status)

            elif len(TrackedObject.filter_lifelines(lifelines, self._info.vp1)) > constants.CORRIDORS_MINIMUM_LIFELINES:
                self._find_corridors(lifelines)

    def _detect_first_vp(self, lifelines):
        """
        Detects first vanishing point using tracked movement of cars.
        Tracked movement lines are accumulated into parallel coordinate space where RANSAC algorithm is used for
        intersection detection - Vanishing Point.

        After vanishing point is being found it propagates this information to InputInfo where it adds VanishingPoint
        to corresponding list.

        :param lifelines: movement of cars
        """

        if len(lifelines) > constants.CALIBRATOR_VP1_TRACK_MINIMUM:

            for history in lifelines:
                line, value = ransac(history, history, 1)

                if line is not None and value > 5:
                    self._pc_lines.add_to_pc_space(line=line)

            # for history in lifelines:
            #     try:
            #         self._pc_lines.add_to_pc_space(point1=history[0], point2=history[-1])
            #     except IndexError:
            #         pass

            if self._pc_lines.count > constants.CALIBRATOR_VP1_TRACK_MINIMUM:
                new_vanishing_point = self._pc_lines.find_most_lines_cross(write=True)
                self._info.vanishing_points.append(VanishingPoint(point=new_vanishing_point))

            self._pc_lines.clear()

    def _detect_second_vanishing_point(self, new_frame, boxes_mask, boxes_mask_no_border, light_status) -> None:
        """
        Calculates second vanishing point using information about car positions and detection of edges supporting
        second vanishing point. It is being detected only if green light status is present on current frame.
        Detected lines from edges are accumulated into parallel coordinate space - RANSAC algorithm is used
        for intersection detection - Vanishing Point.


        After vanishing point is being found it propagates this information to InputInfo where it adds VanishingPoint
        to corresponding list

        :param new_frame: examined frame
        :param boxes_mask: mask used for selecting parts of image where cars exists
        :param boxes_mask_no_border: mask used for selecting parts of image where cars exists
        :param light_status: current light status
        """

        if light_status in [Color.RED, Color.RED_ORANGE]:
            return

        selected_areas = cv2.bitwise_and(new_frame, cv2.cvtColor(boxes_mask, cv2.COLOR_GRAY2RGB))
        blured = cv2.GaussianBlur(selected_areas, (7, 7), 0)

        canny = cv2.Canny(blured, 50, 150, apertureSize=3)
        no_border_canny = cv2.bitwise_and(canny, boxes_mask_no_border)

        no_border_canny = cv2.bitwise_and(no_border_canny, no_border_canny, mask=self._info.update_area.mask())
        lines = cv2.HoughLinesP(image=no_border_canny,
                                rho=1,
                                theta=np.pi / 350,
                                threshold=constants.CALIBRATOR_HLP_THRESHOLD,
                                minLineLength=constants.CALIBRATOR_MIN_LINE_LENGTH,
                                maxLineGap=constants.CALIBRATOR_MAX_LINE_GAP)

        vp1 = self._info.vanishing_points[0]

        canny = cv2.cvtColor(no_border_canny, cv2.COLOR_GRAY2RGB)

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
                        cv2.line(canny, point1, point2, constants.COLOR_RED, 2)
                        continue

                    self._pc_lines.add_to_pc_space(point1, point2)
                    cv2.line(canny, point1, point2, constants.COLOR_BLUE, 2)
                except SamePointError:
                    continue

            cv2.imwrite("test.jpg", canny)

        if self._pc_lines.count > constants.CALIBRATOR_VP2_TRACK_MINIMUM:
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

    def _calculate_third_vp(self):
        """
        Because principal point is said to be in midle, the third vanishing point can be obtained by simple
        mathematics.

        After vanishing point is being found it propagates this information to InputInfo where it adds VanishingPoint
        to corresponding list
        """

        vp1 = self._info.vanishing_points[0].point

        try:
            vp2 = self._info.vanishing_points[1].point
            vp1_to_vp2 = Line(point1=vp1,
                              point2=vp2)

        except VanishingPointError:
            vp1_to_vp2 = Line(point1=vp1,
                              direction=self._info.vanishing_points[1].direction)

        self._info.vanishing_points.append(VanishingPoint(direction=vp1_to_vp2.normal_direction()))

    def _find_corridors(self, lifelines):
        """
        After both vanishing points are found corridors can be constructed with the information of car trajectories.

        :param lifelines: trajectories of cars
        """

        filtered_lifelines = TrackedObject.filter_lifelines(lifelines, self._info.vp1)
        mask = np.zeros(shape=(self._info.height, self._info.width, 3), dtype=np.uint8)

        # for history in filtered_lifelines:
        #     line, value = ransac(history, history, 1)
        #
        #     if line is not None and value > 5:
        #         bottom_point = Coordinates(*line.find_coordinate(y=self._info.height)).tuple()
        #
        #         cv2.line(img=mask,
        #                  pt1=bottom_point,
        #                  pt2=self._info.vp1.point,
        #                  color=params.COLOR_LIFELINE,
        #                  thickness=100)

        for history in filtered_lifelines:
            helper_mask = np.zeros_like(mask)

            first_point = history[0]

            line = Line(first_point, self._info.vp1.point)
            line.draw(image=helper_mask,
                      color=constants.COLOR_LIFELINE,
                      thickness=100)

            mask = cv2.add(mask, helper_mask)

        cv2.imwrite("lifeline.jpg", mask)

        self._info.corridors_repository.find_corridors(lifelines_mask=mask,
                                                       vp1=self._info.vanishing_points[0])
