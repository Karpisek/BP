from queue import Queue
from threading import Thread
import cv2
import numpy as np

from bbox import Box2D
from pipeline import PipeBlock

from munkres import Munkres

DISALLOWED = 10000
TRACKER_INPUTS = 2

APROXIMATION_FRAME_COUNT = 2
OPTICAL_FLOW_PAUSE = 3

LK_PARAMS = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

FEATURE_PARAMS = dict(maxCorners=150,
                      qualityLevel=0.3,
                      minDistance=3,
                      blockSize=7)


def transpose_matrix(matrix):
    return list(map(list, zip(*matrix)))


class Tracker(PipeBlock):

    def __init__(self, area_of_detection, info, output=None, track_boxes=True):
        super().__init__(output, number_of_inputs=TRACKER_INPUTS)

        self.new_positions = []
        self.old_positions = []

        self._area_of_detection = area_of_detection
        self._info = info
        self._points_to_track = None

        self._track_boxes = track_boxes

        self._optical_flow_grey = None

        self._thread = Thread(target=self._run)
        self._thread.daemon = True
        self._thread.start()

    def _run(self):
        sequence_number = 0
        while True:
            sequence_number += 1

            if sequence_number % OPTICAL_FLOW_PAUSE == 0:
                self._update_flow()

            if sequence_number % APROXIMATION_FRAME_COUNT == 0:
                self._update_from_detector(sequence_number)
            else:
                self._update_from_predictor(sequence_number)

            self._control_boxes()

    def _update_from_detector(self, sequence_number):

        detected_boxes = self.next(pipe=0)

        if self._track_boxes:
            self._update_from_predictor(sequence_number)

            if len(Box2D.boxes) != 0:
                self._hungarian_method(detected_boxes)
            else:
                Box2D.boxes = [Box2D(*new_box, self._info, self) for new_box in detected_boxes]

        else:
            Box2D.boxes = [Box2D(*new_box, self._info, self) for new_box in detected_boxes]
            self.send_to((sequence_number, [box.serialize() for box in Box2D.boxes], [], []), out_pipe=0, in_pipe=1)

    def _update_from_predictor(self, sequence_number):

        for box in Box2D.boxes:
            if box.lifetime < 0:
                Box2D.boxes.remove(box)

        if self._track_boxes:
            for box in Box2D.boxes:
                box.predict()

            self.send_to((sequence_number, [box.serialize() for box in Box2D.boxes], self.old_positions[:], self.new_positions[:]), out_pipe=0, in_pipe=1)
        else:
            self.send_to((sequence_number, [], [], []), out_pipe=0, in_pipe=1)

    def __str__(self):
        return super().__str__() + f'Boxes [{len(Box2D.boxes)}]'

    def _hungarian_method(self, detected_boxes):
        # matrix = [
        #     [old_box.center.pixel_distance(new_coordinates, self._info) if new_coordinates.convert_to_fixed(self._info) and old_box.in_radius(new_coordinates) else DISALLOWED for new_coordinates, _, _, in detected_boxes] for old_box in Box2D.boxes
        # ]

        matrix = []
        for old_box in Box2D.boxes:
            row = []
            for new_box in detected_boxes:
                new_coordinates, new_size, new_score = new_box

                new_coordinates.convert_to_fixed(self._info)
                if old_box.in_radius(new_coordinates):
                    row.append(old_box.center.distance(new_coordinates))
                else:
                    row.append(DISALLOWED)
            matrix.append(row)

        munkres = Munkres()
        indexes = munkres.compute(matrix)

        detected_boxes_copy = detected_boxes[:]

        for index, row in enumerate(transpose_matrix(matrix)):
            if sum(row) != DISALLOWED * len(row):
                detected_boxes.remove(detected_boxes_copy[index])

        for old_index, new_index in indexes:

            value = matrix[old_index][new_index]

            if value < DISALLOWED:
                old_box = Box2D.boxes[old_index]
                new_box = detected_boxes_copy[new_index]

                try:
                    detected_boxes.remove(new_box)
                except ValueError:
                    pass

                new_coordinates, size, score = new_box
                old_box.update_position(size, score, new_coordinates)

        Box2D.boxes += [Box2D(*new_box, self._info, self) for new_box in detected_boxes if new_box[2] > Box2D.MINIMAL_SCORE_NEW]

    def _control_boxes(self):
        for box in Box2D.boxes[:]:
            if not self._area_of_detection.contains(box.center):
                Box2D.boxes.remove(box)

    def switch_tracking(self):
        self._track_boxes = not self._track_boxes

    def _update_flow(self):
        seq, new_frame = self.next(pipe=1)

        new_frame_gray = cv2.cvtColor(new_frame, cv2.COLOR_RGB2GRAY)
        mask_for_detection = np.zeros_like(new_frame_gray)

        if self._optical_flow_grey is not None:

            for box in Box2D.boxes:
                cv2.circle(mask_for_detection, box.center.tuple(), box.area_of_interest(), 255, -1)

            if self._points_to_track is not None:
                moved_grid, st, err = cv2.calcOpticalFlowPyrLK(self._optical_flow_grey, new_frame_gray, self._points_to_track, None, **LK_PARAMS)

                self.new_positions = moved_grid[st == 1]
                self.old_positions = self._points_to_track[st == 1]
            else:
                self.new_positions = []
                self.old_positions = []

        self._optical_flow_grey = new_frame_gray
        self._points_to_track = cv2.goodFeaturesToTrack(self._optical_flow_grey, mask=mask_for_detection, **FEATURE_PARAMS)

        for box in Box2D.boxes:
            box.update_flow()
