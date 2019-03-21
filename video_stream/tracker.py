from bbox import Box2D
from bbox.optical_flow import OpticalFlow
from params import FRAME_LOADER_ID, DETECTOR_ID, VIDEO_PLAYER_ID, TRACKER_ID
from pipeline import ThreadedPipeBlock

from munkres import Munkres

DISALLOWED = 10000
TRACKER_INPUTS = 2

APROXIMATION_FRAME_COUNT = 3
OPTICAL_FLOW_PAUSE = 3


def transpose_matrix(matrix):
    return list(map(list, zip(*matrix)))


class Tracker(ThreadedPipeBlock):

    def __init__(self, area_of_detection, info, output=None):
        super().__init__(pipe_id=TRACKER_ID, output=output)

        self.new_positions = []
        self.old_positions = []

        self._area_of_detection = area_of_detection
        self._info = info
        self._points_to_track = None

        self._optical_flow = OpticalFlow()

    def _step(self, seq):
        if seq % OPTICAL_FLOW_PAUSE == 0:

            _, new_frame = self.receive(pipe_id=FRAME_LOADER_ID)
            self._optical_flow.update(new_frame)

        if seq % APROXIMATION_FRAME_COUNT == 0:
            self._update_from_detector(seq)
        else:
            self._update_from_predictor(seq)

        self._control_boxes()

    def _update_from_detector(self, sequence_number) -> None:

        detected_boxes = self.receive(pipe_id=DETECTOR_ID)

        if self._info.track_boxes:
            self._update_from_predictor(sequence_number)

            if len(Box2D.boxes):
                self._hungarian_method(detected_boxes)
            else:
                Box2D.boxes = [Box2D(*new_box, self._info, self) for new_box in detected_boxes]

        else:
            Box2D.boxes = [Box2D(*new_box, self._info, self) for new_box in detected_boxes]
            self.send((sequence_number, [box.serialize() for box in Box2D.boxes], ([], [])), pipe_id=VIDEO_PLAYER_ID)

    def _update_from_predictor(self, sequence_number) -> None:

        for box in Box2D.boxes:
            if box.lifetime < 0:
                Box2D.boxes.remove(box)

        if self._info.track_boxes:
            for box in Box2D.boxes:
                box.predict()

            self.send((sequence_number, [box.serialize() for box in Box2D.boxes], self._optical_flow.serialize()), pipe_id=VIDEO_PLAYER_ID)
        else:
            self.send((sequence_number, [], ([], [])), pipe_id=VIDEO_PLAYER_ID)

    def _hungarian_method(self, detected_boxes) -> None:

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

    def _control_boxes(self) -> None:
        [Box2D.boxes.remove(box) for box in Box2D.boxes if not self._area_of_detection.contains(box.center)]

