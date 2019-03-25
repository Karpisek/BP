import params

from bbox import Box2D
from bbox.optical_flow import OpticalFlow
from pipeline import ThreadedPipeBlock
from munkres import Munkres
from pipeline.pipeline import is_frequency

DISALLOWED = 10000
TRACKER_INPUTS = 2


def transpose_matrix(matrix):
    return list(map(list, zip(*matrix)))


class Tracker(ThreadedPipeBlock):

    def __init__(self, area_of_detection, info, calibrator, output=None):
        super().__init__(pipe_id=params.TRACKER_ID, output=output)

        self.new_positions = []
        self.old_positions = []

        self._area_of_detection = area_of_detection
        self._info = info
        self._points_to_track = None

        self._calibrator = calibrator

        self._optical_flow = OpticalFlow(info)

    def _step(self, seq):
        if is_frequency(seq, params.DETECTOR_FREQUENCY):
            self._update_from_detector(seq)
        else:
            self._update_from_predictor(seq)

        self._control_boxes()

    def _update_from_detector(self, sequence_number) -> None:

        detected_boxes = self.receive(pipe_id=params.DETECTOR_ID)

        if self._info.track_boxes:
            self._update_from_predictor(sequence_number)

            if len(Box2D.boxes):
                self._hungarian_method(detected_boxes)
            else:
                Box2D.boxes = [Box2D(*new_box, self._info, self) for new_box in detected_boxes]

        else:
            self.receive(pipe_id=params.FRAME_LOADER_ID)

            Box2D.boxes = [Box2D(*new_box, self._info, self) for new_box in detected_boxes]

            message = sequence_number, [box.serialize() for box in Box2D.boxes], Box2D.lifelines()
            self.send(message, pipe_id=params.VIDEO_PLAYER_ID)

            if is_frequency(sequence_number, params.CALIBRATOR_FREQUENCY):
                message = sequence_number, \
                          Box2D.all_boxes_mask(info=self._info, area_size="outer"), \
                          Box2D.all_boxes_mask(info=self._info, area_size="small-outer"), \
                          self._optical_flow.serialize()

                self.send(message, pipe_id=params.CALIBRATOR_ID, block=False)

    def _update_from_predictor(self, sequence_number) -> None:

        for box in Box2D.boxes:
            if box.lifetime < 0:
                Box2D.boxes.remove(box)

        if self._info.track_boxes:
            for box in Box2D.boxes:
                box.predict()

            if is_frequency(sequence_number, params.TRACKER_OPTICAL_FLOW_FREQUENCY):
                _, new_frame = self.receive(pipe_id=params.FRAME_LOADER_ID)
                self._optical_flow.update(new_frame)

            message = sequence_number, [box.serialize() for box in Box2D.boxes], Box2D.lifelines()
            self.send(message, pipe_id=params.VIDEO_PLAYER_ID)

            if is_frequency(sequence_number, params.CALIBRATOR_FREQUENCY):
                message = sequence_number, \
                          Box2D.all_boxes_mask(info=self._info, area_size="outer"), \
                          Box2D.all_boxes_mask(info=self._info, area_size="small-outer"), \
                          self._optical_flow.serialize()

                self.send(message, pipe_id=params.CALIBRATOR_ID, block=False)
        else:
            self.receive(pipe_id=params.FRAME_LOADER_ID)

            message = sequence_number, [], []
            self.send(message, pipe_id=params.VIDEO_PLAYER_ID)

            if is_frequency(sequence_number, params.CALIBRATOR_FREQUENCY):
                message = sequence_number, []
                self.send(message, pipe_id=params.CALIBRATOR_ID, block=False)

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

