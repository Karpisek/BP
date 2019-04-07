import params

from bbox import TrackedObject, TrackedObjectsRepository
from bbox.optical_flow import OpticalFlow
from pipeline import ThreadedPipeBlock
from munkres import Munkres
from pipeline.pipeline import is_frequency


def transpose_matrix(matrix):
    return list(map(list, zip(*matrix)))


class Tracker(ThreadedPipeBlock):

    def __init__(self, info, output=None):
        super().__init__(pipe_id=params.TRACKER_ID, output=output)

        self.new_positions = []
        self.old_positions = []

        self._info = info
        self._points_to_track = None

        self._tracked_object_repository = TrackedObjectsRepository(info)
        self._optical_flow = OpticalFlow(info, self._tracked_object_repository)

        self._munkres = Munkres()

    def _step(self, seq):
        if is_frequency(seq, params.DETECTOR_CAR_FREQUENCY):
            self._update_from_detector(seq)

        else:
            self._update_from_predictor(seq)

        self._tracked_object_repository.control_boxes()

    def _send_message(self, target, sequence_number, block=True):
        message = None

        if target == params.VIDEO_PLAYER_ID:
            serialized_tracked_objects = self._tracked_object_repository.serialize()
            tracked_object_lifelines = self._tracked_object_repository.lifelines
            message = sequence_number, serialized_tracked_objects, tracked_object_lifelines

        elif target == params.CALIBRATOR_ID:
            outer_masks = self._tracked_object_repository.all_boxes_mask(area_size="outer")
            outer_masks_no_border = self._tracked_object_repository.all_boxes_mask(area_size="small-outer")
            lifelines = self._tracked_object_repository.lifelines

            message = sequence_number, outer_masks, outer_masks_no_border, lifelines

        elif target == params.OBSERVER_ID:
            serialized_tracked_objects = self._tracked_object_repository.serialize()

            message = sequence_number, serialized_tracked_objects

        if message is not None:
            self.send(message, pipe_id=target, block=block)

    def _update_from_detector(self, sequence_number) -> None:

        detected_objects = self.receive(pipe_id=params.DETECTOR_CAR_ID)

        self._update_from_predictor(sequence_number)

        if self._tracked_object_repository.count():
            self._hungarian_method(detected_objects)

        else:
            for detected_object in detected_objects:
                coordinates, size, confident_score = detected_object

                if self._info.start_area.contains(coordinates):
                    self._tracked_object_repository.new_tracked_object(*detected_object)

    def _update_from_predictor(self, sequence_number) -> None:

        self._tracked_object_repository.predict()

        if is_frequency(sequence_number, params.TRACKER_OPTICAL_FLOW_FREQUENCY):
            _, new_frame = self.receive(pipe_id=params.FRAME_LOADER_ID)
            self._optical_flow.update(new_frame)

        if is_frequency(sequence_number, params.CALIBRATOR_FREQUENCY):
            self._send_message(target=params.CALIBRATOR_ID,
                               sequence_number=sequence_number,
                               block=False)

        if is_frequency(sequence_number, params.OBSERVER_FREQUENCY):
            self._send_message(target=params.OBSERVER_ID,
                               sequence_number=sequence_number)

    def _hungarian_method(self, detected_boxes) -> None:

        matrix = []
        for old_object in self._tracked_object_repository.list:
            row = []
            for new_box in detected_boxes:
                new_coordinates, new_size, new_score = new_box

                if old_object.in_radius(new_coordinates):
                    row.append(old_object.center.distance(new_coordinates))
                else:
                    row.append(params.TRACKER_DISALLOWED)
            matrix.append(row)

        indexes = self._munkres.compute(matrix)
        detected_boxes_copy = detected_boxes[:]

        for index, row in enumerate(transpose_matrix(matrix)):
            if sum(row) != params.TRACKER_DISALLOWED * len(row):
                detected_boxes.remove(detected_boxes_copy[index])

        for old_index, new_index in indexes:

            value = matrix[old_index][new_index]

            if value < params.TRACKER_DISALLOWED:
                old_box = self._tracked_object_repository.list[old_index]
                new_box = detected_boxes_copy[new_index]

                try:
                    detected_boxes.remove(new_box)
                except ValueError:
                    pass

                new_coordinates, size, score = new_box
                old_box.update_position(size, score, new_coordinates)

        for new_box in detected_boxes:
            if new_box[2] > params.TRACKER_MINIMAL_SCORE:
                coordinates, size, confident_score = new_box

                if self._info.start_area.contains(coordinates):
                    self._tracked_object_repository.new_tracked_object(*new_box)



