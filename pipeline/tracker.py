from primitives import constants

from copy import deepcopy
from primitives.optical_flow import OpticalFlow
from munkres import Munkres
from pipeline.base.pipeline import is_frequency, ThreadedPipeBlock
from repositories.tracked_object_repository import TrackedObjectsRepository


def transpose_matrix(matrix):
    """
    :param matrix: matrix to transpose
    :return: transposed matrix
    """

    return list(map(list, zip(*matrix)))


class Tracker(ThreadedPipeBlock):
    """
    Tracks detected object thru the scene. Uses Kalman filter and hungarian algorithm for predicting and
    assigning tracked objects.
    """

    def __init__(self, info, output=None):
        """
        :param info: instance of InputInfo
        :param output: list of PipeBlock outputs
        """

        super().__init__(info=info, pipe_id=constants.TRACKER_ID, output=output)

        self.new_positions = []
        self.old_positions = []

        self._points_to_track = None

        self._tracked_object_repository = TrackedObjectsRepository(info)
        self._optical_flow = OpticalFlow(info, self._tracked_object_repository)

        self._munkres = Munkres()

    def _mode_changed(self, new_mode):
        """
        On mode changed clears all tracked objects

        :param new_mode: nwe mode
        """

        super()._mode_changed(new_mode)

        self._tracked_object_repository.restart()

    def _step(self, seq):
        """
        On every step new detections are read - if possible. Then predicted position is computed on each tracked
        object instance using Kalman filter. Optical flow is used as secondary mesurement.
        Detection are assigned to existing instances of tracked object using hungarian algorithm
        New detected instances are added to Tracked object repository.

        :param seq: current sequence number
        """

        if is_frequency(seq, constants.DETECTOR_CAR_FREQUENCY):
            self._update_from_detector(seq)

        else:
            self._update_from_predictor(seq)

        self._tracked_object_repository.control_boxes(mode=self._mode)

        if is_frequency(seq, constants.CALIBRATOR_FREQUENCY):
            self._send_message(target=constants.CALIBRATOR_ID,
                               sequence_number=seq,
                               block=False)

        if is_frequency(seq, constants.OBSERVER_FREQUENCY):
            self._send_message(target=constants.OBSERVER_ID,
                               sequence_number=seq)

        if is_frequency(seq, constants.VIDEO_PLAYER_FREQUENCY):
            self._send_message(target=constants.VIDEO_PLAYER_ID,
                               sequence_number=seq)

    def _send_message(self, target, sequence_number, block=True):
        """
        Helper method for sending messages to output PipeBlocks
        depending on target receiver it generates certain message and sends it using send() method.

        :param target: targeted receiver
        :param sequence_number: current sequence number
        :param block: if sending message should wait until receiver could receive
        """

        message = None

        if target == constants.VIDEO_PLAYER_ID:
            serialized_tracked_objects = self._tracked_object_repository.serialize()
            tracked_object_lifelines = self._tracked_object_repository.lifelines
            flows = self._tracked_object_repository.flows
            message = sequence_number, serialized_tracked_objects, tracked_object_lifelines, flows

        elif target == constants.CALIBRATOR_ID:
            outer_masks = self._tracked_object_repository.all_boxes_mask(area_size="outer")
            outer_masks_no_border = self._tracked_object_repository.all_boxes_mask(area_size="small-outer")
            lifelines = deepcopy(self._tracked_object_repository.lifelines)

            message = sequence_number, outer_masks, outer_masks_no_border, lifelines

        elif target == constants.OBSERVER_ID:
            serialized_tracked_objects = self._tracked_object_repository.serialize()

            message = sequence_number, serialized_tracked_objects

        if message is not None:
            self.send(message, pipe_id=target, block=block)

    def _update_from_detector(self, sequence_number) -> None:
        """
        Updates position of tracked objects by using detected objects.
        For assigning problem is called hungarian_method(). For unassigned detections are generated new tracked object
        instances.

        :param sequence_number: current sequnce number
        """

        detected_objects = self.receive(pipe_id=constants.DETECTOR_CAR_ID)

        self._update_from_predictor(sequence_number)

        if self._tracked_object_repository.count():
            self._hungarian_method(detected_objects)

        else:
            for detected_object in detected_objects:
                coordinates, size, confident_score, class_id = detected_object

                if coordinates in self._info.update_area:
                    self._tracked_object_repository.new_tracked_object(*detected_object)

    def _update_from_predictor(self, sequence_number) -> None:
        """
        Updates position by using predictor on each instance of tracked object.

        :param sequence_number: current sequence number
        """

        self._tracked_object_repository.predict()

        if is_frequency(sequence_number, constants.TRACKER_OPTICAL_FLOW_FREQUENCY):
            _, new_frame = self.receive(pipe_id=constants.FRAME_LOADER_ID)
            self._optical_flow.update(new_frame)

            # image = np.copy(new_frame)
            # cv2.imwrite("optical_flow.png", OpticalFlow.draw(image, self._optical_flow.serialize()))

    def _hungarian_method(self, detected_boxes) -> None:
        """
        Solves position assigning problem.
        Modification: doesnt allow to assigne position which are too far away from each other.
        Assigned positions are updated by new detected.
        Unassigned are generated if some conditions are met.

        :param detected_boxes: detect boxes by detector
        """

        matrix = []
        for old_object in self._tracked_object_repository.list:
            row = []
            for new_box in detected_boxes:
                new_coordinates, new_size, new_score, _ = new_box

                if old_object.in_radius(new_coordinates):
                    row.append(old_object.center.distance(new_coordinates))
                else:
                    row.append(constants.TRACKER_DISALLOWED)

            matrix.append(row)

        indexes = self._munkres.compute(matrix)
        detected_boxes_copy = detected_boxes[:]

        for index, row in enumerate(transpose_matrix(matrix)):
            if sum(row) != constants.TRACKER_DISALLOWED * len(row):
                detected_boxes.remove(detected_boxes_copy[index])

        for old_index, new_index in indexes:

            value = matrix[old_index][new_index]

            if value < constants.TRACKER_DISALLOWED:
                old_box = self._tracked_object_repository.list[old_index]
                new_box = detected_boxes_copy[new_index]

                try:
                    detected_boxes.remove(new_box)
                except ValueError:
                    pass

                new_coordinates, size, score, _ = new_box
                old_box.update_position(size, score, new_coordinates)

        for new_box in detected_boxes:
            if new_box[2] > constants.TRACKER_MINIMAL_SCORE:
                coordinates, size, confident_score, _ = new_box

                if coordinates in self._info.update_area:
                    self._tracked_object_repository.new_tracked_object(*new_box)



