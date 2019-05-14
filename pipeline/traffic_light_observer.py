import numpy as np

from primitives import constants
from primitives.enums import Color, Mode
from pipeline.base.pipeline import ThreadedPipeBlock, is_frequency


class TrafficLightsObserver(ThreadedPipeBlock):
    """
    Observes detected light and decides what state is on them.
    Delegates this message to output PipeBlocks.
    For light state classification uses color spectre filtering.
    """

    colors = [Color.RED_ORANGE, Color.RED, Color.ORANGE, Color.GREEN]

    def __init__(self, info, output):
        """
        :param info: InputInfo instance for getting traffic light localization in scene
        :param output: list of PipeBlock output instances
        """

        super().__init__(info=info, output=output, pipe_id=constants.TRAFFIC_LIGHT_OBSERVER_ID)

        self._state = None
        self._state_values = [0, 0, 0]

        self._previous_frame = np.zeros(shape=(info.height, info.width, 3), dtype=np.uint8)
        self._state_candidate = None
        self._state_candidate_count = 0

    def _mode_changed(self, new_mode):
        """
        if mode changed to detection current state is forgotten

        :param new_mode: new mode
        """

        super()._mode_changed(new_mode)

        if new_mode == Mode.DETECTION:
            self._state = None
            self._state_values = [0, 0, 0]

            self._state_candidate = None
            self._state_candidate_count = 0

    def _step(self, seq):
        """
        On each step current traffic light status is computed, then the result is send to ouputs.

        :param seq: current sequnece number
        """

        loader_seq, new_frame = self.receive(pipe_id=constants.FRAME_LOADER_ID)

        new_status = self.status(current_frame=new_frame, previous_frame=self._previous_frame)
        self._previous_frame = new_frame

        message = seq, new_status

        if is_frequency(seq, constants.OBSERVER_FREQUENCY):
            self.send(message, pipe_id=constants.OBSERVER_ID)

        if is_frequency(seq, constants.CALIBRATOR_FREQUENCY):
            self.send(message, pipe_id=constants.CALIBRATOR_ID, block=False)

    def status(self, current_frame, previous_frame):
        """
        Calculates current light status, using color analyses and knowing behaviour of traffic light.
        Change is detected only, if condition is satisfied for more then one defined number of frames in a row

        :param current_frame: current frame
        :param previous_frame: previous frame
        :return: current light state
        """

        if self._info.traffic_lights_repository.size == 0:
            return None

        new_status = np.array(self._info.traffic_lights_repository.state(current_frame, previous_frame))

        diff = np.around(np.absolute(new_status - self._state_values), decimals=1)

        if self._state == Color.RED:
            if diff[1] > constants.TRAFFIC_LIGHT_DEFAULT_THRESHOLD or diff[0] > constants.TRAFFIC_LIGHT_DEFAULT_THRESHOLD:
                if self._state_candidate == Color.RED_ORANGE:
                    self._state_candidate_count += 1
                else:
                    self._state_candidate_count = 0
                self._state_candidate = Color.RED_ORANGE

        elif self._state == Color.RED_ORANGE:
            if diff[1] > constants.TRAFFIC_LIGHT_DEFAULT_THRESHOLD or diff[0] > constants.TRAFFIC_LIGHT_GREEN_THRESHOLD or diff[2] > constants.TRAFFIC_LIGHT_DEFAULT_THRESHOLD:
                if self._state_candidate == Color.GREEN:
                    self._state_candidate_count += 1
                else:
                    self._state_candidate_count = 0
                self._state_candidate = Color.GREEN

        elif self._state == Color.GREEN:
            if new_status.max() == 0:
                self._state = Color.GREEN

            elif diff[1] > constants.TRAFFIC_LIGHT_DEFAULT_THRESHOLD or diff[2] > constants.TRAFFIC_LIGHT_GREEN_THRESHOLD:
                if self._state_candidate == Color.ORANGE:
                    self._state_candidate_count += 1
                else:
                    self._state_candidate_count = 0

                self._state_candidate = Color.ORANGE

        elif self._state == Color.ORANGE:
            if diff[1] > constants.TRAFFIC_LIGHT_DEFAULT_THRESHOLD or diff[0] > constants.TRAFFIC_LIGHT_DEFAULT_THRESHOLD:
                if self._state_candidate == Color.RED:
                    self._state_candidate_count += 1
                else:
                    self._state_candidate_count = 0

                self._state_candidate = Color.RED

        if new_status[2] > 0.9:
            self._state_candidate = Color.GREEN
            self._state_candidate_count = 5

        if new_status[0] > 0.9:
            self._state_candidate = Color.RED
            self._state_candidate_count = 5

        if self._state_candidate_count > 2:
            self._state_candidate_count = 0
            self._state_values = new_status
            self._state = self._state_candidate

        return self._state
