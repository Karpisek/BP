import numpy as np

import params
from pipeline import ThreadedPipeBlock, is_frequency
from pipeline.base.pipeline import Mode
from repositories.traffic_light_repository import Color


class TrafficLightsObserver(ThreadedPipeBlock):

    colors = [Color.RED_ORANGE, Color.RED, Color.ORANGE, Color.GREEN]

    def __init__(self, info, output):
        super().__init__(output=output, pipe_id=params.TRAFFIC_LIGHT_OBSERVER_ID)

        self._info = info
        self._state = None
        self._state_values = [0, 0, 0]

        self._previous_frame = np.zeros(shape=(info.height, info.width, 3), dtype=np.uint8)
        self._state_candidate = None
        self._state_candidate_count = 0

    def _mode_changed(self, new_mode):
        if new_mode == Mode.DETECTION:
            self._state = None
            self._state_values = [0, 0, 0]

            self._state_candidate = None
            self._state_candidate_count = 0

    def _step(self, seq):
        loader_seq, new_frame = self.receive(pipe_id=params.FRAME_LOADER_ID)

        new_status = self.status(current_frame=new_frame, previous_frame=self._previous_frame)
        self._previous_frame = new_frame

        message = seq, new_status

        if is_frequency(seq, params.OBSERVER_FREQUENCY):
            self.send(message, pipe_id=params.OBSERVER_ID)

        if is_frequency(seq, params.CALIBRATOR_FREQUENCY):
            self.send(message, pipe_id=params.CALIBRATOR_ID, block=False)

    def status(self, current_frame, previous_frame):
        if self._info.traffic_lights_repository.size == 0:
            return None

        new_status = np.array(self._info.traffic_lights_repository.state(current_frame, previous_frame))

        diff = np.around(np.absolute(new_status - self._state_values), decimals=1)

        if self._state == Color.RED:
            if diff[1] > params.TRAFFIC_LIGHT_DEFAULT_THRESHOLD or diff[0] > params.TRAFFIC_LIGHT_DEFAULT_THRESHOLD:
                if self._state_candidate == Color.RED_ORANGE:
                    self._state_candidate_count += 1
                else:
                    self._state_candidate_count = 0
                self._state_candidate = Color.RED_ORANGE

        elif self._state == Color.RED_ORANGE:
            if diff[1] > params.TRAFFIC_LIGHT_DEFAULT_THRESHOLD or diff[0] > params.TRAFFIC_LIGHT_GREEN_THRESHOLD or diff[2] > params.TRAFFIC_LIGHT_DEFAULT_THRESHOLD:
                if self._state_candidate == Color.GREEN:
                    self._state_candidate_count += 1
                else:
                    self._state_candidate_count = 0
                self._state_candidate = Color.GREEN

        elif self._state == Color.GREEN:
            if new_status.max() == 0:
                self._state = Color.GREEN
                return None

            if diff[1] > params.TRAFFIC_LIGHT_DEFAULT_THRESHOLD or diff[2] > params.TRAFFIC_LIGHT_GREEN_THRESHOLD:
                if self._state_candidate == Color.ORANGE:
                    self._state_candidate_count += 1
                else:
                    self._state_candidate_count = 0

                self._state_candidate = Color.ORANGE

        elif self._state == Color.ORANGE:
            if diff[1] > params.TRAFFIC_LIGHT_DEFAULT_THRESHOLD or diff[0] > params.TRAFFIC_LIGHT_DEFAULT_THRESHOLD:
                if self._state_candidate == Color.RED:
                    self._state_candidate_count += 1
                else:
                    self._state_candidate_count = 0

                self._state_candidate = Color.RED

        if new_status[2] > 0.7:
            self._state_candidate = Color.GREEN
            self._state_candidate_count = 5

        if self._state_candidate_count > 0:
            self._state_candidate_count = 0
            self._state_values = new_status
            self._state = self._state_candidate

        return self._state
