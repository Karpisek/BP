from enum import Enum

import cv2
import numpy as np

import params
from pipeline import ThreadedPipeBlock, is_frequency


class Color(Enum):
    RED = 0
    RED_ORANGE = 1
    ORANGE = 2
    GREEN = 3


class TrafficLightsObserver(ThreadedPipeBlock):

    colors = [Color.RED_ORANGE, Color.RED, Color.ORANGE, Color.GREEN]

    def __init__(self, info, output, tweak_roi=None):
        super().__init__(output=output, pipe_id=params.TRAFFIC_LIGHT_OBSERVER_ID)

        self._traffic_lights = []
        self._info = info

        self._current_status = np.array([1, 1, 1, 1])
        self._next_status = np.array([0, 0, 0, 0])

        if tweak_roi is not None:
            self.new_traffic_light(*tweak_roi)

    def _step(self, seq):
        loader_seq, new_frame = self.receive(pipe_id=params.FRAME_LOADER_ID)

        if is_frequency(seq, params.OBSERVER_FREQUENCY):
            new_status = self.status(image=new_frame)

            message = seq, new_status
            self.send(message, pipe_id=params.OBSERVER_ID)

    def new_traffic_light(self, x, y, width, height):
        top_left = x, y
        bottom_right = x + width, y + height

        new_traffic_light = TrafficLight(top_left=top_left,
                                         bottom_right=bottom_right)

        self._traffic_lights.append(new_traffic_light)

    def status(self, image):
        if len(self._traffic_lights) == 0:
            return None

        new_status = self._traffic_lights[0].state(image)
        new_status *= self._current_status

        self._next_status -= [1, 1, 1, 1]
        self._next_status += new_status
        self._next_status[self._next_status < 0] = 0

        maximum = np.amax(self._next_status)
        index = np.argmax(self._next_status)
        if maximum > 3:
            self._current_status = np.array([1, 1, 1, 1])
            self._current_status[index] = 0

            print(TrafficLightsObserver.colors[int(index)])
            return TrafficLightsObserver.colors[int(index)]
        return None


class TrafficLight:
    def __init__(self, top_left, bottom_right):
        self._top_left = top_left
        self._bottom_right = bottom_right

    def state(self, image):
        light_roi = image[self._top_left[1]: self._bottom_right[1], self._top_left[0]: self._bottom_right[0]]

        smoothed_light_roi = cv2.medianBlur(light_roi, 3)
        hsv_light_roi = cv2.cvtColor(smoothed_light_roi, cv2.COLOR_BGR2HSV)

        lower = np.array([0, 40, 150])
        upper = np.array([180, 255, 255])
        light_state_mask = cv2.inRange(hsv_light_roi, lower, upper)

        height, width = light_state_mask.shape

        red_count = np.count_nonzero(light_state_mask[:int(height / 3), :])
        orange_count = np.count_nonzero(light_state_mask[int(height / 3):int(-height / 3), :])
        green_count = np.count_nonzero(light_state_mask[-int(height / 3):, :])

        maximum = np.amax([red_count, orange_count, green_count])

        if maximum > 0:
            if green_count == maximum:
                return [0, 0, 0, 2]

            if red_count == maximum and orange_count > maximum/2 and green_count < orange_count > maximum/2:
                return [2, 0, 0, 0]

            if red_count == maximum and orange_count < maximum/2 and green_count < maximum/2:
                return [0, 2, 0, 0]

            if orange_count == maximum and red_count < maximum/2 and green_count < maximum/2:
                return [0, 0, 2, 0]

        return [0, 0, 0, 0]
