import cv2
import numpy as np

from primitives import constants


class TrafficLight:
    """
    Represents detected traffic light. Allows to detect current state of this light.
    """

    def __init__(self, top_left, bottom_right):
        """
        :param top_left: top left anchor of detected traffic light
        :param bottom_right: bottom right anchor of detected traffic light
        """
        self._top_left = top_left.tuple()
        self._bottom_right = bottom_right.tuple()

    def state_counts(self, current_frame, previous_frame):
        """
        Returns relative counts of colors in every part of specified color spectre.
        Uses HSV color model for filtering colors.
        For error correction uses combination of current and previous frame

        :param current_frame: current fram
        :param previous_frame: previous frame
        :return:
        """

        current_light = current_frame[self._top_left[1]: self._bottom_right[1], self._top_left[0]: self._bottom_right[0]]
        previous_light = previous_frame[self._top_left[1]: self._bottom_right[1], self._top_left[0]: self._bottom_right[0]]

        smoothed_current_light = cv2.blur(current_light, (3, 3))
        smoothed_previous_light = cv2.blur(previous_light, (3, 3))

        combined_image = cv2.max(smoothed_current_light, smoothed_previous_light)

        hsv_image = cv2.cvtColor(combined_image, cv2.COLOR_BGR2HSV)

        low_red_mask = cv2.inRange(hsv_image, (0, 100, 10), (5, 255, 255))
        up_red_mask = cv2.inRange(hsv_image, (160, 100, 10), (180, 255, 255))
        orange_mask = cv2.inRange(hsv_image, (5, 100, 10), (25, 255, 255))
        green_mask = cv2.inRange(hsv_image, (30, 100, 10), (95, 255, 255))

        red_mask = np.maximum(low_red_mask, up_red_mask)

        red_count = cv2.countNonZero(red_mask)
        orange_count = cv2.countNonZero(orange_mask)
        green_count = cv2.countNonZero(green_mask)

        all_count = red_count + orange_count + green_count

        if all_count < 10:  # 30
            return 0, 0, 0, 0
        else:
            return all_count, red_count/all_count, orange_count/all_count, green_count/all_count

    def draw_contours(self, image):
        """
        Helper function for contoure drawing of detected traffic light

        :param image: selected image to draw on
        """

        cv2.rectangle(img=image,
                      pt1=self._top_left,
                      pt2=self._bottom_right,
                      thickness=3,
                      color=constants.COLOR_YELLOW)

    def serialize(self):
        """
        :return: serialized traffic light
        """

        return {"top left": list(self._top_left), "bottom right": list(self._bottom_right)}



