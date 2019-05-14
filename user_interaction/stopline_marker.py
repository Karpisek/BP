import cv2
import numpy as np

from primitives import constants
from primitives.line import SamePointError, Line
from user_interaction.helper_classes.linedrag import LineDrag
from user_interaction.helper_classes.mouse_callback import mouse_callback


class StopLineMaker(LineDrag):
    """
    Used for stop line creation
    """

    def __init__(self, image):
        super().__init__(image)

        self.line = None

    def add_line(self):
        """
        Adds stop line
        """

        try:
            self.line = Line(self.point1, self.point2)
        except SamePointError:
            pass

    def draw(self, image):
        """
        Helper function to draw stop line

        :param image: selected image
        """

        super().draw(image)

        if self.line is not None:
            self.line.draw(image, color=constants.COLOR_RED, thickness=constants.CORRIDORS_LINE_SELECTOR_THICKNESS)

    def run(self, info) -> []:
        """
        Infinite loop until user selects stop line

        :param info: instance of InputInfo
        """

        cv2.namedWindow("select_stop_line")
        cv2.setMouseCallback("select_stop_line", mouse_callback, self)

        while True:
            image_copy = np.copy(self.base_image)

            self.draw(image_copy)
            cv2.imshow("select_stop_line", image_copy)

            key = cv2.waitKey(1)

            if key == 13:
                self.line.draw(image_copy, constants.COLOR_BLUE, constants.CORRIDORS_LINE_SELECTOR_THICKNESS)
                cv2.imshow("select_stop_line", image_copy)

                key = cv2.waitKey(0)
                if key == 13:
                    cv2.destroyWindow("select_stop_line")
                    return self.line

                if key == 8:
                    self.clear()

            if key == 8:
                self.erase_last()

    def erase_last(self):
        """
        Deletes last selected stopline
        """

        self.line = None
