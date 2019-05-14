import cv2
import numpy as np

from primitives import constants
from primitives.line import Line, SamePointError
from user_interaction.helper_classes.linedrag import LineDrag
from user_interaction.helper_classes.mouse_callback import mouse_callback


class CorridorMaker(LineDrag):
    """
    Used for corridor marking by user
    """

    def __init__(self, image):
        super().__init__(image)
        self.selected = []
        self.base_image = image

    def add_line(self):
        """
        Adds new line to storage
        """

        try:
            new_line = Line(self.point1, self.point2)

            if new_line.angle(Line.horizontal_line()) != 0:
                self.selected.append(Line(self.point1, self.point2))

        except SamePointError:
            pass

    def draw(self, image):
        """
        Helper method for drawing selected corridor lines

        :param image: selected image
        """

        super().draw(image)

        for line in self.selected:
            line.draw(image, color=constants.COLOR_RED, thickness=constants.CORRIDORS_LINE_SELECTOR_THICKNESS)

    def __iter__(self):
        return iter(self.selected)

    def clear(self):
        """
        Clears all selected corridor lines
        """

        self.selected = []
        super().clear()

    def erase_last(self):
        """
        Erases last selected corridor line
        """

        try:
            self.selected.pop()
        except IndexError:
            pass

    def run(self, info) -> []:
        """
        Runs infinite loop until user commits selected corridor lines.

        :param info: instance of InputInfo
        """

        cv2.namedWindow("select_corridors")
        cv2.setMouseCallback("select_corridors", mouse_callback, self)

        while True:
            image_copy = np.copy(self.base_image)

            self.draw(image_copy)
            cv2.imshow("select_corridors", image_copy)

            key = cv2.waitKey(1)

            if key == 13:
                for line in self:
                    line.draw(image_copy, constants.COLOR_BLUE, constants.CORRIDORS_LINE_SELECTOR_THICKNESS)
                    cv2.imshow("select_corridors", image_copy)

                key = cv2.waitKey(0)
                if key == 13:
                    cv2.destroyWindow("select_corridors")

                    if len(self.selected) < 2:
                        print("Not minimal number of lines selected. Please select at least 2 lines")
                        continue
                    else:
                        return sorted(self.selected, key=lambda l: l.find_coordinate(y=info.height)[0])

                if key == 8:
                    self.clear()

            if key == 8:
                self.erase_last()
