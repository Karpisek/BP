import cv2
import numpy as np
from primitives import constants

from primitives.line import Line


class Area:
    """
    Representing interesting area on image. Specified by anchors. Can be any 4 corner object.
    """

    def __init__(self, info, top_left, bottom_right, top_right, bottom_left):
        """
        :param info: instance of InputInfo
        :param top_left: top left anchor of area
        :param bottom_right: bottom right anchor of area
        :param top_right: top right anchor of area
        :param bottom_left: bottom left anchor of area
        """

        self._top_left = top_left.tuple()
        self._bot_right = bottom_right.tuple()
        self._top_right = top_right.tuple()
        self._bot_left = bottom_left.tuple()

        self.top_line = Line(point1=self._top_left,
                             point2=self._top_right)
        self.left_line = Line(point1=self._top_left,
                              point2=self._bot_left)
        self.bottom_line = Line(point1=self._bot_left,
                                point2=self._bot_right)
        self.right_line = Line(point1=self._top_right,
                               point2=self._bot_right)

        self._info = info

    @property
    def middle_point(self):
        """
        :return: middle point of area
        """

        l1 = Line(point1=self._top_left, point2=self._bot_right)
        l2 = Line(point1=self._bot_left, point2=self._top_right)

        return l1.intersection(l2)

    @property
    def defined(self):
        """
        :return: if area is defined
        """

        return self._top_left is not None and self._bot_right is not None

    def change_area(self, top_line=None, bot_line=None, left_line=None, right_line=None):
        """
        Changes area shape

        :param top_line: new top line
        :param bot_line: new bot line
        :param left_line: new left line
        :param right_line: new right line
        """

        self.top_line = top_line if top_line is not None else self.top_line
        self.bottom_line = bot_line if bot_line is not None else self.bottom_line
        self.right_line = right_line if right_line is not None else self.right_line
        self.left_line = left_line if left_line is not None else self.left_line

    def draw(self, image, color=constants.COLOR_AREA):
        """
        Helper function to draw an area

        :param image: selected image to draw on
        :param color: selected color to draw
        :return: updated image
        """

        if self.defined:
            self.top_line.draw(image, color, constants.AREA_THICKNESS)

        return image

    def anchors(self):
        """
        :return: top left nad bottom right anchor of the area
        """

        return self._top_left, self._bot_right

    def __contains__(self, coordinates):
        if not self.defined:
            return False

        if self.top_line.find_coordinate(x=coordinates.x)[1] > coordinates.y:
            return False

        if self.bottom_line.find_coordinate(x=coordinates.x)[1] < coordinates.y:
            return False

        if self.right_line.find_coordinate(y=coordinates.y)[0] < coordinates.x:
            return False

        if self.left_line.find_coordinate(y=coordinates.y)[0] > coordinates.x:
            return False

        return True

    def mask(self):
        """
        Generates binary mask of area

        :return: binary mask of area
        """

        mask = np.zeros(shape=(self._info.height, self._info.width), dtype=np.uint8)

        self.draw(image=mask, color=constants.COLOR_WHITE_MONO)

        mask_with_border = np.pad(mask, 1, 'constant', constant_values=255)

        cv2.floodFill(image=mask,
                      mask=mask_with_border,
                      seedPoint=(int(self.middle_point[0]), int(self.middle_point[1])),
                      newVal=constants.COLOR_WHITE_MONO)

        return mask
