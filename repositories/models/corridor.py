import cv2
import numpy as np

from primitives import constants


class TrafficCorridor:
    """
    Representation of single traffic corridor.
    """

    def __init__(self, index, left_line, right_line, middle_point):
        """
        :param index: corridor id
        :param left_line: left line of corridor
        :param right_line: right line of corridor
        :param middle_point: middle point to use flood fill
        """

        self._id = index

        self.left_line = left_line
        self.right_line = right_line
        self.middle_point = middle_point

    @property
    def id(self):
        """
        :return: corridor id
        """

        return self._id

    def draw_corridor(self, image, info, color=None, fill=True, thickness=constants.DEFAULT_THICKNESS):
        """
        Helper function for drawing corridor

        :param image: image to draw on
        :param info: instance of InputInfo
        :param color: selected color
        :param fill: if flood fill should be used
        :param thickness: thickness of line defining corridors
        """

        if color is None:
            color = self.id

        if fill:
            mask = np.zeros(shape=(info.height, info.width),
                            dtype=np.uint8)

            self.left_line.draw(image, color, thickness)
            self.left_line.draw(mask, color, thickness)

            self.right_line.draw(image, color, thickness)
            self.right_line.draw(mask, color, thickness)

            mask_with_border = np.pad(mask, 1, 'constant', constant_values=255)

            cv2.floodFill(image=image,
                          mask=mask_with_border,
                          seedPoint=self.middle_point,
                          newVal=color)
        else:
            self.left_line.draw(image=image,
                                color=color,
                                thickness=5)

            self.right_line.draw(image=image,
                                 color=color,
                                 thickness=5)

    def serialize(self):
        """
        :return: serialized Traffic corridor in form of dictionary
        """

        return {"left_line": self.left_line.serialize(),
                "right_line": self.right_line.serialize()}

    def __str__(self):
        return f"corridor: id [{self.id}]"

