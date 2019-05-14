import cv2
from primitives import constants


class LineDrag:
    """
    Used for user interaction (dragging line)
    """

    def __init__(self, image):
        """
        :param image: selected image to draw on
        """

        self.selected = []
        self.base_image = image

        self.point1 = None
        self.point2 = None

    def click(self, point):
        """
        click response on user click

        :param point: coordinates where was clicked
        """

        if self.point1 is None:
            self.point1 = point

        else:
            self.__add_line()

    def move(self, point):
        """
        response for mouse move

        :param point: current coordinates of mouse
        """

        self.point2 = point

    def __add_line(self):
        self.add_line()

        self.point1 = None
        self.point2 = None

    def draw(self, image):
        """
        Helper function to draw selected line on selected image.

        :param image: selected image.
        """

        if self.point1 is not None:
            cv2.line(img=image,
                     pt1=self.point1,
                     pt2=self.point2,
                     color=constants.COLOR_GREEN,
                     thickness=constants.CORRIDORS_LINE_SELECTOR_THICKNESS)

    def clear(self):
        """
        clears selected points
        """

        self.point1 = None
        self.point2 = None

    def run(self, info) -> []:
        raise NotImplementedError

    def add_line(self):
        raise NotImplementedError

    def erase_last(self):
        raise NotImplementedError

