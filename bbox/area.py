import cv2

from bbox.coordinates import Coordinates

COLOR_AREA = (255, 255, 204)
AREA_THICKNESS = 3


class Area:
    def __init__(self):

        self.top_left = None
        self.bot_right = None
        self.info = None

    @property
    def defined(self):
        return self.top_left is not None and self.bot_right is not None

    def select(self, info):
        self.top_left = Coordinates(0, int(info.height/4))
        self.bot_right = Coordinates(info.width, info.height)
        self.info = info

    def draw(self, image, color=COLOR_AREA):

        if self.defined:
            top_left, bot_right = self.anchors()
            cv2.rectangle(image, top_left.tuple(), bot_right.tuple(), color, AREA_THICKNESS)

    def anchors(self):
        return self.top_left, self.bot_right

    def contains(self, coordinates, relative=False):
        if not self.defined:
            return False

        if relative:
            coordinates.convert_to_fixed(self.info)

        if coordinates.x < self.top_left.x:
            return False

        if coordinates.y < self.top_left.y:
            return False

        if coordinates.x > self.bot_right.x:
            return False

        if coordinates.y > self.bot_right.y:
            return False

        return True
