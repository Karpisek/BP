import cv2


COLOR_AREA = (255, 255, 204)
AREA_THICKNESS = 3


class Area:
    def __init__(self, info, top_left, bottom_right):

        self._top_left = top_left

        self._bot_right = bottom_right

        self._info = info

    @property
    def defined(self):
        return self._top_left is not None and self._bot_right is not None

    def draw(self, image, color=COLOR_AREA):

        if self.defined:
            top_left, bot_right = self.anchors()
            cv2.rectangle(image, top_left.tuple(), bot_right.tuple(), color, AREA_THICKNESS)

    def anchors(self):
        return self._top_left, self._bot_right

    def contains(self, coordinates):
        if not self.defined:
            return False

        if coordinates.is_relative:
            coordinates.convert_to_fixed(self._info)

        if coordinates.x < self._top_left.x:
            return False

        if coordinates.y < self._top_left.y:
            return False

        if coordinates.x > self._bot_right.x:
            return False

        if coordinates.y > self._bot_right.y:
            return False

        return True
