import cv2

import params
from pipeline import ThreadedPipeBlock, is_frequency


class Box2D:
    def __init__(self, anchors, car_info):
        self._top_left, self._bottom_right, _ = anchors
        self._car_info = car_info

    @property
    def tracker_point(self):
        return int((self._bottom_right[0] + self._top_left[0]) / 2), self._bottom_right[1]

    def draw(self, image):
        cv2.rectangle(img=image,
                      pt1=self._top_left,
                      pt2=self._bottom_right,
                      color=params.COLOR_GREEN,
                      thickness=params.DEFAULT_THICKNESS)

        cv2.rectangle(img=image,
                      pt1=self._top_left,
                      pt2=(self._top_left[0] + 30, self._top_left[1] - 15),
                      color=params.COLOR_GREEN,
                      thickness=params.FILL)

        cv2.circle(img=image,
                   center=self.tracker_point,
                   color=params.COLOR_RED,
                   radius=5,
                   thickness=params.FILL)

        cv2.putText(img=image,
                    text=self._car_info,
                    org=self._top_left,
                    fontFace=1,
                    fontScale=1,
                    color=params.COLOR_BLACK,
                    thickness=2)


class Observer(ThreadedPipeBlock):
    def __init__(self, info, output, pipe_id=params.OBSERVER_ID):

        super().__init__(pipe_id=pipe_id,
                         output=output)

        self._info = info

    def _step(self, seq):
        sequence_number, tracked_objects = self.receive(pipe_id=params.TRACKER_ID)

        boxes = []
        for tracked_object in tracked_objects:
            anchors, _, car_info = tracked_object

            boxes.append(Box2D(anchors=anchors,
                               car_info=car_info))

        if is_frequency(seq, params.VIDEO_PLAYER_FREQUENCY):
            message = sequence_number, boxes
            self.send(message, pipe_id=params.VIDEO_PLAYER_ID)


        # print("observer_step")
