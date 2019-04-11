import cv2

import params
from detectors import Coordinates
from pipeline import ThreadedPipeBlock, is_frequency
from pipeline.traffic_light_observer import Color


class Box2D:
    def __init__(self, anchors, car_info, info, light_status):
        self._top_left, self._bottom_right, _ = anchors
        self._car_info = car_info

        self._red_rider = light_status == Color.RED and info.corridors_repository.red_line_crossed(self.tracker_point)

    @property
    def tracker_point(self):
        return Coordinates((self._bottom_right[0] + self._top_left[0]) / 2, self._bottom_right[1])

    def draw(self, image):
        color = params.COLOR_RED if self._red_rider else params.COLOR_GREEN

        cv2.rectangle(img=image,
                      pt1=self._top_left,
                      pt2=self._bottom_right,
                      color=color,
                      thickness=params.OBSERVER_BOX_THICKNESS)

        cv2.rectangle(img=image,
                      pt1=self._top_left,
                      pt2=(self._top_left[0] + 30, self._top_left[1] - 15),
                      color=color,
                      thickness=params.FILL)

        cv2.circle(img=image,
                   center=self.tracker_point.tuple(),
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
        self._current_lights_status = None

    def _step(self, seq):
        tracker_seq, tracked_objects = self.receive(pipe_id=params.TRACKER_ID)
        lights_seq, new_lights_status = self.receive(pipe_id=params.TRAFFIC_LIGHT_OBSERVER_ID)

        if new_lights_status is not None:
            self._current_lights_status = new_lights_status

        boxes = []
        for tracked_object in tracked_objects:
            anchors, _, car_info = tracked_object

            boxes.append(Box2D(anchors=anchors,
                               car_info=car_info,
                               info=self._info,
                               light_status=self._current_lights_status))

        corridors = {index: [] for index in range(self._info.corridors_repository.count)}
        boxes.sort(key=lambda b: b.tracker_point.y)

        for box in boxes[:]:
            corridor_id = self._info.corridors_repository.get_corridor(box.tracker_point)

            if corridor_id == -1:
                boxes.remove(box)
            else:
                corridors[corridor_id].append(box)

        if not self._info.corridors_repository.ready:
            if new_lights_status == Color.GREEN:
                for _, corridor in corridors.items():

                    # get the first car in each corridor (if there is any)
                    if len(corridor):
                        first_box = corridor[0]
                        self._info.corridors_repository.add_stop_point(first_box.tracker_point)

        if is_frequency(seq, params.VIDEO_PLAYER_FREQUENCY):
            message = seq, boxes, self._current_lights_status
            self.send(message, pipe_id=params.VIDEO_PLAYER_ID)
