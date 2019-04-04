import time
import cv2
import params

from bbox import TrackedObject
from pipeline import PipeBlock
from pipeline.pipeline import is_frequency


class VideoPlayer(PipeBlock):
    def __init__(self, info):
        super().__init__(pipe_id=params.VIDEO_PLAYER_ID)

        self._detector = None
        self._loader = None
        self._tracker = None

        self._info = info

    def start(self):

        cv2.namedWindow("image")

        clock = time.time()
        frame_counter = 0

        seq, image = self.receive(params.FRAME_LOADER_ID)
        frame_counter += 1

        while True:
            observer_seq, boxes, lights_state = self.receive(pipe_id=params.OBSERVER_ID)

            [box.draw(image) for box in boxes]

            # self._info.draw_vanishing_points(image)

            image_with_corridors = self._info.draw_corridors(image)
            self._info.draw(image_with_corridors, lights_state)

            cv2.imshow("image", image_with_corridors)

            key = cv2.waitKey(params.VIDEO_PLAYER_SPEED)

            #  commands
            if key & 0xFF == ord("q"):
                break

            elif key & 0xFF == ord("d"):
                self._info.track_boxes = not self._info.track_boxes

            if frame_counter > 100:
                print("FPS: ", 1000 / (((time.time() - clock) / frame_counter) * 1000), self, self._detector, self._loader, self._tracker)

                frame_counter = 0
                clock = time.time()

            if seq == self._info.frame_count:
                break

            seq, image = self.receive(params.FRAME_LOADER_ID)
            frame_counter += 1

        cv2.destroyAllWindows()
