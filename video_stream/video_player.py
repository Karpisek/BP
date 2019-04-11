import time
import cv2
import params

from pipeline import PipeBlock


class VideoPlayer(PipeBlock):
    def __init__(self, info):
        super().__init__(pipe_id=params.VIDEO_PLAYER_ID)

        self._detector = None
        self._loader = None
        self._tracker = None

        self._info = info

    def start(self):
        clock = time.time()
        frame_counter = 0

        while True:
            seq, image = self.receive(params.FRAME_LOADER_ID)
            frame_counter += 1

            observer_seq, boxes, lights_state = self.receive(pipe_id=params.OBSERVER_ID)

            [box.draw(image) for box in boxes]

            self._info.draw_vanishing_points(image)

            image_with_corridors = self._info.draw_corridors(image)
            image_with_traffic_lights = self._info.draw_detected_traffic_lights(image_with_corridors)

            # self._info.draw(image_with_corridors, lights_state)

            cv2.imshow("image", image_with_traffic_lights)

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

        cv2.destroyAllWindows()
