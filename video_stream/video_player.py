import time
import cv2
import params

from bbox import Box2D
from pipeline import PipeBlock
from pipeline.pipeline import is_frequency


class VideoPlayer(PipeBlock):
    def __init__(self, area_of_detection, info):
        super().__init__(pipe_id=params.VIDEO_PLAYER_ID)

        self._detector = None
        self._loader = None
        self._tracker = None
        self.calibrator = None

        self._info = info

        self._area_of_detection = area_of_detection

    def start(self):

        cv2.namedWindow("image")

        clock = time.time()
        frame_counter = 0

        seq, image, foreground = self.receive(params.FRAME_LOADER_ID)
        frame_counter += 1

        self._area_of_detection.select(self._info)

        while True:
            tracker_seq, boxes, lifelines = self.receive(pipe_id=params.TRACKER_ID)

            Box2D.draw(image, boxes)

            self._info.draw_vanishing_points(image)

            image_with_corridors = self._info.draw_corridors(image)
            # self._area_of_detection.draw(image)

            background = cv2.bitwise_not(foreground)

            foreground_cars = cv2.bitwise_and(image, image, mask=foreground)
            background_corridors = cv2.bitwise_and(image_with_corridors, image_with_corridors, mask=background)

            image = cv2.add(foreground_cars, background_corridors)

            image_calibrator = None
            if is_frequency(seq, params.CALIBRATOR_FREQUENCY):
                image_calibrator = self.receive(pipe_id=params.CALIBRATOR_ID, block=False)

            if image_calibrator is not None:
                cv2.imshow("image", image_calibrator)
            else:
                cv2.imshow("image", image)

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

            seq, image, foreground = self.receive(params.FRAME_LOADER_ID)
            frame_counter += 1

        cv2.destroyAllWindows()




