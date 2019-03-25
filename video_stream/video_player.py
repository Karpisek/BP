import time
import cv2
import params

from bbox import Box2D
from bbox.optical_flow import OpticalFlow
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

        seq, image = self.receive(params.FRAME_LOADER_ID)
        frame_counter += 1

        self._area_of_detection.select(cv2.selectROI("image", image), self._info)

        while True:
            tracker_seq, boxes, lifelines = self.receive(pipe_id=params.TRACKER_ID)

            Box2D.draw(image, boxes, lifelines)
            self.calibrator.draw_vanishing_points(image, self._info)

            self._area_of_detection.draw(image)

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

            seq, image = self.receive(params.FRAME_LOADER_ID)
            frame_counter += 1

        cv2.destroyAllWindows()




