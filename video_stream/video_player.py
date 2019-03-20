import time
import cv2

from bbox import Box2D
from bbox.optical_flow import OpticalFlow
from pipeline import PipeBlock


class VideoPlayer(PipeBlock):
    def __init__(self, area_of_detection, info):
        super().__init__()

        self._detector = None
        self._loader = None
        self._tracker = None
        self._info = info

        self._area_of_detection = area_of_detection

    def start(self):

        cv2.namedWindow("image")

        clock = time.time()
        frame_counter = 0

        seq, image = self.next()
        frame_counter += 1

        self._area_of_detection.select(cv2.selectROI("image", image), self._info)

        while True:
            tracker_seq, boxes, serialized_optical_flow = self.next(pipe=1)

            [Box2D.draw(image, *serialized_box) for serialized_box in boxes]

            self._area_of_detection.draw(image)

            mask = OpticalFlow.draw(image, serialized_optical_flow=serialized_optical_flow)

            cv2.imshow("image", cv2.add(image, mask))
            key = cv2.waitKey(10)

            #  commands
            if key & 0xFF == ord("q"):
                break

            elif key & 0xFF == ord("d"):
                self._info.track_boxes = not self._info.track_boxes

            if frame_counter > 100:
                print("FPS: ", 1000 / (((time.time() - clock) / frame_counter) * 1000), self, self._detector, self._loader, self._tracker)

                frame_counter = 0
                clock = time.time()

            seq, image = self.next()
            frame_counter += 1

        cv2.destroyAllWindows()




