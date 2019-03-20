import time
from queue import Queue
import cv2
import numpy as np

from bbox import Box2D
from pipeline import PipeBlock

color = np.random.randint(0, 255, (10000, 3))


class VideoPlayer(PipeBlock):
    def __init__(self, area_of_detection, info):
        super().__init__()

        self._input = [Queue(20), Queue(20)]

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
            tracker_seq, boxes, old_positions, new_positions = self.next(pipe=1)
            mask = np.zeros_like(image)

            for serialized_box in boxes:
                Box2D.draw(image, *serialized_box, center=True)

            self._area_of_detection.draw(image)

            for i, (new, old) in enumerate(zip(new_positions, old_positions)):
                a, b = new.ravel()
                c, d = old.ravel()
                cv2.line(mask, (a, b), (c, d), color[i].tolist(), 1)

            cv2.imshow("image", cv2.add(image, mask))
            key = cv2.waitKey(20)

            if key & 0xFF == ord("q"):
                break

            elif key & 0xFF == ord("d"):
                self._tracker.switch()

            if frame_counter > 100:
                print("FPS: ", 1000 / (((time.time() - clock) / frame_counter) * 1000), self._detector, self._loader)

                frame_counter = 0
                clock = time.time()

            seq, image = self.next()
            frame_counter += 1

        cv2.destroyAllWindows()




