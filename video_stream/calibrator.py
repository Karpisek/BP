import params
import time

from queue import Full
from pipeline import ThreadedPipeBlock


class Calibrator(ThreadedPipeBlock):
    def __init__(self, output=None):
        super().__init__(pipe_id=params.CALIBRATOR_ID, output=output)

        self._vanishing_points = [VanishingPoint() for _ in range(3)]

    def _step(self, seq):


        if all(vp.found() for vp in self._vanishing_points):
            exit()

        seq_loader, image = self.receive(pipe_id=params.FRAME_LOADER_ID)
        seq_tracker, boxes = self.receive(pipe_id=params.TRACKER_ID)

        print(seq_tracker, seq_loader)

    def find_vanishing_point(self):
        time.sleep(2)
        self._vanishing_points.append(VanishingPoint())


class VanishingPoint:
    def __init__(self):
        self._position = None

    def found(self):
        return False


