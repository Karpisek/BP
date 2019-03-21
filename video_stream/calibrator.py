from queue import Full

from params import CALIBRATOR_ID
from pipeline import ThreadedPipeBlock

import time


class Calibrator(ThreadedPipeBlock):
    def __init__(self, output=None):

        super().__init__(pipe_id=CALIBRATOR_ID, output=output)
        self._vanishing_points = [VanishingPoint() for _ in range(3)]

    def _step(self, seq):

        # searches for vanishing points
        print("Search for VP begins")

        print(all(vp.found for vp in self._vanishing_points))

        if all(vp.found for vp in self._vanishing_points):
            exit()
        print(f"Search for VP ended]")

    def find_vanishing_point(self):
        time.sleep(2)
        self._vanishing_points.append(VanishingPoint())

    def deliver(self, message, pipe=0):
        try:
            self._input.put_nowait(message)
        except Full:
            pass


class VanishingPoint:
    def __init__(self):
        self._position = None

    def found(self):
        return False


