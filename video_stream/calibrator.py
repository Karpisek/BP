from queue import Full
from threading import Thread
from pipeline import PipeBlock

import time


class Calibrator(PipeBlock):
    def __init__(self, output=None):

        super().__init__(output)

        self._vanishing_points = [VanishingPoint() for _ in range(3)]

        self._thread = Thread(target=self._run)
        self._thread.daemon = True
        self._thread.start()

    def _run(self):

        # searches for vanishing points
        print("Search for VP begins")
        while all(vp.found for vp in self._vanishing_points):
            time.sleep(1)
            pass

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


