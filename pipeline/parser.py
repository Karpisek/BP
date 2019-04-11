import getopt
import sys


class ParametersError(Exception):
    pass


class InputParser:
    def __init__(self, argv):
        try:
            opts, args = getopt.getopt(argv, "lcs", ["light", "corridors", "stop-line"])

        except getopt.GetoptError:
            raise ParametersError

        self._light = False
        self._corridors = False
        self._stop_line = False

        for opt, arg in opts:
                
            if opt in ("-l", "--light"):
                self._light = True

            if opt in ("-c", "--corridors"):
                self._corridors = True
                
            if opt in ("-s", "--stop-line"):
                self._stop_line = True

    @property
    def insert_light(self):
        return self._light

    @property
    def insert_corridors(self):
        return self._corridors

    @property
    def insert_stop_line(self):
        return self._stop_line
