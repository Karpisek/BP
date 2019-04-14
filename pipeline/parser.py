import getopt
import sys


class ParametersError(Exception):
    pass


class InputParser:
    def __init__(self, argv):
        try:
            opts, args = getopt.getopt(argv, "lcs", ["light", "corridors", "input", "output"])

        except getopt.GetoptError:
            raise ParametersError

        self._light = False
        self._corridors = False

        for opt, arg in opts:
                
            if opt in ("-l", "--light"):
                self._light = True

            if opt in ("-c", "--corridors"):
                self._corridors = True

    @property
    def insert_light(self):
        return self._light

    @property
    def insert_corridors(self):
        return self._corridors
