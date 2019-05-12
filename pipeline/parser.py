import getopt
import sys


class ParametersError(Exception):
    pass


class InputParser:
    """
    Used for program argument parsing
    """

    def __init__(self, argv):
        """
        :param argv: program arguments
        """

        try:
            opts, args = getopt.getopt(argv, "lc", ["light", "corridors", "input=", "output="])

        except getopt.GetoptError:
            raise ParametersError

        self._light = False
        self._corridors = False
        self._input = None
        self._output = None

        for opt, arg in opts:

            if opt in ("-l", "--light"):
                self._light = True

            if opt in ("-c", "--corridors"):
                self._corridors = True

            if opt in "--input":
                self._input = arg

            if opt in "--output":
                self._output = arg

    @property
    def insert_light(self):
        """
        :return: if user wants to select lights manualy
        """

        return self._light

    @property
    def insert_corridors(self):
        """
        :return: if user wants to select corridors manualy
        """

        return self._corridors

    @property
    def input_video(self):
        """
        :return: path of input video
        """

        return self._input

    @property
    def output_dir(self):
        """
        :return: output directory path
        """

        return self._output
