import getopt
import json
import sys

import params
from pipeline.parser import ParametersError
from video_stream.input_info import VideoInfo


class BaseAnnotator:
    def __init__(self):

        self.video_input = None
        self.annotation_output_dir = None

        try:
            opts, args = getopt.getopt(sys.argv[1:], "", ["input=", "output-dir="])

        except getopt.GetoptError:
            raise ParametersError

        for opt, arg in opts:

            if opt in "--input":
                self.video_input = arg

            if opt in "--output-dir":
                self.annotation_output_dir = arg

            else:
                print("help")

        if None in [self.video_input, self.annotation_output_dir]:
            exit(1)

        self.video_info = VideoInfo(video_path=self.video_input)
        self.directory_output = f"{self.annotation_output_dir}/{self.video_info.filename}"

        try:
            with open(self.directory_output + "/" + params.ANNOTATIONS_FILENAME, "r") as file:
                self.annotations = json.load(file)
        except FileNotFoundError:
            self.clear_annotations()

    def run(self):
        seq = 0

        self._before()
        while True:
            image = self.video_info.read()
            try:
                self._step(seq, image)
            except EOFError:
                break

            seq += 1

        self._after()

    def _add_annotation(self, key, seq, key2=None):

        if key2 is None:
            self.annotations[key].append(seq)
            print(len(self.annotations[key]))

        else:
            self.annotations[key][key2].append(seq)
            print(len(self.annotations[key][key2]))

    def _step(self, image, seq):
        raise NotImplementedError

    def _before(self):
        raise NotImplementedError

    def _after(self):
        with open(self.directory_output + "/" + params.ANNOTATIONS_FILENAME, "w") as file:
            json.dump(self.annotations, file)

    def clear_annotations(self):
        self.annotations = {"cars": [],
                            "lights": {
                                "green": [],
                                "orange": [],
                                "red": [],
                                "red_orange": []
                            },
                            "violations": []}
