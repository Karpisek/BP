#!/usr/bin/env python3

"""
The main thread used for initialization of proposed system
"""
from pipeline.detector import Detector

__author__ = "Miroslav Karpisek"
__email__ = "xkarpi05@stud.fit.vutbr.cz"
__date__ = "14.5.2019"

import sys
from primitives import constants

from primitives.parser import InputParser
from pipeline.observer import Observer
from pipeline.traffic_light_observer import TrafficLightsObserver
from pipeline.calibrator import Calibrator
from pipeline.tracker import Tracker
from video_stream.frame_loader import FrameLoader
from video_stream.input_info import Info
from video_stream.video_player import VideoPlayer
from video_stream.violation_writer import ViolationWriter

PATH_TO_CAR_MODEL = 'pipeline/models/car_detectors/ssd_mobilenet_v2_COCO.pb'
PATH_TO_LIGHTS_MODEL = 'pipeline/models/traffic_light_detectors/faster_RCNN_COCO.pb'


def main(argv):
    """
    Creates instances of PipeBlocks used in this project.
    Starts each instance and waits until VideoWriter instance finishes his work.

    :param argv: program arguments
    """

    program_arguments = InputParser(argv=argv)

    video_info = Info(video_path=program_arguments.input_video,
                      light_detection_model=PATH_TO_LIGHTS_MODEL,
                      program_arguments=program_arguments)

    # video writer
    video_writer = ViolationWriter(info=video_info,
                                   program_arguments=program_arguments)

    # video playback
    video_player = VideoPlayer(info=video_info, print_fps=True, output=[video_writer])

    # calibrator
    calibrator = Calibrator(info=video_info,
                            output=[])

    # observer
    observer = Observer(info=video_info,
                        output=[video_player,
                                video_writer])

    # traffic light observer
    traffic_lights_observer = TrafficLightsObserver(info=video_info,
                                                    output=[observer, calibrator])

    # car tracker
    tracker = Tracker(info=video_info,
                      output=[observer, calibrator, video_player])

    # car detector
    car_detector = Detector(model=PATH_TO_CAR_MODEL,
                            info=video_info,
                            output=[tracker],
                            detector_type_id=constants.DETECTOR_CAR_ID)
    # frame loader
    frame_loader = FrameLoader(info=video_info,
                               output=[car_detector,
                                       video_player,
                                       tracker,
                                       calibrator,
                                       traffic_lights_observer,
                                       video_writer])

    frame_loader.start()
    car_detector.start()

    tracker.start()
    calibrator.start()
    observer.start()
    video_writer.start()
    traffic_lights_observer.start()

    video_player._loader = frame_loader
    video_player._detector = car_detector
    video_player._tracker = tracker

    video_player.start()
    video_writer.join()


if __name__ == '__main__':
    main(sys.argv[1:])
