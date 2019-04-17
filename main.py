import sys

from pipeline.parser import InputParser
from pipeline.observer import Observer
from pipeline.traffic_light_observer import TrafficLightsObserver
from video_stream import FrameLoader, Info, VideoPlayer
from pipeline import Tracker, Calibrator
from detectors import Detector

import params


# VIDEO_PATH = "/Users/miro/Desktop/00004.MTS"
# VIDEO_PATH = "/Users/miro/Desktop/x.mov"
# VIDEO_PATH = "/Volumes/Miro/FIT/BP/Dataset/RedLightViolation/right_camera/00003.MTS"
# VIDEO_PATH = "/Volumes/Miro/FIT/BP/Dataset/RedLightViolation/zoom_camera/00002.MTS"
# VIDEO_PATH = "/Users/miro/Desktop/v2.mp4"

# VIDEO_PATH = "/Volumes/Miro/FIT/BP/Dataset/iARTIST_crossroads/1.mp4"
# VIDEO_PATH = "/Volumes/Miro/FIT/BP/Dataset/iARTIST_crossroads/1a.mp4"
# VIDEO_PATH = "/Volumes/Miro/FIT/BP/Dataset/iARTIST_crossroads/1b.mp4"
# VIDEO_PATH = "/Volumes/Miro/FIT/BP/Dataset/iARTIST_crossroads/2.mp4"
# VIDEO_PATH = "/Volumes/Miro/FIT/BP/Dataset/iARTIST_crossroads/2a.mp4"
# VIDEO_PATH = "/Volumes/Miro/FIT/BP/Dataset/iARTIST_crossroads/2b.mp4"
# VIDEO_PATH = "/Volumes/Miro/FIT/BP/Dataset/iARTIST_crossroads/3.mp4"
# VIDEO_PATH = "/Volumes/Miro/FIT/BP/Dataset/iARTIST_crossroads/3a.mp4"
# VIDEO_PATH = "/Volumes/Miro/FIT/BP/Dataset/iARTIST_crossroads/3b.mp4"
# VIDEO_PATH = "/Volumes/KINGSTON/cam5-1548565201.mp4"
# VIDEO_PATH = "/Volumes/KINGSTON/cam5-1548342000.mp4"
VIDEO_PATH = "/Volumes/KINGSTON/cam5-1548518400.mp4"

PATH_TO_CAR_MODEL = 'detectors/models/car_detectors/ssd2.pb'
# PATH_TO_CAR_MODEL = 'detectors/models/car_detectors/ssd_car_detection_model.pb'

PATH_TO_LIGHTS_MODEL = 'detectors/models/traffic_light_detectors/rcnn_coco_model.pb'


def main(argv):

    program_arguments = InputParser(argv=argv)

    video_info = Info(video_path=VIDEO_PATH,
                      light_detection_model=PATH_TO_LIGHTS_MODEL,
                      program_arguments=program_arguments)

    # video playback
    video_player = VideoPlayer(info=video_info, print_fps=True)

    # video writer
    # video_writer = VideoWriter(info=video_info)

    # calibrator
    calibrator = Calibrator(info=video_info,
                            output=[video_player])

    # observer
    observer = Observer(info=video_info,
                        output=[video_player])

    # traffic light observer
    traffic_lights_observer = TrafficLightsObserver(info=video_info,
                                                    output=[observer, calibrator])

    # car tracker
    tracker = Tracker(info=video_info,
                      output=[observer, calibrator])

    # car detector
    car_detector = Detector(model=PATH_TO_CAR_MODEL,
                            info=video_info,
                            output=[tracker],
                            detector_type_id=params.DETECTOR_CAR_ID)
    # frame loader
    frame_loader = FrameLoader(info=video_info,
                               output=[car_detector,
                                       video_player,
                                       tracker,
                                       calibrator,
                                       traffic_lights_observer])

    frame_loader.start()
    car_detector.start()

    tracker.start()
    calibrator.start()
    observer.start()
    traffic_lights_observer.start()

    video_player._loader = frame_loader
    video_player._detector = car_detector
    video_player._tracker = tracker

    video_player.start()


if __name__ == '__main__':
    main(sys.argv[1:])
