import time

import cv2

import params
from bbox import Area
from pipeline.observer import Observer
from pipeline.traffic_lights_finder import TrafficLightsFinder
from video_stream import FrameLoader, Info, VideoPlayer
from pipeline import Tracker, Calibrator
from detectors import Detector, TrafficLightsObserver

# VIDEO_PATH = "/Users/miro/Desktop/00004.MTS"
# VIDEO_PATH = "/Volumes/Miro/FIT/BP/Dataset/RedLightViolation/right_camera/00002.MTS"
# VIDEO_PATH = "/Volumes/Miro/FIT/BP/Dataset/RedLightViolation/zoom_camera/00002.MTS"
# VIDEO_PATH = "/Users/miro/Desktop/v2.mp4"q

# VIDEO_PATH = "/Volumes/Miro/FIT/BP/Dataset/iARTIST_crossroads/1.mp4"
# VIDEO_PATH = "/Volumes/Miro/FIT/BP/Dataset/iARTIST_crossroads/1a.mp4"
# VIDEO_PATH = "/Volumes/Miro/FIT/BP/Dataset/iARTIST_crossroads/1b.mp4"
# VIDEO_PATH = "/Volumes/Miro/FIT/BP/Dataset/iARTIST_crossroads/2.mp4"
# VIDEO_PATH = "/Volumes/Miro/FIT/BP/Dataset/iARTIST_crossroads/2a.mp4"
# VIDEO_PATH = "/Volumes/Miro/FIT/BP/Dataset/iARTIST_crossroads/2b.mp4"
VIDEO_PATH = "/Volumes/Miro/FIT/BP/Dataset/iARTIST_crossroads/3.mp4"
# VIDEO_PATH = "/Volumes/Miro/FIT/BP/Dataset/iARTIST_crossroads/3a.mp4"
# VIDEO_PATH = "/Volumes/Miro/FIT/BP/Dataset/iARTIST_crossroads/3b.mp4"

PATH_TO_CAR_MODEL = 'detectors/models/car_detectors/small_longer.pb'
PATH_TO_LIGHTS_MODEL = 'detectors/models/traffic_light_detectors/bad_detector.pb'

video_info = Info(VIDEO_PATH)

# video playback
video_player = VideoPlayer(info=video_info)

# calibrator
calibrator = Calibrator(info=video_info,
                        output=[video_player])

# TODO!!! tweak

tape = cv2.VideoCapture(VIDEO_PATH)
_, image = tape.read()
roi_light = cv2.selectROI("image", image)
cv2.destroyAllWindows()
tape.release()

# observer
observer = Observer(info=video_info,
                    output=[video_player])

# traffic light observer
traffic_lights_observer = TrafficLightsObserver(info=video_player,
                                                output=[observer],
                                                tweak_roi=roi_light)

# car tracker
tracker = Tracker(info=video_info,
                  output=[calibrator, observer],
                  calibrator=calibrator)

# traffic lights finder
traffic_lights_finder = TrafficLightsFinder(info=video_info)

# car detector
car_detector = Detector(model=PATH_TO_CAR_MODEL,
                        info=video_info,
                        output=[tracker],
                        detector_type_id=params.DETECTOR_CAR_ID)

# lights detector TODO !!!
light_detector = Detector(model=PATH_TO_LIGHTS_MODEL,
                          info=video_info,
                          output=[traffic_lights_finder],
                          detector_type_id=params.DETECTOR_LIGHT_ID,
                          block=False)

# frame loader
frame_loader = FrameLoader(info=video_info,
                           output=[car_detector, light_detector, video_player, tracker, calibrator, traffic_lights_observer])


frame_loader.start()
car_detector.start()
light_detector.start()
tracker.start()
calibrator.start()
traffic_lights_finder.start()
observer.start()
traffic_lights_observer.start()

video_player._loader = frame_loader
video_player._detector = car_detector
video_player._tracker = tracker

video_player.start()
