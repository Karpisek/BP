import time

import cv2

import params
from bbox import Area
from video_stream import FrameLoader, Info, VideoPlayer, Tracker, Calibrator
from detectors import Detector

# VIDEO_PATH = "/Users/miro/Desktop/00004.MTS"
# VIDEO_PATH = "/Volumes/Miro/FIT/BP/Dataset/RedLightViolation/right_camera/00002.MTS"
# VIDEO_PATH = "/Volumes/Miro/FIT/BP/Dataset/RedLightViolation/zoom_camera/00002.MTS"
# VIDEO_PATH = "/Users/miro/Desktop/v2.mp4"q

# VIDEO_PATH = "/Volumes/Miro/FIT/BP/Dataset/iARTIST_crossroads/1.mp4"
VIDEO_PATH = "/Volumes/Miro/FIT/BP/Dataset/iARTIST_crossroads/1a.mp4"
# VIDEO_PATH = "/Volumes/Miro/FIT/BP/Dataset/iARTIST_crossroads/1b.mp4"
# VIDEO_PATH = "/Volumes/Miro/FIT/BP/Dataset/iARTIST_crossroads/2.mp4"
# VIDEO_PATH = "/Volumes/Miro/FIT/BP/Dataset/iARTIST_crossroads/2a.mp4"
# VIDEO_PATH = "/Volumes/Miro/FIT/BP/Dataset/iARTIST_crossroads/2b.mp4"
# VIDEO_PATH = "/Volumes/Miro/FIT/BP/Dataset/iARTIST_crossroads/3.mp4"
# VIDEO_PATH = "/Volumes/Miro/FIT/BP/Dataset/iARTIST_crossroads/3a.mp4"
# VIDEO_PATH = "/Volumes/Miro/FIT/BP/Dataset/iARTIST_crossroads/3b.mp4"

PATH_TO_CAR_MODEL = 'detectors/models/small_longer.pb'
PATH_TO_LIGHTS_MODEL = 'detectors/models/small_longer.pb'

input_info = Info(VIDEO_PATH)
area_of_detection = Area()

# video playback
video_player = VideoPlayer(area_of_detection=area_of_detection,
                           info=input_info)

# calibrator
calibrator = Calibrator(info=input_info,
                        output=[video_player])

video_player.calibrator = calibrator

# foreground detector
tracker = Tracker(area_of_detection=area_of_detection,
                  info=input_info,
                  output=[video_player, calibrator],
                  calibrator=calibrator)

# car detector
car_detector = Detector(model=PATH_TO_CAR_MODEL,
                        detection_area=area_of_detection,
                        output=[tracker],
                        detector_type_id=params.DETECTOR_CAR_ID)

# lights detector TODO !!!
light_detector = Detector(model=PATH_TO_LIGHTS_MODEL,
                          detection_area=area_of_detection,
                          output=[tracker],
                          detector_type_id=params.DETECTOR_LIGHT_ID)


# frame loader
frame_loader = FrameLoader([car_detector, video_player, tracker, calibrator], input_info)

frame_loader.start()
car_detector.start()
tracker.start()
calibrator.start()

video_player._loader = frame_loader
video_player._detector = car_detector
video_player._tracker = tracker

video_player.start()
