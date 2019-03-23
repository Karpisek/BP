import time

import cv2

from bbox import Area
from video_stream import FrameLoader, Info, VideoPlayer, Tracker, Calibrator
from detectors import Detector

VIDEO_PATH = "/Users/miro/Desktop/00004.MTS"
# VIDEO_PATH = "/Volumes/Miro/FIT/BP/Dataset/RedLightViolation/right_camera/00002.MTS"
# VIDEO_PATH = "/Volumes/Miro/FIT/BP/Dataset/RedLightViolation/zoom_camera/00002.MTS"
# VIDEO_PATH = "/Users/miro/Desktop/v2.mp4"

# VIDEO_PATH = "/Volumes/Miro/FIT/BP/Dataset/iARTIST_crossroads/1.mp4"

# VIDEO_PATH = "/Volumes/Miro/FIT/BP/Dataset/iARTIST_crossroads/2.mp4"
# VIDEO_PATH = "/Volumes/Miro/FIT/BP/Dataset/iARTIST_crossroads/3.mp4"

PATH_TO_MODEL = 'detectors/models/small_longer.pb'

input_info = Info(VIDEO_PATH)
area_of_detection = Area()

# video playback
video_player = VideoPlayer(area_of_detection=area_of_detection, info=input_info)

# calibrator
calibrator = Calibrator(info=input_info)

# foreground detector
tracker = Tracker(area_of_detection=area_of_detection, info=input_info, output=[video_player, calibrator])

# car detector
detector = Detector(model=PATH_TO_MODEL, detection_area=area_of_detection, output=[tracker])

# frame loader
frame_loader = FrameLoader([detector, video_player, tracker, calibrator], input_info)

frame_loader.start()
detector.start()
tracker.start()
calibrator.start()

video_player._loader = frame_loader
video_player._detector = detector
video_player._tracker = tracker

video_player.start()
