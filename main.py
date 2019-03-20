from bbox import Area
from video_stream import FrameLoader, Info, VideoPlayer, Tracker
from detectors import Detector

VIDEO_PATH = "/Users/miro/Desktop/00004.MTS"
# VIDEO_PATH = "/Volumes/Miro/FIT/BP/Dataset/RedLightViolation/right_camera/00002.MTS"
# VIDEO_PATH = "/Volumes/Miro/FIT/BP/Dataset/RedLightViolation/zoom_camera/00002.MTS"
# VIDEO_PATH = "/Users/miro/Desktop/v2.mp4"

PATH_TO_MODEL = 'detectors/models/small_longer.pb'


input_info = Info()
area_of_detection = Area()

# video playback
video_player = VideoPlayer(area_of_detection=area_of_detection, info=input_info)

# foreground detector
tracker = Tracker(area_of_detection=area_of_detection, info=input_info, output=[video_player])

# cardetector
detector = Detector(model=PATH_TO_MODEL, detection_area=area_of_detection, output=[tracker])

# calibrator
# calibrator = Calibrator([video_player])

# frame loader
frame_loader = FrameLoader(VIDEO_PATH, [detector, video_player, tracker], input_info)

video_player._loader = frame_loader
video_player._detector = detector
video_player._tracker = tracker

video_player.start()