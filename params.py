import numpy as np

# pipeline block id's

VIDEO_PLAYER_ID = 0
FRAME_LOADER_ID = 1
DETECTOR_CAR_ID = 2
TRACKER_ID = 3
CALIBRATOR_ID = 4
DETECTOR_LIGHT_ID = 5
LIGHT_FINDER_ID = 6
OBSERVER_ID = 7


# calibrator
CALIBRATOR_FREQUENCY = 40
CALIBRATOR_FLOW_THRESHOLD = 0
CALIBRATOR_RANSAC_THRESHOLD_RATIO = 0.015   # ransac acceptance of point threshold 0.03 (o.O15)

CALIBRATOR_VP1_TRACK_MINIMUM = 100          # not-used 300 (100)
CALIBRATOR_VP2_TRACK_MINIMUM = 100
CALIBRATOR_ANGLE_MIN = 30                   # minimal angle for the second vanishing point
CALIBRATOR_GRID_DENSITY = 5                 # density of prepared vanishing point grid 10
CALIBRATOR_MIN_LINE_LENGTH = 30             # 50
CALIBRATOR_MAX_LINE_GAP = 2                 # 5
CALIBRATOR_HLP_THRESHOLD = 20
CALIBRATOR_LIFELINE_THICKNESS = 100

# detector
DETECTOR_CAR_FREQUENCY = 2
DETECTOR_LIGHT_FREQUENCY = 30
DETECTOR_MINIMAL_SCORE = 0.5
DETECTOR_IOU_THRESHOLD = 0.5
DETECTOR_NON_MAX_SUPPRESSION_COUNT = 10


# tracker
TRACKER_OPTICAL_FLOW_FREQUENCY = 1
TRACKER_DISALLOWED = 10000
TRACKER_LIFELINE = 20
TRACKER_MINIMAL_SCORE = 0.5
TRACKER_SUPPRESSION_MIN = 0.5


# video player
VIDEO_PLAYER_FREQUENCY = 1
VIDEO_PLAYER_SPEED = 10


# frame loader
FRAME_LOADER_SUBTRACTOR_HISTORY = 100
FRAME_LOADER_THRESHOLD = 50


# optical flow
OPTICAL_FLOW_GRID_DENSITY = 11
OPTICAL_FLOW_COLOR = (200, 200, 50)


# traffic light
TRAFFIC_LIGHT_FINDER_DEFAULT_VALUE = 10
TRAFFIC_LIGHT_MINIMAL_SCORE = 0.7
TRAFFIC_LIGHT_GREEN = 1
TRAFFIC_LIGHT_ORANGE = 2
TRAFFIC_LIGHT_RED = 3

# observer
OBSERVER_FREQUENCY = 1

# colors
COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (255, 0, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_WHITE_MONO = 255
COLOR_BLACK = (0, 0, 0)
COLOR_LIFELINE = (10, 10, 10)

FILL = -1
DEFAULT_THICKNESS = 1


# helpers
UINT_MIN = -2147483648
UINT_MAX = 2147483647
RANDOM_COLOR = [(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) for _ in range(100)]

