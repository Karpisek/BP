import numpy as np

# pipeline block id's

VIDEO_PLAYER_ID = 0
FRAME_LOADER_ID = 1
DETECTOR_CAR_ID = 2
TRACKER_ID = 3
CALIBRATOR_ID = 4
DETECTOR_LIGHT_ID = 5


# calibrator
CALIBRATOR_FREQUENCY = 20
CALIBRATOR_FLOW_THRESHOLD = 0
# CALIBRATOR_RANSAC_STEP_POINTS_COUNT = 300 # number of car tracks before estimating
CALIBRATOR_RANSAC_THRESHOLD_RATIO = 0.015    # ransac acceptance of point threshold 0.03 (o.O15)

CALIBRATOR_TRACK_MINIMUM = 10               # not-used 300 (100)
CALIBRATOR_ANGLE_MIN = 30                   # minimal angle for the second vanishing point
CALIBRATOR_GRID_DENSITY = 5                 # density of prepared vanishing point grid 10
CALIBRATOR_MIN_LINE_LENGTH = 50             # 50
CALIBRATOR_MAX_LINE_GAP = 5                 # 5
CALIBRATOR_HLP_THRESHOLD = 20
CALIBRATOR_LIFELINE_THICKNESS = 100

# detector
DETECTOR_FREQUENCY = 3


# tracker
TRACKER_OPTICAL_FLOW_FREQUENCY = 1


# video player
VIDEO_PLAYER_FREQUENCY = 1
VIDEO_PLAYER_SPEED = 1


# frame loader
FRAME_LOADER_SUBTRACTOR_HISTORY = 100
FRAME_LOADER_THRESHOLD = 50


# optical flow
OPTICAL_FLOW_GRID_DENSITY = 11


# bounding box
BOX_HISTORY = 5


# colors
COLOR_RED = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (0, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_LIFELINE = (10, 10, 10)


# helpers
UINT_MIN = -2147483648
UINT_MAX = 2147483647
RANDOM_COLOR = [(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) for _ in range(100)]
