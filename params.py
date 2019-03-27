# pipeline block id's
VIDEO_PLAYER_ID = 0
FRAME_LOADER_ID = 1
DETECTOR_ID = 2
TRACKER_ID = 3
CALIBRATOR_ID = 4


# calibrator
CALIBRATOR_FREQUENCY = 100
CALIBRATOR_FLOW_THRESHOLD = 0
CALIBRATOR_RANSAC_STEP_POINTS_COUNT = 100   # number of car tracks before estimating
CALIBRATOR_RANSAC_THRESHOLD_RATIO = 0.03    # ransac acceptance of point threshold

CALIBRATOR_TRACK_MINIMUM = 100              # not-used
CALIBRATOR_ANGLE_MIN = 30                   # minimal angle for the second vanishing point
CALIBRATOR_GRID_DENSITY = 10                # density of prepared vanishing point grid

# detector
DETECTOR_FREQUENCY = 3


# tracker
TRACKER_OPTICAL_FLOW_FREQUENCY = 1


# video player
VIDEO_PLAYER_FREQUENCY = 1
VIDEO_PLAYER_SPEED = 1


# frame loader
FRAME_LOADER_SUBTRACTOR_HISTORY = 10
FRAME_LOADER_THRESHOLD = 50


# optical flow
OPTICAL_FLOW_GRID_DENSITY = 11


# bounding box
BOX_HISTORY = 5


# colors
COLOR_RED = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (0, 0, 255)


# helpers
UINT_MIN = -2147483648
UINT_MAX = 2147483647
