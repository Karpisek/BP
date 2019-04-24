import numpy as np

# pipeline block id's
from pipeline.base.pipeline import Mode

VIDEO_PLAYER_ID = 0
FRAME_LOADER_ID = 1
DETECTOR_CAR_ID = 2
TRACKER_ID = 3
CALIBRATOR_ID = 4
DETECTOR_LIGHT_ID = 5
LIGHT_FINDER_ID = 6
OBSERVER_ID = 7
TRAFFIC_LIGHT_OBSERVER_ID = 8
VIDEO_WRITER_ID = 9
VIOLATION_WRITER_ID = 10

# traffic violation writer
VIOLATION_WRITER_FREQUENCY = 1
VIOLATION_WRITER_WORKMODES = [Mode.DETECTION]

# calibrator
CALIBRATOR_FREQUENCY = 2  # 50
CALIBRATOR_RANSAC_THRESHOLD_RATIO = 0.01   # ransac acceptance of point threshold 0.03 (o.O15)

CALIBRATOR_VP1_TRACK_MINIMUM = 20
CALIBRATOR_VP2_TRACK_MINIMUM = 30

CALIBRATOR_MIN_LINE_LENGTH = 10             # 30
CALIBRATOR_MAX_LINE_GAP = 0                # 5
CALIBRATOR_HLP_THRESHOLD = 50
CALIBRATOR_LIFELINE_THICKNESS = 100

# corridors
CORRIDORS_MINIMUM_LIFELINES = 50
CORRIDORS_STOP_POINTS_MINIMAL = 5
CORRIDORS_RANSAC_THRESHOLD = 10

CORRIDORS_LINE_SELECTOR_THICKNESS = 3

# detector
DETECTOR_CAR_FREQUENCY = 3
DETECTOR_CAR_CLASSES_IDS = [2, 3, 4, 6, 8]
DETECTOR_MINIMAL_SCORE = 0.1

DETECTOR_LIGHT_MINIMAL_SCORE = 0.8
DETECTOR_LIGHT_IMAGE_ROW = 0

DETECTOR_IOU_THRESHOLD = 0.5
DETECTOR_NON_MAX_SUPPRESSION_COUNT = 10
DETECTOR_IMAGE_WIDTH = 640

# tracker
TRACKER_OPTICAL_FLOW_FREQUENCY = 1
TRACKER_DISALLOWED = 10000
TRACKER_LIFELINE = 20
TRACKER_MINIMAL_SCORE = 0.5
TRACKER_SUPPRESSION_MIN = 0.5


# video player
VIDEO_PLAYER_FREQUENCY = 1
VIDEO_PLAYER_SPEED = 10

# video writer
VIDEO_WRITER_FREQUENCY = 1

# frame loader
FRAME_LOADER_SUBTRACTOR_HISTORY = 100
FRAME_LOADER_THRESHOLD = 50
FRAME_LOADER_MAX_WIDTH = 1100


# optical flow
OPTICAL_FLOW_GRID_DENSITY = 7
OPTICAL_FLOW_COLOR = (200, 200, 50)


# traffic light
TRAFFIC_LIGHT_DEFAULT_THRESHOLD = 0.3
TRAFFIC_LIGHT_GREEN_THRESHOLD = 0.4
TRAFFIC_LIGHT_FINDER_DEFAULT_VALUE = 10
TRAFFIC_LIGHT_MINIMAL_SCORE = 0.7
TRAFFIC_LIGHT_GREEN = 1
TRAFFIC_LIGHT_ORANGE = 2
TRAFFIC_LIGHT_RED = 3
TRAFFIC_LIGHT_OBSERVER_FREQUENCY = 1

# observer
OBSERVER_RED_STANDER_MAX_TRAVEL = 10
OBSERVER_FREQUENCY = 1
OBSERVER_BOX_THICKNESS = 3

# colors
COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (255, 0, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_WHITE_MONO = 255
COLOR_BLACK = (0, 0, 0)
COLOR_LIFELINE = (10, 10, 10)
COLOR_ORANGE = (66, 212, 244)
COLOR_GRAY = (50, 50, 50)
COLOR_YELLOW = (32, 244, 66)

FILL = -1
DEFAULT_THICKNESS = 1

STATISTICS_LOG_FILENAME = "statistics.log"
CALIBRATION_FILENAME = "calibration.log"


# helpers
UINT_MIN = -2147483648
UINT_MAX = 2147483647
RANDOM_COLOR = [(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) for _ in range(100)]
COLOR_VANISHING_DIRECTIONS = [COLOR_GREEN, COLOR_BLUE, COLOR_RED]
